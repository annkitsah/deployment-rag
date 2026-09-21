import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.dependencies import get_container
from app.api.schemas import DocumentListResponse, DocumentResponse
from app.container import ApplicationContainer
from app.api.auth import require_app_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"],
    dependencies=[Depends(require_app_password)],)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile,
    container: ApplicationContainer = Depends(get_container),
) -> DocumentResponse:
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw_dir = Path(container.settings.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{uuid4().hex}_{file.filename}"

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    try:
        result = container.ingestion_service.ingest(
            destination,
            original_filename=file.filename,
        )
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    if result.duplicate:
        destination.unlink(missing_ok=True)

    return DocumentResponse(
        document_id=result.document.document_id,
        filename=result.document.filename,
        page_count=result.document.page_count,
        status=result.document.status,
        created_at=result.document.created_at,
        duplicate=result.duplicate,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    container: ApplicationContainer = Depends(get_container),
) -> DocumentListResponse:
    documents = container.repository.list_all()
    return DocumentListResponse(
        documents=tuple(
            DocumentResponse(
                document_id=d.document_id,
                filename=d.filename,
                page_count=d.page_count,
                status=d.status,
                created_at=d.created_at,
            )
            for d in documents
        ),
        count=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> DocumentResponse:
    document = container.repository.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        status=document.status,
        created_at=document.created_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> None:
    document = container.repository.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    try:
        container.index_lifecycle.remove_document(document_id)
    except Exception:
        logger.exception("Index remove failed for %s", document_id)

    try:
        container.page_store.delete_document(document_id)
    except Exception:
        logger.exception("Page store delete failed for %s", document_id)

    try:
        source = Path(document.source_path)
        raw_root = Path(container.settings.raw_data_dir).resolve()
        if source.exists() and str(source.resolve()).startswith(str(raw_root)):
            source.unlink(missing_ok=True)
    except Exception:
        logger.exception("Source file delete failed for %s", document_id)

    if not hasattr(container.repository, "delete"):
        raise HTTPException(status_code=500, detail="repository.delete is not deployed")
    container.repository.delete(document_id)