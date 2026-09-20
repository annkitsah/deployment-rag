import logging
import shutil
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_container
from app.api.schemas import DocumentListResponse, DocumentResponse
from app.container import ApplicationContainer
from app.ingestion.progress import progress_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentProgressResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    status: str
    total_pages: int = Field(ge=0)
    processed_pages: int = Field(ge=0)
    percent: int = Field(ge=0, le=100)
    message: str = ""
    error: str | None = None


def _friendly_ingest_error(exc: Exception) -> tuple[int, str]:
    msg = str(exc)
    lower = msg.lower()
    if "429" in msg or "rate" in lower or "rate_limited" in lower:
        return (
            status.HTTP_429_TOO_MANY_REQUESTS,
            "OCR rate limit reached (Mistral). Wait a minute and retry, "
            "or upload a smaller PDF (fewer pages).",
        )
    if "timeout" in lower or "timed out" in lower:
        return (
            status.HTTP_504_GATEWAY_TIMEOUT,
            "OCR timed out. Try a smaller PDF.",
        )
    if "api key" in lower or "unauthorized" in lower or "401" in msg:
        return (
            status.HTTP_502_BAD_GATEWAY,
            "OCR provider authentication failed. Check MISTRAL_API_KEY on the server.",
        )
    return (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"Failed to ingest the document: {msg[:300]}",
    )


def _run_ingest_job(
    container: ApplicationContainer,
    document_id: str,
    file_path: Path,
) -> None:
    try:
        container.ingestion_service.process_registered(
            document_id=document_id,
            file_path=file_path,
        )
    except Exception:
        logger.exception("Background ingest failed for %s", document_id)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
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

    max_mb = container.settings.max_upload_mb
    size_bytes = destination.stat().st_size
    if size_bytes > max_mb * 1024 * 1024:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({size_bytes / (1024 * 1024):.1f} MB). Max {max_mb} MB.",
        )

    max_pages = container.settings.max_upload_pages
    try:
        with pymupdf.open(destination) as doc:
            page_count = len(doc)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {exc}") from exc

    if page_count > max_pages:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"This PDF has {page_count} pages. Maximum allowed is {max_pages}.",
        )
    if page_count == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="PDF has no pages.")

    try:
        result = container.ingestion_service.register_document(
            destination,
            original_filename=file.filename,
            total_pages=page_count,
        )
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        code, detail = _friendly_ingest_error(exc)
        raise HTTPException(status_code=code, detail=detail) from exc

    if result.duplicate:
        destination.unlink(missing_ok=True)
        return DocumentResponse(
            document_id=result.document.document_id,
            filename=result.document.filename,
            page_count=result.document.page_count,
            status=result.document.status,
            created_at=result.document.created_at,
            duplicate=True,
        )

    background_tasks.add_task(
        _run_ingest_job,
        container,
        result.document.document_id,
        destination,
    )

    return DocumentResponse(
        document_id=result.document.document_id,
        filename=result.document.filename,
        page_count=result.document.page_count,
        status=result.document.status,
        created_at=result.document.created_at,
        duplicate=False,
    )


@router.get("/{document_id}/progress", response_model=DocumentProgressResponse)
async def get_document_progress(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> DocumentProgressResponse:
    item = progress_registry.get(document_id)
    document = container.repository.get_by_id(document_id)

    if item is None and document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    if item is not None:
        return DocumentProgressResponse(
            document_id=document_id,
            status=item.status,
            total_pages=item.total_pages,
            processed_pages=item.processed_pages,
            percent=item.percent,
            message=item.message,
            error=item.error,
        )

    assert document is not None
    status_value = document.status.value
    percent = 100 if status_value == "processed" else (0 if status_value == "failed" else 50)
    return DocumentProgressResponse(
        document_id=document_id,
        status=status_value,
        total_pages=document.page_count,
        processed_pages=document.page_count if status_value == "processed" else 0,
        percent=percent,
        message=status_value,
        error=None,
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    try:
        container.index_lifecycle.remove_document(document_id)
    except Exception:
        logger.exception("Failed removing document from index: %s", document_id)

    try:
        container.page_store.delete_document(document_id)
    except Exception:
        logger.exception("Failed deleting page files: %s", document_id)

    try:
        source = Path(document.source_path)
        raw_root = Path(container.settings.raw_data_dir).resolve()
        if source.exists() and str(source.resolve()).startswith(str(raw_root)):
            source.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed deleting source file for %s", document_id)

    container.repository.delete(document_id)