import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.dependencies import get_container
from app.api.schemas import DocumentListResponse, DocumentResponse
from app.container import ApplicationContainer

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile,
    container: ApplicationContainer = Depends(get_container),
) -> DocumentResponse:
    """Upload and ingest a PDF document."""

    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest the uploaded document.",
        ) from exc

    if result.duplicate:
        # The content already exists under a different stored copy;
        # this upload's copy is redundant, so don't leave it on disk.
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
    """List every ingested document."""

    documents = container.repository.list_all()

    return DocumentListResponse(
        documents=tuple(
            DocumentResponse(
                document_id=document.document_id,
                filename=document.filename,
                page_count=document.page_count,
                status=document.status,
                created_at=document.created_at,
            )
            for document in documents
        ),
        count=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> DocumentResponse:
    """Retrieve a single document's metadata by ID."""

    document = container.repository.get_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        status=document.status,
        created_at=document.created_at,
    )