from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.documents.models import (
    DocumentRecord,
    DocumentStatus,
    PageRecord,
)
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.hashing import calculate_file_sha256
from app.ingestion.pdf_parser import extract_pages
from app.ocr.pipeline import OCRPipeline
from app.retrieval.index_lifecycle import IndexLifecycle


class IngestionResult(BaseModel):
    """Result returned after processing a document."""

    model_config = ConfigDict(frozen=True)

    document: DocumentRecord
    page_count: int
    duplicate: bool


class IngestionService:
    """Coordinates document registration, OCR, and page persistence."""

    def __init__(
        self,
        repository: DocumentRepository,
        page_store: PageStore,
        ocr_pipeline: OCRPipeline | None = None,
        processed_root: Path | None = None,
        ocr_dpi: int | None = None,
        index_lifecycle: IndexLifecycle | None = None,
    ) -> None:
        self.repository = repository
        self.page_store = page_store
        self.ocr_pipeline = ocr_pipeline
        self.processed_root = processed_root
        self.ocr_dpi = ocr_dpi
        self.index_lifecycle = index_lifecycle

    def ingest(
        self,
        file_path: Path,
        *,
        original_filename: str | None = None,
    ) -> IngestionResult:
        """Ingest a PDF and persist its canonical page records.

        `original_filename`, when given, is stored as the document's
        display filename instead of `file_path.name` — useful when the
        caller stores the file on disk under a generated/prefixed name
        (e.g. to avoid collisions) but wants the user-facing filename
        preserved.
        """

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Document does not exist: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported document type: {file_path.suffix}"
            )

        file_hash = calculate_file_sha256(file_path)

        existing_document = self.repository.get_by_hash(file_hash)

        if existing_document is not None:
            return IngestionResult(
                document=existing_document,
                page_count=existing_document.page_count,
                duplicate=True,
            )

        document_id = f"doc_{uuid4().hex}"

        if self.ocr_pipeline is not None:
            if self.processed_root is None:
                raise RuntimeError(
                    "processed_root is required when OCR pipeline is enabled"
                )

            pipeline_result = self.ocr_pipeline.process_document(
                pdf_path=file_path,
                document_id=document_id,
                output_root=self.processed_root,
                dpi=self.ocr_dpi,
            )

            pages = [
                PageRecord(
                    document_id=document_id,
                    page_number=page.page_number,
                    text=page.text,
                    width=page.classification.content.width,
                    height=page.classification.content.height,
                )
                for page in pipeline_result.pages
            ]
        else:
            pages = extract_pages(
                file_path,
                document_id,
            )

        self.page_store.save_pages(pages)

        if self.index_lifecycle is not None:
            for page in pages:
                self.index_lifecycle.index_page(page)
            self.index_lifecycle.save()

        document = DocumentRecord(
            document_id=document_id,
            filename=original_filename or file_path.name,
            source_path=str(file_path.resolve()),
            file_hash=file_hash,
            file_size_bytes=file_path.stat().st_size,
            page_count=len(pages),
            status=DocumentStatus.PROCESSED,
        )

        self.repository.add(document)

        return IngestionResult(
            document=document,
            page_count=len(pages),
            duplicate=False,
        )