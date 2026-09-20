from pathlib import Path
from uuid import uuid4

import pymupdf
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
from app.ingestion.progress import progress_registry
from app.ocr.pipeline import OCRPipeline
from app.retrieval.index_lifecycle import IndexLifecycle


class IngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: DocumentRecord
    page_count: int
    duplicate: bool


class IngestionService:
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
        registered = self.register_document(
            file_path,
            original_filename=original_filename,
        )
        if registered.duplicate:
            return registered
        return self.process_registered(
            document_id=registered.document.document_id,
            file_path=file_path,
        )

    def register_document(
        self,
        file_path: Path,
        *,
        original_filename: str | None = None,
        total_pages: int = 0,
    ) -> IngestionResult:
        if not file_path.is_file():
            raise FileNotFoundError(f"Document does not exist: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported document type: {file_path.suffix}")

        file_hash = calculate_file_sha256(file_path)
        existing_document = self.repository.get_by_hash(file_hash)
        if existing_document is not None:
            return IngestionResult(
                document=existing_document,
                page_count=existing_document.page_count,
                duplicate=True,
            )

        document_id = f"doc_{uuid4().hex}"
        document = DocumentRecord(
            document_id=document_id,
            filename=original_filename or file_path.name,
            source_path=str(file_path.resolve()),
            file_hash=file_hash,
            file_size_bytes=file_path.stat().st_size,
            page_count=total_pages,
            status=DocumentStatus.PROCESSING,
        )
        self.repository.add(document)
        progress_registry.start(document_id, document.filename, total_pages=total_pages)
        return IngestionResult(document=document, page_count=total_pages, duplicate=False)

    def process_registered(
        self,
        *,
        document_id: str,
        file_path: Path,
    ) -> IngestionResult:
        document = self.repository.get_by_id(document_id)
        if document is None:
            raise ValueError(f"Unknown document: {document_id}")

        try:
            progress_registry.update(document_id, message="Extracting / OCR pages…")

            if self.ocr_pipeline is not None:
                if self.processed_root is None:
                    raise RuntimeError(
                        "processed_root is required when OCR pipeline is enabled"
                    )

                with pymupdf.open(file_path) as doc:
                    page_count = len(doc)

                progress_registry.update(
                    document_id,
                    total_pages=page_count,
                    message=f"Processing {page_count} pages…",
                )

                page_results = []
                for page_number in range(1, page_count + 1):
                    progress_registry.update(
                        document_id,
                        processed_pages=page_number - 1,
                        message=f"OCR page {page_number} of {page_count}…",
                    )
                    page_results.append(
                        self.ocr_pipeline.process_page(
                            pdf_path=file_path,
                            document_id=document_id,
                            page_number=page_number,
                            output_root=self.processed_root,
                            dpi=self.ocr_dpi,
                        )
                    )
                    progress_registry.update(
                        document_id,
                        processed_pages=page_number,
                        message=f"Finished page {page_number} of {page_count}",
                    )

                pages = [
                    PageRecord(
                        document_id=document_id,
                        page_number=page.page_number,
                        text=page.text,
                        width=page.classification.content.width,
                        height=page.classification.content.height,
                    )
                    for page in page_results
                ]
            else:
                pages = extract_pages(file_path, document_id)
                progress_registry.update(
                    document_id,
                    total_pages=len(pages),
                    processed_pages=len(pages),
                    message="Native text extracted",
                )

            progress_registry.update(
                document_id, message="Saving pages and building index…"
            )
            self.page_store.save_pages(pages)

            if self.index_lifecycle is not None:
                for page in pages:
                    self.index_lifecycle.index_page(page)
                self.index_lifecycle.save()

            self.repository.update_status(
                document_id,
                DocumentStatus.PROCESSED,
                page_count=len(pages),
            )
            progress_registry.update(
                document_id,
                status="processed",
                processed_pages=len(pages),
                total_pages=len(pages),
                message="Ingestion complete",
            )

            updated = self.repository.get_by_id(document_id)
            assert updated is not None
            return IngestionResult(
                document=updated, page_count=len(pages), duplicate=False
            )

        except Exception as exc:
            self.repository.update_status(document_id, DocumentStatus.FAILED)
            progress_registry.update(
                document_id,
                status="failed",
                message="Ingestion failed",
                error=str(exc)[:500],
            )
            raise