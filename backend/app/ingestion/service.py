import time
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
        ocr_chunk_size: int = 5,
        ocr_chunk_pause_ms: int = 1500,
    ) -> None:
        self.repository = repository
        self.page_store = page_store
        self.ocr_pipeline = ocr_pipeline
        self.processed_root = processed_root
        self.ocr_dpi = ocr_dpi
        self.index_lifecycle = index_lifecycle
        self.ocr_chunk_size = ocr_chunk_size
        self.ocr_chunk_pause_ms = ocr_chunk_pause_ms

    def ingest(
        self,
        file_path: Path,
        *,
        original_filename: str | None = None,
    ) -> IngestionResult:
        """Synchronous ingest (legacy / tests). Prefer register + process_async."""

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
        """Register document metadata quickly; OCR can run later."""

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

        progress_registry.start(
            document_id,
            document.filename,
            total_pages=total_pages,
        )

        return IngestionResult(
            document=document,
            page_count=total_pages,
            duplicate=False,
        )

    def process_registered(
        self,
        *,
        document_id: str,
        file_path: Path,
    ) -> IngestionResult:
        """Run OCR/index for a document already marked PROCESSING."""

        document = self.repository.get_by_id(document_id)
        if document is None:
            raise ValueError(f"Unknown document: {document_id}")

        try:
            progress_registry.update(
                document_id,
                message="Extracting / OCR pages…",
            )

            if self.ocr_pipeline is not None:
                if self.processed_root is None:
                    raise RuntimeError(
                        "processed_root is required when OCR pipeline is enabled"
                    )

                with pymupdf.open(file_path) as doc:
                    page_count = len(doc)

                # Resume: skip pages already on disk
                already = {
                    n
                    for n in range(1, page_count + 1)
                    if self.page_store.page_exists(document_id, n)
                }

                progress_registry.update(
                    document_id,
                    total_pages=page_count,
                    processed_pages=len(already),
                    message=(
                        f"Resuming OCR ({len(already)}/{page_count} pages already done)…"
                        if already
                        else f"Processing {page_count} pages in chunks…"
                    ),
                )

                chunk_size = max(1, int(self.ocr_chunk_size))
                chunk_pause_sec = max(0.0, float(self.ocr_chunk_pause_ms) / 1000.0)

                pending = [n for n in range(1, page_count + 1) if n not in already]
                done_count = len(already)

                for i in range(0, len(pending), chunk_size):
                    chunk = pending[i : i + chunk_size]

                    for page_number in chunk:
                        progress_registry.update(
                            document_id,
                            processed_pages=done_count,
                            message=f"Processing page {page_number} of {page_count} (native text first)…",
                        )
                        result = self.ocr_pipeline.process_page(
                            pdf_path=file_path,
                            document_id=document_id,
                            page_number=page_number,
                            output_root=self.processed_root,
                            dpi=self.ocr_dpi,
                        )
                        page_rec = PageRecord(
                            document_id=document_id,
                            page_number=result.page_number,
                            text=result.text,
                            width=result.classification.content.width,
                            height=result.classification.content.height,
                        )
                        self.page_store.save_page(page_rec)
                        if self.index_lifecycle is not None:
                            self.index_lifecycle.index_page(page_rec)
                        done_count += 1
                        progress_registry.update(
                            document_id,
                            processed_pages=done_count,
                            message=f"Finished page {page_number} of {page_count}",
                        )

                    if self.index_lifecycle is not None:
                        self.index_lifecycle.save()

                    # Pause between chunks to reduce OCR rate limits
                    remaining = len(pending) - (i + len(chunk))
                    if remaining > 0 and chunk_pause_sec > 0:
                        progress_registry.update(
                            document_id,
                            message=f"Pausing {chunk_pause_sec:.1f}s before next chunk…",
                        )
                        time.sleep(chunk_pause_sec)

                pages = self.page_store.get_pages(document_id)
            else:
                pages = extract_pages(file_path, document_id)
                self.page_store.save_pages(pages)
                if self.index_lifecycle is not None:
                    for page in pages:
                        self.index_lifecycle.index_page(page)
                    self.index_lifecycle.save()
                progress_registry.update(
                    document_id,
                    total_pages=len(pages),
                    processed_pages=len(pages),
                    message="Native text extracted",
                )

            progress_registry.update(
                document_id,
                message="Finalizing index…",
            )
            if self.index_lifecycle is not None:
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
                document=updated,
                page_count=len(pages),
                duplicate=False,
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