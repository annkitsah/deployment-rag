from pathlib import Path
import tempfile

from app.config.settings import get_settings
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.ocr.pipeline import OCRPipeline
from app.ocr.providers.mistral import MistralOCRProvider


PDF_NAME = "Chapter 1 - The Overview of Map of GenAI.pdf"


def main() -> None:
    settings = get_settings()

    if not settings.mistral_api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not configured."
        )

    pdf_path = Path("data/raw/Chapter 1 - The Overview of Map of GenAI.pdf")

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    print("=== Module 2.9: End-to-End Ingestion Smoke Test ===")
    print(f"PDF: {pdf_path}")
    print(f"PDF size: {pdf_path.stat().st_size:,} bytes")
    print(f"OCR model: {settings.mistral_ocr_model}")

    provider = MistralOCRProvider(
        api_key=settings.mistral_api_key,
        model=settings.mistral_ocr_model,
        timeout_ms=settings.mistral_ocr_timeout_ms,
    )

    pipeline = OCRPipeline(
        provider=provider,
    )

    with tempfile.TemporaryDirectory(
        prefix="agentic-rag-smoke-"
    ) as temp_dir:
        temp_root = Path(temp_dir)

        repository = DocumentRepository(
            database_path=temp_root / "metadata" / "documents.db",
        )

        repository.initialize()

        page_store = PageStore(
            root_dir=temp_root / "pages",
        )

        ingestion_service = IngestionService(
            repository=repository,
            page_store=page_store,
            ocr_pipeline=pipeline,
            processed_root=temp_root / "processed",
            ocr_dpi=200,
        )

        # ---------------------------------------------------------
        # First ingestion
        # ---------------------------------------------------------

        print("\n[1/5] Ingesting document...")

        first_result = ingestion_service.ingest(
            pdf_path,
        )

        print("      Ingestion completed.")

        # ---------------------------------------------------------
        # Document validation
        # ---------------------------------------------------------

        print("\n[2/5] Validating document...")

        document = first_result.document

        print(f"      Document ID: {document.document_id}")
        print(f"      Filename: {document.filename}")
        print(f"      Pages: {document.page_count}")
        print(f"      Status: {document.status}")
        print(f"      Duplicate: {first_result.duplicate}")

        if first_result.duplicate:
            raise RuntimeError(
                "First ingestion was unexpectedly detected as a duplicate."
            )

        if document.page_count <= 0:
            raise RuntimeError(
                "Document contains zero pages."
            )

        # ---------------------------------------------------------
        # Page persistence validation
        # ---------------------------------------------------------

        print("\n[3/5] Validating persisted pages...")

        pages = page_store.get_pages(
            document.document_id,
        )

        print(f"      Persisted pages: {len(pages)}")

        if len(pages) != document.page_count:
            raise RuntimeError(
                "Persisted page count does not match "
                "DocumentRecord.page_count."
            )

        for page in pages:
            print(
                f"      Page {page.page_number}: "
                f"{len(page.text):,} characters"
            )

            if page.document_id != document.document_id:
                raise RuntimeError(
                    f"Page {page.page_number} has an incorrect "
                    "document_id."
                )

            if page.page_number < 1:
                raise RuntimeError(
                    f"Invalid page number: {page.page_number}"
                )

            if not page.text.strip():
                print(
                    f"      Page {page.page_number}: "
                    "no text (blank page preserved)."
                )

            if page.width <= 0:
                raise RuntimeError(
                    f"Page {page.page_number} has invalid width."
                )

            if page.height <= 0:
                raise RuntimeError(
                    f"Page {page.page_number} has invalid height."
                )

        # ---------------------------------------------------------
        # Repository validation
        # ---------------------------------------------------------

        print("\n[4/5] Validating repository...")

        stored_document = repository.get_by_id(
            document.document_id,
        )

        if stored_document is None:
            raise RuntimeError(
                "Document was not persisted in the repository."
            )

        if stored_document.document_id != document.document_id:
            raise RuntimeError(
                "Repository returned a different document ID."
            )

        if stored_document.file_hash != document.file_hash:
            raise RuntimeError(
                "Stored document hash does not match."
            )

        if stored_document.page_count != len(pages):
            raise RuntimeError(
                "Repository page count does not match "
                "persisted page count."
            )

        print("      Repository validation passed.")

        # ---------------------------------------------------------
        # Idempotency validation
        # ---------------------------------------------------------

        print("\n[5/5] Testing idempotency...")

        second_result = ingestion_service.ingest(
            pdf_path,
        )

        if not second_result.duplicate:
            raise RuntimeError(
                "Second ingestion should have been detected "
                "as a duplicate."
            )

        if (
            second_result.document.document_id
            != document.document_id
        ):
            raise RuntimeError(
                "Duplicate ingestion returned a different "
                "document ID."
            )

        if second_result.page_count != document.page_count:
            raise RuntimeError(
                "Duplicate ingestion returned a different "
                "page count."
            )

        pages_after_duplicate = page_store.get_pages(
            document.document_id,
        )

        if len(pages_after_duplicate) != len(pages):
            raise RuntimeError(
                "Duplicate ingestion changed the persisted "
                "page count."
            )

        print("      Idempotency validation passed.")

        print("\n=== SUCCESS ===")
        print(
            "End-to-end document ingestion is working correctly."
        )


if __name__ == "__main__":
    main()