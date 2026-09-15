from pathlib import Path

import pymupdf

from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.ocr.pipeline import OCRPipeline
from app.ocr.providers.mock import MockOCRProvider


def create_native_pdf(path: Path) -> None:
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "This is reliable native PDF text with enough content "
        "to satisfy the page quality classifier threshold.",
    )

    document.save(path)
    document.close()


def create_scanned_pdf(path: Path) -> None:
    source = pymupdf.open()

    source_page = source.new_page()

    source_page.insert_text(
        (72, 72),
        "Scanned OCR target content",
    )

    pixmap = source_page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    image_path = path.parent / "source.png"
    pixmap.save(image_path)

    source.close()

    document = pymupdf.open()

    page = document.new_page()

    page.insert_image(
        page.rect,
        filename=str(image_path),
    )

    document.save(path)
    document.close()

    image_path.unlink()


def build_service(
    tmp_path: Path,
    *,
    ocr_text: str = "OCR extracted document content.",
) -> tuple[
    IngestionService,
    DocumentRepository,
    PageStore,
]:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    page_store = PageStore(
        tmp_path / "pages",
    )

    pipeline = OCRPipeline(
        provider=MockOCRProvider(
            text=ocr_text,
            confidence=0.95,
        ),
    )

    service = IngestionService(
        repository=repository,
        page_store=page_store,
        ocr_pipeline=pipeline,
        processed_root=tmp_path / "processed",
    )

    return service, repository, page_store


def test_ingestion_persists_native_text(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "native.pdf"

    create_native_pdf(pdf_path)

    service, _, page_store = build_service(tmp_path)

    result = service.ingest(pdf_path)

    assert result.duplicate is False
    assert result.page_count == 1

    page = page_store.get_page(
        result.document.document_id,
        1,
    )

    assert "reliable native PDF text" in page.text


def test_ingestion_uses_ocr_for_scanned_page(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "scanned.pdf"

    create_scanned_pdf(pdf_path)

    service, _, page_store = build_service(
        tmp_path,
        ocr_text="OCR extracted document content.",
    )

    result = service.ingest(pdf_path)

    assert result.duplicate is False
    assert result.page_count == 1

    page = page_store.get_page(
        result.document.document_id,
        1,
    )

    assert page.text == "OCR extracted document content."


def test_duplicate_document_does_not_run_again(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "document.pdf"

    create_native_pdf(pdf_path)

    service, repository, page_store = build_service(tmp_path)

    first = service.ingest(pdf_path)
    second = service.ingest(pdf_path)

    assert first.duplicate is False
    assert second.duplicate is True

    assert second.document.document_id == (
        first.document.document_id
    )

    pages = page_store.get_pages(
        first.document.document_id,
    )

    assert len(pages) == 1

    stored = repository.get_by_id(
        first.document.document_id,
    )

    assert stored is not None
    assert stored.page_count == 1