from pathlib import Path

import pymupdf
import pytest

from app.ocr.models import PageProcessingDecision
from app.ocr.pipeline import OCRPipeline
from app.ocr.providers.mock import MockOCRProvider


def create_text_pdf(path: Path) -> None:
    document = pymupdf.open()

    page = document.new_page()

    native_text = (
        "This page contains usable native text extracted directly "
        "from the PDF document. The text is intentionally long"
    )

    page.insert_textbox(
        pymupdf.Rect(72, 72, 540, 200),
        native_text,
        fontsize=10,
    )

    document.save(path)
    document.close()

    
def create_image_pdf(path: Path) -> None:
    source = pymupdf.open()

    source_page = source.new_page()

    source_page.insert_text(
        (72, 72),
        "OCR target content",
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


def test_native_text_page_does_not_call_ocr(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "native.pdf"

    create_text_pdf(pdf_path)

    provider = MockOCRProvider(
        text="OCR should not be called",
    )

    pipeline = OCRPipeline(
        provider=provider,
    )

    result = pipeline.process_document(
        pdf_path=pdf_path,
        document_id="doc_native",
        output_root=tmp_path / "processed",
    )

    assert result.page_count == 1
    assert result.native_text_page_count == 1
    assert result.ocr_page_count == 0

    page = result.pages[0]

    assert page.decision == PageProcessingDecision.NATIVE_TEXT
    assert page.ocr_result is None
    assert page.render_result is None
    assert "usable native text" in page.text


def test_image_page_is_rendered_and_sent_to_ocr(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "image.pdf"

    create_image_pdf(pdf_path)

    provider = MockOCRProvider(
        text="OCR extracted content",
        confidence=0.95,
    )

    pipeline = OCRPipeline(
        provider=provider,
    )

    result = pipeline.process_document(
        pdf_path=pdf_path,
        document_id="doc_image",
        output_root=tmp_path / "processed",
    )

    assert result.page_count == 1
    assert result.ocr_page_count == 1

    page = result.pages[0]

    assert page.decision == PageProcessingDecision.OCR_REQUIRED

    assert page.render_result is not None
    assert page.render_result.output_path.is_file()

    assert page.ocr_result is not None
    assert page.ocr_result.text == "OCR extracted content"
    assert page.ocr_result.confidence == 0.95

    assert page.text == "OCR extracted content"


def test_pipeline_combines_page_text(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "native.pdf"

    create_text_pdf(pdf_path)

    pipeline = OCRPipeline(
        provider=MockOCRProvider(),
    )

    result = pipeline.process_document(
        pdf_path=pdf_path,
        document_id="doc_combined",
        output_root=tmp_path / "processed",
    )

    assert " ".join(result.text.split()) == (
    "This page contains usable native text extracted directly "
    "from the PDF document. The text is intentionally long"
)


def test_pipeline_rejects_missing_pdf(
    tmp_path: Path,
) -> None:
    pipeline = OCRPipeline(
        provider=MockOCRProvider(),
    )

    with pytest.raises(FileNotFoundError):
        pipeline.process_document(
            pdf_path=tmp_path / "missing.pdf",
            document_id="doc_missing",
            output_root=tmp_path / "processed",
        )


def test_pipeline_rejects_invalid_document_id(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "native.pdf"

    create_text_pdf(pdf_path)

    pipeline = OCRPipeline(
        provider=MockOCRProvider(),
    )

    with pytest.raises(ValueError):
        pipeline.process_document(
            pdf_path=pdf_path,
            document_id="../unsafe",
            output_root=tmp_path / "processed",
        )