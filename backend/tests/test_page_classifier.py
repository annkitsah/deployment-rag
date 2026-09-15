from app.ocr.classifier import PageContentClassifier
from app.ocr.models import PageProcessingDecision


def test_good_text_is_native() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=1,
        text=(
            "Artificial intelligence systems use machine learning "
            "models to process information and generate responses."
        ),
        image_count=0,
        drawing_count=0,
        width=612,
        height=792,
    )

    assert result.decision == PageProcessingDecision.NATIVE_TEXT
    assert "usable_native_text" in result.reasons


def test_blank_page_is_empty() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=20,
        text=" \n",
        image_count=0,
        drawing_count=0,
        width=612,
        height=792,
    )

    assert result.decision == PageProcessingDecision.EMPTY
    assert "empty_page" in result.reasons


def test_image_only_page_requires_ocr() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=2,
        text="",
        image_count=1,
        drawing_count=0,
        width=612,
        height=792,
    )

    assert result.decision == PageProcessingDecision.OCR_REQUIRED
    assert "raster_image_without_usable_text" in result.reasons


def test_drawing_page_requires_review() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=3,
        text="",
        image_count=0,
        drawing_count=4,
        width=612,
        height=792,
    )

    assert result.decision == PageProcessingDecision.REVIEW_REQUIRED
    assert "vector_content_without_usable_text" in result.reasons


def test_poor_text_without_visual_content_requires_review() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=4,
        text="1234567890 !!!!! 999999",
        image_count=0,
        drawing_count=0,
        width=612,
        height=792,
    )

    assert result.decision == PageProcessingDecision.REVIEW_REQUIRED


def test_page_metadata_is_preserved() -> None:
    classifier = PageContentClassifier()

    result = classifier.classify(
        page_number=7,
        text="This is valid extracted text from a PDF page.",
        image_count=2,
        drawing_count=3,
        width=612,
        height=792,
    )

    assert result.content.page_number == 7
    assert result.content.image_count == 2
    assert result.content.drawing_count == 3
    assert result.content.width == 612
    assert result.content.height == 792