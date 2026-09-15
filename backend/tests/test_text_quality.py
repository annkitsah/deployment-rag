import pytest

from app.ocr.models import TextExtractionMode
from app.ocr.quality import TextQualityAnalyzer


@pytest.fixture
def analyzer() -> TextQualityAnalyzer:
    return TextQualityAnalyzer()


def test_good_text_uses_native_extraction(
    analyzer: TextQualityAnalyzer,
) -> None:
    text = (
        "Artificial intelligence systems use machine learning "
        "models to process information, reason over context, "
        "and generate useful responses for users."
    )

    result = analyzer.analyze(text)

    assert result.ocr_required is False
    assert result.extraction_mode == TextExtractionMode.NATIVE
    assert result.character_count == len(text)
    assert result.quality_score > 0.45
    assert result.reasons == ()


def test_empty_text_requires_ocr(
    analyzer: TextQualityAnalyzer,
) -> None:
    result = analyzer.analyze("")

    assert result.ocr_required is True
    assert result.extraction_mode == TextExtractionMode.OCR
    assert result.quality_score == 0.0
    assert "no_text_extracted" in result.reasons


def test_short_text_requires_ocr(
    analyzer: TextQualityAnalyzer,
) -> None:
    result = analyzer.analyze("Page 1")

    assert result.ocr_required is True
    assert result.extraction_mode == TextExtractionMode.OCR
    assert "insufficient_text" in result.reasons


def test_garbage_text_requires_ocr(
    analyzer: TextQualityAnalyzer,
) -> None:
    text = "1234567890 !!!!!! @@@@@@ 000000 111111 222222"

    result = analyzer.analyze(text)

    assert result.ocr_required is True
    assert result.extraction_mode == TextExtractionMode.OCR
    assert "low_alphabetic_ratio" in result.reasons


def test_replacement_characters_are_detected(
    analyzer: TextQualityAnalyzer,
) -> None:
    text = (
        "This is a valid sentence with corrupted "
        "characters � � �"
    )

    result = analyzer.analyze(text)

    assert result.replacement_character_count == 3
    assert result.ocr_required is True
    assert "replacement_characters_present" in result.reasons


def test_custom_thresholds() -> None:
    analyzer = TextQualityAnalyzer(
        minimum_characters=10,
        minimum_alphabetic_ratio=0.10,
        minimum_printable_ratio=0.80,
        minimum_quality_score=0.20,
    )

    result = analyzer.analyze(
        "This is a short but readable page."
    )

    assert result.ocr_required is False
    assert result.extraction_mode == TextExtractionMode.NATIVE


def test_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        TextQualityAnalyzer(
            minimum_alphabetic_ratio=1.5,
        )

    with pytest.raises(ValueError):
        TextQualityAnalyzer(
            minimum_printable_ratio=-0.1,
        )

    with pytest.raises(ValueError):
        TextQualityAnalyzer(
            minimum_quality_score=2.0,
        )

    with pytest.raises(ValueError):
        TextQualityAnalyzer(
            minimum_characters=-1,
        )


def test_non_string_input(
    analyzer: TextQualityAnalyzer,
) -> None:
    with pytest.raises(TypeError):
        analyzer.analyze(None)  # type: ignore[arg-type]