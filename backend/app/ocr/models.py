from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TextExtractionMode(StrEnum):
    NATIVE = "native"
    OCR = "ocr"


class PageProcessingDecision(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR_REQUIRED = "ocr_required"
    EMPTY = "empty"
    REVIEW_REQUIRED = "review_required"


class TextQualityResult(BaseModel):
    """Quality assessment of text extracted from a PDF page."""

    model_config = ConfigDict(frozen=True)

    character_count: int = Field(ge=0)
    alphabetic_character_count: int = Field(ge=0)
    alphanumeric_character_count: int = Field(ge=0)
    whitespace_count: int = Field(ge=0)
    replacement_character_count: int = Field(ge=0)

    alphabetic_ratio: float = Field(ge=0.0, le=1.0)
    printable_ratio: float = Field(ge=0.0, le=1.0)

    quality_score: float = Field(ge=0.0, le=1.0)

    extraction_mode: TextExtractionMode
    ocr_required: bool

    reasons: tuple[str, ...]


class PageContentMetadata(BaseModel):
    """Content metadata extracted directly from a PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text_character_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    drawing_count: int = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class PageProcessingDecisionResult(BaseModel):
    """Final processing decision for a PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    decision: PageProcessingDecision

    text_quality: TextQualityResult
    content: PageContentMetadata

    reasons: tuple[str, ...]

class OCRProviderMetadata(BaseModel):
    """Metadata describing an OCR provider execution."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    processing_time_ms: float | None = Field(
        default=None,
        ge=0.0,
    )


class OCRResult(BaseModel):
    """Normalized OCR result returned by an OCR provider."""

    model_config = ConfigDict(frozen=True)

    text: str
    page_number: int = Field(ge=1)

    source_image: Path

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: OCRProviderMetadata