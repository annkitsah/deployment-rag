from pathlib import Path

from app.ocr.models import OCRProviderMetadata, OCRResult


class MockOCRProvider:
    """Deterministic OCR provider used for testing."""

    def __init__(
        self,
        *,
        text: str = "Mock OCR extracted text.",
        confidence: float = 1.0,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        self._text = text
        self._confidence = confidence

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-ocr"

    def extract(
        self,
        *,
        image_path: Path,
        page_number: int,
    ) -> OCRResult:
        if not image_path.is_file():
            raise FileNotFoundError(
                f"OCR input image does not exist: {image_path}"
            )

        if page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

        return OCRResult(
            text=self._text,
            page_number=page_number,
            source_image=image_path,
            confidence=self._confidence,
            metadata=OCRProviderMetadata(
                provider=self.provider_name,
                model=self.model_name,
            ),
        )