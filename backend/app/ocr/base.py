from pathlib import Path
from typing import Protocol, runtime_checkable

from app.ocr.models import OCRResult


@runtime_checkable
class OCRProvider(Protocol):
    """Interface implemented by all OCR providers."""

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    def extract(
        self,
        *,
        image_path: Path,
        page_number: int,
    ) -> OCRResult:
        """Extract text from a rendered page image."""
        ...