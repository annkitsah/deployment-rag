from pathlib import Path

from app.ocr.base import OCRProvider
from app.ocr.models import OCRResult


class OCRService:
    """Application service responsible for OCR execution."""

    def __init__(
        self,
        provider: OCRProvider,
    ) -> None:
        self.provider = provider

    def extract_page(
        self,
        *,
        image_path: Path,
        page_number: int,
    ) -> OCRResult:
        """Extract OCR text from one rendered page."""

        return self.provider.extract(
            image_path=image_path,
            page_number=page_number,
        )