from __future__ import annotations

import base64
import time
from pathlib import Path

from mistralai import Mistral

from app.ocr.models import OCRProviderMetadata, OCRResult


class MistralOCRProvider:
    """OCR provider backed by the Mistral OCR API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "mistral-ocr-latest",
        timeout_ms: int = 120_000,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Mistral API key must not be empty")

        if not model.strip():
            raise ValueError("Mistral OCR model must not be empty")

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self._model = model
        self._client = Mistral(
            api_key=api_key,
            timeout_ms=timeout_ms,
        )

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def model_name(self) -> str:
        return self._model

    def extract(
        self,
        *,
        image_path: Path,
        page_number: int,
    ) -> OCRResult:
        """Run Mistral OCR against a rendered page image."""

        if not image_path.is_file():
            raise FileNotFoundError(
                f"OCR input image does not exist: {image_path}"
            )

        if page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

        started_at = time.perf_counter()

        image_bytes = image_path.read_bytes()

        if not image_bytes:
            raise ValueError(
                f"OCR input image is empty: {image_path}"
            )

        image_base64 = base64.b64encode(image_bytes).decode("ascii")

        response = self._client.ocr.process(
            model=self._model,
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{image_base64}",
            },
            include_image_base64=False,
        )

        processing_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        text = self._extract_text(response)

        return OCRResult(
            text=text,
            page_number=page_number,
            source_image=image_path,
            confidence=None,
            metadata=OCRProviderMetadata(
                provider=self.provider_name,
                model=self.model_name,
                processing_time_ms=processing_time_ms,
            ),
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        """Extract normalized text from a Mistral OCR response."""

        pages = getattr(response, "pages", None)

        if not pages:
            return ""

        page_texts: list[str] = []

        for page in pages:
            markdown = getattr(page, "markdown", None)

            if markdown:
                page_texts.append(str(markdown))

        return "\n\n".join(page_texts).strip()