from pathlib import Path

import pytest

from app.ocr.base import OCRProvider
from app.ocr.models import OCRResult
from app.ocr.providers.mock import MockOCRProvider
from app.ocr.service import OCRService


def test_mock_provider_implements_ocr_provider(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake image")

    provider = MockOCRProvider(
        text="Extracted text",
        confidence=0.95,
    )

    assert isinstance(provider, OCRProvider)


def test_mock_provider_returns_normalized_result(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake image")

    provider = MockOCRProvider(
        text="Extracted text",
        confidence=0.95,
    )

    result = provider.extract(
        image_path=image_path,
        page_number=7,
    )

    assert isinstance(result, OCRResult)

    assert result.text == "Extracted text"
    assert result.page_number == 7
    assert result.source_image == image_path
    assert result.confidence == 0.95

    assert result.metadata.provider == "mock"
    assert result.metadata.model == "mock-ocr"


def test_ocr_service_delegates_to_provider(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake image")

    provider = MockOCRProvider(
        text="Service result",
    )

    service = OCRService(provider)

    result = service.extract_page(
        image_path=image_path,
        page_number=3,
    )

    assert result.text == "Service result"
    assert result.page_number == 3


def test_missing_image_is_rejected(
    tmp_path: Path,
) -> None:
    provider = MockOCRProvider()

    with pytest.raises(FileNotFoundError):
        provider.extract(
            image_path=tmp_path / "missing.png",
            page_number=1,
        )


def test_invalid_page_number_is_rejected(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake image")

    provider = MockOCRProvider()

    with pytest.raises(ValueError):
        provider.extract(
            image_path=image_path,
            page_number=0,
        )


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        MockOCRProvider(
            confidence=1.5,
        )