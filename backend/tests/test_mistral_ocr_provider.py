from pathlib import Path

import pytest

from app.ocr.base import OCRProvider
from app.ocr.providers.mistral import MistralOCRProvider


class FakeOCRPage:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown


class FakeOCRResponse:
    def __init__(self, pages: list[FakeOCRPage]) -> None:
        self.pages = pages


class FakeOCRClient:
    def __init__(self, response: FakeOCRResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def process(self, **kwargs: object) -> FakeOCRResponse:
        self.calls.append(kwargs)
        return self.response


def test_mistral_provider_implements_protocol(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake png")

    provider = MistralOCRProvider(
        api_key="test-key",
    )

    assert isinstance(provider, OCRProvider)


def test_provider_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        MistralOCRProvider(api_key="")


def test_provider_rejects_invalid_page_number(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake png")

    provider = MistralOCRProvider(
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="page_number"):
        provider.extract(
            image_path=image_path,
            page_number=0,
        )


def test_provider_rejects_missing_image(
    tmp_path: Path,
) -> None:
    provider = MistralOCRProvider(
        api_key="test-key",
    )

    with pytest.raises(FileNotFoundError):
        provider.extract(
            image_path=tmp_path / "missing.png",
            page_number=1,
        )


def test_provider_rejects_empty_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "empty.png"
    image_path.write_bytes(b"")

    provider = MistralOCRProvider(
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="empty"):
        provider.extract(
            image_path=image_path,
            page_number=1,
        )


def test_extract_text_normalizes_markdown() -> None:
    response = FakeOCRResponse(
        pages=[
            FakeOCRPage("First page"),
            FakeOCRPage("Second page"),
        ]
    )

    text = MistralOCRProvider._extract_text(response)

    assert text == "First page\n\nSecond page"


def test_extract_text_handles_empty_response() -> None:
    response = FakeOCRResponse(pages=[])

    text = MistralOCRProvider._extract_text(response)

    assert text == ""