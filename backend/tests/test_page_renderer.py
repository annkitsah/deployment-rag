from pathlib import Path

import pymupdf
import pytest

from app.ocr.renderer import (
    PageRenderer,
    PageRenderError,
)


@pytest.fixture
def pdf_path() -> Path:
    return Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )


def test_render_single_page(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    result = renderer.render_page(
        pdf_path=pdf_path,
        document_id="doc-render-test",
        page_number=1,
        output_root=tmp_path,
    )

    assert result.document_id == "doc-render-test"
    assert result.page_number == 1

    assert result.output_path.exists()
    assert result.output_path.suffix == ".png"

    assert result.width > 0
    assert result.height > 0

    assert result.dpi == 200
    assert result.file_size_bytes > 0


def test_rendered_dimensions_match_dpi(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    result = renderer.render_page(
        pdf_path=pdf_path,
        document_id="doc-render-test",
        page_number=1,
        output_root=tmp_path,
        dpi=144,
    )

    assert result.dpi == 144

    with pymupdf.open(pdf_path) as document:
        page = document[0]

        expected_width = round(
            page.rect.width * 144 / 72
        )

        expected_height = round(
            page.rect.height * 144 / 72
        )

    assert result.width == expected_width
    assert result.height == expected_height


def test_render_multiple_pages(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    results = renderer.render_pages(
        pdf_path=pdf_path,
        document_id="doc-render-test",
        page_numbers=[3, 1, 2],
        output_root=tmp_path,
    )

    assert len(results) == 3

    assert [
        result.page_number
        for result in results
    ] == [1, 2, 3]

    assert all(
        result.output_path.exists()
        for result in results
    )


def test_duplicate_page_numbers_are_rendered_once(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    results = renderer.render_pages(
        pdf_path=pdf_path,
        document_id="doc-render-test",
        page_numbers=[1, 1, 2, 2, 3],
        output_root=tmp_path,
    )

    assert [
        result.page_number
        for result in results
    ] == [1, 2, 3]


def test_empty_page_list_returns_empty(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    results = renderer.render_pages(
        pdf_path=pdf_path,
        document_id="doc-render-test",
        page_numbers=[],
        output_root=tmp_path,
    )

    assert results == []


def test_missing_pdf_raises(
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    with pytest.raises(FileNotFoundError):
        renderer.render_page(
            pdf_path=tmp_path / "missing.pdf",
            document_id="doc-test",
            page_number=1,
            output_root=tmp_path,
        )


def test_invalid_page_number_raises(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    with pytest.raises(ValueError):
        renderer.render_page(
            pdf_path=pdf_path,
            document_id="doc-test",
            page_number=0,
            output_root=tmp_path,
        )


def test_nonexistent_page_raises(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    with pytest.raises(PageRenderError):
        renderer.render_page(
            pdf_path=pdf_path,
            document_id="doc-test",
            page_number=999,
            output_root=tmp_path,
        )


def test_invalid_document_id_is_rejected(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    with pytest.raises(ValueError):
        renderer.render_page(
            pdf_path=pdf_path,
            document_id="../unsafe",
            page_number=1,
            output_root=tmp_path,
        )


def test_invalid_dpi_is_rejected(
    pdf_path: Path,
    tmp_path: Path,
) -> None:
    renderer = PageRenderer()

    with pytest.raises(ValueError):
        renderer.render_page(
            pdf_path=pdf_path,
            document_id="doc-test",
            page_number=1,
            output_root=tmp_path,
            dpi=0,
        )