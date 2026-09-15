import pytest

from app.documents.models import PageRecord
from app.retrieval.metadata import PageMetadataFilter


@pytest.fixture
def pages() -> tuple[PageRecord, ...]:
    return (
        PageRecord(
            document_id="doc-001",
            page_number=1,
            text="retrieval systems",
            width=600.0,
            height=800.0,
        ),
        PageRecord(
            document_id="doc-001",
            page_number=2,
            text="retrieval architecture",
            width=700.0,
            height=900.0,
        ),
        PageRecord(
            document_id="doc-002",
            page_number=1,
            text="retrieval pipeline",
            width=600.0,
            height=800.0,
        ),
    )


def test_empty_filter_matches_all_pages(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter()

    assert metadata_filter.apply(pages) == pages


def test_filters_by_document_id(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        document_id="doc-001",
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:1",
        "doc-001:page:2",
    )


def test_filters_by_page_number(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        page_number=2,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:2",
    )


def test_filters_by_page_number_range(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        min_page_number=1,
        max_page_number=1,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:1",
        "doc-002:page:1",
    )


def test_filters_by_minimum_width(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        min_width=650.0,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:2",
    )


def test_filters_by_maximum_width(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        max_width=650.0,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:1",
        "doc-002:page:1",
    )


def test_filters_by_minimum_height(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        min_height=850.0,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:2",
    )


def test_filters_by_maximum_height(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        max_height=850.0,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:1",
        "doc-002:page:1",
    )


def test_combines_multiple_filters(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        document_id="doc-001",
        min_page_number=2,
        min_width=650.0,
        min_height=850.0,
    )

    result = metadata_filter.apply(pages)

    assert tuple(page.page_id for page in result) == (
        "doc-001:page:2",
    )


def test_filter_preserves_input_order(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        min_width=600.0,
    )

    result = metadata_filter.apply(
        tuple(reversed(pages)),
    )

    assert tuple(page.page_id for page in result) == (
        "doc-002:page:1",
        "doc-001:page:2",
        "doc-001:page:1",
    )


def test_filter_returns_empty_when_nothing_matches(
    pages: tuple[PageRecord, ...],
) -> None:
    metadata_filter = PageMetadataFilter(
        document_id="does-not-exist",
    )

    assert metadata_filter.apply(pages) == ()


def test_rejects_empty_document_id() -> None:
    with pytest.raises(ValueError):
        PageMetadataFilter(document_id="")


def test_rejects_invalid_page_range() -> None:
    with pytest.raises(ValueError):
        PageMetadataFilter(
            min_page_number=3,
            max_page_number=1,
        )


@pytest.mark.parametrize(
    "field",
    [
        "min_width",
        "max_width",
        "min_height",
        "max_height",
    ],
)
def test_rejects_non_positive_dimensions(field: str) -> None:
    with pytest.raises(ValueError):
        PageMetadataFilter(**{field: 0.0})