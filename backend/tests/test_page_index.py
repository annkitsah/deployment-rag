import pytest

from app.documents.models import PageRecord
from app.documents.page_store import PageNotFoundError, PageStore
from app.retrieval.inverted_index import InvertedIndex
from app.retrieval.page_index import PageIndex


@pytest.fixture
def page_store(tmp_path) -> PageStore:
    return PageStore(tmp_path)


@pytest.fixture
def pages() -> tuple[PageRecord, PageRecord]:
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
            width=600.0,
            height=800.0,
        ),
    )


def test_add_page_indexes_page(page_store: PageStore) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="retrieval systems",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    assert index.lookup("retrieval") == (
        "doc-001:page:1",
    )


def test_lookup_is_case_insensitive(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="Retrieval SYSTEMS",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    assert index.lookup("RETRIEVAL") == (
        "doc-001:page:1",
    )


def test_get_page_resolves_page_id(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=3,
        text="retrieval architecture",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    result = index.get_page("doc-001:page:3")

    assert result == page


def test_get_page_raises_for_missing_page(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    with pytest.raises(PageNotFoundError):
        index.get_page("doc-001:page:1")


def test_remove_page_removes_searchability(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="retrieval systems",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    index.remove_page(page.page_id)

    assert index.lookup("retrieval") == ()


def test_remove_page_does_not_delete_persisted_page(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="retrieval systems",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    index.remove_page(page.page_id)

    assert page_store.get_page(
        "doc-001",
        1,
    ) == page


def test_contains_reports_indexed_terms(
    page_store: PageStore,
) -> None:
    index = PageIndex(page_store=page_store)

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="retrieval systems",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    assert index.contains("retrieval")
    assert not index.contains("unknown")


def test_clear_removes_all_indexed_pages(
    page_store: PageStore,
    pages: tuple[PageRecord, PageRecord],
) -> None:
    index = PageIndex(page_store=page_store)

    for page in pages:
        page_store.save_page(page)
        index.add_page(page)

    index.clear()

    assert index.lookup("retrieval") == ()


def test_clear_does_not_delete_persisted_pages(
    page_store: PageStore,
    pages: tuple[PageRecord, PageRecord],
) -> None:
    index = PageIndex(page_store=page_store)

    for page in pages:
        page_store.save_page(page)
        index.add_page(page)

    index.clear()

    assert page_store.get_page("doc-001", 1) == pages[0]
    assert page_store.get_page("doc-001", 2) == pages[1]


def test_custom_inverted_index_is_used(
    page_store: PageStore,
) -> None:
    inverted_index = InvertedIndex()
    index = PageIndex(
        page_store=page_store,
        inverted_index=inverted_index,
    )

    page = PageRecord(
        document_id="doc-001",
        page_number=1,
        text="retrieval",
        width=600.0,
        height=800.0,
    )

    page_store.save_page(page)
    index.add_page(page)

    assert index.inverted_index is inverted_index
    assert index.lookup("retrieval") == (
        "doc-001:page:1",
    )


@pytest.mark.parametrize(
    "page_id",
    [
        "",
        " ",
        "invalid",
        "doc-001:page:",
        "doc-001:page:abc",
        "doc-001:page:0",
    ],
)
def test_invalid_page_id_is_rejected(
    page_store: PageStore,
    page_id: str,
) -> None:
    index = PageIndex(page_store=page_store)

    with pytest.raises(ValueError):
        index.get_page(page_id)
