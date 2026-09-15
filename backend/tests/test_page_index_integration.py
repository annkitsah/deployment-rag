from pathlib import Path

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.page_index import PageIndex


def create_page(
    document_id: str,
    page_number: int,
    text: str,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


def test_add_pages_indexes_all_pages(
    tmp_path: Path,
) -> None:
    page_store = PageStore(tmp_path)
    page_index = PageIndex(page_store=page_store)

    pages = [
        create_page(
            "doc-001",
            1,
            "retrieval architecture",
        ),
        create_page(
            "doc-001",
            2,
            "retrieval pipeline",
        ),
    ]

    page_store.save_pages(pages)

    for page in pages:
        page_index.add_page(page)

    assert page_index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
    )


def test_indexed_page_can_be_resolved_from_store(
    tmp_path: Path,
) -> None:
    page_store = PageStore(tmp_path)
    page_index = PageIndex(page_store=page_store)

    page = create_page(
        "doc-001",
        1,
        "retrieval architecture",
    )

    page_store.save_page(page)
    page_index.add_page(page)

    result = page_index.get_page(page.page_id)

    assert result == page


def test_reindexing_page_updates_searchable_content(
    tmp_path: Path,
) -> None:
    page_store = PageStore(tmp_path)
    page_index = PageIndex(page_store=page_store)

    original = create_page(
        "doc-001",
        1,
        "retrieval architecture",
    )

    updated = create_page(
        "doc-001",
        1,
        "database indexing",
    )

    page_store.save_page(original)
    page_index.add_page(original)

    page_store.save_page(updated)
    page_index.add_page(updated)

    assert page_index.lookup("retrieval") == ()
    assert page_index.lookup("database") == (
        "doc-001:page:1",
    )


def test_remove_page_only_removes_search_index_entry(
    tmp_path: Path,
) -> None:
    page_store = PageStore(tmp_path)
    page_index = PageIndex(page_store=page_store)

    page = create_page(
        "doc-001",
        1,
        "retrieval architecture",
    )

    page_store.save_page(page)
    page_index.add_page(page)

    page_index.remove_page(page.page_id)

    assert page_index.lookup("retrieval") == ()
    assert page_store.get_page(
        "doc-001",
        1,
    ) == page


def test_clear_only_clears_search_index(
    tmp_path: Path,
) -> None:
    page_store = PageStore(tmp_path)
    page_index = PageIndex(page_store=page_store)

    pages = [
        create_page(
            "doc-001",
            1,
            "retrieval architecture",
        ),
        create_page(
            "doc-001",
            2,
            "retrieval pipeline",
        ),
    ]

    page_store.save_pages(pages)

    for page in pages:
        page_index.add_page(page)

    page_index.clear()

    assert page_index.lookup("retrieval") == ()

    assert page_store.get_page(
        "doc-001",
        1,
    ) == pages[0]

    assert page_store.get_page(
        "doc-001",
        2,
    ) == pages[1]