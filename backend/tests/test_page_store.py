from pathlib import Path

import pytest

from app.documents.models import PageRecord
from app.documents.page_store import (
    PageNotFoundError,
    PageStore,
)


def create_page(
    document_id: str = "doc-test-001",
    page_number: int = 1,
    text: str = "Example page content",
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


def test_save_and_get_page(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    page = create_page()

    saved_path = store.save_page(page)

    assert saved_path.exists()
    assert saved_path.name == "000001.json"

    loaded_page = store.get_page(
        page.document_id,
        page.page_number,
    )

    assert loaded_page == page


def test_save_multiple_pages(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    pages = [
        create_page(page_number=1, text="Page one"),
        create_page(page_number=2, text="Page two"),
        create_page(page_number=3, text="Page three"),
    ]

    paths = store.save_pages(pages)

    assert len(paths) == 3
    assert all(path.exists() for path in paths)

    loaded_pages = store.get_pages(
        "doc-test-001",
    )

    assert loaded_pages == pages


def test_get_page_range(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    pages = [
        create_page(page_number=1),
        create_page(page_number=2),
        create_page(page_number=3),
        create_page(page_number=4),
        create_page(page_number=5),
    ]

    store.save_pages(pages)

    result = store.get_pages(
        "doc-test-001",
        start_page=2,
        end_page=4,
    )

    assert [page.page_number for page in result] == [2, 3, 4]


def test_missing_page_raises(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    with pytest.raises(PageNotFoundError):
        store.get_page(
            "doc-test-001",
            99,
        )


def test_page_exists(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    page = create_page(page_number=7)

    assert not store.page_exists(
        page.document_id,
        7,
    )

    store.save_page(page)

    assert store.page_exists(
        page.document_id,
        7,
    )


def test_document_exists(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    assert not store.document_exists(
        "doc-test-001",
    )

    store.save_page(
        create_page(),
    )

    assert store.document_exists(
        "doc-test-001",
    )


def test_pages_must_belong_to_same_document(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    pages = [
        create_page(
            document_id="doc-001",
            page_number=1,
        ),
        create_page(
            document_id="doc-002",
            page_number=1,
        ),
    ]

    with pytest.raises(ValueError):
        store.save_pages(pages)


def test_invalid_document_id_is_rejected(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    page = create_page(
        document_id="../unsafe",
    )

    with pytest.raises(ValueError):
        store.save_page(page)


def test_delete_document(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(page_number=1),
            create_page(page_number=2),
        ]
    )

    assert store.document_exists(
        "doc-test-001",
    )

    store.delete_document(
        "doc-test-001",
    )

    assert not store.document_exists(
        "doc-test-001",
    )

    with pytest.raises(PageNotFoundError):
        store.get_page(
            "doc-test-001",
            1,
        )
def test_get_all_pages_returns_pages_across_documents(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    pages = [
        PageRecord(
            document_id="doc_b",
            page_number=2,
            text="Document B page two",
            width=612.0,
            height=792.0,
        ),
        PageRecord(
            document_id="doc_a",
            page_number=2,
            text="Document A page two",
            width=612.0,
            height=792.0,
        ),
        PageRecord(
            document_id="doc_a",
            page_number=1,
            text="Document A page one",
            width=612.0,
            height=792.0,
        ),
    ]

    store.save_pages([pages[0]])
    store.save_pages([pages[1], pages[2]])

    result = store.get_all_pages()

    assert [
        (page.document_id, page.page_number)
        for page in result
    ] == [
        ("doc_a", 1),
        ("doc_a", 2),
        ("doc_b", 2),
    ]


def test_count_pages_for_document(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page("doc-a", 1),
            create_page("doc-a", 2),
            create_page("doc-a", 3),
        ]
    )

    assert store.count_pages("doc-a") == 3


def test_count_pages_for_missing_document_is_zero(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    assert store.count_pages("does-not-exist") == 0


def test_count_all_pages_across_documents(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page("doc-a", 1),
            create_page("doc-a", 2),
        ]
    )
    store.save_pages([create_page("doc-b", 1)])

    assert store.count_all_pages() == 3


def test_count_all_pages_on_empty_store(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    assert store.count_all_pages() == 0


def test_count_pages_matches_get_pages_length(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page("doc-a", 1),
            create_page("doc-a", 2),
            create_page("doc-a", 3),
            create_page("doc-a", 4),
        ]
    )

    assert store.count_pages("doc-a") == len(
        store.get_pages("doc-a")
    )
    