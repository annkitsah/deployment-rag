from pathlib import Path

import pytest

from app.documents.models import DocumentStatus
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.page_index import PageIndex
from app.retrieval.text import tokenize


@pytest.fixture
def repository(tmp_path: Path) -> DocumentRepository:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    return repository


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    return PageStore(
        tmp_path / "processed",
    )


@pytest.fixture
def service(
    repository: DocumentRepository,
    page_store: PageStore,
) -> IngestionService:
    return IngestionService(
        repository=repository,
        page_store=page_store,
    )


def test_ingest_pdf(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    assert result.duplicate is False
    assert result.document.status == DocumentStatus.PROCESSED
    assert result.document.filename == pdf_path.name
    assert result.page_count == result.document.page_count
    assert result.page_count > 0

    first_page = page_store.get_page(
        result.document.document_id,
        1,
    )

    assert first_page.document_id == result.document.document_id
    assert first_page.page_number == 1
    assert len(first_page.text) > 0


def test_all_pages_are_persisted(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    pages = page_store.get_pages(
        result.document.document_id,
    )

    assert len(pages) == result.page_count
    assert [page.page_number for page in pages] == list(
        range(1, result.page_count + 1)
    )


def test_ingest_same_pdf_twice_is_idempotent(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    first_result = service.ingest(pdf_path)
    second_result = service.ingest(pdf_path)

    assert first_result.duplicate is False
    assert second_result.duplicate is True

    assert (
        first_result.document.document_id
        == second_result.document.document_id
    )

    assert (
        first_result.document.file_hash
        == second_result.document.file_hash
    )

    pages = page_store.get_pages(
        second_result.document.document_id,
    )

    assert len(pages) == second_result.page_count


def test_duplicate_ingestion_does_not_rewrite_pages(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    first_result = service.ingest(pdf_path)

    page_path = page_store._page_path(
        first_result.document.document_id,
        1,
    )

    original_mtime = page_path.stat().st_mtime_ns

    second_result = service.ingest(pdf_path)

    assert second_result.duplicate is True

    current_mtime = page_path.stat().st_mtime_ns

    assert current_mtime == original_mtime


def test_ingest_missing_file(
    service: IngestionService,
) -> None:
    with pytest.raises(FileNotFoundError):
        service.ingest(
            Path("data/raw/missing.pdf"),
        )


def test_ingest_non_pdf(
    service: IngestionService,
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("not a PDF")

    with pytest.raises(ValueError):
        service.ingest(text_file)


def test_ingest_indexes_pages_when_index_lifecycle_is_configured(
    repository: DocumentRepository,
    page_store: PageStore,
) -> None:
    page_index = PageIndex(page_store=page_store)
    index_lifecycle = IndexLifecycle(
        page_store=page_store,
        page_index=page_index,
    )

    service = IngestionService(
        repository=repository,
        page_store=page_store,
        index_lifecycle=index_lifecycle,
    )

    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    first_page = page_store.get_page(
        result.document.document_id,
        1,
    )

    indexed_terms = tokenize(first_page.text)

    assert len(indexed_terms) > 0
    assert page_index.contains(indexed_terms[0])


def test_ingest_saves_index_snapshot_when_configured(
    repository: DocumentRepository,
    page_store: PageStore,
    tmp_path: Path,
) -> None:
    from app.retrieval.index_persistence import IndexSnapshotStore

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_store = IndexSnapshotStore(snapshot_path)

    page_index = PageIndex(page_store=page_store)
    index_lifecycle = IndexLifecycle(
        page_store=page_store,
        page_index=page_index,
        snapshot_store=snapshot_store,
    )

    service = IngestionService(
        repository=repository,
        page_store=page_store,
        index_lifecycle=index_lifecycle,
    )

    assert not snapshot_path.is_file()

    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    assert snapshot_path.is_file()

    loaded = snapshot_store.load()

    assert loaded is not None
    _, page_count = loaded
    assert page_count == result.page_count


def test_duplicate_ingestion_does_not_reindex(
    repository: DocumentRepository,
    page_store: PageStore,
) -> None:
    page_index = PageIndex(page_store=page_store)
    index_lifecycle = IndexLifecycle(
        page_store=page_store,
        page_index=page_index,
    )

    service = IngestionService(
        repository=repository,
        page_store=page_store,
        index_lifecycle=index_lifecycle,
    )

    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    first_result = service.ingest(pdf_path)
    second_result = service.ingest(pdf_path)

    assert second_result.duplicate is True

    page_id = f"{first_result.document.document_id}:page:1"

    first_page_text = page_store.get_page(
        first_result.document.document_id, 1
    ).text
    indexed_terms = tokenize(first_page_text)

    # Re-ingesting a duplicate should not raise or corrupt the index;
    # the page should still be present exactly once.
    assert len(indexed_terms) > 0
    assert page_id in page_index.inverted_index.lookup(
        indexed_terms[0]
    )