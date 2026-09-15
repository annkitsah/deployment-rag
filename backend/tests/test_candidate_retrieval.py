import pytest

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.candidates import CandidateRetriever
from app.retrieval.metadata import PageMetadataFilter
from app.retrieval.page_index import PageIndex


@pytest.fixture
def page_store(tmp_path) -> PageStore:
    return PageStore(tmp_path)


@pytest.fixture
def page_index(page_store: PageStore) -> PageIndex:
    index = PageIndex(page_store=page_store)

    pages = (
        PageRecord(
            document_id="doc-001",
            page_number=1,
            text="retrieval systems architecture",
            width=600.0,
            height=800.0,
        ),
        PageRecord(
            document_id="doc-001",
            page_number=2,
            text="retrieval pipeline",
            width=600.0,
            height=800.0,
        ),
        PageRecord(
            document_id="doc-002",
            page_number=1,
            text="retrieval architecture",
            width=700.0,
            height=900.0,
        ),
        PageRecord(
            document_id="doc-003",
            page_number=1,
            text="database systems",
            width=600.0,
            height=800.0,
        ),
    )

    for page in pages:
        page_store.save_page(page)
        index.add_page(page)

    return index


def test_retrieves_candidates_for_single_term(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    result = retriever.retrieve(("retrieval",))

    assert result == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_retrieves_union_for_multiple_terms(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    result = retriever.retrieve(
        ("retrieval", "database"),
    )

    assert result == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
        "doc-003:page:1",
    )


def test_duplicate_candidates_are_returned_once(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    result = retriever.retrieve(
        ("retrieval", "architecture"),
    )

    assert result == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_candidates_are_returned_in_deterministic_order(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    result = retriever.retrieve(
        ("database", "retrieval"),
    )

    assert result == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
        "doc-003:page:1",
    )


def test_unknown_terms_produce_no_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    assert retriever.retrieve(("unknown",)) == ()


def test_empty_terms_produce_no_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    assert retriever.retrieve(()) == ()


def test_metadata_filter_restricts_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    metadata_filter = PageMetadataFilter(
        document_id="doc-001",
    )

    result = retriever.retrieve(
        ("retrieval",),
        metadata_filter=metadata_filter,
    )

    assert result == (
        "doc-001:page:1",
        "doc-001:page:2",
    )


def test_metadata_filter_can_remove_all_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    metadata_filter = PageMetadataFilter(
        document_id="doc-999",
    )

    assert (
        retriever.retrieve(
            ("retrieval",),
            metadata_filter=metadata_filter,
        )
        == ()
    )


def test_page_number_filter_restricts_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    metadata_filter = PageMetadataFilter(
        document_id="doc-001",
        page_number=2,
    )

    result = retriever.retrieve(
        ("retrieval",),
        metadata_filter=metadata_filter,
    )

    assert result == (
        "doc-001:page:2",
    )


def test_metadata_filter_is_applied_after_term_candidates(
    page_index: PageIndex,
) -> None:
    retriever = CandidateRetriever(page_index)

    metadata_filter = PageMetadataFilter(
        min_width=650.0,
    )

    result = retriever.retrieve(
        ("retrieval",),
        metadata_filter=metadata_filter,
    )

    assert result == (
        "doc-002:page:1",
    )


@pytest.mark.parametrize(
    "terms",
    [
        ("RETRIEVAL",),
        ("Retrieval",),
        ("retrieval", "RETRIEVAL"),
    ],
)
def test_term_lookup_is_case_insensitive(
    page_index: PageIndex,
    terms: tuple[str, ...],
) -> None:
    retriever = CandidateRetriever(page_index)

    assert retriever.retrieve(terms) == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )
