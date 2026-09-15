import pytest

from app.citations.models import Citation
from app.citations.service import build_citations
from app.retrieval.models import RetrievalResult, RetrievedContext


def make_result(
    *,
    document_id: str = "doc-1",
    page_number: int = 1,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        page_number=page_number,
        text="Some page text.",
        score=score,
        matched_terms=("retrieval",),
    )


def test_rejects_non_retrieved_context() -> None:
    with pytest.raises(TypeError):
        build_citations("not a context")  # type: ignore[arg-type]


def test_empty_context_produces_no_citations() -> None:
    context = RetrievedContext(
        query="test",
        results=(),
        text="",
    )

    assert build_citations(context) == ()


def test_builds_one_citation_per_result() -> None:
    context = RetrievedContext(
        query="test",
        results=(
            make_result(page_number=1, score=2.0),
            make_result(page_number=2, score=1.0),
        ),
        text="combined text",
    )

    citations = build_citations(context)

    assert citations == (
        Citation(document_id="doc-1", page_number=1, score=2.0),
        Citation(document_id="doc-1", page_number=2, score=1.0),
    )


def test_citations_are_sorted_by_score_descending() -> None:
    context = RetrievedContext(
        query="test",
        results=(
            make_result(page_number=1, score=1.0),
            make_result(page_number=2, score=3.0),
            make_result(page_number=3, score=2.0),
        ),
        text="combined text",
    )

    citations = build_citations(context)

    assert [citation.page_number for citation in citations] == [2, 3, 1]


def test_duplicate_document_and_page_is_deduplicated() -> None:
    context = RetrievedContext(
        query="test",
        results=(
            make_result(
                document_id="doc-1",
                page_number=1,
                score=2.0,
            ),
            make_result(
                document_id="doc-1",
                page_number=1,
                score=2.0,
            ),
        ),
        text="combined text",
    )

    citations = build_citations(context)

    assert len(citations) == 1


def test_citations_span_multiple_documents() -> None:
    context = RetrievedContext(
        query="test",
        results=(
            make_result(document_id="doc-a", page_number=1, score=2.0),
            make_result(document_id="doc-b", page_number=1, score=1.0),
        ),
        text="combined text",
    )

    citations = build_citations(context)

    document_ids = {citation.document_id for citation in citations}

    assert document_ids == {"doc-a", "doc-b"}