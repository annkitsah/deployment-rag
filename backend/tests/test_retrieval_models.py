import pytest
from pydantic import ValidationError

from app.retrieval.models import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    RetrievedContext,
)


def test_retrieval_query_defaults() -> None:
    query = RetrievalQuery(text="What is generative AI?")

    assert query.text == "What is generative AI?"
    assert query.top_k == 10
    assert query.document_id is None


def test_retrieval_query_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(text="")


def test_retrieval_query_validates_top_k() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(
            text="test",
            top_k=0,
        )

    with pytest.raises(ValidationError):
        RetrievalQuery(
            text="test",
            top_k=101,
        )


def test_retrieval_result_page_id() -> None:
    result = RetrievalResult(
        document_id="doc_123",
        page_number=7,
        text="Relevant content",
        score=4.5,
        matched_terms=("relevant", "content"),
    )

    assert result.page_id == "doc_123:page:7"


def test_retrieval_result_rejects_negative_score() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            document_id="doc_123",
            page_number=1,
            text="content",
            score=-1,
        )


def test_retrieval_response_result_count() -> None:
    query = RetrievalQuery(text="test")

    results = (
        RetrievalResult(
            document_id="doc_1",
            page_number=1,
            text="first",
            score=2.0,
        ),
        RetrievalResult(
            document_id="doc_1",
            page_number=2,
            text="second",
            score=1.0,
        ),
    )

    response = RetrievalResponse(
        query=query,
        results=results,
    )

    assert response.result_count == 2


def test_retrieved_context_page_count() -> None:
    results = (
        RetrievalResult(
            document_id="doc_1",
            page_number=1,
            text="first",
            score=2.0,
        ),
    )

    context = RetrievedContext(
        query="test",
        results=results,
        text="first",
    )

    assert context.page_count == 1
    assert context.text == "first"