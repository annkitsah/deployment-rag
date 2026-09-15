import pytest

from app.agents.answerer import Answerer, ContextAnswerer
from app.retrieval.models import RetrievalResult, RetrievedContext


def make_context(
    *,
    text: str,
    page_number: int = 1,
) -> RetrievedContext:
    result = RetrievalResult(
        document_id="test-document",
        page_number=page_number,
        text=text,
        score=1.0,
    )

    return RetrievedContext(
        query="test query",
        results=(result,),
        text=text,
    )


def test_context_answerer_is_answerer() -> None:
    answerer = ContextAnswerer()

    assert isinstance(answerer, Answerer)


def test_context_answerer_returns_context_text() -> None:
    context = make_context(
        text="Retrieved context."
    )

    answerer = ContextAnswerer()

    assert (
        answerer.answer(
            "test query",
            context,
        )
        == "Retrieved context."
    )


def test_context_answerer_rejects_empty_query() -> None:
    context = make_context(
        text="Retrieved context."
    )

    answerer = ContextAnswerer()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        answerer.answer(
            "   ",
            context,
        )


def test_context_answerer_rejects_empty_context() -> None:
    context = make_context(
        text="   "
    )

    answerer = ContextAnswerer()

    with pytest.raises(
        ValueError,
        match="cannot build answer from empty retrieved context",
    ):
        answerer.answer(
            "test query",
            context,
        )


def test_context_answerer_rejects_invalid_query() -> None:
    context = make_context(
        text="Retrieved context."
    )

    answerer = ContextAnswerer()

    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        answerer.answer(
            123,  # type: ignore[arg-type]
            context,
        )


def test_context_answerer_rejects_invalid_context() -> None:
    answerer = ContextAnswerer()

    with pytest.raises(
        TypeError,
        match="context must be a RetrievedContext",
    ):
        answerer.answer(
            "test query",
            "invalid context",  # type: ignore[arg-type]
        )