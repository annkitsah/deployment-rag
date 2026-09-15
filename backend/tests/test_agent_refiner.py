from app.agents.refiner import AgentQueryRefiner
from app.agents.state import AgentState
from app.retrieval.models import RetrievalResult, RetrievedContext


def _empty_context() -> RetrievedContext:
    return RetrievedContext(
        query="vectorless RAG",
        results=(),
        text="",
    )


def _retrieved_context() -> RetrievedContext:
    return RetrievedContext(
        query="vectorless RAG",
        results=(
            RetrievalResult(
                document_id="doc-1",
                page_number=1,
                text="Vectorless RAG uses lexical retrieval.",
                score=0.95,
                matched_terms=("vectorless", "rag", "retrieval"),
            ),
        ),
        text="Vectorless RAG uses lexical retrieval.",
    )


def test_refiner_rejects_invalid_state() -> None:
    refiner = AgentQueryRefiner()

    try:
        refiner.refine("invalid")
    except TypeError as exc:
        assert str(exc) == "state must be an AgentState"
    else:
        raise AssertionError("TypeError was not raised")


def test_refiner_adds_retrieval_instruction_without_context() -> None:
    refiner = AgentQueryRefiner()

    state = AgentState(
        original_query="vectorless RAG",
        current_query="vectorless RAG",
    )

    refined_query = refiner.refine(state)

    assert refined_query == (
        "vectorless RAG relevant definition explanation details"
    )


def test_refiner_adds_retrieval_instruction_for_empty_context() -> None:
    refiner = AgentQueryRefiner()

    state = AgentState(
        original_query="vectorless RAG",
        current_query="vectorless RAG",
        contexts=[_empty_context()],
    )

    refined_query = refiner.refine(state)

    assert refined_query == (
        "vectorless RAG relevant definition explanation details"
    )


def test_refiner_does_not_change_query_when_context_exists() -> None:
    refiner = AgentQueryRefiner()

    state = AgentState(
        original_query="vectorless RAG",
        current_query="vectorless RAG",
        contexts=[_retrieved_context()],
    )

    refined_query = refiner.refine(state)

    assert refined_query == "vectorless RAG"


def test_refiner_does_not_mutate_state() -> None:
    refiner = AgentQueryRefiner()

    state = AgentState(
        original_query="vectorless RAG",
        current_query="vectorless RAG",
        contexts=[_empty_context()],
    )

    original_query = state.current_query
    original_contexts = list(state.contexts)

    refiner.refine(state)

    assert state.current_query == original_query
    assert state.contexts == original_contexts