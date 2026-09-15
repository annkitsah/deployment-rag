from app.agents.decision import AgentDecisionEngine
from app.agents.models import AgentDecisionType
from app.agents.state import AgentState
from app.retrieval.models import RetrievedContext


def create_context(
    *,
    query: str = "What is RAG?",
    text: str = "RAG combines retrieval with generation.",
    page_count: int = 1,
) -> RetrievedContext:
    from app.retrieval.models import RetrievalResult

    results = tuple(
        RetrievalResult(
            document_id=f"doc-{index}",
            page_number=1,
            text=text,
            score=1.0,
        )
        for index in range(page_count)
    )

    return RetrievedContext(
        query=query,
        results=results,
        text=text if page_count else "",
    )


def test_decision_engine_answers_when_context_exists() -> None:
    engine = AgentDecisionEngine()

    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
    )

    state.add_context(create_context())

    decision = engine.decide(state)

    assert decision.decision_type == AgentDecisionType.ANSWER
    assert decision.next_query is None


def test_decision_engine_refines_when_no_context_exists() -> None:
    engine = AgentDecisionEngine()

    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
    )

    decision = engine.decide(state)

    assert decision.decision_type == AgentDecisionType.REFINE
    assert decision.next_query == "What is RAG?"


def test_decision_engine_refines_when_latest_context_is_empty() -> None:
    engine = AgentDecisionEngine()

    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
    )

    state.add_context(
        create_context(
            page_count=0,
        )
    )

    decision = engine.decide(state)

    assert decision.decision_type == AgentDecisionType.REFINE
    assert decision.next_query == "What is RAG?"


def test_decision_engine_stops_at_iteration_limit() -> None:
    engine = AgentDecisionEngine(
        max_iterations=3,
    )

    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
        iteration=3,
    )

    decision = engine.decide(state)

    assert decision.decision_type == AgentDecisionType.STOP
    assert decision.next_query is None


def test_decision_engine_stops_before_context_evaluation() -> None:
    engine = AgentDecisionEngine(
        max_iterations=2,
    )

    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
        iteration=2,
    )

    state.add_context(create_context())

    decision = engine.decide(state)

    assert decision.decision_type == AgentDecisionType.STOP


def test_decision_engine_rejects_invalid_max_iterations() -> None:
    try:
        AgentDecisionEngine(max_iterations=0)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid max_iterations"
    )