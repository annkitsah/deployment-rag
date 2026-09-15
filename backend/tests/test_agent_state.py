import pytest
from pydantic import ValidationError

from app.agents.models import (
    AgentAction,
    AgentActionType,
    AgentDecision,
    AgentDecisionType,
    AgentPlan,
)
from app.agents.state import AgentState
from app.retrieval.models import (
    RetrievalResult,
    RetrievedContext,
)


def create_context(
    query: str = "retrieval",
) -> RetrievedContext:
    result = RetrievalResult(
        document_id="doc-001",
        page_number=1,
        text="retrieval architecture",
        score=2.5,
        matched_terms=("retrieval",),
    )

    return RetrievedContext(
        query=query,
        results=(result,),
        text=result.text,
    )


def create_plan(
    query: str = "retrieval architecture",
) -> AgentPlan:
    return AgentPlan(
        query=query,
        actions=(
            AgentAction(
                action_type=AgentActionType.RETRIEVE,
                query=query,
            ),
        ),
    )


def create_decision() -> AgentDecision:
    return AgentDecision(
        decision_type=AgentDecisionType.ANSWER,
        reason="Retrieved evidence is sufficient.",
    )


def test_agent_state_initializes_with_query() -> None:
    state = AgentState(
        original_query="What is vectorless RAG?",
        current_query="What is vectorless RAG?",
    )

    assert state.original_query == "What is vectorless RAG?"
    assert state.current_query == "What is vectorless RAG?"
    assert state.plan is None
    assert state.contexts == []
    assert state.decision is None
    assert state.iteration == 0


def test_agent_state_can_store_plan() -> None:
    state = AgentState(
        original_query="retrieval architecture",
        current_query="retrieval architecture",
    )

    plan = create_plan()

    state.set_plan(plan)

    assert state.plan == plan


def test_agent_state_can_add_retrieved_context() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    context = create_context()

    state.add_context(context)

    assert state.contexts == [context]


def test_agent_state_preserves_context_history() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    first = create_context("retrieval")
    second = create_context("lexical retrieval")

    state.add_context(first)
    state.add_context(second)

    assert state.contexts == [first, second]


def test_agent_state_can_update_current_query() -> None:
    state = AgentState(
        original_query="What is RAG?",
        current_query="What is RAG?",
    )

    state.set_query("How does lexical retrieval work?")

    assert state.original_query == "What is RAG?"
    assert state.current_query == "How does lexical retrieval work?"


def test_agent_state_rejects_empty_updated_query() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    try:
        state.set_query("   ")
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError("Expected ValueError")


def test_agent_state_can_store_decision() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    decision = create_decision()

    state.set_decision(decision)

    assert state.decision == decision


def test_agent_state_can_advance_iteration() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    assert state.iteration == 0

    state.advance_iteration()

    assert state.iteration == 1

    state.advance_iteration()

    assert state.iteration == 2


def test_agent_state_rejects_empty_original_query() -> None:
    with pytest.raises(ValidationError):
        AgentState(
            original_query="",
            current_query="retrieval",
        )


def test_agent_state_rejects_empty_current_query() -> None:
    with pytest.raises(ValidationError):
        AgentState(
            original_query="retrieval",
            current_query="",
        )


def test_agent_state_document_id_defaults_to_none() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
    )

    assert state.document_id is None


def test_agent_state_accepts_explicit_document_id() -> None:
    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
        document_id="doc-123",
    )

    assert state.document_id == "doc-123"