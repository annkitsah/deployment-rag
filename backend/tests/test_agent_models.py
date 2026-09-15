import pytest
from pydantic import ValidationError

from app.agents.models import (
    AgentAction,
    AgentActionType,
    AgentDecision,
    AgentDecisionType,
    AgentPlan,
    AgentResponse,
)
from app.citations.models import Citation


def test_agent_action_defaults_and_values() -> None:
    action = AgentAction(
        action_type=AgentActionType.RETRIEVE,
        query="What is vectorless RAG?",
    )

    assert action.action_type == AgentActionType.RETRIEVE
    assert action.query == "What is vectorless RAG?"


def test_agent_action_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        AgentAction(
            action_type=AgentActionType.RETRIEVE,
            query="",
        )


def test_agent_plan_accepts_multiple_actions() -> None:
    plan = AgentPlan(
        query="Explain retrieval architecture",
        actions=(
            AgentAction(
                action_type=AgentActionType.RETRIEVE,
                query="retrieval architecture",
            ),
            AgentAction(
                action_type=AgentActionType.ANSWER,
                query="Explain the retrieved evidence",
            ),
        ),
    )

    assert plan.query == "Explain retrieval architecture"
    assert len(plan.actions) == 2


def test_agent_plan_requires_at_least_one_action() -> None:
    with pytest.raises(ValidationError):
        AgentPlan(
            query="test",
            actions=(),
        )


def test_agent_decision_for_answer() -> None:
    decision = AgentDecision(
        decision_type=AgentDecisionType.ANSWER,
        reason="Retrieved evidence is sufficient.",
    )

    assert decision.decision_type == AgentDecisionType.ANSWER
    assert decision.next_query is None


def test_agent_decision_can_request_refinement() -> None:
    decision = AgentDecision(
        decision_type=AgentDecisionType.REFINE,
        reason="Retrieved evidence is insufficient.",
        next_query="What are the candidate retrieval strategies?",
    )

    assert decision.decision_type == AgentDecisionType.REFINE
    assert decision.next_query == (
        "What are the candidate retrieval strategies?"
    )


def test_agent_decision_rejects_empty_reason() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(
            decision_type=AgentDecisionType.STOP,
            reason="",
        )


def test_agent_response() -> None:
    response = AgentResponse(
        query="What is vectorless RAG?",
        answer="Vectorless RAG retrieves information without embeddings.",
        iterations=1,
    )

    assert response.query == "What is vectorless RAG?"
    assert response.answer == (
        "Vectorless RAG retrieves information without embeddings."
    )
    assert response.iterations == 1


def test_agent_response_rejects_negative_iterations() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            query="test",
            answer="answer",
            iterations=-1,
        )


def test_agent_response_citations_default_to_empty() -> None:
    response = AgentResponse(
        query="test",
        answer="answer",
        iterations=1,
    )

    assert response.citations == ()


def test_agent_response_accepts_citations() -> None:
    citation = Citation(
        document_id="doc-1",
        page_number=4,
        score=3.09,
    )

    response = AgentResponse(
        query="test",
        answer="answer",
        iterations=1,
        citations=(citation,),
    )

    assert response.citations == (citation,)