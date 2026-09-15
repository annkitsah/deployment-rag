import pytest

from app.agents.models import (
    AgentAction,
    AgentActionType,
)
from app.agents.planner import AgentPlanner


def test_planner_creates_retrieve_plan() -> None:
    planner = AgentPlanner()

    plan = planner.plan("What is vectorless RAG?")

    assert plan.query == "What is vectorless RAG?"
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type is AgentActionType.RETRIEVE
    assert plan.actions[0].query == "What is vectorless RAG?"


def test_planner_strips_query_whitespace() -> None:
    planner = AgentPlanner()

    plan = planner.plan("   retrieval architecture   ")

    assert plan.query == "retrieval architecture"
    assert plan.actions[0].query == "retrieval architecture"


def test_planner_rejects_empty_query() -> None:
    planner = AgentPlanner()

    with pytest.raises(ValueError, match="query cannot be empty"):
        planner.plan("")


def test_planner_rejects_whitespace_only_query() -> None:
    planner = AgentPlanner()

    with pytest.raises(ValueError, match="query cannot be empty"):
        planner.plan("   ")


def test_planner_rejects_non_string_query() -> None:
    planner = AgentPlanner()

    with pytest.raises(TypeError, match="query must be a string"):
        planner.plan(None)  # type: ignore[arg-type]


def test_planner_is_deterministic() -> None:
    planner = AgentPlanner()

    first = planner.plan("Explain lexical retrieval")
    second = planner.plan("Explain lexical retrieval")

    assert first == second

def test_planner_creates_retrieval_only_execution_plan() -> None:
    planner = AgentPlanner()

    plan = planner.plan(
        "What does vectorless RAG use for retrieval?"
    )

    assert plan.actions == (
        AgentAction(
            action_type=AgentActionType.RETRIEVE,
            query="What does vectorless RAG use for retrieval?",
        ),
    )
