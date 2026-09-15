from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.citations.models import Citation


class AgentActionType(StrEnum):
    """Supported actions that the agent can request."""

    RETRIEVE = "retrieve"
    ANSWER = "answer"
    REFINE = "refine"


class AgentAction(BaseModel):
    """A single action produced by the agent planner."""

    model_config = ConfigDict(frozen=True)

    action_type: AgentActionType
    query: str = Field(min_length=1, max_length=10_000)


class AgentPlan(BaseModel):
    """Structured execution plan produced for an agent query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=10_000)
    actions: tuple[AgentAction, ...] = Field(min_length=1)


class AgentDecisionType(StrEnum):
    """Possible decisions after evaluating retrieved evidence."""

    ANSWER = "answer"
    REFINE = "refine"
    STOP = "stop"


class AgentDecision(BaseModel):
    """Decision produced after evaluating the current agent state."""

    model_config = ConfigDict(frozen=True)

    decision_type: AgentDecisionType
    reason: str = Field(min_length=1, max_length=10_000)
    next_query: str | None = Field(
        default=None,
        max_length=10_000,
    )


class AgentResponse(BaseModel):
    """Final response returned by the agent orchestration layer."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=10_000)
    answer: str = Field(min_length=1, max_length=100_000)
    iterations: int = Field(ge=0)
    citations: tuple[Citation, ...] = ()