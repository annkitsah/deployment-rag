from pydantic import BaseModel, ConfigDict, Field

from app.agents.models import AgentDecision, AgentPlan
from app.retrieval.models import RetrievedContext


class AgentState(BaseModel):
    """Mutable execution state for one agent run."""

    model_config = ConfigDict(validate_assignment=True)

    original_query: str = Field(
        min_length=1,
        max_length=10_000,
    )
    current_query: str = Field(
        min_length=1,
        max_length=10_000,
    )
    plan: AgentPlan | None = None
    contexts: list[RetrievedContext] = Field(default_factory=list)
    decision: AgentDecision | None = None
    iteration: int = Field(default=0, ge=0)
    document_id: str | None = None

    def add_context(
        self,
        context: RetrievedContext,
    ) -> None:
        """Append retrieved context to the execution history."""

        self.contexts.append(context)

    def advance_iteration(self) -> None:
        """Advance the agent execution iteration."""

        self.iteration += 1

    def set_query(self, query: str) -> None:
        """Update the current query used by the agent."""

        if not query.strip():
            raise ValueError("query cannot be empty")

        self.current_query = query

    def set_plan(self, plan: AgentPlan) -> None:
        """Store the current execution plan."""

        self.plan = plan

    def set_decision(
        self,
        decision: AgentDecision,
    ) -> None:
        """Store the latest agent decision."""

        self.decision = decision