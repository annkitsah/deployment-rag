from app.agents.models import (
    AgentAction,
    AgentActionType,
    AgentPlan,
)


class AgentPlanner:
    """Create deterministic execution plans for agent queries."""

    def plan(self, query: str) -> AgentPlan:
        """Build a retrieval plan for an agent query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty")

        return AgentPlan(
            query=normalized_query,
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query=normalized_query,
                ),
            ),
        )