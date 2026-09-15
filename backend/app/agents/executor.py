from app.agents.models import AgentActionType
from app.agents.state import AgentState
from app.retrieval.models import RetrievalQuery
from app.retrieval.service import RetrievalService


class AgentExecutor:
    """Execute retrieval actions against application services."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        *,
        top_k: int = 10,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        self.retrieval_service = retrieval_service
        self.top_k = top_k

    def execute(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute retrieval actions and update agent state."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")

        if state.plan is None:
            raise ValueError("agent state must contain a plan")

        for action in state.plan.actions:
            if action.action_type is AgentActionType.RETRIEVE:
                self._execute_action(
                    state,
                    action.action_type,
                    action.query,
                )
                continue

            raise ValueError(
                f"unsupported executor action: {action.action_type}"
            )

        state.advance_iteration()

        return state

    def _execute_action(
        self,
        state: AgentState,
        action_type: AgentActionType,
        query: str,
    ) -> None:
        if action_type is not AgentActionType.RETRIEVE:
            raise ValueError(
                f"unsupported executor action: {action_type}"
            )

        context = self.retrieval_service.retrieve(
            RetrievalQuery(
                text=query,
                top_k=self.top_k,
                document_id=state.document_id,
            )
        )

        state.add_context(context)