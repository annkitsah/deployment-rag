from app.agents.answerer import Answerer, ContextAnswerer
from app.agents.decision import AgentDecisionEngine, DecisionEngine
from app.agents.executor import AgentExecutor
from app.agents.models import AgentDecisionType, AgentResponse
from app.agents.planner import AgentPlanner
from app.agents.refiner import AgentQueryRefiner, QueryRefiner
from app.agents.state import AgentState
from app.citations.models import Citation
from app.citations.service import build_citations
from app.retrieval.models import RetrievedContext


class AgentOrchestrator:
    """Coordinate planning, retrieval, decision-making, and refinement."""

    def __init__(
        self,
        *,
        planner: AgentPlanner | None = None,
        executor: AgentExecutor,
        decision_engine: DecisionEngine | None = None,
        answerer: Answerer | None = None,
        refiner: QueryRefiner | None = None,
    ) -> None:
        self.planner = planner or AgentPlanner()
        self.executor = executor
        self.decision_engine = (
            decision_engine or AgentDecisionEngine()
        )
        self.answerer = answerer or ContextAnswerer()
        self.refiner = refiner or AgentQueryRefiner()

    def run(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> AgentResponse:
        """Run the agent lifecycle for a user query.

        `document_id`, when given, scopes retrieval to that single
        document for every retrieval attempt across the run, including
        any refine-and-retry iterations.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty")

        state = AgentState(
            original_query=normalized_query,
            current_query=normalized_query,
            document_id=document_id,
        )

        self._plan_current_query(state)

        while True:
            state = self.executor.execute(state)

            decision = self.decision_engine.decide(state)
            state.set_decision(decision)

            if decision.decision_type is AgentDecisionType.ANSWER:
                if not state.contexts:
                    raise ValueError(
                        "cannot answer without retrieved context"
                    )

                latest_context = state.contexts[-1]

                return self._build_response(
                    state,
                    answer=self.answerer.answer(
                        state.current_query,
                        latest_context,
                    ),
                    citations=build_citations(latest_context),
                )

            if decision.decision_type is AgentDecisionType.STOP:
                # Prefer answering from the best non-empty context if any.
                usable = [
                    ctx
                    for ctx in state.contexts
                    if ctx.page_count > 0 and ctx.text.strip()
                ]
                if usable:
                    latest_context = usable[-1]
                    try:
                        return self._build_response(
                            state,
                            answer=self.answerer.answer(
                                state.current_query,
                                latest_context,
                            ),
                            citations=build_citations(latest_context),
                        )
                    except Exception:
                        pass
                return self._build_response(
                    state,
                    answer=self._build_stop_response(state),
                )

            if decision.decision_type is AgentDecisionType.REFINE:
                next_query = decision.next_query

                if next_query is None:
                    next_query = self.refiner.refine(state)

                state.set_query(next_query)
                self._plan_current_query(state)
                continue

            raise ValueError(
                f"unsupported agent decision: {decision.decision_type}"
            )

    def _plan_current_query(
        self,
        state: AgentState,
    ) -> None:
        plan = self.planner.plan(state.current_query)
        state.set_plan(plan)

    @staticmethod
    def _build_stop_response(
        state: AgentState,
    ) -> str:
        """Build a deterministic response when execution must stop."""

        usable = [
            ctx
            for ctx in state.contexts
            if ctx.page_count > 0 and (ctx.text or "").strip()
        ]
        if usable and usable[-1].text.strip():
            return usable[-1].text

        if state.decision is not None and state.decision.reason:
            reason = state.decision.reason.strip()
            # Never surface the bare max-iterations string alone.
            if reason.lower().startswith("maximum agent iterations"):
                return (
                    "No relevant pages were found in the indexed documents "
                    "for this question. Try different keywords, a broader "
                    "phrasing, or scope to a specific document."
                )
            return reason

        return (
            "No relevant pages were found in the indexed documents "
            "for this question. Try different keywords, a broader "
            "phrasing, or scope to a specific document."
        )

    @staticmethod
    def _build_response(
        state: AgentState,
        *,
        answer: str,
        citations: tuple[Citation, ...] = (),
    ) -> AgentResponse:
        return AgentResponse(
            query=state.original_query,
            answer=answer,
            iterations=state.iteration,
            citations=citations,
        )

    @staticmethod
    def _latest_context(
        state: AgentState,
    ) -> RetrievedContext:
        """Return the latest retrieved context."""

        if not state.contexts:
            raise ValueError(
                "cannot answer without retrieved context"
            )

        return state.contexts[-1]