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
        """Build a deterministic response when execution must stop.

        Prefers the triggering decision's reason (a clear, human-facing
        explanation of why the agent gave up -- e.g. "off-topic
        excerpts" or "iteration cap reached") over dumping the latest
        raw retrieved page text, which previously surfaced verbatim,
        `[Source: ...]` markup and all, whenever *any* context existed
        at stop time -- even context an LLM judge had just determined
        didn't actually answer the question.
        """

        if state.decision is not None and state.decision.reason:
            return state.decision.reason

        if state.contexts:
            latest_context = state.contexts[-1]

            if latest_context.text.strip():
                return latest_context.text

        return "The agent could not retrieve sufficient context."

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