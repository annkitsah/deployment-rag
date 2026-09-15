from unittest.mock import Mock

import pytest

from app.agents.decision import AgentDecisionEngine
from app.agents.models import AgentDecision, AgentDecisionType
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.agents.state import AgentState
from app.retrieval.models import RetrievalResult, RetrievedContext


def _context(
    *,
    query: str = "test query",
    text: str = "Relevant retrieved context.",
    with_result: bool = True,
) -> RetrievedContext:
    results: tuple[RetrievalResult, ...] = ()

    if with_result:
        results = (
            RetrievalResult(
                document_id="doc-1",
                page_number=1,
                text=text,
                score=1.0,
                matched_terms=("retrieved",),
            ),
        )

    return RetrievedContext(
        query=query,
        results=results,
        text=text,
    )


class StubExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        state: AgentState,
    ) -> AgentState:
        self.calls += 1

        state.add_context(
            _context(
                query=state.current_query,
            )
        )

        state.advance_iteration()

        return state


class EmptyContextExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        state: AgentState,
    ) -> AgentState:
        self.calls += 1

        state.add_context(
            _context(
                query=state.current_query,
                text="",
                with_result=False,
            )
        )

        state.advance_iteration()

        return state


def test_orchestrator_returns_answer_from_retrieved_context() -> None:
    executor = StubExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
    )

    response = orchestrator.run(
        "  What is vectorless RAG?  "
    )

    assert response.query == "What is vectorless RAG?"
    assert response.answer == "Relevant retrieved context."
    assert response.iterations == 1
    assert executor.calls == 1


def test_orchestrator_strips_query_whitespace() -> None:
    orchestrator = AgentOrchestrator(
        executor=StubExecutor(),
    )

    response = orchestrator.run("   test query   ")

    assert response.query == "test query"


def test_orchestrator_rejects_empty_query() -> None:
    orchestrator = AgentOrchestrator(
        executor=StubExecutor(),
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        orchestrator.run("   ")


def test_orchestrator_rejects_non_string_query() -> None:
    orchestrator = AgentOrchestrator(
        executor=StubExecutor(),
    )

    with pytest.raises(TypeError, match="query must be a string"):
        orchestrator.run(None)  # type: ignore[arg-type]


def test_orchestrator_uses_custom_planner() -> None:
    planner = Mock(spec=AgentPlanner)
    planner.plan.side_effect = lambda query: AgentPlanner().plan(query)

    orchestrator = AgentOrchestrator(
        planner=planner,
        executor=StubExecutor(),
    )

    orchestrator.run("test query")

    assert planner.plan.call_count == 1
    planner.plan.assert_called_once_with("test query")


def test_orchestrator_refines_when_context_is_empty() -> None:
    class RefiningExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(
            self,
            state: AgentState,
        ) -> AgentState:
            self.calls += 1

            if self.calls == 1:
                state.add_context(
                    _context(
                        query=state.current_query,
                        text="",
                        with_result=False,
                    )
                )
            else:
                state.add_context(
                    _context(
                        query=state.current_query,
                        text="Final context.",
                        with_result=True,
                    )
                )

            state.advance_iteration()

            return state

    class RefiningDecisionEngine:
        def __init__(self) -> None:
            self.calls = 0

        def decide(
            self,
            state: AgentState,
        ) -> AgentDecision:
            self.calls += 1

            if self.calls == 1:
                return AgentDecision(
                    decision_type=AgentDecisionType.REFINE,
                    reason="Need better context.",
                    next_query="refined query",
                )

            return AgentDecision(
                decision_type=AgentDecisionType.ANSWER,
                reason="Context available.",
            )

    executor = RefiningExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,  # type: ignore[arg-type]
        decision_engine=RefiningDecisionEngine(),  # type: ignore[arg-type]
    )

    response = orchestrator.run("original query")

    assert response.query == "original query"
    assert response.answer == "Final context."
    assert response.iterations == 2
    assert executor.calls == 2


def test_orchestrator_preserves_original_query_after_refinement() -> None:
    class RefiningExecutor:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(
            self,
            state: AgentState,
        ) -> AgentState:
            self.queries.append(state.current_query)

            if len(self.queries) == 1:
                state.add_context(
                    _context(
                        query=state.current_query,
                        text="",
                        with_result=False,
                    )
                )
            else:
                state.add_context(
                    _context(
                        query=state.current_query,
                        text="Final context.",
                        with_result=True,
                    )
                )

            state.advance_iteration()

            return state

    class RefiningDecisionEngine:
        def __init__(self) -> None:
            self.calls = 0

        def decide(
            self,
            state: AgentState,
        ) -> AgentDecision:
            self.calls += 1

            if self.calls == 1:
                return AgentDecision(
                    decision_type=AgentDecisionType.REFINE,
                    reason="Refine query.",
                    next_query="better query",
                )

            return AgentDecision(
                decision_type=AgentDecisionType.ANSWER,
                reason="Answer.",
            )

    executor = RefiningExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,  # type: ignore[arg-type]
        decision_engine=RefiningDecisionEngine(),  # type: ignore[arg-type]
    )

    response = orchestrator.run("original query")

    assert executor.queries == [
        "original query",
        "better query",
    ]

    assert response.query == "original query"
    assert response.answer == "Final context."
    assert response.iterations == 2


def test_orchestrator_stops_at_iteration_limit() -> None:
    executor = EmptyContextExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
        decision_engine=AgentDecisionEngine(
            max_iterations=1,
        ),
    )

    response = orchestrator.run("test query")

    assert response.query == "test query"
    assert response.iterations == 1
    assert response.answer == (
        "Maximum agent iterations reached."
    )
    assert executor.calls == 1


def test_stop_response_prefers_decision_reason_over_raw_context() -> None:
    # StubExecutor always adds non-empty context, unlike
    # EmptyContextExecutor above -- this proves the stop response now
    # explains *why* the agent stopped instead of silently dumping the
    # last retrieved page text (with its "[Source: ...]" markup) as if
    # it were an answer.
    executor = StubExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
        decision_engine=AgentDecisionEngine(max_iterations=1),
    )

    response = orchestrator.run("test query")

    assert response.answer == "Maximum agent iterations reached."
    assert executor.calls == 1


def test_orchestrator_stores_decision_in_state() -> None:
    captured_states: list[AgentState] = []

    class CapturingDecisionEngine:
        def decide(
            self,
            state: AgentState,
        ) -> AgentDecision:
            captured_states.append(state)

            return AgentDecision(
                decision_type=AgentDecisionType.ANSWER,
                reason="Answer.",
            )

    orchestrator = AgentOrchestrator(
        executor=StubExecutor(),
        decision_engine=CapturingDecisionEngine(),  # type: ignore[arg-type]
    )

    orchestrator.run("test query")

    assert len(captured_states) == 1
    assert captured_states[0].decision is not None
    assert (
        captured_states[0].decision.decision_type
        is AgentDecisionType.ANSWER
    )


def test_orchestrator_propagates_executor_failure() -> None:
    class FailingExecutor:
        def execute(
            self,
            state: AgentState,
        ) -> AgentState:
            raise RuntimeError("retrieval failed")

    orchestrator = AgentOrchestrator(
        executor=FailingExecutor(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        orchestrator.run("test query")

class StubAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, RetrievedContext]] = []

    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        self.calls.append((query, context))
        return "Generated answer."


def test_orchestrator_uses_custom_answerer() -> None:
    executor = StubExecutor()
    answerer = StubAnswerer()

    orchestrator = AgentOrchestrator(
        executor=executor,
        answerer=answerer,
    )

    response = orchestrator.run(
        "What is vectorless RAG?"
    )

    assert response.answer == "Generated answer."
    assert len(answerer.calls) == 1
    assert answerer.calls[0][0] == "What is vectorless RAG?"


class RecordingExecutor:
    def __init__(self) -> None:
        self.seen_document_ids: list[str | None] = []

    def execute(
        self,
        state: AgentState,
    ) -> AgentState:
        self.seen_document_ids.append(state.document_id)

        state.add_context(
            _context(query=state.current_query)
        )

        state.advance_iteration()

        return state


def test_orchestrator_passes_document_id_to_state() -> None:
    executor = RecordingExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
    )

    orchestrator.run(
        "What is vectorless RAG?",
        document_id="doc-123",
    )

    assert executor.seen_document_ids == ["doc-123"]


def test_orchestrator_defaults_document_id_to_none() -> None:
    executor = RecordingExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
    )

    orchestrator.run("What is vectorless RAG?")

    assert executor.seen_document_ids == [None]


def test_orchestrator_keeps_document_id_across_refine_iterations() -> None:
    class RefineOnceThenAnswerDecisionEngine:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, state: AgentState) -> AgentDecision:
            self.calls += 1

            if self.calls == 1:
                return AgentDecision(
                    decision_type=AgentDecisionType.REFINE,
                    reason="try again",
                )

            return AgentDecision(
                decision_type=AgentDecisionType.ANSWER,
                reason="found it",
            )

    executor = RecordingExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
        decision_engine=RefineOnceThenAnswerDecisionEngine(),
    )

    orchestrator.run(
        "What is vectorless RAG?",
        document_id="doc-123",
    )

    assert executor.seen_document_ids == ["doc-123", "doc-123"]


def test_answer_response_includes_citations_from_used_context() -> None:
    executor = StubExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
    )

    response = orchestrator.run("test query")

    assert len(response.citations) == 1
    assert response.citations[0].document_id == "doc-1"
    assert response.citations[0].page_number == 1


def test_stop_response_has_no_citations() -> None:
    executor = StubExecutor()

    orchestrator = AgentOrchestrator(
        executor=executor,
        decision_engine=AgentDecisionEngine(max_iterations=1),
    )

    response = orchestrator.run("test query")

    assert response.answer == "Maximum agent iterations reached."
    assert response.citations == ()