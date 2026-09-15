from pathlib import Path

import pytest

from app.agents.executor import AgentExecutor
from app.agents.models import (
    AgentAction,
    AgentActionType,
    AgentPlan,
)
from app.agents.state import AgentState
from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.models import RetrievalQuery
from app.retrieval.service import RetrievalService


def create_page(
    document_id: str,
    page_number: int,
    text: str,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc-001",
                1,
                "Retrieval augmented generation combines retrieval "
                "with language models.",
            ),
            create_page(
                "doc-001",
                2,
                "Vectorless retrieval uses lexical matching.",
            ),
        ]
    )

    return store


@pytest.fixture
def executor(page_store: PageStore) -> AgentExecutor:
    return AgentExecutor(
        retrieval_service=RetrievalService(page_store),
    )


def test_executor_retrieves_context(
    executor: AgentExecutor,
) -> None:
    state = AgentState(
        original_query="lexical retrieval",
        current_query="lexical retrieval",
    )

    state.set_plan(
        AgentPlan(
            query="lexical retrieval",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="lexical retrieval",
                ),
            ),
        )
    )

    result = executor.execute(state)

    assert result is state
    assert len(state.contexts) == 1
    assert state.contexts[0].query == "lexical retrieval"
    assert state.contexts[0].page_count > 0


def test_executor_advances_iteration_after_execution(
    executor: AgentExecutor,
) -> None:
    state = AgentState(
        original_query="lexical retrieval",
        current_query="lexical retrieval",
    )

    state.set_plan(
        AgentPlan(
            query="lexical retrieval",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="lexical retrieval",
                ),
            ),
        )
    )

    assert state.iteration == 0

    executor.execute(state)

    assert state.iteration == 1


def test_executor_uses_action_query(
    executor: AgentExecutor,
) -> None:
    state = AgentState(
        original_query="original query",
        current_query="current query",
    )

    state.set_plan(
        AgentPlan(
            query="current query",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="lexical retrieval",
                ),
            ),
        )
    )

    executor.execute(state)

    assert len(state.contexts) == 1
    assert state.contexts[0].query == "lexical retrieval"


def test_executor_preserves_existing_context_history(
    executor: AgentExecutor,
) -> None:
    state = AgentState(
        original_query="lexical retrieval",
        current_query="lexical retrieval",
    )

    state.set_plan(
        AgentPlan(
            query="lexical retrieval",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="lexical retrieval",
                ),
            ),
        )
    )

    executor.execute(state)
    executor.execute(state)

    assert len(state.contexts) == 2
    assert state.contexts[0] == state.contexts[1]


def test_executor_requires_plan(
    executor: AgentExecutor,
) -> None:
    state = AgentState(
        original_query="lexical retrieval",
        current_query="lexical retrieval",
    )

    with pytest.raises(ValueError, match="plan"):
        executor.execute(state)


def test_executor_rejects_non_agent_state(
    executor: AgentExecutor,
) -> None:
    with pytest.raises(TypeError, match="state"):
        executor.execute("invalid")  # type: ignore[arg-type]


def test_executor_supports_custom_retrieval_service() -> None:
    class StubRetrievalService:
        def __init__(self) -> None:
            self.queries: list[RetrievalQuery] = []

        def retrieve(self, query: RetrievalQuery):
            from app.retrieval.models import RetrievedContext

            self.queries.append(query)

            return RetrievedContext(
                query=query.text,
                results=(),
                text="stub context",
            )

    retrieval_service = StubRetrievalService()
    executor = AgentExecutor(
        retrieval_service=retrieval_service,
    )

    state = AgentState(
        original_query="test",
        current_query="test",
    )

    state.set_plan(
        AgentPlan(
            query="test",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="test",
                ),
            ),
        )
    )

    executor.execute(state)

    assert len(retrieval_service.queries) == 1
    assert retrieval_service.queries[0].text == "test"
    assert state.contexts[0].text == "stub context"


def test_executor_does_not_advance_on_failed_retrieval() -> None:
    class FailingRetrievalService:
        def retrieve(self, query: RetrievalQuery):
            raise RuntimeError("retrieval failed")

    executor = AgentExecutor(
        retrieval_service=FailingRetrievalService(),
    )

    state = AgentState(
        original_query="test",
        current_query="test",
    )

    state.set_plan(
        AgentPlan(
            query="test",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="test",
                ),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        executor.execute(state)

    assert state.contexts == []
    assert state.iteration == 0


def test_executor_rejects_unsupported_action() -> None:
    class StubRetrievalService:
        def retrieve(self, query: RetrievalQuery):
            from app.retrieval.models import RetrievedContext

            return RetrievedContext(
                query=query.text,
                results=(),
                text="stub",
            )

    executor = AgentExecutor(
        retrieval_service=StubRetrievalService(),
    )

    state = AgentState(
        original_query="test",
        current_query="test",
    )

    state.set_plan(
        AgentPlan(
            query="test",
            actions=(
                AgentAction(
                    action_type=AgentActionType.ANSWER,
                    query="test",
                ),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="unsupported executor action",
    ):
        executor.execute(state)


def test_executor_passes_configured_top_k_to_retrieval(
    page_store: PageStore,
) -> None:
    executor = AgentExecutor(
        retrieval_service=RetrievalService(page_store),
        top_k=1,
    )

    state = AgentState(
        original_query="lexical retrieval",
        current_query="lexical retrieval",
    )

    state.set_plan(
        AgentPlan(
            query="lexical retrieval",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="lexical retrieval",
                ),
            ),
        )
    )

    result_state = executor.execute(state)

    # Both pages match "retrieval"; top_k=1 must cap the result count.
    assert result_state.contexts[-1].page_count == 1


def test_executor_defaults_top_k_to_ten(
    page_store: PageStore,
) -> None:
    executor = AgentExecutor(
        retrieval_service=RetrievalService(page_store),
    )

    assert executor.top_k == 10


def test_executor_rejects_invalid_top_k(
    page_store: PageStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        AgentExecutor(
            retrieval_service=RetrievalService(page_store),
            top_k=0,
        )


def test_executor_scopes_retrieval_to_state_document_id(
    tmp_path: Path,
) -> None:
    scoped_store = PageStore(tmp_path)

    scoped_store.save_pages(
        [
            create_page(
                "doc-a",
                1,
                "Retrieval scoped to document A.",
            ),
        ]
    )
    scoped_store.save_pages(
        [
            create_page(
                "doc-b",
                1,
                "Retrieval scoped to document B.",
            ),
        ]
    )

    executor = AgentExecutor(
        retrieval_service=RetrievalService(scoped_store),
    )

    state = AgentState(
        original_query="retrieval",
        current_query="retrieval",
        document_id="doc-a",
    )

    state.set_plan(
        AgentPlan(
            query="retrieval",
            actions=(
                AgentAction(
                    action_type=AgentActionType.RETRIEVE,
                    query="retrieval",
                ),
            ),
        )
    )

    result_state = executor.execute(state)

    context = result_state.contexts[-1]

    assert context.page_count == 1
    assert context.results[0].document_id == "doc-a"