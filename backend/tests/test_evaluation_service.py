from pathlib import Path

import pytest

from app.agents.answerer import ContextAnswerer
from app.agents.decision import AgentDecisionEngine
from app.agents.executor import AgentExecutor
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.agents.refiner import AgentQueryRefiner
from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.evaluation.models import EvaluationCase
from app.evaluation.service import EvaluationService
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
                "Vectorless retrieval uses lexical matching and "
                "traditional information retrieval techniques.",
            ),
            create_page(
                "doc-001",
                3,
                "Agentic systems can plan, retrieve, and refine "
                "queries iteratively.",
            ),
        ]
    )

    return store


@pytest.fixture
def retrieval_service(page_store: PageStore) -> RetrievalService:
    return RetrievalService(page_store)


@pytest.fixture
def agent_orchestrator(
    retrieval_service: RetrievalService,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=AgentPlanner(),
        executor=AgentExecutor(retrieval_service=retrieval_service),
        decision_engine=AgentDecisionEngine(),
        answerer=ContextAnswerer(),
        refiner=AgentQueryRefiner(),
    )


def test_constructor_requires_at_least_one_service() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EvaluationService()


def test_evaluate_retrieval_requires_retrieval_service(
    agent_orchestrator: AgentOrchestrator,
) -> None:
    service = EvaluationService(agent_orchestrator=agent_orchestrator)

    with pytest.raises(RuntimeError, match="retrieval_service"):
        service.evaluate_retrieval(
            (
                EvaluationCase(
                    case_id="c1",
                    question="q",
                    expected_pages=(1,),
                ),
            )
        )


def test_evaluate_retrieval_rejects_empty_cases(
    retrieval_service: RetrievalService,
) -> None:
    service = EvaluationService(retrieval_service=retrieval_service)

    with pytest.raises(ValueError, match="cases cannot be empty"):
        service.evaluate_retrieval(())


def test_evaluate_retrieval_hit_and_recall(
    retrieval_service: RetrievalService,
) -> None:
    service = EvaluationService(retrieval_service=retrieval_service)

    cases = (
        EvaluationCase(
            case_id="lexical-matching",
            question="lexical matching techniques",
            document_id="doc-001",
            expected_pages=(2,),
        ),
    )

    summary = service.evaluate_retrieval(cases)

    assert summary.results[0].hit is True
    assert summary.results[0].recall == 1.0
    assert summary.hit_rate == 1.0
    assert summary.mean_recall == 1.0


def test_evaluate_retrieval_miss_produces_zero_recall(
    retrieval_service: RetrievalService,
) -> None:
    service = EvaluationService(retrieval_service=retrieval_service)

    cases = (
        EvaluationCase(
            case_id="nonexistent-page",
            question="lexical matching techniques",
            document_id="doc-001",
            expected_pages=(99,),
        ),
    )

    summary = service.evaluate_retrieval(cases)

    assert summary.results[0].hit is False
    assert summary.results[0].recall == 0.0
    assert summary.hit_rate == 0.0


def test_evaluate_agent_requires_agent_orchestrator(
    retrieval_service: RetrievalService,
) -> None:
    service = EvaluationService(retrieval_service=retrieval_service)

    with pytest.raises(RuntimeError, match="agent_orchestrator"):
        service.evaluate_agent(
            (
                EvaluationCase(
                    case_id="c1",
                    question="q",
                    expected_pages=(1,),
                ),
            )
        )


def test_evaluate_agent_rejects_empty_cases(
    agent_orchestrator: AgentOrchestrator,
) -> None:
    service = EvaluationService(agent_orchestrator=agent_orchestrator)

    with pytest.raises(ValueError, match="cases cannot be empty"):
        service.evaluate_agent(())


def test_evaluate_agent_page_hit_and_keyword_coverage(
    agent_orchestrator: AgentOrchestrator,
) -> None:
    service = EvaluationService(agent_orchestrator=agent_orchestrator)

    cases = (
        EvaluationCase(
            case_id="lexical-matching",
            question="lexical matching techniques",
            document_id="doc-001",
            expected_pages=(2,),
            expected_keywords=("lexical", "vectorless"),
        ),
    )

    summary = service.evaluate_agent(cases)

    result = summary.results[0]

    assert result.page_hit is True
    assert 2 in result.cited_pages
    assert result.keyword_coverage == 1.0
    assert set(result.matched_keywords) == {"lexical", "vectorless"}
    assert summary.page_hit_rate == 1.0
    assert summary.mean_keyword_coverage == 1.0


def test_evaluate_agent_partial_keyword_coverage(
    agent_orchestrator: AgentOrchestrator,
) -> None:
    service = EvaluationService(agent_orchestrator=agent_orchestrator)

    cases = (
        EvaluationCase(
            case_id="lexical-matching",
            question="lexical matching techniques",
            document_id="doc-001",
            expected_pages=(2,),
            expected_keywords=("lexical", "nonexistent-term-xyz"),
        ),
    )

    summary = service.evaluate_agent(cases)

    assert summary.results[0].keyword_coverage == 0.5


def test_evaluate_agent_no_expected_keywords_gives_full_coverage(
    agent_orchestrator: AgentOrchestrator,
) -> None:
    service = EvaluationService(agent_orchestrator=agent_orchestrator)

    cases = (
        EvaluationCase(
            case_id="lexical-matching",
            question="lexical matching techniques",
            document_id="doc-001",
            expected_pages=(2,),
        ),
    )

    summary = service.evaluate_agent(cases)

    assert summary.results[0].keyword_coverage == 1.0