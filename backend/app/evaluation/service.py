from app.agents.orchestrator import AgentOrchestrator
from app.evaluation.models import (
    AgentCaseResult,
    AgentEvaluationSummary,
    EvaluationCase,
    RetrievalCaseResult,
    RetrievalEvaluationSummary,
)
from app.retrieval.models import RetrievalQuery
from app.retrieval.service import RetrievalService


class EvaluationService:
    """Runs golden-dataset cases against retrieval and/or the full agent."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService | None = None,
        agent_orchestrator: AgentOrchestrator | None = None,
    ) -> None:
        if retrieval_service is None and agent_orchestrator is None:
            raise ValueError(
                "at least one of retrieval_service or agent_orchestrator "
                "must be provided"
            )

        self._retrieval_service = retrieval_service
        self._agent_orchestrator = agent_orchestrator

    def evaluate_retrieval(
        self,
        cases: tuple[EvaluationCase, ...],
    ) -> RetrievalEvaluationSummary:
        """Check whether expected pages appear in retrieved results.

        Network-free and deterministic (no LLM involved) -- exercises
        only the lexical retrieval layer (candidate generation + BM25
        ranking), so this is safe to run in the normal test suite on
        every change.
        """

        if self._retrieval_service is None:
            raise RuntimeError(
                "retrieval_service was not configured on this "
                "EvaluationService"
            )

        if not cases:
            raise ValueError("cases cannot be empty")

        results = tuple(
            self._evaluate_retrieval_case(case) for case in cases
        )

        mean_recall = sum(result.recall for result in results) / len(
            results
        )
        hit_rate = sum(1 for result in results if result.hit) / len(
            results
        )

        return RetrievalEvaluationSummary(
            results=results,
            mean_recall=mean_recall,
            hit_rate=hit_rate,
        )

    def _evaluate_retrieval_case(
        self,
        case: EvaluationCase,
    ) -> RetrievalCaseResult:
        assert self._retrieval_service is not None

        context = self._retrieval_service.retrieve(
            RetrievalQuery(
                text=case.question,
                document_id=case.document_id,
            )
        )

        retrieved_pages = tuple(
            result.page_number for result in context.results
        )

        expected_set = set(case.expected_pages)
        matched = expected_set & set(retrieved_pages)

        recall = (
            len(matched) / len(expected_set) if expected_set else 0.0
        )

        return RetrievalCaseResult(
            case_id=case.case_id,
            question=case.question,
            expected_pages=case.expected_pages,
            retrieved_pages=retrieved_pages,
            recall=recall,
            hit=len(matched) > 0,
        )

    def evaluate_agent(
        self,
        cases: tuple[EvaluationCase, ...],
    ) -> AgentEvaluationSummary:
        """Run cases through the full agent pipeline.

        Exercises planning, retrieval, decision-making, refinement, and
        answer generation exactly as a real request would. When the
        configured orchestrator uses LLM-backed components (the default
        production wiring), this requires a reachable generation
        provider (e.g. Ollama running) -- this is why it is not run
        automatically as part of the normal test suite; use
        `scripts/run_evaluation.py` to run it against a live system.
        """

        if self._agent_orchestrator is None:
            raise RuntimeError(
                "agent_orchestrator was not configured on this "
                "EvaluationService"
            )

        if not cases:
            raise ValueError("cases cannot be empty")

        results = tuple(
            self._evaluate_agent_case(case) for case in cases
        )

        page_hit_rate = sum(
            1 for result in results if result.page_hit
        ) / len(results)

        mean_keyword_coverage = sum(
            result.keyword_coverage for result in results
        ) / len(results)

        mean_iterations = sum(
            result.iterations for result in results
        ) / len(results)

        return AgentEvaluationSummary(
            results=results,
            page_hit_rate=page_hit_rate,
            mean_keyword_coverage=mean_keyword_coverage,
            mean_iterations=mean_iterations,
        )

    def _evaluate_agent_case(
        self,
        case: EvaluationCase,
    ) -> AgentCaseResult:
        assert self._agent_orchestrator is not None

        response = self._agent_orchestrator.run(
            case.question,
            document_id=case.document_id,
        )

        cited_pages = tuple(
            citation.page_number for citation in response.citations
        )

        page_hit = bool(set(case.expected_pages) & set(cited_pages))

        answer_lower = response.answer.lower()

        matched_keywords = tuple(
            keyword
            for keyword in case.expected_keywords
            if keyword.lower() in answer_lower
        )

        keyword_coverage = (
            len(matched_keywords) / len(case.expected_keywords)
            if case.expected_keywords
            else 1.0
        )

        return AgentCaseResult(
            case_id=case.case_id,
            question=case.question,
            answer=response.answer,
            iterations=response.iterations,
            expected_pages=case.expected_pages,
            cited_pages=cited_pages,
            page_hit=page_hit,
            expected_keywords=case.expected_keywords,
            matched_keywords=matched_keywords,
            keyword_coverage=keyword_coverage,
        )