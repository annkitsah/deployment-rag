import pytest
from pydantic import ValidationError

from app.evaluation.models import (
    AgentCaseResult,
    AgentEvaluationSummary,
    EvaluationCase,
    RetrievalCaseResult,
    RetrievalEvaluationSummary,
)


def test_evaluation_case_requires_at_least_one_expected_page() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            case_id="c1",
            question="q",
            expected_pages=(),
        )


def test_evaluation_case_defaults() -> None:
    case = EvaluationCase(
        case_id="c1",
        question="q",
        expected_pages=(1,),
    )

    assert case.source_filename is None
    assert case.document_id is None
    assert case.expected_keywords == ()


def test_evaluation_case_is_frozen() -> None:
    case = EvaluationCase(
        case_id="c1",
        question="q",
        expected_pages=(1,),
    )

    with pytest.raises(ValidationError):
        case.question = "changed"  # type: ignore[misc]


def test_retrieval_case_result_recall_bounds() -> None:
    with pytest.raises(ValidationError):
        RetrievalCaseResult(
            case_id="c1",
            question="q",
            expected_pages=(1,),
            retrieved_pages=(1,),
            recall=1.5,
            hit=True,
        )


def test_retrieval_evaluation_summary_requires_results() -> None:
    with pytest.raises(ValidationError):
        RetrievalEvaluationSummary(
            results=(),
            mean_recall=0.0,
            hit_rate=0.0,
        )


def test_agent_case_result_accepts_valid_data() -> None:
    result = AgentCaseResult(
        case_id="c1",
        question="q",
        answer="answer text",
        iterations=1,
        expected_pages=(1,),
        cited_pages=(1,),
        page_hit=True,
        expected_keywords=("foo",),
        matched_keywords=("foo",),
        keyword_coverage=1.0,
    )

    assert result.page_hit is True


def test_agent_evaluation_summary_requires_results() -> None:
    with pytest.raises(ValidationError):
        AgentEvaluationSummary(
            results=(),
            page_hit_rate=0.0,
            mean_keyword_coverage=0.0,
            mean_iterations=0.0,
        )