from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    """A known question with the pages/keywords its answer should draw on.

    `source_filename`, when set, is resolved to a concrete `document_id`
    at load time (via `app.evaluation.dataset.resolve_document_ids`)
    rather than hardcoded, since document IDs are generated per-ingestion
    and are not stable across runs. `document_id` can also be set
    directly for programmatic use (e.g. in tests).
    """

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=10_000)
    source_filename: str | None = None
    document_id: str | None = None
    expected_pages: tuple[int, ...] = Field(min_length=1)
    expected_keywords: tuple[str, ...] = ()


class RetrievalCaseResult(BaseModel):
    """Retrieval-only evaluation outcome for one case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    recall: float = Field(ge=0, le=1)
    hit: bool


class RetrievalEvaluationSummary(BaseModel):
    """Aggregate retrieval-only evaluation outcome across all cases."""

    model_config = ConfigDict(frozen=True)

    results: tuple[RetrievalCaseResult, ...] = Field(min_length=1)
    mean_recall: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)


class AgentCaseResult(BaseModel):
    """Full agent-pipeline evaluation outcome for one case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    answer: str
    iterations: int = Field(ge=0)
    expected_pages: tuple[int, ...]
    cited_pages: tuple[int, ...]
    page_hit: bool
    expected_keywords: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    keyword_coverage: float = Field(ge=0, le=1)


class AgentEvaluationSummary(BaseModel):
    """Aggregate full-pipeline evaluation outcome across all cases."""

    model_config = ConfigDict(frozen=True)

    results: tuple[AgentCaseResult, ...] = Field(min_length=1)
    page_hit_rate: float = Field(ge=0, le=1)
    mean_keyword_coverage: float = Field(ge=0, le=1)
    mean_iterations: float = Field(ge=0)