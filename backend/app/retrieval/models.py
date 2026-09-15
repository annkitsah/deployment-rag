from pydantic import BaseModel, ConfigDict, Field


class RetrievalQuery(BaseModel):
    """Validated user query entering the retrieval layer."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=10, ge=1, le=100)
    document_id: str | None = None


class RetrievalResult(BaseModel):
    """A single ranked page returned by retrieval."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    score: float = Field(ge=0)
    matched_terms: tuple[str, ...] = ()

    @property
    def page_id(self) -> str:
        return f"{self.document_id}:page:{self.page_number}"


class RetrievalResponse(BaseModel):
    """Complete retrieval response."""

    model_config = ConfigDict(frozen=True)

    query: RetrievalQuery
    results: tuple[RetrievalResult, ...]

    @property
    def result_count(self) -> int:
        return len(self.results)


class RetrievedContext(BaseModel):
    """Bounded context assembled from retrieved pages."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[RetrievalResult, ...]
    text: str

    @property
    def page_count(self) -> int:
        return len(self.results)