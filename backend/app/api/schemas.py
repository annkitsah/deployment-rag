from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.documents.models import DocumentStatus


class DocumentResponse(BaseModel):
    """API representation of an ingested document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    page_count: int
    status: DocumentStatus
    created_at: datetime
    duplicate: bool = False


class DocumentListResponse(BaseModel):
    """API representation of a collection of ingested documents."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[DocumentResponse, ...]
    count: int = Field(ge=0)


class QueryRequest(BaseModel):
    """Incoming question for the agentic query endpoint."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, max_length=10_000)
    document_id: str | None = Field(
        default=None,
        description=(
            "When set, scopes retrieval to this document only, "
            "for every retrieval attempt including refine iterations."
        ),
    )


class CitationResponse(BaseModel):
    """API representation of a source page an answer was grounded in."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str | None = Field(
        default=None,
        description=(
            "The source document's filename, when it could still be "
            "resolved at response time."
        ),
    )
    page_number: int = Field(ge=1)
    score: float = Field(ge=0)


class QueryResponse(BaseModel):
    """API representation of an agent answer."""

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    iterations: int = Field(ge=0)
    citations: tuple[CitationResponse, ...] = ()