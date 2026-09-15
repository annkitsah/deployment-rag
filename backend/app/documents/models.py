from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    REGISTERED = "registered"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Metadata describing an ingested source document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    source_path: str
    file_hash: str = Field(min_length=64, max_length=64)
    file_size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    status: DocumentStatus = DocumentStatus.REGISTERED
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class PageRecord(BaseModel):
    """Canonical page-level representation used by downstream retrieval."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def page_id(self) -> str:
        return f"{self.document_id}:page:{self.page_number}"