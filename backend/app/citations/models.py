from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A single source page an answer was grounded in."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page_number: int = Field(ge=1)
    score: float = Field(ge=0)