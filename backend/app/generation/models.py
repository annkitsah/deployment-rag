from pydantic import BaseModel, ConfigDict, Field


class GenerationMessage(BaseModel):
    """A single message supplied to a generation provider."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=100_000)


class GenerationUsage(BaseModel):
    """Token usage reported by a generation provider."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GenerationRequest(BaseModel):
    """Validated request entering the generation layer."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[GenerationMessage, ...] = Field(min_length=1)
    model: str = Field(min_length=1, max_length=200)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=100_000)


class GenerationResponse(BaseModel):
    """Normalized response returned by a generation provider."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=200)
    usage: GenerationUsage = Field(default_factory=GenerationUsage)