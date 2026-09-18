from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "agentic-true-vectorless-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    cors_allowed_origins: str = "http://localhost:3000"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_generation_timeout_ms: int = Field(
        default=120_000,
        gt=0,
    )

    mistral_api_key: str | None = None

    mistral_ocr_model: str = "mistral-ocr-latest"
    mistral_ocr_timeout_ms: int = Field(
        default=120_000,
        gt=0,
    )

    mistral_generation_model: str = "mistral-small-latest"
    mistral_generation_timeout_ms: int = Field(
        default=120_000,
        gt=0,
    )

    generation_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    generation_max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
    )

    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    index_dir: str = "./data/indexes"
    metadata_dir: str = "./data/metadata"

    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    retrieval_max_pages: int = Field(
        default=20,
        ge=1,
        le=500,
    )

    retrieval_max_chars: int = Field(
        default=100_000,
        ge=1_000,
        le=1_000_000,
    )

    ocr_enabled: bool = True

    # Generation provider: "groq" | "mistral" | "ollama"
    generation_provider: str = "groq"

    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_generation_timeout_ms: int = Field(
        default=120_000,
        gt=0,
    )

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS origins setting into a list."""

        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()