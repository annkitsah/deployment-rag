from __future__ import annotations

import httpx

from app.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from app.generation.provider import GenerationProvider


class OllamaGenerationProvider(GenerationProvider):
    """Generation provider backed by a local Ollama chat completion API."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
        timeout_ms: int = 120_000,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        if not default_model.strip():
            raise ValueError("default_model must not be empty")

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self._default_model = default_model.strip()

        self._client = httpx.Client(
            base_url=base_url.strip().rstrip("/"),
            timeout=timeout_ms / 1000,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Return the configured default model identifier."""
        return self._default_model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate text using a local Ollama chat completion API."""

        if not isinstance(request, GenerationRequest):
            raise TypeError(
                "request must be a GenerationRequest"
            )

        options: dict[str, object] = {
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": False,
            "options": options,
        }

        try:
            http_response = self._client.post(
                "/api/chat",
                json=payload,
            )
            http_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama generation request failed: {exc}"
            ) from exc

        body = http_response.json()

        text = self._extract_text(body)
        usage = self._extract_usage(body)

        return GenerationResponse(
            text=text,
            model=request.model,
            usage=usage,
        )

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Extract generated text from an Ollama response body."""

        message = body.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Ollama response contains no message"
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama response contains no text content"
            )

        return content.strip()

    @staticmethod
    def _extract_usage(body: dict) -> GenerationUsage:
        """Normalize usage information from an Ollama response body."""

        prompt_tokens = body.get("prompt_eval_count") or 0
        completion_tokens = body.get("eval_count") or 0

        return GenerationUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )