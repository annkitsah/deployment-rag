from __future__ import annotations

import httpx

from app.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from app.generation.provider import GenerationProvider


class GroqGenerationProvider(GenerationProvider):
    """Generation provider backed by the Groq OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = "llama-3.1-8b-instant",
        timeout_ms: int = 120_000,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")

        if not default_model.strip():
            raise ValueError("default_model must not be empty")

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self._default_model = default_model.strip()
        self._api_key = api_key.strip()

        self._client = httpx.Client(
            base_url=base_url.strip().rstrip("/"),
            timeout=timeout_ms / 1000,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._default_model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")

        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "temperature": request.temperature,
            "stream": False,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        text = self._extract_text(data)
        usage = self._extract_usage(data)

        return GenerationResponse(
            text=text,
            model=str(data.get("model") or request.model),
            usage=usage,
        )

    @staticmethod
    def _extract_text(data: object) -> str:
        if not isinstance(data, dict):
            raise ValueError("Groq response is not a JSON object")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError("Groq response contains no choices")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError("Groq response choice contains no message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq response contains empty text")

        return content.strip()

    @staticmethod
    def _extract_usage(data: object) -> GenerationUsage:
        if not isinstance(data, dict):
            return GenerationUsage()

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return GenerationUsage()

        return GenerationUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )