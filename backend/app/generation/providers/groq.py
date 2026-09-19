from __future__ import annotations

import time

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
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")

        if not default_model.strip():
            raise ValueError("default_model must not be empty")

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self._default_model = default_model.strip()
        self._api_key = api_key.strip()
        self._max_retries = max(0, max_retries)
        self._retry_backoff_sec = max(0.1, retry_backoff_sec)

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

        response = self._post_with_retries(payload)
        data = response.json()

        text = self._extract_text(data)
        usage = self._extract_usage(data)

        return GenerationResponse(
            text=text,
            model=str(data.get("model") or request.model),
            usage=usage,
        )

    def _post_with_retries(self, payload: dict[str, object]) -> httpx.Response:
        """POST /chat/completions with retries on 429 and 5xx."""

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt >= self._max_retries:
                        response.raise_for_status()

                    # Honor Retry-After when present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after)
                    else:
                        wait = self._retry_backoff_sec * (2**attempt)

                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_sec * (2**attempt))

            except httpx.HTTPStatusError:
                raise

            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_sec * (2**attempt))

        if last_error is not None:
            raise last_error

        raise RuntimeError("Groq request failed after retries")

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