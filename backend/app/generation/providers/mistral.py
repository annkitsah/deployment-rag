from __future__ import annotations

from mistralai import Mistral

from app.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from app.generation.provider import GenerationProvider


class MistralGenerationProvider(GenerationProvider):
    """Generation provider backed by the Mistral chat completion API."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = "mistral-small-latest",
        timeout_ms: int = 120_000,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Mistral API key must not be empty")

        if not default_model.strip():
            raise ValueError("default_model must not be empty")

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self._default_model = default_model.strip()

        self._client = Mistral(
            api_key=api_key,
            timeout_ms=timeout_ms,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "mistral"

    @property
    def model_name(self) -> str:
        """Return the configured default model identifier."""
        return self._default_model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate text using the Mistral chat completion API."""

        if not isinstance(request, GenerationRequest):
            raise TypeError(
                "request must be a GenerationRequest"
            )

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        kwargs: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        response = self._client.chat.complete(**kwargs)

        text = self._extract_text(response)
        usage = self._extract_usage(response)

        return GenerationResponse(
            text=text,
            model=request.model,
            usage=usage,
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        """Extract generated text from a Mistral response."""

        choices = getattr(response, "choices", None)

        if not choices:
            raise ValueError(
                "Mistral response contains no choices"
            )

        message = getattr(choices[0], "message", None)

        if message is None:
            raise ValueError(
                "Mistral response choice contains no message"
            )

        content = getattr(message, "content", None)

        if isinstance(content, str):
            text = content.strip()

            if not text:
                raise ValueError(
                    "Mistral response contains empty text"
                )

            return text

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                item_text = getattr(item, "text", None)

                if item_text:
                    text_parts.append(str(item_text))

            text = "".join(text_parts).strip()

            if text:
                return text

        raise ValueError(
            "Mistral response contains no text content"
        )

    @staticmethod
    def _extract_usage(response: object) -> GenerationUsage:
        """Normalize usage information from a Mistral response."""

        usage = getattr(response, "usage", None)

        if usage is None:
            return GenerationUsage()

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        )

        return GenerationUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )