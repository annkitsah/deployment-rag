from __future__ import annotations

from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.provider import GenerationProvider


class GenerationService:
    """Application service coordinating text generation."""

    def __init__(
        self,
        *,
        provider: GenerationProvider,
    ) -> None:
        if not isinstance(provider, GenerationProvider):
            raise TypeError(
                "provider must implement GenerationProvider"
            )

        self._provider = provider

    @property
    def provider(self) -> GenerationProvider:
        """Return the configured generation provider."""

        return self._provider

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a response through the configured provider."""

        if not isinstance(request, GenerationRequest):
            raise TypeError(
                "request must be a GenerationRequest"
            )

        response = self._provider.generate(request)

        if not isinstance(response, GenerationResponse):
            raise TypeError(
                "generation provider must return GenerationResponse"
            )

        return response
