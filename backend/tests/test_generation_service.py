from types import SimpleNamespace

import pytest

from app.generation.models import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from app.generation.provider import GenerationProvider
from app.generation.service import GenerationService


class StubGenerationProvider(GenerationProvider):
    """Test provider implementing the generation contract."""

    def __init__(
        self,
        response: GenerationResponse,
    ) -> None:
        self.response = response
        self.received_request: GenerationRequest | None = None

    @property
    def provider_name(self) -> str:
        return "stub"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.received_request = request
        return self.response


def make_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            GenerationMessage(
                role="user",
                content="Explain vectorless RAG.",
            ),
        ),
        model="test-model",
        temperature=0.2,
        max_tokens=100,
    )


def make_response() -> GenerationResponse:
    return GenerationResponse(
        text="Vectorless RAG retrieves information without embeddings.",
        model="test-model",
        usage=GenerationUsage(
            prompt_tokens=10,
            completion_tokens=8,
        ),
    )


def test_generation_service_accepts_generation_provider() -> None:
    provider = StubGenerationProvider(make_response())

    service = GenerationService(
        provider=provider,
    )

    assert service.provider is provider


def test_generation_service_rejects_invalid_provider() -> None:
    invalid_provider = SimpleNamespace()

    with pytest.raises(
        TypeError,
        match="provider must implement GenerationProvider",
    ):
        GenerationService(
            provider=invalid_provider,
        )


def test_generation_service_rejects_invalid_request() -> None:
    provider = StubGenerationProvider(make_response())
    service = GenerationService(provider=provider)

    with pytest.raises(
        TypeError,
        match="request must be a GenerationRequest",
    ):
        service.generate("invalid request")  # type: ignore[arg-type]


def test_generation_service_delegates_request_to_provider() -> None:
    provider = StubGenerationProvider(make_response())
    service = GenerationService(provider=provider)

    request = make_request()

    service.generate(request)

    assert provider.received_request is request


def test_generation_service_returns_provider_response() -> None:
    response = make_response()
    provider = StubGenerationProvider(response)
    service = GenerationService(provider=provider)

    result = service.generate(make_request())

    assert result is response
    assert result.text == response.text
    assert result.model == response.model
    assert result.usage == response.usage


class InvalidResponseProvider(GenerationProvider):
    """Provider returning an invalid response for service validation."""

    @property
    def provider_name(self) -> str:
        return "invalid"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return "invalid response"  # type: ignore[return-value]


def test_generation_service_rejects_invalid_provider_response() -> None:
    provider = InvalidResponseProvider()
    service = GenerationService(provider=provider)

    with pytest.raises(
        TypeError,
        match="generation provider must return GenerationResponse",
    ):
        service.generate(make_request())
