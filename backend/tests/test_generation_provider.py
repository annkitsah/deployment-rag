import pytest

from app.generation.models import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from app.generation.provider import GenerationProvider


def make_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            GenerationMessage(
                role="user",
                content="Hello",
            ),
        ),
        model="test-model",
    )


def make_response() -> GenerationResponse:
    return GenerationResponse(
        text="Hello!",
        model="test-model",
        usage=GenerationUsage(
            prompt_tokens=5,
            completion_tokens=2,
        ),
    )


def test_generation_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        GenerationProvider()  # type: ignore[abstract]


def test_concrete_provider_implements_generation_contract() -> None:
    class StubGenerationProvider(GenerationProvider):
        def generate(
            self,
            request: GenerationRequest,
        ) -> GenerationResponse:
            return make_response()

    provider = StubGenerationProvider()

    assert isinstance(provider, GenerationProvider)


def test_concrete_provider_receives_generation_request() -> None:
    captured_requests: list[GenerationRequest] = []

    class StubGenerationProvider(GenerationProvider):
        def generate(
            self,
            request: GenerationRequest,
        ) -> GenerationResponse:
            captured_requests.append(request)
            return make_response()

    request = make_request()
    provider = StubGenerationProvider()

    response = provider.generate(request)

    assert captured_requests == [request]
    assert response.text == "Hello!"
    assert response.model == "test-model"


def test_concrete_provider_returns_generation_response() -> None:
    class StubGenerationProvider(GenerationProvider):
        def generate(
            self,
            request: GenerationRequest,
        ) -> GenerationResponse:
            return make_response()

    provider = StubGenerationProvider()

    response = provider.generate(make_request())

    assert isinstance(response, GenerationResponse)
    assert response.text == "Hello!"
    assert response.usage.total_tokens == 7


def test_provider_can_use_request_configuration() -> None:
    class StubGenerationProvider(GenerationProvider):
        def generate(
            self,
            request: GenerationRequest,
        ) -> GenerationResponse:
            return GenerationResponse(
                text=request.messages[0].content,
                model=request.model,
            )

    request = GenerationRequest(
        messages=(
            GenerationMessage(
                role="user",
                content="Generate this.",
            ),
        ),
        model="configured-model",
        temperature=0.7,
        max_tokens=100,
    )

    provider = StubGenerationProvider()

    response = provider.generate(request)

    assert response.text == "Generate this."
    assert response.model == "configured-model"
