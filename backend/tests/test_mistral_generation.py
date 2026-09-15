from types import SimpleNamespace

import pytest

from app.generation.models import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
)
from app.generation.provider import GenerationProvider
from app.generation.providers.mistral import MistralGenerationProvider


class StubChat:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)
        return self.response


class StubClient:
    def __init__(self, response: object) -> None:
        self.chat = StubChat(response)


def make_response(
    *,
    text: str = "Generated answer.",
    model: str = "mistral-small-latest",
    prompt_tokens: int = 12,
    completion_tokens: int = 7,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def make_request(
    *,
    model: str = "mistral-small-latest",
    temperature: float = 0.2,
    max_tokens: int | None = 100,
) -> GenerationRequest:
    return GenerationRequest(
        messages=(
            GenerationMessage(
                role="system",
                content="You are a precise assistant.",
            ),
            GenerationMessage(
                role="user",
                content="Explain vectorless RAG.",
            ),
        ),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_mistral_generation_provider_is_generation_provider() -> None:
    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    assert isinstance(provider, GenerationProvider)


def test_mistral_generation_provider_rejects_empty_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="Mistral API key must not be empty",
    ):
        MistralGenerationProvider(
            api_key="   ",
        )


def test_mistral_generation_provider_rejects_empty_default_model() -> None:
    with pytest.raises(
        ValueError,
        match="default_model must not be empty",
    ):
        MistralGenerationProvider(
            api_key="test-key",
            default_model="   ",
        )


def test_mistral_generation_provider_exposes_provider_name() -> None:
    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    assert provider.provider_name == "mistral"


def test_mistral_generation_provider_sends_request_to_mistral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = make_response()
    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    request = make_request()

    result = provider.generate(request)

    assert result.text == "Generated answer."
    assert result.model == "mistral-small-latest"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 19

    assert len(client.chat.calls) == 1

    call = client.chat.calls[0]

    assert call["model"] == "mistral-small-latest"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 100
    assert call["messages"] == [
        {
            "role": "system",
            "content": "You are a precise assistant.",
        },
        {
            "role": "user",
            "content": "Explain vectorless RAG.",
        },
    ]


def test_mistral_generation_provider_omits_max_tokens_when_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = make_response()
    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    request = make_request(
        max_tokens=None,
    )

    provider.generate(request)

    call = client.chat.calls[0]

    assert "max_tokens" not in call


def test_mistral_generation_provider_rejects_invalid_request() -> None:
    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    with pytest.raises(
        TypeError,
        match="request must be a GenerationRequest",
    ):
        provider.generate("invalid")  # type: ignore[arg-type]


def test_mistral_generation_provider_rejects_missing_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        model="mistral-small-latest",
        choices=[],
        usage=None,
    )

    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    with pytest.raises(
        ValueError,
        match="response contains no choices",
    ):
        provider.generate(make_request())


def test_mistral_generation_provider_rejects_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = make_response(
        text="   ",
    )

    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    with pytest.raises(
        ValueError,
        match="response contains empty text",
    ):
        provider.generate(make_request())


def test_mistral_generation_provider_defaults_missing_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        model="mistral-small-latest",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Generated answer.",
                )
            )
        ],
        usage=None,
    )

    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
    )

    result = provider.generate(make_request())

    assert result == GenerationResponse(
        text="Generated answer.",
        model="mistral-small-latest",
    )


def test_mistral_generation_provider_uses_request_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = make_response(
        model="mistral-large-latest",
    )
    client = StubClient(response)

    monkeypatch.setattr(
        "app.generation.providers.mistral.Mistral",
        lambda **_: client,
    )

    provider = MistralGenerationProvider(
        api_key="test-key",
        default_model="mistral-small-latest",
    )

    request = make_request(
        model="mistral-large-latest",
    )

    result = provider.generate(request)

    assert result.model == "mistral-large-latest"
    assert client.chat.calls[0]["model"] == "mistral-large-latest"

def test_model_name_returns_configured_default_model() -> None:
    provider = MistralGenerationProvider(
        api_key="test-key",
        default_model="mistral-small-latest",
    )

    assert provider.model_name == "mistral-small-latest"