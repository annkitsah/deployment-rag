from types import SimpleNamespace

import pytest

from app.generation.models import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
)
from app.generation.provider import GenerationProvider
from app.generation.providers.ollama import OllamaGenerationProvider


class StubHTTPResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class StubHTTPClient:
    def __init__(self, response_body: dict) -> None:
        self._response_body = response_body
        self.calls: list[dict[str, object]] = []

    def post(self, path: str, *, json: dict) -> StubHTTPResponse:
        self.calls.append({"path": path, "json": json})
        return StubHTTPResponse(self._response_body)


def make_response_body(
    *,
    text: str = "Generated answer.",
    prompt_tokens: int = 12,
    completion_tokens: int = 7,
) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": text,
        },
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def make_request(
    *,
    model: str = "llama3.2",
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


def test_ollama_generation_provider_is_generation_provider() -> None:
    provider = OllamaGenerationProvider()

    assert isinstance(provider, GenerationProvider)


def test_ollama_generation_provider_rejects_empty_base_url() -> None:
    with pytest.raises(
        ValueError,
        match="base_url must not be empty",
    ):
        OllamaGenerationProvider(base_url="   ")


def test_ollama_generation_provider_rejects_empty_default_model() -> None:
    with pytest.raises(
        ValueError,
        match="default_model must not be empty",
    ):
        OllamaGenerationProvider(default_model="   ")


def test_ollama_generation_provider_exposes_provider_name() -> None:
    provider = OllamaGenerationProvider()

    assert provider.provider_name == "ollama"


def test_ollama_generation_provider_sends_request_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = StubHTTPClient(make_response_body())

    monkeypatch.setattr(
        "app.generation.providers.ollama.httpx.Client",
        lambda **_: client,
    )

    provider = OllamaGenerationProvider(default_model="llama3.2")

    request = make_request()

    result = provider.generate(request)

    assert isinstance(result, GenerationResponse)
    assert result.text == "Generated answer."
    assert result.model == "llama3.2"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 19

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == "/api/chat"
    assert client.calls[0]["json"]["model"] == "llama3.2"
    assert client.calls[0]["json"]["stream"] is False
    assert (
        client.calls[0]["json"]["options"]["temperature"] == 0.2
    )
    assert (
        client.calls[0]["json"]["options"]["num_predict"] == 100
    )


def test_ollama_generation_provider_raises_on_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = StubHTTPClient(make_response_body(text=""))

    monkeypatch.setattr(
        "app.generation.providers.ollama.httpx.Client",
        lambda **_: client,
    )

    provider = OllamaGenerationProvider()

    with pytest.raises(
        ValueError,
        match="Ollama response contains no text content",
    ):
        provider.generate(make_request())


def test_ollama_generation_provider_wraps_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class FailingClient:
        def post(self, path: str, *, json: dict) -> object:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "app.generation.providers.ollama.httpx.Client",
        lambda **_: FailingClient(),
    )

    provider = OllamaGenerationProvider()

    with pytest.raises(
        RuntimeError,
        match="Ollama generation request failed",
    ):
        provider.generate(make_request())