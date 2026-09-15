import pytest
from pydantic import ValidationError

from app.generation.models import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)


def make_message(
    *,
    role: str = "user",
    content: str = "What is vectorless RAG?",
) -> GenerationMessage:
    return GenerationMessage(
        role=role,
        content=content,
    )


def make_request(
    *,
    messages: tuple[GenerationMessage, ...] | None = None,
    model: str = "test-model",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        messages=messages or (make_message(),),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_generation_message_accepts_valid_message() -> None:
    message = make_message()

    assert message.role == "user"
    assert message.content == "What is vectorless RAG?"


def test_generation_message_rejects_empty_role() -> None:
    with pytest.raises(ValidationError):
        GenerationMessage(
            role="",
            content="test",
        )


def test_generation_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        GenerationMessage(
            role="user",
            content="",
        )


def test_generation_usage_defaults_to_zero() -> None:
    usage = GenerationUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_generation_usage_calculates_total_tokens() -> None:
    usage = GenerationUsage(
        prompt_tokens=100,
        completion_tokens=50,
    )

    assert usage.total_tokens == 150


def test_generation_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        GenerationUsage(
            prompt_tokens=-1,
        )


def test_generation_request_accepts_valid_request() -> None:
    request = make_request(
        temperature=0.7,
        max_tokens=500,
    )

    assert request.model == "test-model"
    assert request.temperature == 0.7
    assert request.max_tokens == 500
    assert len(request.messages) == 1


def test_generation_request_requires_messages() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            messages=(),
            model="test-model",
        )


def test_generation_request_rejects_empty_model() -> None:
    with pytest.raises(ValidationError):
        make_request(model="")


def test_generation_request_rejects_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        make_request(temperature=-0.1)

    with pytest.raises(ValidationError):
        make_request(temperature=2.1)


def test_generation_request_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValidationError):
        make_request(max_tokens=0)

    with pytest.raises(ValidationError):
        make_request(max_tokens=-1)


def test_generation_request_is_immutable() -> None:
    request = make_request()

    with pytest.raises(ValidationError):
        request.model = "another-model"  # type: ignore[misc]


def test_generation_response_accepts_valid_response() -> None:
    response = GenerationResponse(
        text="Vectorless RAG retrieves information without embeddings.",
        model="test-model",
    )

    assert response.text.startswith("Vectorless RAG")
    assert response.model == "test-model"
    assert response.usage.total_tokens == 0


def test_generation_response_accepts_usage() -> None:
    response = GenerationResponse(
        text="Generated answer.",
        model="test-model",
        usage=GenerationUsage(
            prompt_tokens=25,
            completion_tokens=10,
        ),
    )

    assert response.usage.prompt_tokens == 25
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 35


def test_generation_response_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        GenerationResponse(
            text="",
            model="test-model",
        )


def test_generation_response_rejects_empty_model() -> None:
    with pytest.raises(ValidationError):
        GenerationResponse(
            text="Generated answer.",
            model="",
        )