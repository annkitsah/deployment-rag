import pytest

from app.generation.models import GenerationMessage, GenerationRequest
from app.generation.prompt import PromptBuilder


def make_builder() -> PromptBuilder:
    return PromptBuilder(
        question="What is vectorless RAG?",
        context="Vectorless RAG retrieves documents without vector embeddings.",
    )


def test_prompt_builder_accepts_valid_input() -> None:
    builder = make_builder()

    assert builder.question == "What is vectorless RAG?"
    assert builder.context == (
        "Vectorless RAG retrieves documents without vector embeddings."
    )


def test_prompt_builder_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        PromptBuilder(
            question="   ",
            context="Some context.",
        )


def test_prompt_builder_rejects_empty_context() -> None:
    with pytest.raises(ValueError, match="context must not be empty"):
        PromptBuilder(
            question="What is RAG?",
            context="   ",
        )


def test_prompt_builder_rejects_non_string_question() -> None:
    with pytest.raises(TypeError, match="question must be a string"):
        PromptBuilder(
            question=123,  # type: ignore[arg-type]
            context="Some context.",
        )


def test_prompt_builder_rejects_non_string_context() -> None:
    with pytest.raises(TypeError, match="context must be a string"):
        PromptBuilder(
            question="What is RAG?",
            context=123,  # type: ignore[arg-type]
        )


def test_prompt_builder_uses_default_system_instruction() -> None:
    builder = make_builder()

    assert builder.system_instruction == PromptBuilder.DEFAULT_SYSTEM_INSTRUCTION


def test_prompt_builder_accepts_custom_system_instruction() -> None:
    builder = PromptBuilder(
        question="What is RAG?",
        context="RAG retrieves relevant information.",
        system_instruction="Answer concisely.",
    )

    assert builder.system_instruction == "Answer concisely."


def test_prompt_builder_rejects_empty_system_instruction() -> None:
    with pytest.raises(
        ValueError,
        match="system_instruction must not be empty",
    ):
        PromptBuilder(
            question="What is RAG?",
            context="Some context.",
            system_instruction="   ",
        )


def test_build_system_message() -> None:
    builder = make_builder()

    message = builder.build_system_message()

    assert isinstance(message, GenerationMessage)
    assert message.role == "system"
    assert message.content == PromptBuilder.DEFAULT_SYSTEM_INSTRUCTION


def test_build_context_message() -> None:
    builder = make_builder()

    message = builder.build_context_message()

    assert isinstance(message, GenerationMessage)
    assert message.role == "system"
    assert "retrieved context" in message.content
    assert builder.context in message.content


def test_build_user_message() -> None:
    builder = make_builder()

    message = builder.build_user_message()

    assert isinstance(message, GenerationMessage)
    assert message.role == "user"
    assert message.content == builder.question


def test_build_messages_returns_deterministic_order() -> None:
    builder = make_builder()

    messages = builder.build_messages()

    assert len(messages) == 3
    assert messages[0].role == "system"
    assert messages[1].role == "system"
    assert messages[2].role == "user"

    assert messages[0].content == builder.system_instruction
    assert builder.context in messages[1].content
    assert messages[2].content == builder.question


def test_build_request_returns_generation_request() -> None:
    builder = make_builder()

    request = builder.build_request(
        model="mistral-small-latest",
        temperature=0.2,
        max_tokens=500,
    )

    assert isinstance(request, GenerationRequest)
    assert request.model == "mistral-small-latest"
    assert request.temperature == 0.2
    assert request.max_tokens == 500
    assert len(request.messages) == 3


def test_build_request_propagates_generation_configuration() -> None:
    builder = make_builder()

    request = builder.build_request(
        model="test-model",
        temperature=0.7,
        max_tokens=None,
    )

    assert request.model == "test-model"
    assert request.temperature == 0.7
    assert request.max_tokens is None


def test_build_request_rejects_invalid_model() -> None:
    builder = make_builder()

    with pytest.raises(ValueError):
        builder.build_request(model="   ")


def test_build_messages_does_not_mutate_builder() -> None:
    builder = make_builder()

    first = builder.build_messages()
    second = builder.build_messages()

    assert first == second