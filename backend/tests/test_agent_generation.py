from types import SimpleNamespace

import pytest

from app.agents.answerer import GenerationAnswerer
from app.generation.models import (
    GenerationRequest,
    GenerationResponse,
)
from app.generation.provider import GenerationProvider
from app.generation.service import GenerationService
from app.retrieval.models import RetrievalResult, RetrievedContext


class StubGenerationProvider(GenerationProvider):
    """Capture generation requests and return a deterministic response."""

    def __init__(self, response_text: str = "Generated answer.") -> None:
        self.response_text = response_text
        self.requests: list[GenerationRequest] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(request)

        return GenerationResponse(
            model=request.model,
            text=self.response_text,
        )


def make_context(
    *,
    text: str = "Vectorless RAG uses lexical retrieval.",
) -> RetrievedContext:
    result = RetrievalResult(
        document_id="doc-001",
        page_number=1,
        text=text,
        score=1.0,
        matched_terms=("retrieval",),
    )

    return RetrievedContext(
        query="What is vectorless RAG?",
        results=(result,),
        text=text,
    )


def make_answerer(
    *,
    response_text: str = "Vectorless RAG uses lexical retrieval.",
    model: str = "mistral-small-latest",
    temperature: float = 0.2,
    max_tokens: int | None = 500,
) -> tuple[GenerationAnswerer, StubGenerationProvider]:
    provider = StubGenerationProvider(
        response_text=response_text,
    )

    service = GenerationService(
        provider=provider,
    )

    answerer = GenerationAnswerer(
        generation_service=service,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return answerer, provider


def test_generation_answerer_implements_answerer_contract() -> None:
    from app.agents.answerer import Answerer

    answerer, _ = make_answerer()

    assert isinstance(answerer, Answerer)


def test_generation_answerer_generates_answer_from_context() -> None:
    answerer, provider = make_answerer(
        response_text="Vectorless RAG uses lexical retrieval.",
    )

    answer = answerer.answer(
        "What is vectorless RAG?",
        make_context(),
    )

    assert answer == "Vectorless RAG uses lexical retrieval."
    assert len(provider.requests) == 1


def test_generation_answerer_builds_grounded_request() -> None:
    answerer, provider = make_answerer()

    answerer.answer(
        "What is vectorless RAG?",
        make_context(),
    )

    request = provider.requests[0]

    assert request.model == "mistral-small-latest"
    assert request.temperature == 0.2
    assert request.max_tokens == 500
    assert len(request.messages) == 3

    assert request.messages[0].role == "system"
    assert request.messages[1].role == "system"
    assert request.messages[2].role == "user"

    assert "Vectorless RAG uses lexical retrieval." in (
        request.messages[1].content
    )

    assert request.messages[2].content == "What is vectorless RAG?"


def test_generation_answerer_supports_custom_system_instruction() -> None:
    answerer, provider = make_answerer()

    answerer = GenerationAnswerer(
        generation_service=answerer.generation_service,
        model="mistral-small-latest",
        system_instruction="Answer only from evidence.",
    )

    answerer.answer(
        "What is vectorless RAG?",
        make_context(),
    )

    request = provider.requests[0]

    assert request.messages[0].content == (
        "Answer only from evidence."
    )


def test_generation_answerer_rejects_empty_query() -> None:
    answerer, _ = make_answerer()

    with pytest.raises(ValueError, match="query cannot be empty"):
        answerer.answer(
            "   ",
            make_context(),
        )


def test_generation_answerer_rejects_non_string_query() -> None:
    answerer, _ = make_answerer()

    with pytest.raises(TypeError, match="query must be a string"):
        answerer.answer(
            123,  # type: ignore[arg-type]
            make_context(),
        )


def test_generation_answerer_rejects_empty_context() -> None:
    answerer, _ = make_answerer()

    context = make_context(text="   ")

    with pytest.raises(
        ValueError,
        match="empty retrieved context",
    ):
        answerer.answer(
            "What is vectorless RAG?",
            context,
        )


def test_generation_answerer_rejects_invalid_context() -> None:
    answerer, _ = make_answerer()

    with pytest.raises(
        TypeError,
        match="context must be a RetrievedContext",
    ):
        answerer.answer(
            "question",
            SimpleNamespace(text="context"),  # type: ignore[arg-type]
        )


def test_generation_answerer_rejects_invalid_model() -> None:
    provider = StubGenerationProvider()
    service = GenerationService(provider=provider)

    with pytest.raises(ValueError, match="model cannot be empty"):
        GenerationAnswerer(
            generation_service=service,
            model="   ",
        )


def test_generation_answerer_rejects_invalid_temperature() -> None:
    provider = StubGenerationProvider()
    service = GenerationService(provider=provider)

    with pytest.raises(
        ValueError,
        match="temperature must be between",
    ):
        GenerationAnswerer(
            generation_service=service,
            model="mistral-small-latest",
            temperature=2.1,
        )


def test_generation_answerer_rejects_invalid_max_tokens() -> None:
    provider = StubGenerationProvider()
    service = GenerationService(provider=provider)

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater",
    ):
        GenerationAnswerer(
            generation_service=service,
            model="mistral-small-latest",
            max_tokens=0,
        )


def test_generation_answerer_rejects_empty_generated_answer() -> None:
    answerer, _ = make_answerer(
        response_text="   ",
    )

    with pytest.raises(
        ValueError,
        match="empty answer",
    ):
        answerer.answer(
            "What is vectorless RAG?",
            make_context(),
        )


def test_generation_answerer_does_not_mutate_context() -> None:
    answerer, provider = make_answerer()

    context = make_context()

    answerer.answer(
        "What is vectorless RAG?",
        context,
    )

    assert context.text == "Vectorless RAG uses lexical retrieval."
    assert len(provider.requests) == 1