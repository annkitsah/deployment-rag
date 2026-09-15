import pytest

from app.agents.models import AgentDecision, AgentDecisionType
from app.agents.refiner import AgentQueryRefiner, LLMQueryRefiner
from app.agents.state import AgentState
from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.provider import GenerationProvider
from app.generation.service import GenerationService
from app.retrieval.models import RetrievalResult, RetrievedContext


class StubGenerationProvider(GenerationProvider):
    """Return a scripted response, or raise if configured to fail."""

    def __init__(
        self,
        *,
        response_text: str | None = None,
        raise_error: bool = False,
    ) -> None:
        self.response_text = response_text
        self.raise_error = raise_error
        self.requests: list[GenerationRequest] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(request)

        if self.raise_error:
            raise RuntimeError("stub provider failure")

        assert self.response_text is not None

        return GenerationResponse(
            model=request.model,
            text=self.response_text,
        )


def make_refiner(
    *,
    response_text: str | None = None,
    raise_error: bool = False,
) -> tuple[LLMQueryRefiner, StubGenerationProvider]:
    provider = StubGenerationProvider(
        response_text=response_text,
        raise_error=raise_error,
    )

    service = GenerationService(provider=provider)

    refiner = LLMQueryRefiner(
        service,
        model="llama3.2:3b",
    )

    return refiner, provider


def _empty_state(
    *,
    reason: str | None = None,
) -> AgentState:
    state = AgentState(
        original_query="What is vectorless RAG?",
        current_query="What is vectorless RAG?",
    )

    state.add_context(
        RetrievedContext(
            query=state.current_query,
            results=(),
            text="",
        )
    )

    if reason is not None:
        state.set_decision(
            AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason=reason,
            )
        )

    return state


def test_rejects_invalid_generation_service() -> None:
    with pytest.raises(TypeError):
        LLMQueryRefiner(
            "not a service",  # type: ignore[arg-type]
            model="llama3.2:3b",
        )


def test_rejects_empty_model() -> None:
    provider = StubGenerationProvider(response_text="x")
    service = GenerationService(provider=provider)

    with pytest.raises(ValueError, match="model cannot be empty"):
        LLMQueryRefiner(service, model="   ")


def test_rejects_empty_current_query() -> None:
    refiner, _ = make_refiner(response_text="rewritten query")

    state = AgentState(
        original_query="q",
        current_query="   ",
    )

    with pytest.raises(ValueError, match="current query cannot be empty"):
        refiner.refine(state)


def test_llm_rewrite_is_used() -> None:
    refiner, provider = make_refiner(
        response_text="generative ai model architecture",
    )

    state = _empty_state(reason="No retrieved context is available.")

    result = refiner.refine(state)

    assert result == "generative ai model architecture"
    assert len(provider.requests) == 1


def test_llm_rewrite_strips_surrounding_quotes() -> None:
    refiner, _ = make_refiner(
        response_text='"generative ai architecture"',
    )

    state = _empty_state()

    result = refiner.refine(state)

    assert result == "generative ai architecture"


def test_identical_rewrite_falls_back_to_heuristic() -> None:
    refiner, _ = make_refiner(
        response_text="What is vectorless RAG?",
    )

    state = _empty_state()

    result = refiner.refine(state)

    # LLM echoed the same query back -> not real progress, use fallback.
    assert result != "What is vectorless RAG?"
    assert "relevant definition explanation details" in result


def test_empty_rewrite_falls_back_to_heuristic() -> None:
    refiner, _ = make_refiner(response_text="   ")

    state = _empty_state()

    result = refiner.refine(state)

    assert "relevant definition explanation details" in result


def test_generation_failure_falls_back_to_heuristic() -> None:
    refiner, _ = make_refiner(raise_error=True)

    state = _empty_state()

    result = refiner.refine(state)

    assert "relevant definition explanation details" in result


def test_prompt_includes_decision_reason_when_available() -> None:
    refiner, provider = make_refiner(response_text="new query")

    state = _empty_state(reason="The excerpts were off-topic.")

    refiner.refine(state)

    sent_request = provider.requests[0]
    user_message = sent_request.messages[-1]

    assert "The excerpts were off-topic." in user_message.content


def test_custom_fallback_is_used() -> None:
    provider = StubGenerationProvider(raise_error=True)
    service = GenerationService(provider=provider)

    custom_fallback = AgentQueryRefiner()

    refiner = LLMQueryRefiner(
        service,
        model="llama3.2:3b",
        fallback=custom_fallback,
    )

    assert refiner.fallback is custom_fallback

    state = _empty_state()

    result = refiner.refine(state)

    assert "relevant definition explanation details" in result


def test_model_and_generation_service_are_exposed() -> None:
    refiner, provider = make_refiner(response_text="x")

    assert refiner.model == "llama3.2:3b"
    assert refiner.generation_service.provider is provider