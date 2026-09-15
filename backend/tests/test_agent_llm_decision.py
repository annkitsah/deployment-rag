import json

import pytest

from app.agents.decision import (
    AgentDecisionEngine,
    LLMDecisionEngine,
)
from app.agents.models import AgentDecisionType
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


def make_engine(
    *,
    response_text: str | None = None,
    raise_error: bool = False,
) -> tuple[LLMDecisionEngine, StubGenerationProvider]:
    provider = StubGenerationProvider(
        response_text=response_text,
        raise_error=raise_error,
    )

    service = GenerationService(provider=provider)

    engine = LLMDecisionEngine(
        service,
        model="llama3.2:3b",
    )

    return engine, provider


def _state_with_context(
    *,
    page_count: int = 1,
    iteration: int = 0,
) -> AgentState:
    state = AgentState(
        original_query="What is vectorless RAG?",
        current_query="What is vectorless RAG?",
    )

    for _ in range(iteration):
        state.advance_iteration()

    if page_count > 0:
        results = tuple(
            RetrievalResult(
                document_id="doc-1",
                page_number=index + 1,
                text="Some retrieved text.",
                score=1.0,
                matched_terms=("retrieval",),
            )
            for index in range(page_count)
        )

        state.add_context(
            RetrievedContext(
                query=state.current_query,
                results=results,
                text="Some retrieved text.",
            )
        )
    else:
        state.add_context(
            RetrievedContext(
                query=state.current_query,
                results=(),
                text="",
            )
        )

    return state


def test_rejects_invalid_generation_service() -> None:
    with pytest.raises(TypeError):
        LLMDecisionEngine(
            "not a service",  # type: ignore[arg-type]
            model="llama3.2:3b",
        )


def test_rejects_empty_model() -> None:
    provider = StubGenerationProvider(response_text="{}")
    service = GenerationService(provider=provider)

    with pytest.raises(ValueError, match="model cannot be empty"):
        LLMDecisionEngine(service, model="   ")


def test_no_context_delegates_to_fallback_without_llm_call() -> None:
    engine, provider = make_engine(response_text="unused")

    state = AgentState(
        original_query="q",
        current_query="q",
    )

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.REFINE
    assert decision.next_query is None
    assert len(provider.requests) == 0


def test_iteration_cap_stops_without_llm_call() -> None:
    engine, provider = make_engine(response_text="unused")

    state = _state_with_context(page_count=1, iteration=3)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.STOP
    assert len(provider.requests) == 0


def test_llm_answer_verdict_is_honored() -> None:
    engine, provider = make_engine(
        response_text=json.dumps(
            {
                "decision": "answer",
                "reason": "The excerpt directly answers the question.",
            }
        ),
    )

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.ANSWER
    assert len(provider.requests) == 1


def test_llm_refine_verdict_carries_next_query() -> None:
    engine, _ = make_engine(
        response_text=json.dumps(
            {
                "decision": "refine",
                "reason": "The excerpt is off-topic.",
                "next_query": "better keywords",
            }
        ),
    )

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.REFINE
    assert decision.next_query == "better keywords"


def test_llm_response_wrapped_in_markdown_fence_is_parsed() -> None:
    fenced = (
        "```json\n"
        + json.dumps({"decision": "answer", "reason": "ok"})
        + "\n```"
    )

    engine, _ = make_engine(response_text=fenced)

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.ANSWER


def test_unparseable_llm_response_falls_back_to_heuristic() -> None:
    engine, _ = make_engine(response_text="not json at all")

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    # Heuristic fallback: non-empty context -> ANSWER.
    assert decision.decision_type is AgentDecisionType.ANSWER
    assert decision.reason == "Relevant retrieved context is available."


def test_generation_failure_falls_back_to_heuristic() -> None:
    engine, _ = make_engine(raise_error=True)

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.ANSWER
    assert decision.reason == "Relevant retrieved context is available."


def test_invalid_decision_value_falls_back_to_heuristic() -> None:
    engine, _ = make_engine(
        response_text=json.dumps(
            {"decision": "maybe", "reason": "unsure"}
        ),
    )

    state = _state_with_context(page_count=1)

    decision = engine.decide(state)

    assert decision.decision_type is AgentDecisionType.ANSWER


def test_custom_fallback_is_used() -> None:
    provider = StubGenerationProvider(raise_error=True)
    service = GenerationService(provider=provider)

    custom_fallback = AgentDecisionEngine(max_iterations=1)

    engine = LLMDecisionEngine(
        service,
        model="llama3.2:3b",
        fallback=custom_fallback,
    )

    assert engine.fallback is custom_fallback

    state = _state_with_context(page_count=1, iteration=1)

    decision = engine.decide(state)

    # custom_fallback has max_iterations=1, so iteration=1 hits the cap.
    assert decision.decision_type is AgentDecisionType.STOP


def test_model_and_generation_service_are_exposed() -> None:
    engine, provider = make_engine(response_text="{}")

    assert engine.model == "llama3.2:3b"
    assert engine.generation_service.provider is provider