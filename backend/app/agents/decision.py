import json
import logging
from abc import ABC, abstractmethod

from app.agents.models import AgentDecision, AgentDecisionType
from app.agents.state import AgentState
from app.generation.models import GenerationMessage, GenerationRequest
from app.generation.service import GenerationService

logger = logging.getLogger(__name__)


class DecisionEngine(ABC):
    """Abstraction for deciding the next agent action from state."""

    @abstractmethod
    def decide(
        self,
        state: AgentState,
    ) -> AgentDecision:
        """Determine whether the agent should answer, refine, or stop."""

        raise NotImplementedError


class AgentDecisionEngine(DecisionEngine):
    """Deterministic heuristic decision engine (no LLM, no network).

    Only inspects whether retrieval produced any pages at all. Used as
    the default fallback for `LLMDecisionEngine`, and directly whenever
    a dependency-free, fully deterministic decision engine is wanted
    (tests, offline environments).
    """

    def __init__(
        self,
        *,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        self.max_iterations = max_iterations

    def decide(
        self,
        state: AgentState,
    ) -> AgentDecision:
        """Determine whether the agent should answer, refine, or stop."""

        if not isinstance(state, AgentState):
            raise TypeError(
                "state must be an AgentState"
            )

        if state.iteration >= self.max_iterations:
            return AgentDecision(
                decision_type=AgentDecisionType.STOP,
                reason="Maximum agent iterations reached.",
            )

        if not state.contexts:
            return AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason="No retrieved context is available.",
                next_query=state.current_query,
            )

        latest_context = state.contexts[-1]

        if latest_context.page_count == 0:
            return AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason=(
                    "The latest retrieval produced no relevant pages."
                ),
                next_query=state.current_query,
            )

        return AgentDecision(
            decision_type=AgentDecisionType.ANSWER,
            reason="Relevant retrieved context is available.",
        )


class LLMDecisionEngine(DecisionEngine):
    """LLM-judged decision engine.

    The iteration-cap case is delegated straight to a deterministic
    `AgentDecisionEngine` -- there is nothing to usefully judge once no
    iterations are left. When the heuristic finds no context at all,
    this clears `next_query` so the configured query refiner actually
    runs (the heuristic's own default of reusing the unchanged query
    would otherwise make the refiner unreachable). The LLM itself is
    only invoked for the one genuinely ambiguous case: retrieval found
    *some* pages, but do they actually answer the question, or only
    happen to share keywords with it?

    If the LLM call fails, times out, or returns an unparseable
    response, this falls back to the deterministic engine's decision
    for that state, so a generation provider outage never breaks the
    agent loop.
    """

    def __init__(
        self,
        generation_service: GenerationService,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 300,
        fallback: DecisionEngine | None = None,
    ) -> None:
        if not isinstance(
            generation_service,
            GenerationService,
        ):
            raise TypeError(
                "generation_service must be a GenerationService"
            )

        if not isinstance(model, str):
            raise TypeError("model must be a string")

        normalized_model = model.strip()

        if not normalized_model:
            raise ValueError("model cannot be empty")

        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0"
            )

        if max_tokens is not None and max_tokens < 1:
            raise ValueError(
                "max_tokens must be greater than or equal to 1"
            )

        self._generation_service = generation_service
        self._model = normalized_model
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._fallback: DecisionEngine = (
            fallback or AgentDecisionEngine()
        )

    @property
    def model(self) -> str:
        """Return the configured judgment model."""

        return self._model

    @property
    def generation_service(self) -> GenerationService:
        """Return the configured generation service."""

        return self._generation_service

    @property
    def fallback(self) -> DecisionEngine:
        """Return the deterministic engine used as a fallback."""

        return self._fallback

    def decide(
        self,
        state: AgentState,
    ) -> AgentDecision:
        """Judge whether retrieved context answers the current query."""

        if not isinstance(state, AgentState):
            raise TypeError(
                "state must be an AgentState"
            )

        fallback_decision = self._fallback.decide(state)

        if fallback_decision.decision_type is AgentDecisionType.STOP:
            return fallback_decision

        if fallback_decision.decision_type is AgentDecisionType.REFINE:
            # The heuristic found no context at all. This is exactly
            # the case where an LLM-driven query rewrite is most
            # valuable, so clear next_query rather than propagating
            # the heuristic's "retry the same query unchanged"
            # default -- that default exists so the plain heuristic
            # pairing (AgentDecisionEngine + AgentQueryRefiner) stays
            # deterministic on its own, but here it would silently
            # make the configured refiner unreachable.
            return AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason=fallback_decision.reason,
                next_query=None,
            )

        try:
            judged_decision = self._judge(state)
        except Exception:
            logger.exception(
                "LLM decision judgment failed for query %r; "
                "falling back to heuristic (%s).",
                state.current_query,
                fallback_decision.decision_type,
            )
            return fallback_decision

        if judged_decision is None:
            logger.warning(
                "LLM decision response was unparseable for query %r; "
                "falling back to heuristic (%s).",
                state.current_query,
                fallback_decision.decision_type,
            )
            return fallback_decision

        logger.info(
            "LLM judged query %r as %s (reason: %s)",
            state.current_query,
            judged_decision.decision_type,
            judged_decision.reason,
        )

        return judged_decision

    def _judge(
        self,
        state: AgentState,
    ) -> AgentDecision | None:
        """Call the LLM to judge sufficiency of the latest context."""

        latest_context = state.contexts[-1]

        request = GenerationRequest(
            messages=self._build_prompt(
                state.current_query,
                latest_context.text,
            ),
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        response = self._generation_service.generate(request)

        return self._parse_response(response.text)

    @staticmethod
    def _build_prompt(
        query: str,
        context_text: str,
    ) -> tuple[GenerationMessage, ...]:
        system = GenerationMessage(
            role="system",
            content=(
                "You are a strict evaluator for a retrieval-augmented "
                "question answering system. Given a user question and "
                "retrieved page excerpts, decide whether the excerpts "
                "genuinely contain enough information to answer the "
                "question -- sharing keywords with the question is not "
                "enough on its own.\n\n"
                "Respond with ONLY a JSON object, no other text, in "
                "exactly this shape:\n"
                '{"decision": "answer" or "refine", '
                '"reason": "<one short sentence>", '
                '"next_query": "<a better search query, or null>"}\n\n'
                'Use "answer" only if the excerpts genuinely address '
                'the question. Use "refine" if the excerpts are '
                "present but off-topic or insufficient, and suggest a "
                "next_query (concrete keywords, not a full sentence) "
                "that would find better results."
            ),
        )

        user = GenerationMessage(
            role="user",
            content=(
                f"Question: {query}\n\n"
                f"Retrieved excerpts:\n{context_text}"
            ),
        )

        return (system, user)

    @staticmethod
    def _parse_response(text: str) -> AgentDecision | None:
        """Parse the LLM's JSON verdict into an AgentDecision."""

        candidate = text.strip()

        if candidate.startswith("```"):
            candidate = candidate.strip("`").strip()

            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        decision_value = payload.get("decision")

        if decision_value not in ("answer", "refine"):
            return None

        reason = payload.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            reason = "LLM-judged decision."
        else:
            reason = reason.strip()[:10_000]

        if decision_value == "answer":
            return AgentDecision(
                decision_type=AgentDecisionType.ANSWER,
                reason=reason,
            )

        next_query = payload.get("next_query")

        if not isinstance(next_query, str) or not next_query.strip():
            next_query = None
        else:
            next_query = next_query.strip()[:10_000]

        return AgentDecision(
            decision_type=AgentDecisionType.REFINE,
            reason=reason,
            next_query=next_query,
        )