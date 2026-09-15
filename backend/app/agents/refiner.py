import logging
from abc import ABC, abstractmethod

from app.agents.state import AgentState
from app.generation.models import GenerationMessage, GenerationRequest
from app.generation.service import GenerationService

logger = logging.getLogger(__name__)


class QueryRefiner(ABC):
    """Abstraction for producing the next retrieval query on refine."""

    @abstractmethod
    def refine(self, state: AgentState) -> str:
        """Generate the next retrieval query from the current agent state."""

        raise NotImplementedError


class AgentQueryRefiner(QueryRefiner):
    """Deterministic follow-up query generator (no LLM, no network).

    Used as the default fallback for `LLMQueryRefiner`, and directly
    whenever a dependency-free, fully deterministic refiner is wanted
    (tests, offline environments).
    """

    def refine(self, state: AgentState) -> str:
        """Generate the next retrieval query from the current agent state."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")

        current_query = state.current_query.strip()

        if not current_query:
            raise ValueError("current query cannot be empty")

        if not state.contexts:
            return self._add_retrieval_instruction(current_query)

        latest_context = state.contexts[-1]

        if latest_context.page_count == 0:
            return self._add_retrieval_instruction(current_query)

        return current_query

    @staticmethod
    def _add_retrieval_instruction(query: str) -> str:
        """Make an unsuccessful query more explicit for lexical retrieval."""

        suffix = " relevant definition explanation details"

        if query.lower().endswith(suffix):
            return query

        return f"{query}{suffix}"


class LLMQueryRefiner(QueryRefiner):
    """LLM-driven query rewriter for lexical (keyword) retrieval.

    Rewrites the query using an LLM, informed by why the previous
    attempt was judged insufficient (the triggering decision's reason,
    when available). Falls back to the deterministic
    `AgentQueryRefiner` if the LLM call fails, returns an empty or
    unusable result, or the rewritten query is identical to the
    current one -- so a generation provider outage never breaks the
    agent loop, and a no-op rewrite never gets treated as progress.
    """

    def __init__(
        self,
        generation_service: GenerationService,
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int | None = 100,
        fallback: QueryRefiner | None = None,
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
        self._fallback: QueryRefiner = (
            fallback or AgentQueryRefiner()
        )

    @property
    def model(self) -> str:
        """Return the configured rewrite model."""

        return self._model

    @property
    def generation_service(self) -> GenerationService:
        """Return the configured generation service."""

        return self._generation_service

    @property
    def fallback(self) -> QueryRefiner:
        """Return the deterministic refiner used as a fallback."""

        return self._fallback

    def refine(self, state: AgentState) -> str:
        """Generate the next retrieval query from the current agent state."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")

        current_query = state.current_query.strip()

        if not current_query:
            raise ValueError("current query cannot be empty")

        try:
            rewritten = self._rewrite(state, current_query)
        except Exception:
            logger.exception(
                "LLM query rewrite failed for query %r; "
                "falling back to heuristic refiner.",
                current_query,
            )
            return self._fallback.refine(state)

        if rewritten is None:
            logger.warning(
                "LLM rewrite of query %r was empty or unchanged; "
                "falling back to heuristic refiner.",
                current_query,
            )
            return self._fallback.refine(state)

        logger.info(
            "LLM rewrote query %r -> %r",
            current_query,
            rewritten,
        )

        return rewritten

    def _rewrite(
        self,
        state: AgentState,
        current_query: str,
    ) -> str | None:
        """Call the LLM to rewrite the current query."""

        reason = state.decision.reason if state.decision else None

        request = GenerationRequest(
            messages=self._build_prompt(
                state.original_query,
                current_query,
                reason,
            ),
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        response = self._generation_service.generate(request)

        candidate = response.text.strip().strip('"').strip()

        if not candidate:
            return None

        if candidate.lower() == current_query.lower():
            return None

        return candidate[:10_000]

    @staticmethod
    def _build_prompt(
        original_query: str,
        current_query: str,
        reason: str | None,
    ) -> tuple[GenerationMessage, ...]:
        system = GenerationMessage(
            role="system",
            content=(
                "You rewrite search queries for a lexical (keyword) "
                "document search engine, not a semantic search engine "
                "-- favor concrete keywords and plausible synonyms "
                "over natural-language phrasing. Respond with ONLY the "
                "rewritten query text, nothing else: no quotes, no "
                "explanation, no preamble."
            ),
        )

        reason_note = (
            f" The previous attempt was insufficient because: {reason}"
            if reason
            else ""
        )

        user = GenerationMessage(
            role="user",
            content=(
                f"Original question: {original_query}\n"
                f"Last search query: {current_query}\n"
                f"{reason_note}\n"
                "Rewrite the search query to find more relevant pages."
            ),
        )

        return (system, user)