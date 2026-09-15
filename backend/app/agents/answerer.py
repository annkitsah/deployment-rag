from abc import ABC, abstractmethod

from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.prompt import PromptBuilder
from app.generation.service import GenerationService
from app.retrieval.models import RetrievedContext

class Answerer(ABC):
    """Abstraction for generating an answer from retrieved context."""

    @abstractmethod
    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Generate an answer for a query using retrieved context."""

        raise NotImplementedError


class ContextAnswerer(Answerer):
    """Return retrieved context directly as the answer."""

    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Return retrieved context text without additional generation."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if not isinstance(context, RetrievedContext):
            raise TypeError(
                "context must be a RetrievedContext"
            )

        if not context.text.strip():
            raise ValueError(
                "cannot build answer from empty retrieved context"
            )

        return context.text


class GenerationAnswerer(Answerer):
    """Generate a grounded answer using the generation service."""

    def __init__(
        self,
        *,
        generation_service: GenerationService,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
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

        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")

        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0"
            )

        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise TypeError("max_tokens must be an integer")

            if max_tokens < 1:
                raise ValueError(
                    "max_tokens must be greater than or equal to 1"
                )

        if system_instruction is not None:
            if not isinstance(system_instruction, str):
                raise TypeError(
                    "system_instruction must be a string"
                )

            system_instruction = system_instruction.strip()

            if not system_instruction:
                raise ValueError(
                    "system_instruction cannot be empty"
                )

        self._generation_service = generation_service
        self._model = normalized_model
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._system_instruction = system_instruction

    @property
    def generation_service(self) -> GenerationService:
        """Return the configured generation service."""

        return self._generation_service

    @property
    def model(self) -> str:
        """Return the configured generation model."""

        return self._model

    @property
    def temperature(self) -> float:
        """Return the configured generation temperature."""

        return self._temperature

    @property
    def max_tokens(self) -> int | None:
        """Return the configured maximum output tokens."""

        return self._max_tokens

    @property
    def system_instruction(self) -> str | None:
        """Return the configured system instruction."""

        return self._system_instruction

    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Generate a grounded answer from retrieved context."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty")

        if not isinstance(context, RetrievedContext):
            raise TypeError(
                "context must be a RetrievedContext"
            )

        if not context.text.strip():
            raise ValueError(
                "cannot generate answer from empty retrieved context"
            )

        builder = PromptBuilder(
            question=normalized_query,
            context=context.text,
            system_instruction=self._system_instruction,
        )

        request: GenerationRequest = builder.build_request(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        response: GenerationResponse = (
            self._generation_service.generate(request)
        )

        if not response.text.strip():
            raise ValueError(
                "generation service returned empty answer"
            )

        return response.text