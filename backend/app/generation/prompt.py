from __future__ import annotations

from app.generation.models import GenerationMessage, GenerationRequest


class PromptBuilder:
    """Build grounded generation prompts from a question and retrieved context."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a precise, grounded question-answering assistant. "
        "Answer the user's question using only the provided context. "
        "Do not invent facts or unsupported information. "
        "If the context does not contain enough information to answer the question, "
        "state that clearly."
    )

    def __init__(
        self,
        *,
        question: str,
        context: str,
        system_instruction: str | None = None,
    ) -> None:
        self._question = self._validate_text(
            question,
            field_name="question",
        )
        self._context = self._validate_text(
            context,
            field_name="context",
        )

        if system_instruction is None:
            self._system_instruction = self.DEFAULT_SYSTEM_INSTRUCTION
        else:
            self._system_instruction = self._validate_text(
                system_instruction,
                field_name="system_instruction",
            )

    @property
    def question(self) -> str:
        """Return the normalized user question."""

        return self._question

    @property
    def context(self) -> str:
        """Return the retrieved context."""

        return self._context

    @property
    def system_instruction(self) -> str:
        """Return the system instruction."""

        return self._system_instruction

    def build_system_message(self) -> GenerationMessage:
        """Build the system message."""

        return GenerationMessage(
            role="system",
            content=self._system_instruction,
        )

    def build_context_message(self) -> GenerationMessage:
        """Build the message containing retrieved evidence."""

        content = (
            "Use the following retrieved context to answer the question.\n\n"
            f"{self._context}"
        )

        return GenerationMessage(
            role="system",
            content=content,
        )

    def build_user_message(self) -> GenerationMessage:
        """Build the user's question message."""

        return GenerationMessage(
            role="user",
            content=self._question,
        )

    def build_messages(self) -> tuple[GenerationMessage, ...]:
        """Build the complete ordered generation message sequence."""

        return (
            self.build_system_message(),
            self.build_context_message(),
            self.build_user_message(),
        )

    def build_request(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> GenerationRequest:
        """Build a validated generation request."""

        normalized_model = self._validate_text(
            model,
            field_name="model",
        )

        return GenerationRequest(
            messages=self.build_messages(),
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _validate_text(
        value: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )

        return normalized