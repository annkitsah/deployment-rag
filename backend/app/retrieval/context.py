from app.retrieval.models import (
    RetrievalResponse,
    RetrievalResult,
    RetrievedContext,
)


class RetrievalContextAssembler:
    """Assemble a bounded, deterministic context from retrieval results."""

    def __init__(
        self,
        *,
        max_chars: int = 12_000,
        max_pages: int | None = None,
        separator: str = "\n\n",
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be greater than zero")

        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be greater than zero")

        if not separator:
            raise ValueError("separator cannot be empty")

        self.max_chars = max_chars
        self.max_pages = max_pages
        self.separator = separator

    def assemble(
        self,
        response: RetrievalResponse,
    ) -> RetrievedContext:
        """Build bounded context while preserving retrieval ranking."""

        selected_results: list[RetrievalResult] = []
        context_parts: list[str] = []
        current_length = 0

        for result in response.results:
            if (
                self.max_pages is not None
                and len(selected_results) >= self.max_pages
            ):
                break

            formatted = self._format_result(result)

            additional_length = len(formatted)

            if context_parts:
                additional_length += len(self.separator)

            if current_length + additional_length > self.max_chars:
                break

            selected_results.append(result)
            context_parts.append(formatted)
            current_length += additional_length

        return RetrievedContext(
            query=response.query.text,
            results=tuple(selected_results),
            text=self.separator.join(context_parts),
        )

    @staticmethod
    def _format_result(result: RetrievalResult) -> str:
        """Format one retrieval result with deterministic source metadata."""

        return (
            f"[Source: {result.document_id} | "
            f"Page: {result.page_number} | "
            f"Score: {result.score:.4f}]\n"
            f"{result.text}"
        )