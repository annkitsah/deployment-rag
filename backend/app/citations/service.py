from app.citations.models import Citation
from app.retrieval.models import RetrievedContext


def build_citations(
    context: RetrievedContext,
) -> tuple[Citation, ...]:
    """Extract deduplicated, score-ranked citations from a context.

    One citation per unique (document, page), highest-scoring pages
    first, mirroring the ranking the retriever already established.
    """

    if not isinstance(context, RetrievedContext):
        raise TypeError("context must be a RetrievedContext")

    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []

    for result in context.results:
        key = (result.document_id, result.page_number)

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            Citation(
                document_id=result.document_id,
                page_number=result.page_number,
                score=result.score,
            )
        )

    citations.sort(
        key=lambda citation: citation.score,
        reverse=True,
    )

    return tuple(citations)