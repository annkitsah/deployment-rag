import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.api.schemas import CitationResponse, QueryRequest, QueryResponse
from app.citations.models import Citation
from app.container import ApplicationContainer
from app.api.auth import require_app_password


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"],
    dependencies=[Depends(require_app_password)],)


@router.post("/query", response_model=QueryResponse)
async def run_query(
    payload: QueryRequest,
    container: ApplicationContainer = Depends(get_container),
) -> QueryResponse:
    """Run the agentic retrieval-and-answer loop for a question."""

    if payload.document_id is not None:
        document = container.repository.get_by_id(payload.document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {payload.document_id}",
            )

    try:
        response = container.agent_orchestrator.run(
            payload.question,
            document_id=payload.document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        msg = str(exc).lower()
        logger.exception("Agent orchestration failed for query.")

        if (
            "429" in msg
            or "rate limit" in msg
            or "rate_limit" in msg
            or "rate_limited" in msg
            or "too many requests" in msg
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "AI rate limit reached (provider quota). "
                    "Wait about 30–60 seconds and try again. "
                    "Large documents or many questions in a row make this more likely."
                ),
            ) from exc

        if "413" in msg or "payload too large" in msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "Retrieved context is too large for the model. "
                    "Try a more specific question or scope to a single document."
                ),
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The generation provider failed to produce an answer. "
                "This is usually a rate limit or upstream outage; "
                "please retry shortly."
            ),
        ) from exc

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        iterations=response.iterations,
        citations=_build_citation_responses(
            response.citations,
            container,
        ),
    )


def _build_citation_responses(
    citations: tuple[Citation, ...],
    container: ApplicationContainer,
) -> tuple[CitationResponse, ...]:
    """Enrich citations with source filenames, caching lookups.

    Citations often repeat the same document across several pages, so
    each unique document is looked up at most once per request rather
    than once per citation.
    """

    filename_cache: dict[str, str | None] = {}

    def resolve_filename(document_id: str) -> str | None:
        if document_id not in filename_cache:
            document = container.repository.get_by_id(document_id)
            filename_cache[document_id] = (
                document.filename if document is not None else None
            )

        return filename_cache[document_id]

    return tuple(
        CitationResponse(
            document_id=citation.document_id,
            filename=resolve_filename(citation.document_id),
            page_number=citation.page_number,
            score=citation.score,
        )
        for citation in citations
    )