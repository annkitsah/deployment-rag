from app.documents.page_store import PageStore
from app.retrieval.context import RetrievalContextAssembler
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievedContext,
)


class RetrievalService:
    """Orchestrate retrieval and context assembly."""

    def __init__(
        self,
        page_store: PageStore,
        *,
        retriever: LexicalRetriever | None = None,
        context_assembler: RetrievalContextAssembler | None = None,
    ) -> None:
        self.retriever = retriever or LexicalRetriever(page_store)
        self.context_assembler = (
            context_assembler or RetrievalContextAssembler()
        )

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievedContext:
        """Retrieve ranked pages and assemble bounded context."""

        response: RetrievalResponse = self.retriever.retrieve(query)

        return self.context_assembler.assemble(response)