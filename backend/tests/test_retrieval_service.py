from pathlib import Path

import pytest

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.context import RetrievalContextAssembler
from app.retrieval.models import RetrievalQuery, RetrievalResponse
from app.retrieval.service import RetrievalService


def create_page(
    document_id: str,
    page_number: int,
    text: str,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc-001",
                1,
                "Retrieval augmented generation combines retrieval "
                "with language models.",
            ),
            create_page(
                "doc-001",
                2,
                "Vectorless retrieval uses lexical matching and "
                "traditional information retrieval.",
            ),
            create_page(
                "doc-001",
                3,
                "The weather today is sunny and warm.",
            ),
        ]
    )

    store.save_pages(
        [
            create_page(
                "doc-002",
                1,
                "BM25 is a lexical ranking algorithm used in "
                "information retrieval.",
            ),
        ]
    )

    return store


def test_service_returns_retrieved_context(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    result = service.retrieve(
        RetrievalQuery(
            text="lexical retrieval",
            top_k=2,
        )
    )

    assert result.query == "lexical retrieval"
    assert result.page_count == 2
    assert result.results[0].document_id == "doc-001"
    assert result.results[0].page_number == 2
    assert "lexical" in result.text
    assert "retrieval" in result.text


def test_service_respects_top_k(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    result = service.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=1,
        )
    )

    assert result.page_count == 1


def test_service_filters_by_document(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    result = service.retrieve(
        RetrievalQuery(
            text="information retrieval",
            document_id="doc-002",
            top_k=10,
        )
    )

    assert result.page_count == 1
    assert result.results[0].document_id == "doc-002"


def test_service_returns_empty_context_for_no_match(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    result = service.retrieve(
        RetrievalQuery(
            text="quantum computing",
            top_k=10,
        )
    )

    assert result.page_count == 0
    assert result.results == ()
    assert result.text == ""


def test_service_preserves_retrieval_order(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    result = service.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=10,
        )
    )

    page_ids = tuple(item.page_id for item in result.results)

    assert page_ids == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )

def test_service_applies_context_size_limit(
    page_store: PageStore,
) -> None:
    service = RetrievalService(
        page_store,
        context_assembler=RetrievalContextAssembler(
            max_chars=100,
        ),
    )

    result = service.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=10,
        )
    )

    assert len(result.text) <= 100
    assert result.page_count >= 0


def test_service_is_deterministic(
    page_store: PageStore,
) -> None:
    service = RetrievalService(page_store)

    query = RetrievalQuery(
        text="retrieval",
        top_k=10,
    )

    first = service.retrieve(query)
    second = service.retrieve(query)

    assert first == second


def test_service_accepts_custom_retriever(
    page_store: PageStore,
) -> None:
    class StubRetriever:
        def retrieve(
            self,
            query: RetrievalQuery,
        ) -> RetrievalResponse:
            return RetrievalResponse(
                query=query,
                results=(),
            )

    service = RetrievalService(
        page_store,
        retriever=StubRetriever(),
    )

    result = service.retrieve(
        RetrievalQuery(text="anything"),
    )

    assert result.page_count == 0
    assert result.text == ""


def test_service_accepts_custom_context_assembler(
    page_store: PageStore,
) -> None:
    class StubAssembler:
        def assemble(
            self,
            response: RetrievalResponse,
        ):
            from app.retrieval.models import RetrievedContext

            return RetrievedContext(
                query=response.query.text,
                results=(),
                text="custom-context",
            )

    service = RetrievalService(
        page_store,
        context_assembler=StubAssembler(),
    )

    result = service.retrieve(
        RetrievalQuery(text="retrieval"),
    )

    assert result.text == "custom-context"


def test_service_rejects_invalid_context_assembler_configuration(
    page_store: PageStore,
) -> None:
    with pytest.raises(ValueError):
        RetrievalService(
            page_store,
            context_assembler=RetrievalContextAssembler(
                max_chars=0,
            ),
        )