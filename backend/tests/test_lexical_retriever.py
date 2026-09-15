from pathlib import Path

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import RetrievalQuery


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


def test_retrieve_returns_matching_pages(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_1",
                1,
                "Agentic retrieval systems use intelligent agents.",
            ),
            create_page(
                "doc_1",
                2,
                "This page discusses database indexing.",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="agentic retrieval",
        )
    )

    assert response.result_count == 1

    result = response.results[0]

    assert result.document_id == "doc_1"
    assert result.page_number == 1
    assert result.score > 0
    assert result.matched_terms == (
        "agentic",
        "retrieval",
    )


def test_retrieve_ranks_more_relevant_page_first(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_1",
                1,
                "agentic retrieval",
            ),
            create_page(
                "doc_1",
                2,
                "agentic agentic retrieval retrieval",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="agentic retrieval",
            top_k=2,
        )
    )

    assert response.result_count == 2
    assert response.results[0].page_number == 2
    assert response.results[0].score > response.results[1].score


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page("doc_1", 1, "agentic retrieval"),
            create_page("doc_1", 2, "agentic systems"),
            create_page("doc_1", 3, "retrieval systems"),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="agentic retrieval",
            top_k=1,
        )
    )

    assert response.result_count == 1


def test_retrieve_can_filter_by_document(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_a",
                1,
                "agentic retrieval architecture",
            ),
        ]
    )
    store.save_pages(
        [ 
            create_page(
                "doc_b",
                1,
                "agentic retrieval architecture",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="agentic retrieval",
            document_id="doc_b",
        )
    )

    assert response.result_count == 1
    assert response.results[0].document_id == "doc_b"


def test_retrieve_returns_empty_for_no_match(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_1",
                1,
                "database indexing and storage",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="quantum computing",
        )
    )

    assert response.results == ()


def test_retrieve_ignores_stopwords(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_1",
                1,
                "agentic retrieval",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="the agentic retrieval system",
        )
    )

    assert response.result_count == 1
    assert response.results[0].matched_terms == (
        "agentic",
        "retrieval",
        
    )


def test_retrieve_all_documents_when_document_id_is_none(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc_a",
                1,
                "agentic retrieval architecture",
            ),
        ]
    )
    store.save_pages( 
        [ 
            create_page(
                "doc_b",
                1,
                "agentic retrieval architecture",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="agentic retrieval",
        )
    )

    assert response.result_count == 2
    assert {
        result.document_id
        for result in response.results
    } == {
        "doc_a",
        "doc_b",
    }