from pathlib import Path

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.candidates import CandidateRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import RetrievalQuery
from app.retrieval.page_index import PageIndex


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


def create_indexed_store(
    tmp_path: Path,
) -> tuple[PageStore, PageIndex]:
    store = PageStore(tmp_path)
    index = PageIndex(page_store=store)

    pages = [
        create_page(
            "doc-001",
            1,
            "retrieval architecture system",
        ),
        create_page(
            "doc-001",
            2,
            "retrieval pipeline implementation",
        ),
        create_page(
            "doc-002",
            1,
            "database architecture",
        ),
        create_page(
            "doc-003",
            1,
            "weather forecast sunny",
        ),
    ]

    

    for page in pages:
        store.save_page(page)
        index.add_page(page)

    return store, index


def test_lexical_retriever_can_use_candidate_retriever(
    tmp_path: Path,
) -> None:
    store, index = create_indexed_store(tmp_path)

    candidate_retriever = CandidateRetriever(index)

    retriever = LexicalRetriever(
        store,
        candidate_retriever=candidate_retriever,
    )

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=10,
        )
    )

    assert tuple(
        result.page_id
        for result in response.results
    ) == (
        "doc-001:page:1",
        "doc-001:page:2",
    )


def test_candidate_retrieval_excludes_unmatched_pages(
    tmp_path: Path,
) -> None:
    store, index = create_indexed_store(tmp_path)

    candidate_retriever = CandidateRetriever(index)

    retriever = LexicalRetriever(
        store,
        candidate_retriever=candidate_retriever,
    )

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=10,
        )
    )

    page_ids = tuple(
        result.page_id
        for result in response.results
    )

    assert "doc-002:page:1" not in page_ids
    assert "doc-003:page:1" not in page_ids


def test_multiple_query_terms_use_index_union(
    tmp_path: Path,
) -> None:
    store, index = create_indexed_store(tmp_path)

    candidate_retriever = CandidateRetriever(index)

    retriever = LexicalRetriever(
        store,
        candidate_retriever=candidate_retriever,
    )

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval database",
            top_k=10,
        )
    )

    page_ids = {
        result.page_id
        for result in response.results
    }

    assert page_ids == {
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    }


def test_indexed_retrieval_preserves_bm25_ranking(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)
    index = PageIndex(page_store=store)

    pages = [
        create_page(
            "doc-001",
            1,
            "retrieval " + " ".join(["common"] * 100),
        ),
        create_page(
            "doc-001",
            2,
            "retrieval",
        ),
    ]

    store.save_pages(pages)

    for page in pages:
        index.add_page(page)

    retriever = LexicalRetriever(
        store,
        candidate_retriever=CandidateRetriever(index),
    )

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=2,
        )
    )

    assert response.results[0].page_id == "doc-001:page:2"


def test_retrieval_without_candidate_retriever_remains_supported(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc-001",
                1,
                "retrieval architecture",
            ),
        ]
    )

    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
        )
    )

    assert response.result_count == 1
    assert response.results[0].page_id == "doc-001:page:1"