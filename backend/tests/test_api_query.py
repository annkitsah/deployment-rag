from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.answerer import ContextAnswerer
from app.agents.decision import AgentDecisionEngine
from app.agents.executor import AgentExecutor
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.agents.refiner import AgentQueryRefiner
from app.api.dependencies import get_container
from app.api.routes.query import router as query_router
from app.config.settings import Settings
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.retrieval.candidates import CandidateRetriever
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.page_index import PageIndex
from app.retrieval.service import RetrievalService
from app.retrieval.text import tokenize

PDF_PATH = Path("data/raw/Chapter 1 - The Overview of Map of GenAI.pdf")


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"),
        index_dir=str(tmp_path / "data" / "indexes"),
        metadata_dir=str(tmp_path / "data" / "metadata"),
    )


def make_test_container(tmp_path: Path) -> SimpleNamespace:
    """Build a lightweight stand-in for ApplicationContainer.

    Uses ContextAnswerer (returns retrieved text directly) instead of
    GenerationAnswerer so the agent loop can be tested without making
    real network calls to a generation provider.
    """

    settings = make_settings(tmp_path)

    repository = DocumentRepository(
        Path(settings.metadata_dir) / "documents.db",
    )
    repository.initialize()

    page_store = PageStore(Path(settings.processed_data_dir))

    page_index = PageIndex(page_store=page_store)
    index_lifecycle = IndexLifecycle(
        page_store=page_store,
        page_index=page_index,
    )

    ingestion_service = IngestionService(
        repository=repository,
        page_store=page_store,
        index_lifecycle=index_lifecycle,
    )

    candidate_retriever = CandidateRetriever(page_index=page_index)
    lexical_retriever = LexicalRetriever(
        page_store=page_store,
        candidate_retriever=candidate_retriever,
    )
    retrieval_service = RetrievalService(
        page_store=page_store,
        retriever=lexical_retriever,
    )

    agent_orchestrator = AgentOrchestrator(
        planner=AgentPlanner(),
        executor=AgentExecutor(retrieval_service=retrieval_service),
        decision_engine=AgentDecisionEngine(),
        answerer=ContextAnswerer(),
        refiner=AgentQueryRefiner(),
    )

    return SimpleNamespace(
        settings=settings,
        repository=repository,
        page_store=page_store,
        ingestion_service=ingestion_service,
        agent_orchestrator=agent_orchestrator,
    )


@pytest.fixture
def container(tmp_path: Path) -> SimpleNamespace:
    return make_test_container(tmp_path)


@pytest.fixture
def client(container: SimpleNamespace) -> TestClient:
    app = FastAPI()
    app.include_router(query_router)
    app.dependency_overrides[get_container] = lambda: container

    return TestClient(app)


def test_query_without_any_ingested_documents_stops_gracefully(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": "What is this document about?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "What is this document about?"
    assert len(body["answer"]) > 0
    assert body["iterations"] >= 1


def test_query_after_ingesting_a_document_returns_grounded_answer(
    client: TestClient,
    container: SimpleNamespace,
) -> None:
    container.ingestion_service.ingest(PDF_PATH)

    response = client.post(
        "/query",
        json={"question": "genai"},
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["answer"]) > 0
    assert body["iterations"] >= 1


def test_query_with_empty_question_returns_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_query_missing_question_field_returns_422(
    client: TestClient,
) -> None:
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_scoped_to_document_id_only_uses_that_document(
    client: TestClient,
    container: SimpleNamespace,
) -> None:
    first_result = container.ingestion_service.ingest(PDF_PATH)

    other_pdf_path = Path(
        "data/raw/Chapter 3 – The Foundation Layer.pdf"
    )
    container.ingestion_service.ingest(other_pdf_path)

    first_page_text = container.page_store.get_page(
        first_result.document.document_id,
        1,
    ).text

    query_term = tokenize(first_page_text)[0]

    response = client.post(
        "/query",
        json={
            "question": query_term,
            "document_id": first_result.document.document_id,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["answer"]) > 0
    # ContextAnswerer returns retrieved page text directly, tagged with
    # its source document, so scoping is verifiable from the answer text.
    assert first_result.document.document_id in body["answer"]


def test_query_with_unknown_document_id_returns_404(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={
            "question": "what is this about",
            "document_id": "doc_does_not_exist",
        },
    )

    assert response.status_code == 404


def test_query_response_includes_citations_with_resolved_filename(
    client: TestClient,
    container: SimpleNamespace,
) -> None:
    result = container.ingestion_service.ingest(PDF_PATH)

    first_page_text = container.page_store.get_page(
        result.document.document_id,
        1,
    ).text

    query_term = tokenize(first_page_text)[0]

    response = client.post(
        "/query",
        json={"question": query_term},
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["citations"]) > 0

    citation = body["citations"][0]

    assert citation["document_id"] == result.document.document_id
    assert citation["filename"] == PDF_PATH.name
    assert citation["page_number"] >= 1
    assert citation["score"] >= 0


def test_query_without_any_ingested_documents_has_no_citations(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": "anything"},
    )

    assert response.status_code == 200
    assert response.json()["citations"] == []