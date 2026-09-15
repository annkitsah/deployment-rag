from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_container
from app.api.routes.documents import router as documents_router
from app.config.settings import Settings
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.page_index import PageIndex

PDF_PATH = Path("data/raw/Chapter 1 - The Overview of Map of GenAI.pdf")


def make_settings(tmp_path: Path) -> Settings:
    """Build isolated settings pointing at a scratch data directory."""

    return Settings(
        data_dir=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"),
        index_dir=str(tmp_path / "data" / "indexes"),
        metadata_dir=str(tmp_path / "data" / "metadata"),
    )


def make_test_container(tmp_path: Path) -> SimpleNamespace:
    """Build a lightweight stand-in for ApplicationContainer.

    Uses native PDF text extraction (no OCR pipeline, no network calls)
    so the API layer can be exercised in isolation and deterministically.
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

    return SimpleNamespace(
        settings=settings,
        repository=repository,
        page_store=page_store,
        page_index=page_index,
        index_lifecycle=index_lifecycle,
        ingestion_service=ingestion_service,
    )


@pytest.fixture
def container(tmp_path: Path) -> SimpleNamespace:
    return make_test_container(tmp_path)


@pytest.fixture
def client(container: SimpleNamespace) -> TestClient:
    app = FastAPI()
    app.include_router(documents_router)
    app.dependency_overrides[get_container] = lambda: container

    return TestClient(app)


def test_upload_document_returns_201_and_metadata(
    client: TestClient,
) -> None:
    with PDF_PATH.open("rb") as pdf_file:
        response = client.post(
            "/documents",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 201

    body = response.json()

    assert body["filename"] == PDF_PATH.name
    assert body["page_count"] > 0
    assert body["status"] == "processed"
    assert body["duplicate"] is False
    assert body["document_id"].startswith("doc_")


def test_upload_non_pdf_returns_400(
    client: TestClient,
) -> None:
    response = client.post(
        "/documents",
        files={
            "file": (
                "notes.txt",
                b"not a pdf",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_upload_duplicate_is_flagged_and_not_rewritten(
    client: TestClient,
    container: SimpleNamespace,
) -> None:
    with PDF_PATH.open("rb") as pdf_file:
        first = client.post(
            "/documents",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    with PDF_PATH.open("rb") as pdf_file:
        second = client.post(
            "/documents",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    assert (
        second.json()["document_id"]
        == first.json()["document_id"]
    )

    raw_dir = Path(container.settings.raw_data_dir)
    stored_copies = list(raw_dir.glob(f"*{PDF_PATH.name}"))

    # The duplicate upload's temporary copy should be cleaned up,
    # leaving only the first copy on disk.
    assert len(stored_copies) == 1


def test_list_documents_returns_uploaded_documents(
    client: TestClient,
) -> None:
    with PDF_PATH.open("rb") as pdf_file:
        client.post(
            "/documents",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    response = client.get("/documents")

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert len(body["documents"]) == 1
    assert body["documents"][0]["filename"] == PDF_PATH.name


def test_list_documents_empty(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "count": 0}


def test_get_document_by_id(client: TestClient) -> None:
    with PDF_PATH.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents",
            files={
                "file": (
                    PDF_PATH.name,
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    document_id = upload_response.json()["document_id"]

    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id


def test_get_unknown_document_returns_404(
    client: TestClient,
) -> None:
    response = client.get("/documents/doc_does_not_exist")

    assert response.status_code == 404