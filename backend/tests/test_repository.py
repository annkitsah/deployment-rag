from pathlib import Path

from app.documents.models import DocumentRecord, DocumentStatus
from app.documents.repository import DocumentRepository


def create_document() -> DocumentRecord:
    return DocumentRecord(
        document_id="doc-test-001",
        filename="test.pdf",
        source_path="data/raw/test.pdf",
        file_hash="a" * 64,
        file_size_bytes=1024,
        page_count=10,
        status=DocumentStatus.REGISTERED,
    )


def test_repository_initialization(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata" / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    assert database_path.exists()


def test_add_and_get_by_hash(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    document = create_document()

    repository.add(document)

    result = repository.get_by_hash(document.file_hash)

    assert result is not None
    assert result.document_id == document.document_id
    assert result.filename == document.filename
    assert result.file_hash == document.file_hash
    assert result.page_count == 10
    assert result.status == DocumentStatus.REGISTERED


def test_get_by_id(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    document = create_document()

    repository.add(document)

    result = repository.get_by_id(document.document_id)

    assert result is not None
    assert result.document_id == document.document_id


def test_unknown_document_returns_none(tmp_path: Path) -> None:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    assert repository.get_by_id("does-not-exist") is None
    assert repository.get_by_hash("b" * 64) is None