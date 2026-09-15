import hashlib
import json
from pathlib import Path

import pytest

from app.documents.models import DocumentRecord, DocumentStatus
from app.documents.repository import DocumentRepository
from app.evaluation.dataset import load_cases, resolve_document_ids
from app.evaluation.models import EvaluationCase


def make_document(
    document_id: str,
    filename: str,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        filename=filename,
        source_path=f"/data/raw/{filename}",
        file_hash=hashlib.sha256(document_id.encode()).hexdigest(),
        file_size_bytes=1024,
        page_count=10,
        status=DocumentStatus.PROCESSED,
    )


@pytest.fixture
def repository(tmp_path: Path) -> DocumentRepository:
    repo = DocumentRepository(tmp_path / "documents.db")
    repo.initialize()

    repo.add(make_document("doc-a", "Chapter 1.pdf"))
    repo.add(make_document("doc-b", "Chapter 3.pdf"))

    return repo


def write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(cases))
    return dataset_path


def test_load_cases_from_file(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [
            {
                "case_id": "c1",
                "question": "what is x",
                "source_filename": "Chapter 1.pdf",
                "expected_pages": [1, 2],
                "expected_keywords": ["x"],
            }
        ],
    )

    cases = load_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].expected_pages == (1, 2)


def test_load_cases_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "does_not_exist.json")


def test_load_cases_rejects_non_array_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"not": "an array"}))

    with pytest.raises(ValueError, match="JSON array"):
        load_cases(dataset_path)


def test_load_default_dataset_is_valid() -> None:
    # The shipped starter golden dataset should always load cleanly.
    cases = load_cases()

    assert len(cases) > 0


def test_resolve_document_ids_matches_by_filename(
    repository: DocumentRepository,
) -> None:
    cases = (
        EvaluationCase(
            case_id="c1",
            question="q",
            source_filename="Chapter 1.pdf",
            expected_pages=(1,),
        ),
    )

    resolved = resolve_document_ids(cases, repository)

    assert resolved[0].document_id == "doc-a"


def test_resolve_document_ids_leaves_explicit_document_id_alone(
    repository: DocumentRepository,
) -> None:
    cases = (
        EvaluationCase(
            case_id="c1",
            question="q",
            document_id="doc-explicit",
            expected_pages=(1,),
        ),
    )

    resolved = resolve_document_ids(cases, repository)

    assert resolved[0].document_id == "doc-explicit"


def test_resolve_document_ids_leaves_unscoped_case_alone(
    repository: DocumentRepository,
) -> None:
    cases = (
        EvaluationCase(
            case_id="c1",
            question="q",
            expected_pages=(1,),
        ),
    )

    resolved = resolve_document_ids(cases, repository)

    assert resolved[0].document_id is None


def test_resolve_document_ids_raises_for_unknown_filename(
    repository: DocumentRepository,
) -> None:
    cases = (
        EvaluationCase(
            case_id="c1",
            question="q",
            source_filename="Not Ingested.pdf",
            expected_pages=(1,),
        ),
    )

    with pytest.raises(ValueError, match="Not Ingested.pdf"):
        resolve_document_ids(cases, repository)


def test_resolve_document_ids_rejects_empty_cases(
    repository: DocumentRepository,
) -> None:
    with pytest.raises(ValueError, match="cases cannot be empty"):
        resolve_document_ids((), repository)