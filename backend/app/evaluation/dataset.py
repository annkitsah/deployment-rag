import json
from pathlib import Path

from app.documents.repository import DocumentRepository
from app.evaluation.models import EvaluationCase

DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_cases(
    path: Path | None = None,
) -> tuple[EvaluationCase, ...]:
    """Load evaluation cases from a JSON golden-dataset file.

    Cases loaded this way typically carry `source_filename` rather than
    a resolved `document_id` -- pass them through `resolve_document_ids`
    before evaluating, using the repository of the environment you want
    to evaluate against.
    """

    dataset_path = path or DEFAULT_DATASET_PATH

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Golden dataset not found: {dataset_path}"
        )

    raw = json.loads(dataset_path.read_text())

    if not isinstance(raw, list):
        raise ValueError(
            "Golden dataset must be a JSON array of case objects"
        )

    return tuple(EvaluationCase(**item) for item in raw)


def resolve_document_ids(
    cases: tuple[EvaluationCase, ...],
    repository: DocumentRepository,
) -> tuple[EvaluationCase, ...]:
    """Resolve each case's `source_filename` to a live `document_id`.

    Document IDs are generated fresh per ingestion, so a golden dataset
    references documents by filename instead of hardcoding an ID that
    would go stale. A case that already has `document_id` set (e.g.
    built programmatically) is left unchanged. Raises `ValueError`
    naming any filename that could not be found, so a stale or
    not-yet-ingested dataset fails loudly rather than silently
    evaluating against the wrong (or no) document.
    """

    if not cases:
        raise ValueError("cases cannot be empty")

    documents_by_filename = {
        document.filename: document.document_id
        for document in repository.list_all()
    }

    resolved_cases = []
    unresolved_filenames = []

    for case in cases:
        if case.document_id is not None or case.source_filename is None:
            resolved_cases.append(case)
            continue

        document_id = documents_by_filename.get(case.source_filename)

        if document_id is None:
            unresolved_filenames.append(case.source_filename)
            continue

        resolved_cases.append(
            case.model_copy(update={"document_id": document_id})
        )

    if unresolved_filenames:
        raise ValueError(
            "Could not resolve document_id for these source "
            f"filenames (not currently ingested?): "
            f"{sorted(set(unresolved_filenames))}"
        )

    return tuple(resolved_cases)