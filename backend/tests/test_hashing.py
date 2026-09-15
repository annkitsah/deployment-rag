from pathlib import Path

from app.ingestion.hashing import calculate_file_sha256


def test_calculate_file_sha256(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_bytes(b"vectorless-rag")

    first_hash = calculate_file_sha256(file_path)
    second_hash = calculate_file_sha256(file_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_hashing_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.pdf"

    try:
        calculate_file_sha256(missing_file)
    except FileNotFoundError:
        return

    raise AssertionError("Expected FileNotFoundError")