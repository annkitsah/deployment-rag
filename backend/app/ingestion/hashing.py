from hashlib import sha256
from pathlib import Path


def calculate_file_sha256(
    file_path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the entire file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    digest = sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()