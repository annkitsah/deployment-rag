from pathlib import Path

import pymupdf

from app.documents.models import PageRecord


class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be processed."""


def extract_pages(
    file_path: Path,
    document_id: str,
) -> list[PageRecord]:
    """Extract text and page metadata from a PDF."""

    if not file_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {file_path}")

    try:
        document = pymupdf.open(file_path)
    except (RuntimeError, ValueError) as exc:
        raise PDFProcessingError(
            f"Unable to open PDF: {file_path}"
        ) from exc

    pages: list[PageRecord] = []

    try:
        for page_index in range(len(document)):
            page = document[page_index]
            rectangle = page.rect

            pages.append(
                PageRecord(
                    document_id=document_id,
                    page_number=page_index + 1,
                    text=page.get_text("text"),
                    width=float(rectangle.width),
                    height=float(rectangle.height),
                )
            )
    finally:
        document.close()

    return pages