"""In-memory ingestion progress for the current process.

Suitable for a single Railway service instance. Progress is lost on restart,
but document status in SQLite remains the source of truth.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class IngestionProgress:
    document_id: str
    filename: str
    status: str  # processing | processed | failed
    total_pages: int = 0
    processed_pages: int = 0
    message: str = "Queued for OCR…"
    error: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def percent(self) -> int:
        if self.status == "processed":
            return 100
        if self.status == "failed":
            return (
                self.processed_pages * 100 // max(self.total_pages, 1)
                if self.total_pages
                else 0
            )
        if self.total_pages <= 0:
            return 5
        return min(99, max(1, int(100 * self.processed_pages / self.total_pages)))


class ProgressRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, IngestionProgress] = {}

    def start(
        self,
        document_id: str,
        filename: str,
        total_pages: int,
    ) -> IngestionProgress:
        item = IngestionProgress(
            document_id=document_id,
            filename=filename,
            status="processing",
            total_pages=total_pages,
            processed_pages=0,
            message="Starting OCR…",
        )
        with self._lock:
            self._items[document_id] = item
        return item

    def update(
        self,
        document_id: str,
        *,
        processed_pages: int | None = None,
        total_pages: int | None = None,
        message: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            item = self._items.get(document_id)
            if item is None:
                return
            if processed_pages is not None:
                item.processed_pages = processed_pages
            if total_pages is not None:
                item.total_pages = total_pages
            if message is not None:
                item.message = message
            if status is not None:
                item.status = status
            if error is not None:
                item.error = error
            item.updated_at = datetime.now(UTC)

    def get(self, document_id: str) -> IngestionProgress | None:
        with self._lock:
            return self._items.get(document_id)


progress_registry = ProgressRegistry()