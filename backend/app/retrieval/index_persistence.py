import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.retrieval.inverted_index import InvertedIndex

_SNAPSHOT_VERSION = 1


class IndexSnapshotStore:
    """Persists an InvertedIndex snapshot to disk for fast startup.

    A snapshot contains only term-to-page-ID mappings, never raw page
    text, so loading one skips re-tokenizing the entire corpus -- the
    expensive part of a full rebuild. Writes are atomic (temp file +
    rename), matching PageStore's durability pattern.
    """

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def save(
        self,
        inverted_index: InvertedIndex,
        *,
        page_count: int,
    ) -> None:
        """Atomically write a snapshot of the index to disk.

        `page_count` is recorded alongside the index so `load()`
        callers can detect staleness (pages added or removed since
        this snapshot was taken) without re-reading any page content.
        """

        if page_count < 0:
            raise ValueError(
                "page_count must be greater than or equal to zero"
            )

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": _SNAPSHOT_VERSION,
            "page_count": page_count,
            "index": inverted_index.to_snapshot(),
        }

        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.snapshot_path.parent,
            prefix=".index-snapshot-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            try:
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        os.replace(temporary_path, self.snapshot_path)

    def load(self) -> tuple[InvertedIndex, int] | None:
        """Load a snapshot, or return None if missing, corrupt, or an
        unrecognized version.

        Returning None (rather than raising) on any problem is
        deliberate: a snapshot is purely a performance optimization,
        never the source of truth (persisted pages are), so any doubt
        about its validity should fall back to a full rebuild rather
        than risk serving a broken or incomplete index.
        """

        if not self.snapshot_path.is_file():
            return None

        try:
            payload = json.loads(
                self.snapshot_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        if payload.get("version") != _SNAPSHOT_VERSION:
            return None

        page_count = payload.get("page_count")
        index_data = payload.get("index")

        if not isinstance(page_count, int) or not isinstance(
            index_data, dict
        ):
            return None

        try:
            inverted_index = InvertedIndex.from_snapshot(index_data)
        except (TypeError, ValueError):
            return None

        return inverted_index, page_count

    def delete(self) -> None:
        """Remove any persisted snapshot, if present."""

        self.snapshot_path.unlink(missing_ok=True)