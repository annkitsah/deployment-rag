from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.index_persistence import IndexSnapshotStore
from app.retrieval.page_index import PageIndex


class IndexLifecycle:
    """Manage synchronization between persisted pages and the search index."""

    def __init__(
        self,
        page_store: PageStore,
        *,
        page_index: PageIndex | None = None,
        snapshot_store: IndexSnapshotStore | None = None,
    ) -> None:
        self.page_store = page_store
        self.page_index = page_index or PageIndex(
            page_store=page_store,
        )
        self.snapshot_store = snapshot_store

    def build(self) -> int:
        """Bring the in-memory index up to date, preferring a snapshot.

        A snapshot is trusted only when its recorded page count exactly
        matches the page store's current total. Any mismatch (pages
        added or removed since the snapshot was saved, a snapshot left
        over from a different data directory, a missing or corrupted
        file) falls back to a full rebuild from persisted pages, so a
        stale snapshot can never silently produce an incomplete index.

        This page-count check is a cheap proxy, not a content hash --
        pages in this system are never edited in place after ingestion
        (only added or removed as whole documents), so a matching count
        is a reliable signal here without the cost of re-reading every
        page to verify content.
        """

        actual_page_count = self.page_store.count_all_pages()

        if self.snapshot_store is not None:
            loaded = self.snapshot_store.load()

            if loaded is not None:
                snapshot_index, snapshot_page_count = loaded

                if snapshot_page_count == actual_page_count:
                    self.page_index.inverted_index = snapshot_index
                    return actual_page_count

        return self._rebuild_from_pages()

    def rebuild(self) -> int:
        """Clear the current index and rebuild it from persisted pages."""

        self.clear()

        return self._rebuild_from_pages()

    def _rebuild_from_pages(self) -> int:
        """Re-tokenize every persisted page and refresh the snapshot."""

        indexed_count = 0

        for document_id in self.page_store.get_document_ids():
            indexed_count += self.index_document(document_id)

        self.save()

        return indexed_count

    def index_page(
        self,
        page: PageRecord,
    ) -> None:
        """Add or replace a single page in the search index.

        This only updates the in-memory index -- call `save()` once
        after indexing a batch of pages (e.g. all pages of a newly
        ingested document) to persist the snapshot, rather than
        serializing the whole index after every single page.
        """

        self.page_index.add_page(page)

    def index_document(
        self,
        document_id: str,
    ) -> int:
        """Index every persisted page belonging to a document."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        pages = self.page_store.get_pages(document_id)

        for page in pages:
            self.index_page(page)

        return len(pages)

    def remove_page(
        self,
        page_id: str,
    ) -> None:
        """Remove a single page from the search index."""

        self.page_index.remove_page(page_id)

    def remove_document(
        self,
        document_id: str,
    ) -> int:
        """Remove every persisted page of a document from the search index."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        pages = self.page_store.get_pages(document_id)

        for page in pages:
            self.remove_page(page.page_id)

        self.save()

        return len(pages)

    def clear(self) -> None:
        """Remove every page from the search index."""

        self.page_index.clear()

    def save(self) -> None:
        """Persist the current in-memory index to disk.

        A no-op when no snapshot_store is configured. Cheap relative to
        a rebuild (no re-tokenization), but still O(index size) to
        serialize -- call this once after a batch of index_page() calls
        rather than after each individual page.
        """

        if self.snapshot_store is None:
            return

        self.snapshot_store.save(
            self.page_index.inverted_index,
            page_count=self.page_store.count_all_pages(),
        )