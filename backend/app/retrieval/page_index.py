from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.inverted_index import InvertedIndex


class PageIndex:
    """Page-level index abstraction over an inverted index and page store."""

    def __init__(
        self,
        *,
        page_store: PageStore,
        inverted_index: InvertedIndex | None = None,
    ) -> None:
        self.page_store = page_store
        self.inverted_index = inverted_index or InvertedIndex()

    def add_page(self, page: PageRecord) -> None:
        """Add a page to the searchable index."""

        self.inverted_index.add(
            page.page_id,
            page.text,
        )

    def lookup(self, term: str) -> tuple[str, ...]:
        """Return page IDs containing the normalized term."""

        return self.inverted_index.lookup(term)

    def get_page(self, page_id: str) -> PageRecord:
        """Resolve a page ID to its persisted page record."""

        document_id, page_number = self._parse_page_id(page_id)

        return self.page_store.get_page(
            document_id,
            page_number,
        )

    def remove_page(self, page_id: str) -> None:
        """Remove a page from the searchable index."""

        self.inverted_index.remove(page_id)

    def contains(self, term: str) -> bool:
        """Return whether a normalized term exists in the index."""

        return self.inverted_index.contains(term)

    def clear(self) -> None:
        """Remove every indexed page."""

        self.inverted_index.clear()

    @staticmethod
    def _parse_page_id(page_id: str) -> tuple[str, int]:
        """Parse the canonical document:page:N identifier."""

        if not page_id.strip():
            raise ValueError("page_id cannot be empty")

        parts = page_id.rsplit(":page:", 1)

        if len(parts) != 2:
            raise ValueError(
                "page_id must have format '<document_id>:page:<page_number>'"
            )

        document_id, page_number_text = parts

        if not document_id:
            raise ValueError("page_id must contain a document_id")

        try:
            page_number = int(page_number_text)
        except ValueError as exc:
            raise ValueError(
                "page_id must contain a valid page number"
            ) from exc

        if page_number < 1:
            raise ValueError(
                "page number must be greater than or equal to 1"
            )

        return document_id, page_number
