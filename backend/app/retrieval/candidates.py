from collections.abc import Iterable

from app.documents.models import PageRecord
from app.retrieval.metadata import PageMetadataFilter
from app.retrieval.page_index import PageIndex


class CandidateRetriever:
    """Retrieve deterministic page candidates from the page index."""

    def __init__(self, page_index: PageIndex) -> None:
        self.page_index = page_index

    def retrieve(
        self,
        terms: Iterable[str],
        *,
        metadata_filter: PageMetadataFilter | None = None,
    ) -> tuple[str, ...]:
        """Return unique page IDs matching at least one supplied term.

        Multiple terms use OR semantics. Metadata filtering is applied
        after lexical candidate generation.
        """

        candidate_ids: set[str] = set()

        for term in terms:
            normalized_term = term.strip()

            if not normalized_term:
                continue

            candidate_ids.update(
                self.page_index.lookup(normalized_term),
            )

        if not candidate_ids:
            return ()

        ordered_candidates = tuple(sorted(candidate_ids))

        if metadata_filter is None:
            return ordered_candidates

        pages = tuple(
            self.page_index.get_page(page_id)
            for page_id in ordered_candidates
        )

        filtered_pages = metadata_filter.apply(pages)

        return tuple(
            page.page_id
            for page in filtered_pages
        )

    def get_page(self, page_id: str) -> PageRecord:
        """Resolve a candidate page ID to its persisted page."""

        return self.page_index.get_page(page_id)