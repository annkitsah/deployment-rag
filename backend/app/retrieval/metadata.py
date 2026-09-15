from collections.abc import Iterable

from app.documents.models import PageRecord


class PageMetadataFilter:
    """Filter page records using metadata constraints."""

    def __init__(
        self,
        *,
        document_id: str | None = None,
        page_number: int | None = None,
        min_page_number: int | None = None,
        max_page_number: int | None = None,
        min_width: float | None = None,
        max_width: float | None = None,
        min_height: float | None = None,
        max_height: float | None = None,
    ) -> None:
        if document_id is not None and not document_id.strip():
            raise ValueError("document_id cannot be empty")

        if page_number is not None and page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

        if min_page_number is not None and min_page_number < 1:
            raise ValueError(
                "min_page_number must be greater than or equal to 1"
            )

        if max_page_number is not None and max_page_number < 1:
            raise ValueError(
                "max_page_number must be greater than or equal to 1"
            )

        if (
            min_page_number is not None
            and max_page_number is not None
            and min_page_number > max_page_number
        ):
            raise ValueError(
                "min_page_number cannot be greater than max_page_number"
            )

        self._validate_dimension(min_width, "min_width")
        self._validate_dimension(max_width, "max_width")
        self._validate_dimension(min_height, "min_height")
        self._validate_dimension(max_height, "max_height")

        if (
            min_width is not None
            and max_width is not None
            and min_width > max_width
        ):
            raise ValueError(
                "min_width cannot be greater than max_width"
            )

        if (
            min_height is not None
            and max_height is not None
            and min_height > max_height
        ):
            raise ValueError(
                "min_height cannot be greater than max_height"
            )

        self.document_id = document_id
        self.page_number = page_number
        self.min_page_number = min_page_number
        self.max_page_number = max_page_number
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def apply(
        self,
        pages: Iterable[PageRecord],
    ) -> tuple[PageRecord, ...]:
        """Return pages matching all configured metadata filters."""

        filtered: list[PageRecord] = []

        for page in pages:
            if (
                self.document_id is not None
                and page.document_id != self.document_id
            ):
                continue

            if (
                self.page_number is not None
                and page.page_number != self.page_number
            ):
                continue

            if (
                self.min_page_number is not None
                and page.page_number < self.min_page_number
            ):
                continue

            if (
                self.max_page_number is not None
                and page.page_number > self.max_page_number
            ):
                continue

            if (
                self.min_width is not None
                and page.width < self.min_width
            ):
                continue

            if (
                self.max_width is not None
                and page.width > self.max_width
            ):
                continue

            if (
                self.min_height is not None
                and page.height < self.min_height
            ):
                continue

            if (
                self.max_height is not None
                and page.height > self.max_height
            ):
                continue

            filtered.append(page)

        return tuple(filtered)

    @staticmethod
    def _validate_dimension(
        value: float | None,
        field_name: str,
    ) -> None:
        if value is not None and value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero"
            )