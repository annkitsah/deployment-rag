from pathlib import Path
from tempfile import NamedTemporaryFile

import pymupdf
from pydantic import BaseModel, ConfigDict, Field


class PageRenderError(RuntimeError):
    """Base exception for PDF page rendering failures."""


class PageRenderResult(BaseModel):
    """Metadata describing a rendered PDF page."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page_number: int = Field(ge=1)

    output_path: Path

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    dpi: int = Field(gt=0)

    file_size_bytes: int = Field(ge=0)


class PageRenderer:
    """Render individual PDF pages to raster images."""

    def __init__(
        self,
        *,
        default_dpi: int = 200,
    ) -> None:
        if default_dpi <= 0:
            raise ValueError(
                "default_dpi must be greater than zero"
            )

        self.default_dpi = default_dpi

    @staticmethod
    def _validate_document_id(
        document_id: str,
    ) -> None:
        if not document_id:
            raise ValueError(
                "document_id cannot be empty"
            )

        if document_id in {".", ".."}:
            raise ValueError(
                "invalid document_id"
            )

        if "/" in document_id or "\\" in document_id:
            raise ValueError(
                "document_id cannot contain path separators"
            )

    @staticmethod
    def _validate_page_number(
        page_number: int,
    ) -> None:
        if page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

    @staticmethod
    def _validate_dpi(
        dpi: int,
    ) -> None:
        if dpi <= 0:
            raise ValueError(
                "dpi must be greater than zero"
            )

    def render_page(
        self,
        pdf_path: Path,
        document_id: str,
        page_number: int,
        output_root: Path,
        *,
        dpi: int | None = None,
    ) -> PageRenderResult:
        """Render one PDF page to a PNG file."""

        self._validate_document_id(document_id)
        self._validate_page_number(page_number)

        effective_dpi = (
            self.default_dpi
            if dpi is None
            else dpi
        )

        self._validate_dpi(effective_dpi)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF does not exist: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got: {pdf_path.suffix}"
            )

        document_output_dir = (
            output_root
            / "documents"
            / document_id
            / "pages"
        )

        document_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            document_output_dir
            / f"{page_number:06d}.png"
        )

        temporary_path: Path | None = None

        try:
            with pymupdf.open(pdf_path) as document:
                page_index = page_number - 1

                if page_index >= len(document):
                    raise PageRenderError(
                        f"Page {page_number} does not exist. "
                        f"PDF contains {len(document)} pages."
                    )

                page = document[page_index]

                scale = effective_dpi / 72.0

                matrix = pymupdf.Matrix(
                    scale,
                    scale,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )

                with NamedTemporaryFile(
                    suffix=".png",
                    prefix=".render-",
                    dir=document_output_dir,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(
                        temporary_file.name
                    )

                pixmap.save(
                    str(temporary_path)
                )

            temporary_path.replace(output_path)

            return PageRenderResult(
                document_id=document_id,
                page_number=page_number,
                output_path=output_path,
                width=pixmap.width,
                height=pixmap.height,
                dpi=effective_dpi,
                file_size_bytes=output_path.stat().st_size,
            )

        except (
            FileNotFoundError,
            ValueError,
            PageRenderError,
        ):
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True
                )
            raise

        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True
                )

            raise PageRenderError(
                f"Failed to render page "
                f"{page_number} from {pdf_path}"
            ) from exc

    def render_pages(
        self,
        pdf_path: Path,
        document_id: str,
        page_numbers: list[int],
        output_root: Path,
        *,
        dpi: int | None = None,
    ) -> list[PageRenderResult]:
        """Render multiple PDF pages."""

        if not page_numbers:
            return []

        normalized_pages = sorted(
            set(page_numbers)
        )

        return [
            self.render_page(
                pdf_path=pdf_path,
                document_id=document_id,
                page_number=page_number,
                output_root=output_root,
                dpi=dpi,
            )
            for page_number in normalized_pages
        ]