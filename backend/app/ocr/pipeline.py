from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict

from app.ocr.base import OCRProvider
from app.ocr.classifier import PageContentClassifier
from app.ocr.models import (
    OCRResult,
    PageProcessingDecision,
    PageProcessingDecisionResult,
)
from app.ocr.renderer import PageRenderer, PageRenderResult


class PagePipelineResult(BaseModel):
    """Result of processing one PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int
    decision: PageProcessingDecision

    classification: PageProcessingDecisionResult

    native_text: str

    ocr_result: OCRResult | None = None

    render_result: PageRenderResult | None = None

    @property
    def text(self) -> str:
        """Return the best available text for the page."""

        if self.ocr_result is not None:
            return self.ocr_result.text

        return self.native_text


class OCRPipelineResult(BaseModel):
    """Result of processing all pages in a PDF."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    pdf_path: Path

    pages: tuple[PagePipelineResult, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(
            page.decision == PageProcessingDecision.OCR_REQUIRED
            for page in self.pages
        )

    @property
    def native_text_page_count(self) -> int:
        return sum(
            page.decision == PageProcessingDecision.NATIVE_TEXT
            for page in self.pages
        )

    @property
    def review_page_count(self) -> int:
        return sum(
            page.decision == PageProcessingDecision.REVIEW_REQUIRED
            for page in self.pages
        )

    @property
    def empty_page_count(self) -> int:
        return sum(
            page.decision == PageProcessingDecision.EMPTY
            for page in self.pages
        )

    @property
    def text(self) -> str:
        """Return the combined text from all processed pages."""

        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )


class OCRPipeline:
    """Coordinate page classification, rendering, and OCR."""

    def __init__(
        self,
        *,
        classifier: PageContentClassifier | None = None,
        renderer: PageRenderer | None = None,
        provider: OCRProvider,
    ) -> None:
        self.classifier = (
            classifier
            if classifier is not None
            else PageContentClassifier()
        )

        self.renderer = (
            renderer
            if renderer is not None
            else PageRenderer()
        )

        self.provider = provider

    @staticmethod
    def _validate_pdf(pdf_path: Path) -> None:
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF does not exist: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got: {pdf_path.suffix}"
            )

    @staticmethod
    def _validate_document_id(document_id: str) -> None:
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
    def _inspect_page(
        page: pymupdf.Page,
        page_number: int,
    ) -> tuple[str, int, int, float, float]:
        """Extract the inputs required by PageContentClassifier."""

        text = page.get_text("text")

        image_count = len(page.get_images(full=True))

        drawing_count = len(page.get_drawings())

        rect = page.rect

        return (
            text,
            image_count,
            drawing_count,
            float(rect.width),
            float(rect.height),
        )

    def process_page(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        page_number: int,
        output_root: Path,
        dpi: int | None = None,
    ) -> PagePipelineResult:
        """Process one page through classification and OCR."""

        self._validate_pdf(pdf_path)
        self._validate_document_id(document_id)

        if page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

        with pymupdf.open(pdf_path) as document:
            page_index = page_number - 1

            if page_index >= len(document):
                raise ValueError(
                    f"Page {page_number} does not exist. "
                    f"PDF contains {len(document)} pages."
                )

            page = document[page_index]

            (
                native_text,
                image_count,
                drawing_count,
                width,
                height,
            ) = self._inspect_page(
                page,
                page_number,
            )

        classification = self.classifier.classify(
            page_number=page_number,
            text=native_text,
            image_count=image_count,
            drawing_count=drawing_count,
            width=width,
            height=height,
        )

        decision = classification.decision

        if decision != PageProcessingDecision.OCR_REQUIRED:
            return PagePipelineResult(
                page_number=page_number,
                decision=decision,
                classification=classification,
                native_text=native_text,
            )

        render_result = self.renderer.render_page(
            pdf_path=pdf_path,
            document_id=document_id,
            page_number=page_number,
            output_root=output_root,
            dpi=dpi,
        )

        ocr_result = self.provider.extract(
            image_path=render_result.output_path,
            page_number=page_number,
        )

        return PagePipelineResult(
            page_number=page_number,
            decision=decision,
            classification=classification,
            native_text=native_text,
            ocr_result=ocr_result,
            render_result=render_result,
        )

    def process_document(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        output_root: Path,
        dpi: int | None = None,
    ) -> OCRPipelineResult:
        """Process every page in a PDF."""

        self._validate_pdf(pdf_path)
        self._validate_document_id(document_id)

        with pymupdf.open(pdf_path) as document:
            page_count = len(document)

        pages = tuple(
            self.process_page(
                pdf_path=pdf_path,
                document_id=document_id,
                page_number=page_number,
                output_root=output_root,
                dpi=dpi,
            )
            for page_number in range(1, page_count + 1)
        )

        return OCRPipelineResult(
            document_id=document_id,
            pdf_path=pdf_path,
            pages=pages,
        )