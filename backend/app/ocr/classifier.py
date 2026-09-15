from app.ocr.models import (
    PageContentMetadata,
    PageProcessingDecision,
    PageProcessingDecisionResult,
)
from app.ocr.quality import TextQualityAnalyzer


class PageContentClassifier:
    """Determine the correct processing strategy for a PDF page."""

    def __init__(
        self,
        quality_analyzer: TextQualityAnalyzer | None = None,
        *,
        empty_text_threshold: int = 5,
    ) -> None:
        if empty_text_threshold < 0:
            raise ValueError(
                "empty_text_threshold must be non-negative"
            )

        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else TextQualityAnalyzer()
        )

        self.empty_text_threshold = empty_text_threshold

    def classify(
        self,
        *,
        page_number: int,
        text: str,
        image_count: int,
        drawing_count: int,
        width: float,
        height: float,
    ) -> PageProcessingDecisionResult:
        """Classify a PDF page based on text and visual content."""

        if page_number < 1:
            raise ValueError(
                "page_number must be greater than or equal to 1"
            )

        if image_count < 0:
            raise ValueError(
                "image_count cannot be negative"
            )

        if drawing_count < 0:
            raise ValueError(
                "drawing_count cannot be negative"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "page dimensions must be positive"
            )

        quality = self.quality_analyzer.analyze(text)

        content = PageContentMetadata(
            page_number=page_number,
            text_character_count=len(text),
            image_count=image_count,
            drawing_count=drawing_count,
            width=width,
            height=height,
        )

        stripped_text_length = len(text.strip())

        # ---------------------------------------------------------
        # 1. Truly empty page
        # ---------------------------------------------------------
        if (
            stripped_text_length <= self.empty_text_threshold
            and image_count == 0
            and drawing_count == 0
        ):
            return PageProcessingDecisionResult(
                page_number=page_number,
                decision=PageProcessingDecision.EMPTY,
                text_quality=quality,
                content=content,
                reasons=("empty_page",),
            )

        # ---------------------------------------------------------
        # 2. Usable native text
        # ---------------------------------------------------------
        if not quality.ocr_required:
            return PageProcessingDecisionResult(
                page_number=page_number,
                decision=PageProcessingDecision.NATIVE_TEXT,
                text_quality=quality,
                content=content,
                reasons=("usable_native_text",),
            )

        # ---------------------------------------------------------
        # 3. Raster image + unusable text
        # ---------------------------------------------------------
        if image_count > 0:
            return PageProcessingDecisionResult(
                page_number=page_number,
                decision=PageProcessingDecision.OCR_REQUIRED,
                text_quality=quality,
                content=content,
                reasons=(
                    "raster_image_without_usable_text",
                ),
            )

        # ---------------------------------------------------------
        # 4. Vector drawing + unusable text
        # ---------------------------------------------------------
        if drawing_count > 0:
            reasons = (
                *quality.reasons,
                "vector_content_without_usable_text",
            )

            return PageProcessingDecisionResult(
                page_number=page_number,
                decision=PageProcessingDecision.REVIEW_REQUIRED,
                text_quality=quality,
                content=content,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        # ---------------------------------------------------------
        # 5. Poor text with no visual content
        # ---------------------------------------------------------
        reasons = (
            *quality.reasons,
            "poor_text_without_visual_content",
        )

        return PageProcessingDecisionResult(
            page_number=page_number,
            decision=PageProcessingDecision.REVIEW_REQUIRED,
            text_quality=quality,
            content=content,
            reasons=tuple(dict.fromkeys(reasons)),
        )