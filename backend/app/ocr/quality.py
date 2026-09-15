import string

from app.ocr.models import TextExtractionMode, TextQualityResult


class TextQualityAnalyzer:
    """Determine whether extracted PDF text is suitable for retrieval."""

    def __init__(
        self,
        *,
        minimum_characters: int = 80,
        minimum_alphabetic_ratio: float = 0.25,
        minimum_printable_ratio: float = 0.90,
        minimum_quality_score: float = 0.45,
    ) -> None:
        if minimum_characters < 0:
            raise ValueError(
                "minimum_characters must be non-negative"
            )

        if not 0.0 <= minimum_alphabetic_ratio <= 1.0:
            raise ValueError(
                "minimum_alphabetic_ratio must be between 0 and 1"
            )

        if not 0.0 <= minimum_printable_ratio <= 1.0:
            raise ValueError(
                "minimum_printable_ratio must be between 0 and 1"
            )

        if not 0.0 <= minimum_quality_score <= 1.0:
            raise ValueError(
                "minimum_quality_score must be between 0 and 1"
            )

        self.minimum_characters = minimum_characters
        self.minimum_alphabetic_ratio = minimum_alphabetic_ratio
        self.minimum_printable_ratio = minimum_printable_ratio
        self.minimum_quality_score = minimum_quality_score

    def analyze(self, text: str) -> TextQualityResult:
        """Analyze extracted text and determine whether OCR is required."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        character_count = len(text)

        if character_count == 0:
            return TextQualityResult(
                character_count=0,
                alphabetic_character_count=0,
                alphanumeric_character_count=0,
                whitespace_count=0,
                replacement_character_count=0,
                alphabetic_ratio=0.0,
                printable_ratio=0.0,
                quality_score=0.0,
                extraction_mode=TextExtractionMode.OCR,
                ocr_required=True,
                reasons=("no_text_extracted",),
            )

        alphabetic_character_count = sum(
            character.isalpha()
            for character in text
        )

        alphanumeric_character_count = sum(
            character.isalnum()
            for character in text
        )

        whitespace_count = sum(
            character.isspace()
            for character in text
        )

        replacement_character_count = text.count("\ufffd")

        printable_count = sum(
            character in string.printable
            or character.isprintable()
            for character in text
        )

        alphabetic_ratio = (
            alphabetic_character_count / character_count
        )

        printable_ratio = (
            printable_count / character_count
        )

        quality_score = self._calculate_quality_score(
            character_count=character_count,
            alphabetic_ratio=alphabetic_ratio,
            printable_ratio=printable_ratio,
            replacement_character_count=replacement_character_count,
        )

        reasons: list[str] = []

        if character_count < self.minimum_characters:
            reasons.append("insufficient_text")

        if alphabetic_ratio < self.minimum_alphabetic_ratio:
            reasons.append("low_alphabetic_ratio")

        if printable_ratio < self.minimum_printable_ratio:
            reasons.append("low_printable_ratio")

        if replacement_character_count > 0:
            reasons.append("replacement_characters_present")

        if quality_score < self.minimum_quality_score:
            reasons.append("low_quality_score")

        ocr_required = bool(reasons)

        return TextQualityResult(
            character_count=character_count,
            alphabetic_character_count=alphabetic_character_count,
            alphanumeric_character_count=alphanumeric_character_count,
            whitespace_count=whitespace_count,
            replacement_character_count=replacement_character_count,
            alphabetic_ratio=alphabetic_ratio,
            printable_ratio=printable_ratio,
            quality_score=quality_score,
            extraction_mode=(
                TextExtractionMode.OCR
                if ocr_required
                else TextExtractionMode.NATIVE
            ),
            ocr_required=ocr_required,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _calculate_quality_score(
        *,
        character_count: int,
        alphabetic_ratio: float,
        printable_ratio: float,
        replacement_character_count: int,
    ) -> float:
        """Calculate a bounded text-quality score."""

        character_score = min(
            character_count / 1000.0,
            1.0,
        )

        replacement_penalty = min(
            replacement_character_count
            / max(character_count, 1),
            1.0,
        )

        score = (
            0.35 * character_score
            + 0.35 * alphabetic_ratio
            + 0.30 * printable_ratio
        )

        score *= 1.0 - replacement_penalty

        return max(
            0.0,
            min(score, 1.0),
        )