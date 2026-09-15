import re
import unicodedata

_TOKEN_PATTERN = re.compile(r"\b[\w]+\b", re.UNICODE)

_DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "with",
    }
)


def normalize_text(text: str) -> str:
    """Normalize text into a deterministic lexical representation."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()

    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    min_token_length: int = 2,
) -> tuple[str, ...]:
    """Convert text into deterministic lexical tokens."""

    if min_token_length < 1:
        raise ValueError("min_token_length must be greater than zero")

    normalized = normalize_text(text)

    tokens = tuple(
        match.group(0)
        for match in _TOKEN_PATTERN.finditer(normalized)
        if len(match.group(0)) >= min_token_length
    )

    if not remove_stopwords:
        return tokens

    return tuple(
        token
        for token in tokens
        if token not in _DEFAULT_STOPWORDS
    )


def term_frequency(
    tokens: tuple[str, ...],
) -> dict[str, int]:
    """Return term frequencies for a token sequence."""

    frequencies: dict[str, int] = {}

    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies