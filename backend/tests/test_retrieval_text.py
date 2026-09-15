import pytest

from app.retrieval.text import (
    normalize_text,
    term_frequency,
    tokenize,
)


def test_normalize_text() -> None:
    result = normalize_text(
        "  Hello   WORLD!\n\nThis is   a test.  "
    )

    assert result == "hello world! this is a test."


def test_normalize_text_uses_unicode_normalization() -> None:
    result = normalize_text("Ｃａｆé")

    assert result == "café"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


def test_tokenize_removes_stopwords() -> None:
    result = tokenize(
        "This is a simple retrieval test."
    )

    assert result == (
        "simple",
        "retrieval",
        "test",
    )


def test_tokenize_can_keep_stopwords() -> None:
    result = tokenize(
        "This is a test.",
        remove_stopwords=False,
    )

    assert result == (
        "this",
        "is",
        "test",
    )


def test_tokenize_respects_minimum_length() -> None:
    result = tokenize(
        "A big AI model is useful.",
        min_token_length=3,
    )

    assert result == (
        "big",
        "model",
        "useful",
    )


def test_tokenize_preserves_unicode_words() -> None:
    result = tokenize(
        "Generative AI और machine learning",
    )

    assert "generative" in result
    assert "machine" in result
    assert "learning" in result
    assert "और" in result


def test_term_frequency() -> None:
    tokens = (
        "rag",
        "retrieval",
        "rag",
        "ocr",
    )

    assert term_frequency(tokens) == {
        "rag": 2,
        "retrieval": 1,
        "ocr": 1,
    }


def test_term_frequency_empty() -> None:
    assert term_frequency(()) == {}


def test_invalid_minimum_token_length() -> None:
    with pytest.raises(ValueError):
        tokenize(
            "test",
            min_token_length=0,
        )


def test_tokenize_removes_question_words() -> None:
    # Interrogative/auxiliary words are grammatically necessary in a
    # natural-language question but carry no topical signal for BM25
    # ranking -- leaving them un-stripped let common question phrasing
    # (e.g. "what ... does ...") skew scores toward pages that simply
    # contain ordinary prose, rather than pages actually relevant to
    # the query's real content words.
    result = tokenize(
        "What models does Perplexity use, and how do they work?"
    )

    assert result == (
        "models",
        "perplexity",
        "use",
        "they",
        "work",
    )
    assert "what" not in result
    assert "does" not in result
    assert "how" not in result
    assert "do" not in result


def test_tokenize_removes_remaining_wh_words() -> None:
    result = tokenize("Who, which, why, when, where, and whom")

    assert result == ()