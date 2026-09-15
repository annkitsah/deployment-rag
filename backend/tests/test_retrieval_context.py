import pytest

from app.retrieval.context import RetrievalContextAssembler
from app.retrieval.models import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)


def create_result(
    document_id: str,
    page_number: int,
    text: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        page_number=page_number,
        text=text,
        score=score,
        matched_terms=("retrieval",),
    )


def create_response(
    results: tuple[RetrievalResult, ...],
    query: str = "what is retrieval?",
) -> RetrievalResponse:
    return RetrievalResponse(
        query=RetrievalQuery(
            text=query,
            top_k=10,
        ),
        results=results,
    )


def test_assemble_includes_query_and_results() -> None:
    response = create_response(
        (
            create_result(
                "doc-001",
                1,
                "Retrieval finds relevant information.",
                2.5,
            ),
        )
    )

    assembler = RetrievalContextAssembler()

    context = assembler.assemble(response)

    assert context.query == "what is retrieval?"
    assert context.page_count == 1
    assert context.results == response.results
    assert "Retrieval finds relevant information." in context.text


def test_assemble_includes_source_metadata() -> None:
    response = create_response(
        (
            create_result(
                "doc-001",
                3,
                "Vectorless retrieval uses lexical matching.",
                1.23456,
            ),
        )
    )

    context = RetrievalContextAssembler().assemble(response)

    assert "[Source: doc-001 | Page: 3 | Score: 1.2346]" in context.text


def test_assemble_preserves_retrieval_order() -> None:
    response = create_response(
        (
            create_result("doc-001", 2, "Second ranked page.", 2.0),
            create_result("doc-001", 1, "First ranked page.", 1.0),
        )
    )

    context = RetrievalContextAssembler().assemble(response)

    assert context.results[0].page_number == 2
    assert context.results[1].page_number == 1

    assert context.text.index("Second ranked page.") < (
        context.text.index("First ranked page.")
    )


def test_assemble_respects_max_chars() -> None:
    response = create_response(
        (
            create_result("doc-001", 1, "A" * 20, 2.0),
            create_result("doc-001", 2, "B" * 20, 1.0),
        )
    )

    assembler = RetrievalContextAssembler(max_chars=70)

    context = assembler.assemble(response)

    assert context.page_count == 1
    assert "A" * 20 in context.text
    assert "B" * 20 not in context.text
    assert len(context.text) <= 70


def test_assemble_does_not_split_a_page() -> None:
    response = create_response(
        (
            create_result(
                "doc-001",
                1,
                "This entire page must remain intact.",
                2.0,
            ),
        )
    )

    assembler = RetrievalContextAssembler(max_chars=30)

    context = assembler.assemble(response)

    assert context.results == ()
    assert context.text == ""


def test_assemble_allows_exact_context_limit() -> None:
    response = create_response(
        (
            create_result(
                "doc-001",
                1,
                "Exact page.",
                2.0,
            ),
        )
    )

    assembler = RetrievalContextAssembler()

    formatted = assembler._format_result(response.results[0])
    assembler = RetrievalContextAssembler(max_chars=len(formatted))

    context = assembler.assemble(response)

    assert context.page_count == 1
    assert len(context.text) == len(formatted)


def test_assemble_empty_results() -> None:
    response = create_response(())

    context = RetrievalContextAssembler().assemble(response)

    assert context.query == "what is retrieval?"
    assert context.results == ()
    assert context.text == ""
    assert context.page_count == 0


def test_assemble_is_deterministic() -> None:
    response = create_response(
        (
            create_result("doc-001", 1, "First page.", 2.0),
            create_result("doc-002", 4, "Second page.", 1.0),
        )
    )

    assembler = RetrievalContextAssembler()

    first = assembler.assemble(response)
    second = assembler.assemble(response)

    assert first == second


def test_assembler_rejects_invalid_max_chars() -> None:
    with pytest.raises(ValueError):
        RetrievalContextAssembler(max_chars=0)

    with pytest.raises(ValueError):
        RetrievalContextAssembler(max_chars=-1)


def test_assembler_rejects_empty_separator() -> None:
    with pytest.raises(ValueError):
        RetrievalContextAssembler(separator="")


def test_custom_separator_is_used() -> None:
    response = create_response(
        (
            create_result("doc-001", 1, "First page.", 2.0),
            create_result("doc-001", 2, "Second page.", 1.0),
        )
    )

    context = RetrievalContextAssembler(
        separator="\n---\n",
    ).assemble(response)

    assert "\n---\n" in context.text


def test_max_pages_caps_the_number_of_included_results() -> None:
    response = create_response(
        (
            create_result("doc-001", 1, "First page.", 3.0),
            create_result("doc-001", 2, "Second page.", 2.0),
            create_result("doc-001", 3, "Third page.", 1.0),
        )
    )

    context = RetrievalContextAssembler(
        max_pages=2,
    ).assemble(response)

    assert context.page_count == 2
    assert context.results[0].page_number == 1
    assert context.results[1].page_number == 2
    assert "Third page." not in context.text


def test_max_pages_none_means_unbounded_by_count() -> None:
    response = create_response(
        tuple(
            create_result(
                "doc-001",
                page_number,
                f"Page {page_number} content.",
                float(10 - page_number),
            )
            for page_number in range(1, 11)
        )
    )

    context = RetrievalContextAssembler(
        max_pages=None,
    ).assemble(response)

    assert context.page_count == 10


def test_assembler_rejects_invalid_max_pages() -> None:
    with pytest.raises(ValueError):
        RetrievalContextAssembler(max_pages=0)

    with pytest.raises(ValueError):
        RetrievalContextAssembler(max_pages=-1)


def test_max_pages_and_max_chars_both_apply() -> None:
    response = create_response(
        (
            create_result("doc-001", 1, "First page.", 2.0),
            create_result("doc-001", 2, "Second page.", 1.0),
        )
    )

    # max_chars alone would allow only the first result through.
    formatted_first = RetrievalContextAssembler._format_result(
        response.results[0]
    )

    context = RetrievalContextAssembler(
        max_pages=2,
        max_chars=len(formatted_first),
    ).assemble(response)

    assert context.page_count == 1