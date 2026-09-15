
import pytest

from app.retrieval.inverted_index import InvertedIndex


def test_add_and_lookup_term() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval systems")

    assert index.lookup("retrieval") == ("doc-001:page:1",)


def test_index_tokenizes_text() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "Retrieval SYSTEMS")

    assert index.lookup("retrieval") == ("doc-001:page:1",)
    assert index.lookup("systems") == ("doc-001:page:1",)


def test_multiple_pages_for_same_term() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "retrieval architecture")
    index.add("doc-002:page:1", "retrieval pipeline")

    assert index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_unknown_term_returns_empty_tuple() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")

    assert index.lookup("unknown") == ()


def test_lookup_is_deterministic() -> None:
    index = InvertedIndex()

    index.add("doc-002:page:1", "retrieval")
    index.add("doc-001:page:2", "retrieval")
    index.add("doc-001:page:1", "retrieval")

    assert index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_duplicate_page_indexing_does_not_duplicate_page_id() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval")
    index.add("doc-001:page:1", "retrieval")

    assert index.lookup("retrieval") == ("doc-001:page:1",)


def test_contains_reports_indexed_terms() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")

    assert index.contains("retrieval")
    assert not index.contains("unknown")


def test_remove_page_removes_all_its_terms() -> None:
    index = InvertedIndex()

    index.add(
        "doc-001:page:1",
        "retrieval system architecture",
    )

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == ()
    assert index.lookup("system") == ()
    assert index.lookup("architecture") == ()


def test_remove_page_preserves_other_pages() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "retrieval architecture")

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == (
        "doc-001:page:2",
    )
    assert index.lookup("system") == ()


def test_remove_unknown_page_is_safe() -> None:
    index = InvertedIndex()

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == ()


def test_empty_text_indexes_no_terms() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "")

    assert index.lookup("retrieval") == ()


def test_clear_removes_all_entries() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "architecture")

    index.clear()

    assert index.lookup("retrieval") == ()
    assert index.lookup("system") == ()
    assert index.lookup("architecture") == ()


@pytest.mark.parametrize(
    "page_id",
    [
        "",
        " ",
    ],
)
def test_invalid_page_id_is_rejected(page_id: str) -> None:
    index = InvertedIndex()

    with pytest.raises(ValueError):
        index.add(page_id, "retrieval")


def test_to_snapshot_and_from_snapshot_round_trip() -> None:
    index = InvertedIndex()
    index.add("doc-001:page:1", "retrieval systems architecture")
    index.add("doc-001:page:2", "vectorless retrieval design")

    snapshot = index.to_snapshot()

    restored = InvertedIndex.from_snapshot(snapshot)

    assert restored.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
    )
    assert restored.lookup("systems") == ("doc-001:page:1",)
    assert restored.lookup("vectorless") == ("doc-001:page:2",)


def test_from_snapshot_preserves_remove_behavior() -> None:
    index = InvertedIndex()
    index.add("doc-001:page:1", "retrieval systems")
    index.add("doc-001:page:2", "retrieval design")

    restored = InvertedIndex.from_snapshot(index.to_snapshot())

    restored.remove("doc-001:page:1")

    assert restored.lookup("systems") == ()
    assert restored.lookup("retrieval") == ("doc-001:page:2",)


def test_to_snapshot_on_empty_index() -> None:
    index = InvertedIndex()

    snapshot = index.to_snapshot()

    assert snapshot == {"index": {}, "page_terms": {}}

    restored = InvertedIndex.from_snapshot(snapshot)

    assert restored.lookup("anything") == ()


def test_from_snapshot_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        InvertedIndex.from_snapshot("not a dict")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"index": {}},
        {"page_terms": {}},
        {"index": "not a dict", "page_terms": {}},
        {"index": {}, "page_terms": "not a dict"},
    ],
)
def test_from_snapshot_rejects_malformed_shape(
    snapshot: dict,
) -> None:
    with pytest.raises(ValueError):
        InvertedIndex.from_snapshot(snapshot)