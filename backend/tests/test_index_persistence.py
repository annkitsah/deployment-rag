import json
from pathlib import Path

import pytest

from app.retrieval.index_persistence import IndexSnapshotStore
from app.retrieval.inverted_index import InvertedIndex


def make_index() -> InvertedIndex:
    index = InvertedIndex()
    index.add("doc-001:page:1", "retrieval systems architecture")
    index.add("doc-001:page:2", "vectorless retrieval design")
    return index


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    store = IndexSnapshotStore(tmp_path / "does-not-exist.json")

    assert store.load() is None


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = IndexSnapshotStore(tmp_path / "snapshot.json")
    index = make_index()

    store.save(index, page_count=2)

    loaded = store.load()

    assert loaded is not None

    restored_index, page_count = loaded

    assert page_count == 2
    assert restored_index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
    )


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "nested" / "dirs" / "snapshot.json"
    store = IndexSnapshotStore(nested_path)

    store.save(make_index(), page_count=2)

    assert nested_path.is_file()


def test_save_is_atomic_no_leftover_temp_files(
    tmp_path: Path,
) -> None:
    store = IndexSnapshotStore(tmp_path / "snapshot.json")

    store.save(make_index(), page_count=2)

    temp_files = list(tmp_path.glob(".index-snapshot-*.tmp"))

    assert temp_files == []


def test_save_rejects_negative_page_count(tmp_path: Path) -> None:
    store = IndexSnapshotStore(tmp_path / "snapshot.json")

    with pytest.raises(ValueError):
        store.save(make_index(), page_count=-1)


def test_load_returns_none_for_corrupt_json(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text("{ not valid json")

    store = IndexSnapshotStore(snapshot_path)

    assert store.load() is None


def test_load_returns_none_for_non_dict_json(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(["not", "a", "dict"]))

    store = IndexSnapshotStore(snapshot_path)

    assert store.load() is None


def test_load_returns_none_for_wrong_version(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "version": 999,
                "page_count": 2,
                "index": {"index": {}, "page_terms": {}},
            }
        )
    )

    store = IndexSnapshotStore(snapshot_path)

    assert store.load() is None


def test_load_returns_none_for_missing_fields(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"version": 1}))

    store = IndexSnapshotStore(snapshot_path)

    assert store.load() is None


def test_load_returns_none_for_malformed_index_payload(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "version": 1,
                "page_count": 2,
                "index": {"index": "not a dict", "page_terms": {}},
            }
        )
    )

    store = IndexSnapshotStore(snapshot_path)

    assert store.load() is None


def test_delete_removes_snapshot_file(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    store = IndexSnapshotStore(snapshot_path)

    store.save(make_index(), page_count=2)
    assert snapshot_path.is_file()

    store.delete()

    assert not snapshot_path.is_file()


def test_delete_is_safe_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    store = IndexSnapshotStore(tmp_path / "does-not-exist.json")

    # Should not raise.
    store.delete()


def test_save_overwrites_previous_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    store = IndexSnapshotStore(snapshot_path)

    store.save(make_index(), page_count=2)

    new_index = InvertedIndex()
    new_index.add("doc-002:page:1", "completely different content")

    store.save(new_index, page_count=1)

    loaded = store.load()

    assert loaded is not None
    restored_index, page_count = loaded

    assert page_count == 1
    assert restored_index.lookup("retrieval") == ()
    assert restored_index.lookup("completely") == (
        "doc-002:page:1",
    )