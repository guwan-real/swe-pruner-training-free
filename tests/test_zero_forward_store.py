from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_forward_pruning.store import RawStore


def test_store_is_byte_exact_and_rejects_traversal(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    text = "源码\nline two\n"
    saved = store.save(text, {"method": "test"})
    assert store.read(saved.raw_id) == text
    assert store.metadata(saved.raw_id)["method"] == "test"
    with pytest.raises(KeyError, match="invalid"):
        store.read("../../etc/passwd")


def test_store_purges_expired_items(tmp_path: Path) -> None:
    store = RawStore(tmp_path, ttl_hours=1)
    saved = store.save("old")
    metadata = store.metadata(saved.raw_id)
    metadata["created_at"] = 1.0
    saved.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert store.purge_expired(now=7200) == 1
    with pytest.raises(KeyError):
        store.read(saved.raw_id)
