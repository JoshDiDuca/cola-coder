"""retrieval_search_view: lexical search over the persisted index. Pure + MAIN-SAFE
(no model/GPU). Missing index → valid empty result; ranking by query-token overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

from cola_coder.ui import retrieval_search_view as rsch


def _write_index(root: Path, texts: list[str], metadata: list[dict] | None = None) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    payload = {
        "ids": [f"doc_{i}" for i in range(len(texts))],
        "texts": texts,
        "metadata": metadata if metadata is not None else [{} for _ in texts],
        "embed_dim": 8,
    }
    (root / "data" / "vector_index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_missing_index_is_empty_not_built(tmp_path: Path) -> None:
    out = rsch.search_index("anything", root=str(tmp_path))
    assert out["exists"] is False
    assert out["total_indexed"] == 0
    assert out["hits"] == []


def test_blank_query_returns_no_hits_but_counts(tmp_path: Path) -> None:
    _write_index(tmp_path, ["def add(a, b): return a + b", "class Foo: pass"])
    out = rsch.search_index("   ", root=str(tmp_path))
    assert out["exists"] is True
    assert out["total_indexed"] == 2
    assert out["hits"] == []


def test_ranks_token_overlap(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        [
            "def add(a, b):\n    return a + b",          # matches add
            "import os\nprint(os.getcwd())",             # no match
            "def add_numbers(x, y):\n    return x + y",  # matches add (as token? 'add_numbers' != 'add')
        ],
        metadata=[{"file_path": "math.py"}, {"file_path": "os_util.py"}, {"file_path": "nums.py"}],
    )
    out = rsch.search_index("add", top_k=5, root=str(tmp_path))
    assert out["exists"] is True
    # The first chunk has the exact token 'add' AND the substring → top hit.
    assert out["hits"], "expected at least one hit"
    top = out["hits"][0]
    assert top["source"] == "math.py"
    assert top["score"] > 0
    assert "add" in top["snippet"].lower()


def test_top_k_limits_results(tmp_path: Path) -> None:
    _write_index(tmp_path, [f"function process item {i}" for i in range(20)])
    out = rsch.search_index("process item", top_k=3, root=str(tmp_path))
    assert len(out["hits"]) == 3


def test_source_falls_back_to_id_when_no_metadata(tmp_path: Path) -> None:
    _write_index(tmp_path, ["query target here"], metadata=[{}])
    out = rsch.search_index("target", root=str(tmp_path))
    assert out["hits"][0]["source"] is None  # no source key in metadata
    assert out["hits"][0]["id"] == "doc_0"
