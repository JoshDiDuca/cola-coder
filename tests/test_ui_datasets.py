"""Tests for the dataset-browsing UI helpers (src/cola_coder/ui/datasets.py).

All fixtures are tiny in-memory arrays written to tmp_path — no GPU, no large
files, no training interaction.
"""

from __future__ import annotations

import json

import numpy as np

from cola_coder.ui.datasets import dataset_preview, list_datasets, score_summary


def _make_fixtures(tmp_path):
    """Create a small .npy, its .weights.npy sibling, and a .jsonl (with a blank)."""
    npy_path = tmp_path / "train.npy"
    np.save(npy_path, np.arange(40, dtype=np.uint16).reshape(5, 8))

    weights_path = tmp_path / "train.weights.npy"
    np.save(weights_path, np.array([0.1, 0.5, 0.9, 0.3, 0.7], dtype=np.float32))

    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        json.dumps({"a": 1}) + "\n" + "\n" + json.dumps({"b": 2}) + "\n",
        encoding="utf-8",
    )
    return npy_path, weights_path, jsonl_path


def test_list_datasets_finds_npy_and_jsonl(tmp_path):
    _make_fixtures(tmp_path)
    entries = list_datasets(str(tmp_path))

    names = {entry["name"] for entry in entries}
    assert "train.npy" in names
    assert "data.jsonl" in names
    # The .weights.npy sidecar must NOT appear in the listing.
    assert "train.weights.npy" not in names

    by_name = {entry["name"]: entry for entry in entries}

    npy_entry = by_name["train.npy"]
    assert npy_entry["kind"] == "npy"
    assert npy_entry["has_weights"] is True
    assert npy_entry["num_samples"] == 5
    assert npy_entry["size_bytes"] > 0
    assert isinstance(npy_entry["mtime"], float)

    jsonl_entry = by_name["data.jsonl"]
    assert jsonl_entry["kind"] == "jsonl"
    assert jsonl_entry["has_weights"] is False
    # 3 lines, one blank -> 2 non-empty samples.
    assert jsonl_entry["num_samples"] == 2


def test_list_datasets_sorted_by_path(tmp_path):
    _make_fixtures(tmp_path)
    entries = list_datasets(str(tmp_path))
    paths = [entry["path"] for entry in entries]
    assert paths == sorted(paths)


def test_list_datasets_missing_root():
    assert list_datasets("does/not/exist/anywhere") == []


def test_dataset_preview_jsonl(tmp_path):
    _, _, jsonl_path = _make_fixtures(tmp_path)
    result = dataset_preview(str(jsonl_path))
    assert result["kind"] == "jsonl"
    assert result["num_samples"] == 2
    # Blank line skipped; both parsed objects present.
    assert result["preview"] == [{"a": 1}, {"b": 2}]


def test_dataset_preview_jsonl_respects_n(tmp_path):
    _, _, jsonl_path = _make_fixtures(tmp_path)
    result = dataset_preview(str(jsonl_path), n=1)
    assert result["num_samples"] == 2
    assert result["preview"] == [{"a": 1}]


def test_dataset_preview_npy(tmp_path):
    npy_path, _, _ = _make_fixtures(tmp_path)
    result = dataset_preview(str(npy_path), n=3)
    assert result["kind"] == "npy"
    assert result["shape"] == [5, 8]
    assert result["dtype"] == "uint16"
    assert result["num_samples"] == 5
    assert len(result["preview"]) == 3
    # Preview rows are plain python lists.
    assert isinstance(result["preview"], list)
    assert isinstance(result["preview"][0], list)
    assert result["preview"][0] == list(range(8))


def test_dataset_preview_missing_path():
    result = dataset_preview("nope/missing.jsonl")
    assert "error" in result


def test_score_summary(tmp_path):
    _, weights_path, _ = _make_fixtures(tmp_path)
    result = score_summary(str(weights_path))
    assert result["n"] == 5
    assert abs(result["mean"] - 0.5) < 1e-6
    assert abs(result["min"] - 0.1) < 1e-6
    assert abs(result["max"] - 0.9) < 1e-6
    assert len(result["histogram"]) == 10
    assert len(result["bins"]) == 11
    assert sum(result["histogram"]) == 5


def test_score_summary_missing():
    result = score_summary("nope/missing.weights.npy")
    assert "error" in result


def test_score_summary_empty(tmp_path):
    empty_path = tmp_path / "empty.weights.npy"
    np.save(empty_path, np.array([], dtype=np.float32))
    result = score_summary(str(empty_path))
    assert "error" in result
