"""BUG-104: combine_datasets.py exact dedup must remove CROSS-dataset duplicates.

The script's `run_pipeline` used `ExactDeduplicator.deduplicate_array` per
dataset, which only removes WITHIN-dataset duplicates (already done by
prepare_data) — leaving cross-dataset duplicate chunks in the combined training
set, despite the menu label "remove dupes across datasets". The minhash path
already deduped across (each secondary vs the primary). The fix unifies both to
CrossDatasetDeduplicator so "exact" is genuinely cross-dataset.

These tests import the script module and drive `run_pipeline` end-to-end.
"""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parent.parent / "scripts" / "combine_datasets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("combine_datasets_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path: Path, rows: list[list[int]]) -> str:
    np.save(str(path), np.array(rows, dtype=np.uint16))
    # np.save appends .npy; return the real path
    return str(path) + (".npy" if not str(path).endswith(".npy") else "")


def _settings(a_path: str, b_path: str, dedup: str, strategy: str = "concat") -> dict:
    return {
        "datasets": [
            {"path": a_path, "name": "A"},
            {"path": b_path, "name": "B"},
        ],
        "weights": [0.5, 0.5],
        "strategy": strategy,
        "dedup_method": dedup,
    }


class TestCrossDatasetExactDedup:
    def _build(self, tmp_path):
        # A and B each have NO within-dataset dupes, but share two chunks
        # ([2,2] and [3,3]) across datasets — the cross-dataset duplicates.
        a = _save(tmp_path / "A", [[1, 1], [2, 2], [3, 3]])
        b = _save(tmp_path / "B", [[2, 2], [3, 3], [4, 4]])
        return a, b

    def test_exact_removes_cross_dataset_duplicates(self, tmp_path):
        mod = _load_module()
        a, b = self._build(tmp_path)
        out = tmp_path / "combined.npy"
        mod.run_pipeline(_settings(a, b, "exact"), str(out))

        result = np.load(str(out))
        # A (3) + B-minus-shared (1: only [4,4]) = 4 chunks. The old within-only
        # path would have produced 6 (no cross-dataset removal).
        assert result.shape[0] == 4
        kept = {tuple(row) for row in result.tolist()}
        assert kept == {(1, 1), (2, 2), (3, 3), (4, 4)}

    def test_no_dedup_keeps_all(self, tmp_path):
        mod = _load_module()
        a, b = self._build(tmp_path)
        out = tmp_path / "combined_none.npy"
        mod.run_pipeline(_settings(a, b, "none"), str(out))
        result = np.load(str(out))
        # Without dedup, all 6 chunks (with the 2 cross-dataset dups) survive.
        assert result.shape[0] == 6

    def test_primary_kept_intact(self, tmp_path):
        # Primary (A) chunks must all survive; only the secondary is deduped.
        mod = _load_module()
        a, b = self._build(tmp_path)
        out = tmp_path / "combined2.npy"
        mod.run_pipeline(_settings(a, b, "exact"), str(out))
        kept = {tuple(row) for row in np.load(str(out)).tolist()}
        for chunk in ((1, 1), (2, 2), (3, 3)):
            assert chunk in kept
