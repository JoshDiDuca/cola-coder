"""Tests for dataset deduplication (data/dedup.py) and its prepare_data wiring.

dedup.py was fully implemented but never invoked by the main prep pipeline —
a dead data-quality feature. These tests lock the ExactDeduplicator behavior
and assert prepare_data.py actually wires it in.
"""

from pathlib import Path

import numpy as np
import pytest

from cola_coder.data.dedup import ExactDeduplicator

ROOT = Path(__file__).parent.parent


class TestExactDeduplicator:
    def test_removes_exact_duplicate_rows(self):
        a = np.array([1, 2, 3, 4], dtype=np.uint16)
        b = np.array([5, 6, 7, 8], dtype=np.uint16)
        data = np.stack([a, b, a, a, b])  # 5 rows, 2 unique
        deduped, removed = ExactDeduplicator().deduplicate_array(data)
        assert removed == 3
        assert len(deduped) == 2
        # First occurrences kept, in order
        assert np.array_equal(deduped[0], a)
        assert np.array_equal(deduped[1], b)

    def test_no_duplicates_returns_all(self):
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        deduped, removed = ExactDeduplicator().deduplicate_array(data)
        assert removed == 0
        assert len(deduped) == 3
        assert np.array_equal(deduped, data)

    def test_all_identical_collapses_to_one(self):
        row = np.array([9, 9, 9, 9], dtype=np.uint16)
        data = np.stack([row] * 10)
        deduped, removed = ExactDeduplicator().deduplicate_array(data)
        assert removed == 9
        assert len(deduped) == 1

    def test_near_duplicate_kept_one_token_differs(self):
        # Exact dedup must NOT collapse rows that differ by a single token.
        a = np.array([1, 2, 3, 4], dtype=np.uint16)
        b = np.array([1, 2, 3, 5], dtype=np.uint16)
        deduped, removed = ExactDeduplicator().deduplicate_array(np.stack([a, b]))
        assert removed == 0
        assert len(deduped) == 2

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            ExactDeduplicator().deduplicate_array(np.arange(4, dtype=np.uint16))

    def test_is_reusable_after_reset(self):
        d = ExactDeduplicator()
        d.deduplicate_array(np.stack([np.array([1, 1], dtype=np.uint16)] * 2))
        # deduplicate_array clears seen_hashes at entry, so a fresh call is clean
        _, removed = d.deduplicate_array(np.array([[1, 1], [2, 2]], dtype=np.uint16))
        assert removed == 0


class TestPrepareDataWiring:
    """prepare_data.py must actually call the deduplicator (not just define it)."""

    def test_prepare_data_invokes_exact_deduplicator(self):
        text = (ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
        assert "from cola_coder.data.dedup import ExactDeduplicator" in text
        assert "deduplicate_array" in text

    def test_dedup_flag_exists_and_defaults_on(self):
        text = (ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
        assert '"--dedup"' in text
        assert '"--no-dedup"' in text
        # Default must be exact (dedup on by default)
        assert 'default="exact"' in text

    def test_menu_passes_no_dedup_when_disabled(self):
        text = (ROOT / "src" / "cola_coder" / "features" / "menus"
                / "data_menu.py").read_text(encoding="utf-8")
        assert "--no-dedup" in text
