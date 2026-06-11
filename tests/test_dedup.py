"""Tests for dataset deduplication (data/dedup.py) and its prepare_data wiring.

dedup.py was fully implemented but never invoked by the main prep pipeline —
a dead data-quality feature. These tests lock the ExactDeduplicator behavior
and assert prepare_data.py actually wires it in.
"""

from pathlib import Path

import numpy as np
import pytest

from cola_coder.data.dedup import (
    _HAS_DATASKETCH,
    CrossDatasetDeduplicator,
    ExactDeduplicator,
    dedup_npy_file,
)

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

    def test_prepare_data_invokes_dedup_helper(self):
        text = (ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
        assert "dedup_npy_file" in text

    def test_dedup_flag_exists_and_defaults_on(self):
        text = (ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
        assert '"--dedup"' in text
        assert '"--no-dedup"' in text
        # Default must be exact (dedup on by default)
        assert 'default="exact"' in text

    def test_minhash_mode_wired(self):
        text = (ROOT / "scripts" / "prepare_data.py").read_text(encoding="utf-8")
        assert '"minhash"' in text
        assert "--dedup-threshold" in text
        # Must warn (not silently degrade) when datasketch is missing
        assert "_HAS_DATASKETCH" in text

    def test_menu_passes_dedup_mode(self):
        text = (ROOT / "src" / "cola_coder" / "features" / "menus"
                / "data_menu.py").read_text(encoding="utf-8")
        assert '"--dedup", dedup_mode' in text
        assert "minhash" in text

    def test_pyproject_declares_dedup_extra(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "dedup = [" in text
        assert "datasketch" in text


class TestSelfMinHashDedup:
    """deduplicate_self_array: near-dup removal, with exact fallback."""

    def test_exact_fallback_removes_identical_keeps_distinct(self):
        # method="exact" forces the no-datasketch path regardless of install.
        a = np.array([1, 2, 3, 4], dtype=np.uint16)
        b = np.array([5, 6, 7, 8], dtype=np.uint16)
        data = np.stack([a, b, a, a])  # 2 unique
        dedup = CrossDatasetDeduplicator(method="exact")
        deduped, removed = dedup.deduplicate_self_array(data)
        assert removed == 2
        assert len(deduped) == 2

    def test_rejects_non_2d(self):
        dedup = CrossDatasetDeduplicator(method="exact")
        with pytest.raises(ValueError):
            dedup.deduplicate_self_array(np.arange(4, dtype=np.uint16))

    def test_missing_datasketch_falls_back_to_exact(self):
        # Constructing with method="minhash" while datasketch is absent must
        # transparently downgrade to exact (and still remove identical rows).
        if _HAS_DATASKETCH:
            pytest.skip("datasketch installed — fallback path not exercised")
        dedup = CrossDatasetDeduplicator(method="minhash")
        assert dedup.method == "exact"  # constructor downgraded
        row = np.array([9, 9, 9, 9], dtype=np.uint16)
        deduped, removed = dedup.deduplicate_self_array(np.stack([row, row, row]))
        assert removed == 2 and len(deduped) == 1

    @pytest.mark.skipif(not _HAS_DATASKETCH, reason="needs datasketch")
    def test_minhash_removes_near_duplicate(self):
        # Two chunks differing by a few tokens out of many → near-dup → one kept.
        base = list(range(1, 60))
        near = base.copy()
        near[0] = 999  # one token differs out of 59
        distinct = list(range(200, 259))
        data = np.array([base, near, distinct], dtype=np.uint16)
        dedup = CrossDatasetDeduplicator(method="minhash", threshold=0.6)
        deduped, removed = dedup.deduplicate_self_array(data)
        assert removed == 1  # near-dup dropped
        assert len(deduped) == 2  # base + distinct kept


class TestDedupNpyFile:
    """In-place file dedup — regression for DATA-004 (Windows mmap+replace lock).

    The bug: prepare_data loaded the file with mmap_mode='r' and then os.replace'd
    it; on Windows the open mmap kept the file locked, so the replace raised
    PermissionError that was swallowed as 'dedup failed' — silently no-op'ing
    dedup on the primary platform. These tests exercise the real mmap+replace
    path on a temp file.
    """

    def _write(self, tmp_path, rows):
        p = tmp_path / "data.npy"
        np.save(p, np.array(rows, dtype=np.uint16))
        return p

    def test_exact_rewrites_file_in_place(self, tmp_path):
        a, b = [1, 2, 3, 4], [5, 6, 7, 8]
        p = self._write(tmp_path, [a, b, a, a, b])  # 2 unique
        result = dedup_npy_file(p, mode="exact")
        assert result.before == 5
        assert result.after == 2
        assert result.removed == 3
        # File was actually rewritten (this is what failed on Windows)
        reloaded = np.load(p)
        assert len(reloaded) == 2
        # No temp file left behind
        assert not p.with_suffix(".dedup_tmp.npy").exists()

    def test_no_duplicates_leaves_file_untouched(self, tmp_path):
        p = self._write(tmp_path, [[1, 1], [2, 2], [3, 3]])
        result = dedup_npy_file(p, mode="exact")
        assert result.removed == 0
        assert len(np.load(p)) == 3

    def test_minhash_mode_label_reflects_datasketch(self, tmp_path):
        p = self._write(tmp_path, [[1, 1], [2, 2]])
        result = dedup_npy_file(p, mode="minhash", threshold=0.8)
        assert result.minhash_active is _HAS_DATASKETCH
        if _HAS_DATASKETCH:
            assert "minhash" in result.mode
        else:
            assert "fallback" in result.mode

    @pytest.mark.skipif(not _HAS_DATASKETCH, reason="needs datasketch")
    def test_minhash_file_removes_near_dups_in_place(self, tmp_path):
        base = list(range(1, 60))
        near = base.copy()
        near[0] = 999
        distinct = list(range(200, 259))
        p = self._write(tmp_path, [base, near, distinct])
        result = dedup_npy_file(p, mode="minhash", threshold=0.6)
        assert result.removed == 1
        assert len(np.load(p)) == 2


class TestDeduplicatePairInPlace:
    """DATA-024: deduplicate_pair's default output_path overwrites secondary;
    on Windows np.save over a still-mmapped file raised PermissionError/OSError
    (the DATA-004 class). The secondary mmap must be released before the save.
    """

    def test_default_output_overwrites_secondary_no_crash(self, tmp_path):
        prim = tmp_path / "p.npy"
        sec = tmp_path / "s.npy"
        np.save(prim, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint16))
        # row 0 of secondary duplicates primary row 0; row 1 is unique.
        np.save(sec, np.array([[1, 2, 3, 4], [9, 9, 9, 9]], dtype=np.uint16))

        dd = CrossDatasetDeduplicator(method="exact")
        # output_path defaults to secondary_path -> in-place overwrite (the
        # path that crashed on Windows before the mmap release).
        result = dd.deduplicate_pair(str(prim), str(sec))

        assert result.duplicates_removed == 1
        assert result.output_chunks == 1
        reloaded = np.load(sec)
        assert len(reloaded) == 1
        assert list(reloaded[0]) == [9, 9, 9, 9]  # the non-dup row survived

    def test_distinct_output_path_keeps_secondary(self, tmp_path):
        prim = tmp_path / "p.npy"
        sec = tmp_path / "s.npy"
        out = tmp_path / "out.npy"
        np.save(prim, np.array([[1, 2], [3, 4]], dtype=np.uint16))
        np.save(sec, np.array([[1, 2], [7, 8]], dtype=np.uint16))

        dd = CrossDatasetDeduplicator(method="exact")
        result = dd.deduplicate_pair(str(prim), str(sec), output_path=str(out))
        assert result.duplicates_removed == 1
        # Secondary untouched; output written separately.
        assert len(np.load(sec)) == 2
        assert len(np.load(out)) == 1
