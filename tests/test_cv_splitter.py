"""Tests for features/cv_splitter.py — Feature 97.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import pytest

from cola_coder.features.cv_splitter import (
    FEATURE_ENABLED,
    CVReport,
    CodeFile,
    CrossValidationSplitter,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_files(n: int, size: int = 100) -> list[CodeFile]:
    return [CodeFile(path=f"file_{i}.py", size=size) for i in range(n)]


@pytest.fixture
def splitter():
    return CrossValidationSplitter(k=5, seed=42)


@pytest.fixture
def files():
    return _make_files(20)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_k_lt_2_raises():
    with pytest.raises(ValueError):
        CrossValidationSplitter(k=1)


def test_k_2_valid():
    s = CrossValidationSplitter(k=2, seed=0)
    assert s.k == 2


# ---------------------------------------------------------------------------
# Basic split
# ---------------------------------------------------------------------------


def test_split_returns_report(splitter, files):
    report = splitter.split(files)
    assert isinstance(report, CVReport)


def test_report_k_matches(splitter, files):
    report = splitter.split(files)
    assert report.k == 5


def test_report_n_files(splitter, files):
    report = splitter.split(files, )
    assert report.n_files == 20


def test_n_folds_equals_k(splitter, files):
    report = splitter.split(files)
    assert len(report.folds) == 5


def test_empty_files_returns_empty_report(splitter):
    report = splitter.split([])
    assert report.n_files == 0
    assert report.folds == []


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------


def test_no_leakage_per_fold(splitter, files):
    report = splitter.split(files)
    for fold in report.folds:
        leaked = CrossValidationSplitter.check_leakage(fold)
        assert leaked == [], f"Fold {fold.fold_index} leaked: {leaked}"


# ---------------------------------------------------------------------------
# Coverage — every file appears in validation exactly once
# ---------------------------------------------------------------------------


def test_every_file_in_val_exactly_once(splitter, files):
    report = splitter.split(files)
    val_paths: list[str] = []
    for fold in report.folds:
        val_paths.extend(f.path for f in fold.val_files)
    # Each file should appear exactly once across all val folds
    from collections import Counter
    counts = Counter(val_paths)
    for path, count in counts.items():
        assert count == 1, f"{path} appears {count} times in val sets"
    # All files covered
    assert set(counts.keys()) == {f.path for f in files}


# ---------------------------------------------------------------------------
# Train / val sizes
# ---------------------------------------------------------------------------


def test_train_val_sum_to_total(splitter, files):
    report = splitter.split(files)
    for fold in report.folds:
        assert fold.n_train + fold.n_val == len(files)


def test_val_fraction_roughly_one_over_k(splitter):
    files = _make_files(50)
    report = splitter.split(files)
    # Expected val fraction ≈ 1/k = 0.2
    assert abs(report.avg_val_fraction - 1 / splitter.k) < 0.1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_splits():
    files = _make_files(15)
    r1 = CrossValidationSplitter(k=3, seed=7).split(files)
    r2 = CrossValidationSplitter(k=3, seed=7).split(files)
    for f1, f2 in zip(r1.folds, r2.folds):
        assert [f.path for f in f1.val_files] == [f.path for f in f2.val_files]


def test_different_seed_different_splits():
    files = _make_files(20)
    r1 = CrossValidationSplitter(k=4, seed=1).split(files)
    r2 = CrossValidationSplitter(k=4, seed=2).split(files)
    same = all(
        [f.path for f in r1.folds[i].val_files] == [f.path for f in r2.folds[i].val_files]
        for i in range(4)
    )
    # Very unlikely to be identical with different seeds
    assert not same


# ---------------------------------------------------------------------------
# Group boundary respected
# ---------------------------------------------------------------------------


def test_group_files_not_split(splitter):
    """Files in the same group must all be in the same fold."""
    files = [
        CodeFile(path=f"repo_a/file_{i}.py", size=100, group="repo_a")
        for i in range(4)
    ] + [
        CodeFile(path=f"repo_b/file_{i}.py", size=100, group="repo_b")
        for i in range(4)
    ] + [
        CodeFile(path=f"other_{i}.py", size=100) for i in range(12)
    ]
    report = splitter.split(files)
    for fold in report.folds:
        val_paths = {f.path for f in fold.val_files}
        # If any repo_a file is in val, ALL repo_a files must be in val
        a_in_val = [p for p in val_paths if p.startswith("repo_a")]
        b_in_val = [p for p in val_paths if p.startswith("repo_b")]
        if a_in_val:
            assert len(a_in_val) == 4
        if b_in_val:
            assert len(b_in_val) == 4


# ---------------------------------------------------------------------------
# Deterministic file_fold_id
# ---------------------------------------------------------------------------


def test_file_fold_id_range():
    for i in range(20):
        fid = CrossValidationSplitter.file_fold_id(f"file_{i}.py", k=5)
        assert 0 <= fid < 5


def test_file_fold_id_deterministic():
    a = CrossValidationSplitter.file_fold_id("main.py", k=5)
    b = CrossValidationSplitter.file_fold_id("main.py", k=5)
    assert a == b


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def test_fold_as_dict(splitter, files):
    report = splitter.split(files)
    d = report.folds[0].as_dict()
    assert "fold_index" in d
    assert "n_train" in d
    assert "n_val" in d


def test_report_as_dict(splitter, files):
    report = splitter.split(files)
    d = report.as_dict()
    assert d["k"] == 5
    assert len(d["folds"]) == 5
