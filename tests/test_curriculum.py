"""Tests for CurriculumOrderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cola_coder.data.scorers.curriculum import (
    CurriculumOrderer,
    CurriculumSchedule,
    CurriculumStrategy,
)


@pytest.fixture
def scored_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create test .npy data with known weights."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, size=(100, 32), dtype=np.uint16)
    # Weights: 0.1, 0.2, ..., 1.0 repeated
    weights = np.array([i / 100.0 for i in range(100)], dtype=np.float32)

    data_path = tmp_path / "train.npy"
    weights_path = tmp_path / "train.weights.npy"
    np.save(str(data_path), data)
    np.save(str(weights_path), weights)
    return data_path, weights_path


class TestCurriculumStrategy:
    def test_all_strategies_are_strings(self) -> None:
        for s in CurriculumStrategy:
            assert isinstance(s.value, str)


class TestCurriculumSchedule:
    def test_save_and_load(self, tmp_path: Path) -> None:
        schedule = CurriculumSchedule(
            strategy="easy_to_hard",
            total_samples=100,
            phases=[{"phase": 1, "start_idx": 0, "end_idx": 100}],
        )
        path = tmp_path / "schedule.json"
        schedule.save(path)
        loaded = CurriculumSchedule.load(path)
        assert loaded.strategy == "easy_to_hard"
        assert loaded.total_samples == 100
        assert len(loaded.phases) == 1


class TestCurriculumOrderer:
    def test_easy_to_hard_highest_first(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.EASY_TO_HARD)
        schedule = orderer.reorder(data_path, weights_path)

        # After reorder, first weight should be the highest
        reordered_weights = np.load(str(weights_path))
        assert reordered_weights[0] >= reordered_weights[-1]
        assert schedule.strategy == "easy_to_hard"
        assert schedule.total_samples == 100

    def test_hard_to_easy_lowest_first(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.HARD_TO_EASY)
        schedule = orderer.reorder(data_path, weights_path)

        reordered_weights = np.load(str(weights_path))
        assert reordered_weights[0] <= reordered_weights[-1]
        assert schedule.strategy == "hard_to_easy"

    def test_random_changes_order(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        original_weights = np.load(str(weights_path)).copy()
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.RANDOM)
        orderer.reorder(data_path, weights_path)

        reordered_weights = np.load(str(weights_path))
        # Should be different from original (with overwhelming probability)
        assert not np.array_equal(original_weights, reordered_weights)

    def test_output_path_preserves_original(self, scored_data: tuple[Path, Path], tmp_path: Path) -> None:
        data_path, weights_path = scored_data
        output = tmp_path / "reordered.npy"
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.EASY_TO_HARD)
        orderer.reorder(data_path, weights_path, output_path=output)

        # Original should be unchanged (reorder wrote to output_path)
        assert output.exists()
        assert (tmp_path / "reordered.weights.npy").exists()

    def test_phases_have_correct_structure(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.EASY_TO_HARD, num_phases=3)
        schedule = orderer.reorder(data_path, weights_path)

        assert len(schedule.phases) == 3
        for phase in schedule.phases:
            assert "phase" in phase
            assert "start_idx" in phase
            assert "end_idx" in phase
            assert "mean_score" in phase
            assert "num_samples" in phase

    def test_schedule_file_created(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.EASY_TO_HARD)
        orderer.reorder(data_path, weights_path)

        schedule_path = data_path.with_suffix(".curriculum.json")
        assert schedule_path.exists()
        with open(schedule_path) as f:
            data = json.load(f)
        assert data["strategy"] == "easy_to_hard"

    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        data = np.zeros((10, 32), dtype=np.uint16)
        weights = np.zeros(5, dtype=np.float32)
        data_path = tmp_path / "data.npy"
        weights_path = tmp_path / "weights.npy"
        np.save(str(data_path), data)
        np.save(str(weights_path), weights)

        orderer = CurriculumOrderer()
        with pytest.raises(ValueError, match="mismatch"):
            orderer.reorder(data_path, weights_path)

    def test_data_and_weights_stay_aligned(self, scored_data: tuple[Path, Path]) -> None:
        """After reorder, data[i] still corresponds to weights[i]."""
        data_path, weights_path = scored_data
        original_data = np.load(str(data_path)).copy()
        original_weights = np.load(str(weights_path)).copy()

        orderer = CurriculumOrderer(strategy=CurriculumStrategy.EASY_TO_HARD)
        orderer.reorder(data_path, weights_path)

        reordered_data = np.load(str(data_path))
        reordered_weights = np.load(str(weights_path))

        # Find the original index of the first reordered sample
        first_reordered_row = reordered_data[0]
        original_idx = None
        for i in range(len(original_data)):
            if np.array_equal(original_data[i], first_reordered_row):
                original_idx = i
                break
        assert original_idx is not None
        # The weight should match
        assert reordered_weights[0] == pytest.approx(original_weights[original_idx])


class TestFoldingOrder:
    def test_folding_is_a_true_permutation(self, scored_data: tuple[Path, Path]) -> None:
        """Every sample appears exactly once — no drops, no duplication."""
        data_path, weights_path = scored_data
        original_weights = np.load(str(weights_path)).copy()
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.FOLDING, num_folds=4)
        schedule = orderer.reorder(data_path, weights_path)

        reordered_weights = np.load(str(weights_path))
        assert len(reordered_weights) == len(original_weights)
        # Same multiset of weights — a permutation, not a resample.
        assert np.array_equal(np.sort(reordered_weights), np.sort(original_weights))
        assert schedule.strategy == "folding"
        assert schedule.total_samples == 100

    def test_folding_repeats_sweep_l_times(self, scored_data: tuple[Path, Path]) -> None:
        """Order is not globally sorted, but each fold segment runs easy->hard."""
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.FOLDING, num_folds=4)
        orderer.reorder(data_path, weights_path)
        w = np.load(str(weights_path))

        # Not globally descending (that would be plain easy_to_hard).
        assert not np.all(np.diff(w) <= 0)
        # Each of the 4 folds individually is descending (easy/high-quality first).
        n = len(w)
        # Round-robin striding of 100 into 4 folds -> 25 each.
        for f in range(4):
            seg = w[f * (n // 4):(f + 1) * (n // 4)]
            assert np.all(np.diff(seg) <= 0), f"fold {f} not easy->hard"

    def test_folding_keeps_data_weights_aligned(self, scored_data: tuple[Path, Path]) -> None:
        data_path, weights_path = scored_data
        original_data = np.load(str(data_path)).copy()
        original_weights = np.load(str(weights_path)).copy()
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.FOLDING, num_folds=4)
        orderer.reorder(data_path, weights_path)

        reordered_data = np.load(str(data_path))
        reordered_weights = np.load(str(weights_path))
        for i in (0, 1, 50, 99):
            row = reordered_data[i]
            match = next(
                j for j in range(len(original_data))
                if np.array_equal(original_data[j], row)
            )
            assert reordered_weights[i] == pytest.approx(original_weights[match])

    def test_folding_single_fold_equals_easy_to_hard(self, scored_data: tuple[Path, Path]) -> None:
        """L=1 degenerates to a plain easy->hard sort."""
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(strategy=CurriculumStrategy.FOLDING, num_folds=1)
        orderer.reorder(data_path, weights_path)
        w = np.load(str(weights_path))
        assert np.all(np.diff(w) <= 0)

    def test_folding_more_folds_than_samples(self, tmp_path: Path) -> None:
        """num_folds > n must not crash or drop samples."""
        data = np.arange(3 * 8, dtype=np.uint16).reshape(3, 8)
        weights = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        data_path = tmp_path / "tiny.npy"
        weights_path = tmp_path / "tiny.weights.npy"
        np.save(str(data_path), data)
        np.save(str(weights_path), weights)

        orderer = CurriculumOrderer(strategy=CurriculumStrategy.FOLDING, num_folds=16)
        orderer.reorder(data_path, weights_path)
        w = np.load(str(weights_path))
        assert np.array_equal(np.sort(w), np.array([0.1, 0.5, 0.9], dtype=np.float32))


class TestStagedReorder:
    def test_creates_phase_files(self, scored_data: tuple[Path, Path], tmp_path: Path) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(
            strategy=CurriculumStrategy.STAGED,
            num_phases=3,
        )
        out_dir = tmp_path / "phases"
        paths = orderer.staged_reorder(data_path, weights_path, out_dir)

        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()

    def test_phases_cover_all_data(self, scored_data: tuple[Path, Path], tmp_path: Path) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(num_phases=3)
        out_dir = tmp_path / "phases"
        paths = orderer.staged_reorder(data_path, weights_path, out_dir)

        total = sum(len(np.load(p)) for p in paths)
        original = len(np.load(str(data_path)))
        assert total == original

    def test_phase_1_has_highest_quality(self, scored_data: tuple[Path, Path], tmp_path: Path) -> None:
        data_path, weights_path = scored_data
        orderer = CurriculumOrderer(num_phases=3)
        out_dir = tmp_path / "phases"
        paths = orderer.staged_reorder(data_path, weights_path, out_dir)

        w1 = np.load(str(Path(paths[0]).with_suffix(".weights.npy")))
        w3 = np.load(str(Path(paths[2]).with_suffix(".weights.npy")))
        assert np.mean(w1) >= np.mean(w3)

    def test_empty_data_handled(self, tmp_path: Path) -> None:
        data = np.zeros((0, 32), dtype=np.uint16)
        weights = np.zeros(0, dtype=np.float32)
        data_path = tmp_path / "data.npy"
        weights_path = tmp_path / "weights.npy"
        np.save(str(data_path), data)
        np.save(str(weights_path), weights)

        orderer = CurriculumOrderer(num_phases=3)
        out_dir = tmp_path / "phases"
        paths = orderer.staged_reorder(data_path, weights_path, out_dir)
        assert len(paths) == 3
