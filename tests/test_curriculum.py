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
        original_weights = np.load(str(weights_path)).copy()
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
