"""Tests for EarlyStopping.

Tests the patience counter, min/max modes, state dict save/load,
best model saving, and integration behaviour.
"""

import json
import tempfile
from pathlib import Path

import torch.nn as nn
import pytest

from cola_coder.training.early_stopping import EarlyStopping


# ── Simple model fixture ──────────────────────────────────────────────────────

class TinyModel(nn.Module):
    """Minimal model for testing save_best functionality."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPatienceCounter:
    """Test that the patience counter correctly counts non-improvements."""

    def test_no_stop_with_improving_metrics(self):
        """Counter stays at 0 and should_stop stays False while improving."""
        stopper = EarlyStopping(patience=3, min_delta=0.0, mode="min", save_best=False)
        for loss in [5.0, 4.0, 3.0, 2.0, 1.0]:
            stopped = stopper.step(loss)
            assert not stopped
        assert stopper.counter == 0
        assert not stopper.should_stop

    def test_stop_after_patience_exceeded(self):
        """should_stop becomes True exactly after patience non-improvements."""
        stopper = EarlyStopping(patience=3, min_delta=0.0, mode="min", save_best=False)
        stopper.step(2.0)  # Initial best
        stopper.step(2.5)  # counter = 1
        assert not stopper.should_stop
        stopper.step(2.5)  # counter = 2
        assert not stopper.should_stop
        stopped = stopper.step(2.5)  # counter = 3 — triggers stop
        assert stopped
        assert stopper.should_stop

    def test_counter_resets_on_improvement(self):
        """Counter resets to 0 when a new best is found."""
        stopper = EarlyStopping(patience=3, min_delta=0.0, mode="min", save_best=False)
        stopper.step(2.0)  # best = 2.0
        stopper.step(2.5)  # counter = 1
        stopper.step(2.5)  # counter = 2
        stopper.step(1.9)  # new best — counter resets
        assert stopper.counter == 0
        assert stopper.best_score == pytest.approx(1.9)

    def test_patience_of_one(self):
        """With patience=1, stops after a single non-improvement."""
        stopper = EarlyStopping(patience=1, min_delta=0.0, mode="min", save_best=False)
        stopper.step(3.0)  # initial
        stopped = stopper.step(3.5)  # counter = 1 → stop
        assert stopped

    def test_exact_patience_boundary(self):
        """stop triggers on the Nth non-improvement, not before."""
        patience = 5
        stopper = EarlyStopping(patience=patience, mode="min", save_best=False)
        stopper.step(1.0)  # sets best

        for i in range(patience - 1):
            result = stopper.step(2.0)
            assert not result, f"Stopped too early at non-improvement {i + 1}"

        result = stopper.step(2.0)  # The Nth non-improvement
        assert result, "Should have stopped on the Nth non-improvement"


class TestModeMinMax:
    """Test that min and max modes work correctly."""

    def test_min_mode_monitors_decreasing_metric(self):
        """Min mode: lower is better (e.g., validation loss)."""
        stopper = EarlyStopping(patience=2, mode="min", min_delta=0.0, save_best=False)
        stopper.step(10.0)
        stopper.step(8.0)   # improvement
        stopper.step(7.0)   # improvement
        assert stopper.counter == 0

        stopper.step(7.5)   # regression → counter = 1
        assert stopper.counter == 1

    def test_max_mode_monitors_increasing_metric(self):
        """Max mode: higher is better (e.g., accuracy)."""
        stopper = EarlyStopping(patience=3, mode="max", min_delta=0.0, save_best=False)
        stopper.step(0.5)
        stopper.step(0.6)   # improvement
        stopper.step(0.7)   # improvement
        assert stopper.counter == 0

        stopper.step(0.65)  # regression → counter = 1
        assert stopper.counter == 1

    def test_max_mode_stops_correctly(self):
        """Max mode: stops when accuracy stops increasing."""
        stopper = EarlyStopping(patience=2, mode="max", min_delta=0.0, save_best=False)
        stopper.step(0.7)   # best = 0.7
        stopper.step(0.65)  # counter = 1
        stopped = stopper.step(0.65)  # counter = 2 → stop
        assert stopped

    def test_invalid_mode_raises(self):
        """Passing an invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(patience=3, mode="banana", save_best=False)


class TestMinDelta:
    """Test that min_delta creates a dead zone for tiny improvements."""

    def test_improvement_below_min_delta_not_counted(self):
        """An improvement smaller than min_delta doesn't count as improvement."""
        stopper = EarlyStopping(patience=3, min_delta=0.1, mode="min", save_best=False)
        stopper.step(2.0)   # best = 2.0
        # 1.95 improves by 0.05, which is < min_delta=0.1 → NOT an improvement
        stopper.step(1.95)  # counter = 1
        assert stopper.counter == 1

    def test_improvement_above_min_delta_counted(self):
        """An improvement larger than min_delta resets the counter."""
        stopper = EarlyStopping(patience=3, min_delta=0.1, mode="min", save_best=False)
        stopper.step(2.0)   # best = 2.0
        stopper.step(1.5)   # improves by 0.5 > 0.1 → improvement, counter = 0
        assert stopper.counter == 0
        assert stopper.best_score == pytest.approx(1.5)


class TestStateDictSaveLoad:
    """Test state_dict and load_state_dict for checkpoint resume."""

    def test_state_dict_contains_required_keys(self):
        """state_dict() must contain counter, best_score, should_stop."""
        stopper = EarlyStopping(patience=5, mode="min", save_best=False)
        stopper.step(3.0)
        stopper.step(3.5)
        state = stopper.state_dict()
        assert "counter" in state
        assert "best_score" in state
        assert "should_stop" in state
        assert "patience" in state
        assert "mode" in state

    def test_load_state_dict_restores_counter(self):
        """load_state_dict restores the counter and best_score."""
        stopper = EarlyStopping(patience=5, mode="min", save_best=False)
        stopper.step(2.0)   # best = 2.0
        stopper.step(3.0)   # counter = 1
        stopper.step(3.0)   # counter = 2

        state = stopper.state_dict()

        # Create a fresh stopper and restore state
        new_stopper = EarlyStopping(patience=5, mode="min", save_best=False)
        new_stopper.load_state_dict(state)

        assert new_stopper.counter == 2
        assert new_stopper.best_score == pytest.approx(2.0)
        assert not new_stopper.should_stop

    def test_load_state_dict_restores_should_stop(self):
        """load_state_dict restores should_stop=True state."""
        stopper = EarlyStopping(patience=2, mode="min", save_best=False)
        stopper.step(1.0)
        stopper.step(2.0)
        stopper.step(2.0)  # should_stop = True

        state = stopper.state_dict()
        assert state["should_stop"] is True

        new_stopper = EarlyStopping(patience=2, mode="min", save_best=False)
        new_stopper.load_state_dict(state)
        assert new_stopper.should_stop

    def test_round_trip_preserves_all_state(self):
        """Saving and loading state is a perfect round-trip."""
        stopper = EarlyStopping(
            patience=7, min_delta=0.005, mode="max",
            save_best=False, verbose=False
        )
        stopper.step(0.5)
        stopper.step(0.6)
        stopper.step(0.55)

        state = stopper.state_dict()
        new_stopper = EarlyStopping(patience=1, mode="min", save_best=False)
        new_stopper.load_state_dict(state)

        assert new_stopper.patience == stopper.patience
        assert new_stopper.min_delta == pytest.approx(stopper.min_delta)
        assert new_stopper.mode == stopper.mode
        assert new_stopper.counter == stopper.counter
        assert new_stopper.best_score == pytest.approx(stopper.best_score)
        assert new_stopper.num_calls == stopper.num_calls


class TestBestModelSaving:
    """Test that the best model is saved when save_best=True."""

    def test_saves_best_model_on_improvement(self):
        """Best model is saved to disk when metric improves."""
        model = TinyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = EarlyStopping(
                patience=5, mode="min", save_best=True,
                best_model_path=tmpdir, verbose=False
            )
            stopper.step(3.0, model=model, step=10)
            # Either safetensors or .pt file should exist
            files = list(Path(tmpdir).iterdir())
            assert len(files) > 0, "No files saved to best_model_path"

    def test_saves_metadata_alongside_model(self):
        """A metadata JSON is saved with the best model."""
        model = TinyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = EarlyStopping(
                patience=5, mode="min", save_best=True,
                best_model_path=tmpdir, verbose=False
            )
            stopper.step(2.5, model=model, step=100)
            meta_path = Path(tmpdir) / "best_metadata.json"
            assert meta_path.exists(), "best_metadata.json not created"
            meta = json.loads(meta_path.read_text())
            assert meta["step"] == 100
            assert meta["metric"] == pytest.approx(2.5)

    def test_no_save_when_no_improvement(self):
        """Model is NOT re-saved when the metric doesn't improve."""
        model = TinyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = EarlyStopping(
                patience=5, mode="min", save_best=True,
                best_model_path=tmpdir, verbose=False
            )
            stopper.step(2.0, model=model, step=10)  # saves
            # Record mtime
            best_file = next(
                (p for p in Path(tmpdir).iterdir() if "best_model" in p.name), None
            )
            assert best_file is not None
            mtime_after_first = best_file.stat().st_mtime

            stopper.step(3.0, model=model, step=20)  # regression, should NOT save
            mtime_after_second = best_file.stat().st_mtime
            assert mtime_after_second == mtime_after_first, (
                "Best model was overwritten on a non-improving step"
            )


class TestResetAndRepr:
    """Test reset() and __repr__."""

    def test_reset_clears_all_state(self):
        """reset() returns the stopper to its initial state."""
        stopper = EarlyStopping(patience=3, mode="min", save_best=False)
        stopper.step(1.0)
        stopper.step(2.0)
        stopper.step(2.0)
        stopper.reset()
        assert stopper.counter == 0
        assert stopper.best_score is None
        assert not stopper.should_stop
        assert stopper.num_calls == 0

    def test_repr_contains_key_info(self):
        """__repr__ includes patience, mode, and counter."""
        stopper = EarlyStopping(patience=5, mode="max", save_best=False)
        r = repr(stopper)
        assert "patience=5" in r
        assert "mode='max'" in r
        assert "counter=" in r

    def test_invalid_patience_raises(self):
        """patience < 1 raises ValueError."""
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0, save_best=False)
