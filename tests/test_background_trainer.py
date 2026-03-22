"""Tests for background training system.

Tests the throttle controller, training session management, lock files,
stop conditions, and duration parsing — without running actual training.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

from cola_coder.features.background_trainer import (
    BackgroundTrainingConfig,
    GPUThrottleController,
    TrainingSession,
    is_background_running,
    parse_duration,
    read_background_status,
    send_stop_signal,
)


# ── Duration Parsing ──────────────────────────────────────────────────────


class TestParseDuration:
    def test_hours(self):
        assert parse_duration("8h") == 8 * 3600

    def test_minutes(self):
        assert parse_duration("30m") == 30 * 60

    def test_hours_and_minutes(self):
        assert parse_duration("2h30m") == 2 * 3600 + 30 * 60

    def test_seconds(self):
        assert parse_duration("90s") == 90

    def test_bare_number_treated_as_hours(self):
        assert parse_duration("4") == 4 * 3600

    def test_empty_returns_none(self):
        assert parse_duration("") is None

    def test_none_input(self):
        assert parse_duration(None) is None  # type: ignore[arg-type]


# ── GPU Throttle Controller ───────────────────────────────────────────────


class TestGPUThrottleController:
    def test_compute_sleep_idle(self):
        """When GPU is idle, no sleep needed."""
        ctrl = GPUThrottleController(max_sleep_ms=200, busy_threshold=40.0)
        ctrl._other_gpu_util = 5.0
        assert ctrl.compute_sleep_ms() == 0

    def test_compute_sleep_light_load(self):
        """Between 15% and busy threshold: proportional sleep."""
        ctrl = GPUThrottleController(max_sleep_ms=200, busy_threshold=40.0)
        ctrl._other_gpu_util = 27.5  # halfway between 15 and 40
        sleep = ctrl.compute_sleep_ms()
        assert 90 <= sleep <= 110  # ~100ms (half of 200)

    def test_compute_sleep_busy(self):
        """Above busy threshold: max sleep."""
        ctrl = GPUThrottleController(max_sleep_ms=200, busy_threshold=40.0)
        ctrl._other_gpu_util = 60.0
        assert ctrl.compute_sleep_ms() == 200

    def test_compute_sleep_saturated(self):
        """Above 90%: long pause."""
        ctrl = GPUThrottleController(max_sleep_ms=200, busy_threshold=40.0)
        ctrl._other_gpu_util = 95.0
        assert ctrl.compute_sleep_ms() == 30_000

    def test_should_pause(self):
        """should_pause when GPU >90%."""
        ctrl = GPUThrottleController()
        ctrl._other_gpu_util = 95.0
        assert ctrl.should_pause() is True
        ctrl._other_gpu_util = 50.0
        assert ctrl.should_pause() is False

    @patch("cola_coder.features.background_trainer.GPUThrottleController._run_nvidia_smi")
    def test_apply_limits_success(self, mock_smi):
        """When nvidia-smi succeeds, clocks and power are applied."""
        mock_smi.return_value = 0
        ctrl = GPUThrottleController(clock_mhz=1500, power_watts=200)
        results = ctrl.apply_limits()
        assert results["clock"] is True
        assert results["power"] is True
        assert ctrl._clocks_applied is True
        assert ctrl._power_applied is True

    @patch("cola_coder.features.background_trainer.GPUThrottleController._run_nvidia_smi")
    def test_apply_limits_failure(self, mock_smi):
        """When nvidia-smi fails (no admin), graceful fallback."""
        mock_smi.return_value = 1
        ctrl = GPUThrottleController(clock_mhz=1500, power_watts=200)
        results = ctrl.apply_limits()
        assert results["clock"] is False
        assert results["power"] is False
        assert ctrl._clocks_applied is False
        assert ctrl._power_applied is False

    @patch("cola_coder.features.background_trainer.GPUThrottleController._run_nvidia_smi")
    def test_restore_only_if_applied(self, mock_smi):
        """restore_defaults should only run if we actually applied limits."""
        mock_smi.return_value = 0
        ctrl = GPUThrottleController()
        # Nothing applied yet
        ctrl.restore_defaults()
        mock_smi.assert_not_called()

        # Now apply and restore
        ctrl._clocks_applied = True
        ctrl._power_applied = True
        ctrl.restore_defaults()
        assert mock_smi.call_count == 2


# ── Lock File ─────────────────────────────────────────────────────────────


class TestLockFile:
    def test_acquire_and_release(self, tmp_path):
        """Basic lock acquire/release cycle."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)

        assert session.acquire_lock() is True
        assert (tmp_path / ".background_train.lock").exists()

        # Read lock data
        data = json.loads((tmp_path / ".background_train.lock").read_text())
        assert data["pid"] == os.getpid()
        assert data["config_path"] == "configs/tiny.yaml"

        session.release_lock()
        assert not (tmp_path / ".background_train.lock").exists()

    def test_double_acquire_fails(self, tmp_path):
        """Cannot acquire lock twice (same PID)."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session1 = TrainingSession(config)
        session2 = TrainingSession(config)

        assert session1.acquire_lock() is True
        # Second acquire should fail (our PID is still alive)
        assert session2.acquire_lock() is False

        session1.release_lock()

    def test_stale_lock_recovery(self, tmp_path):
        """Stale lock from dead PID should be cleaned up."""
        lock_path = tmp_path / ".background_train.lock"
        lock_path.write_text(json.dumps({
            "pid": 99999999,  # Almost certainly not running
            "start_time": "2020-01-01T00:00:00",
            "config_path": "configs/old.yaml",
        }))

        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            lock_file=str(lock_path),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)

        # Should recover stale lock and acquire
        assert session.acquire_lock() is True
        session.release_lock()


# ── Status File ───────────────────────────────────────────────────────────


class TestStatusFile:
    def test_write_and_read_status(self, tmp_path):
        """Status file write/read cycle."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)
        session._start_time = time.time()

        session.write_status(step=1000, loss=2.345, tokens_per_sec=5000.0)

        status = read_background_status(str(tmp_path))
        assert status is not None
        assert status["step"] == 1000
        assert status["loss"] == 2.345
        assert status["tokens_per_sec"] == 5000.0

    def test_read_nonexistent_returns_none(self, tmp_path):
        assert read_background_status(str(tmp_path)) is None


# ── Stop Conditions ───────────────────────────────────────────────────────


class TestStopConditions:
    def test_stop_signal_file(self, tmp_path):
        """Stop when signal file exists."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)
        session._start_time = time.time()

        assert session.should_stop() is False

        # Create stop signal
        (tmp_path / ".background_stop").write_text("stop")
        assert session.should_stop() is True
        # Signal file should be cleaned up
        assert not (tmp_path / ".background_stop").exists()

    def test_duration_limit(self, tmp_path):
        """Stop when duration exceeded."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            max_duration_seconds=1,  # 1 second
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)
        session._start_time = time.time() - 2  # Started 2 seconds ago

        assert session.should_stop() is True

    def test_no_duration_no_stop(self, tmp_path):
        """No duration limit means don't stop."""
        config = BackgroundTrainingConfig(
            config_path="configs/tiny.yaml",
            max_duration_seconds=None,
            lock_file=str(tmp_path / ".background_train.lock"),
            status_file=str(tmp_path / ".background_status.json"),
        )
        session = TrainingSession(config)
        session._start_time = time.time()

        assert session.should_stop() is False


# ── Helper Functions ──────────────────────────────────────────────────────


class TestHelpers:
    def test_send_stop_signal(self, tmp_path):
        """send_stop_signal creates the stop file."""
        assert send_stop_signal(str(tmp_path)) is True
        assert (tmp_path / ".background_stop").exists()

    def test_is_background_running_no_lock(self, tmp_path):
        """No lock file means not running."""
        running, status = is_background_running(str(tmp_path))
        assert running is False
        assert status is None

    def test_is_background_running_stale(self, tmp_path):
        """Stale lock (dead PID) means not running."""
        (tmp_path / ".background_train.lock").write_text(json.dumps({
            "pid": 99999999,
            "start_time": "2020-01-01T00:00:00",
        }))
        running, status = is_background_running(str(tmp_path))
        assert running is False


# ── Step Callback ─────────────────────────────────────────────────────────


class TestStepCallback:
    def test_trainer_has_callback_attribute(self):
        """Trainer should have _step_callback attribute."""
        # Verify the attribute exists without needing CUDA
        from cola_coder.training.trainer import Trainer
        assert hasattr(Trainer, "__init__")
        # We can't instantiate without a full config, but we can check the code
        import inspect
        source = inspect.getsource(Trainer.__init__)
        assert "_step_callback" in source

    def test_callback_in_training_loop(self):
        """Verify the callback call exists in the train method."""
        from cola_coder.training.trainer import Trainer
        import inspect
        source = inspect.getsource(Trainer.train)
        assert "_step_callback" in source
        assert "self._step_callback(step, avg_loss)" in source
