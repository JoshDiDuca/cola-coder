"""Tests for checkpoint_scheduler.py."""

from __future__ import annotations


from cola_coder.features.checkpoint_scheduler import (
    FEATURE_ENABLED,
    CheckpointRecord,
    CheckpointScheduler,
    compute_checkpoint_density,
    estimate_checkpoints_needed,
    exponential_decay_interval,
    is_enabled,
    linear_decay_interval,
    loss_adaptive_interval,
)


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestExponentialDecayInterval:
    def test_initial_interval_at_zero(self):
        result = exponential_decay_interval(0, initial_interval=100, min_interval=50, decay_rate=1.5)
        assert result == 100

    def test_grows_over_time(self):
        early = exponential_decay_interval(0, 100, 50, 1.5)
        late = exponential_decay_interval(1000, 100, 50, 1.5)
        assert late >= early

    def test_min_interval_respected(self):
        # Even at step 0, should not go below min
        result = exponential_decay_interval(0, 100, 200, 1.5)  # min > initial
        assert result >= 200

    def test_warmup_keeps_initial(self):
        result = exponential_decay_interval(50, 100, 50, 2.0, warmup_steps=100)
        assert result == 100


class TestLinearDecayInterval:
    def test_start_equals_initial(self):
        result = linear_decay_interval(0, 100, 500, 1000)
        assert result == 100

    def test_end_equals_final(self):
        result = linear_decay_interval(1000, 100, 500, 1000)
        assert result == 500

    def test_monotone_increasing(self):
        vals = [linear_decay_interval(s, 100, 500, 1000) for s in range(0, 1001, 100)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a


class TestLossAdaptiveInterval:
    def test_stable_loss_gives_longer_interval(self):
        stable = [2.0, 2.0, 2.0, 2.0, 2.0]
        volatile = [3.0, 1.0, 4.0, 1.0, 3.0]
        stable_int = loss_adaptive_interval(stable, base_interval=100)
        volatile_int = loss_adaptive_interval(volatile, base_interval=100)
        assert stable_int >= volatile_int

    def test_single_point_returns_base(self):
        result = loss_adaptive_interval([2.5], base_interval=100)
        assert result == 100


class TestCheckpointScheduler:
    def test_should_checkpoint_at_interval(self):
        sched = CheckpointScheduler(initial_interval=100, min_interval=100, decay_rate=1.0, save_best=False)
        # Step 0 is already "last checkpoint", so step 100 should trigger
        assert sched.should_checkpoint(100, loss=2.0)

    def test_should_not_checkpoint_before_interval(self):
        sched = CheckpointScheduler(initial_interval=100, min_interval=100, decay_rate=1.0, save_best=False)
        assert not sched.should_checkpoint(50, loss=2.0)

    def test_save_best_triggers_checkpoint(self):
        sched = CheckpointScheduler(initial_interval=1000, save_best=True)
        sched.state.best_loss = 3.0
        assert sched.should_checkpoint(50, loss=1.0)  # new best

    def test_add_override_triggers_checkpoint(self):
        sched = CheckpointScheduler(initial_interval=1000, save_best=False)
        sched.add_override(42)
        assert sched.should_checkpoint(42, loss=2.5)

    def test_record_checkpoint_updates_state(self):
        sched = CheckpointScheduler(initial_interval=100)
        record = sched.record_checkpoint(100, 2.5)
        assert isinstance(record, CheckpointRecord)
        assert sched.state.last_checkpoint_step == 100

    def test_preview_schedule_nonempty(self):
        sched = CheckpointScheduler(initial_interval=100, total_steps=1000, decay_rate=1.0)
        schedule = sched.preview_schedule(1000)
        assert len(schedule) > 0

    def test_reset_clears_state(self):
        sched = CheckpointScheduler(initial_interval=100)
        sched.record_checkpoint(100, 2.0)
        sched.reset()
        assert sched.state.last_checkpoint_step == 0
        assert len(sched.state.checkpoints) == 0

    def test_step_returns_record_when_saving(self):
        sched = CheckpointScheduler(initial_interval=100, min_interval=100, decay_rate=1.0, save_best=False)
        record = sched.step(100, 2.0)
        assert record is not None
        assert record.step == 100

    def test_summary_contains_info(self):
        sched = CheckpointScheduler(initial_interval=100)
        s = sched.summary()
        assert "CheckpointScheduler" in s


class TestUtilities:
    def test_compute_checkpoint_density_length(self):
        schedule = list(range(100, 1001, 100))
        density = compute_checkpoint_density(schedule, 1000)
        assert len(density) == 10

    def test_estimate_checkpoints_positive(self):
        count = estimate_checkpoints_needed(10000, 100, 1.5)
        assert count > 0

    def test_max_interval_caps_growth(self):
        sched = CheckpointScheduler(
            initial_interval=100, max_interval=200, decay_rate=100.0
        )
        # Even with huge decay rate, interval should be capped
        interval = sched._current_interval(100000)
        assert interval <= 200
