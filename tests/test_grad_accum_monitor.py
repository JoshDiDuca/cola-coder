"""Tests for GradAccumMonitor (features/grad_accum_monitor.py)."""

from __future__ import annotations

import math

import pytest

from cola_coder.features.grad_accum_monitor import (
    FEATURE_ENABLED,
    AccumStepReport,
    GradAccumMonitor,
    MicroBatchRecord,
    check_accumulation,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_feature_enabled_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_construction(self):
        m = GradAccumMonitor()
        assert m.accumulation_steps == 4
        assert m.drift_threshold == 0.15

    def test_custom_params(self):
        m = GradAccumMonitor(accumulation_steps=8, drift_threshold=0.2)
        assert m.accumulation_steps == 8
        assert m.drift_threshold == 0.2

    def test_invalid_accum_steps(self):
        with pytest.raises(ValueError):
            GradAccumMonitor(accumulation_steps=0)

    def test_invalid_drift_threshold(self):
        with pytest.raises(ValueError):
            GradAccumMonitor(drift_threshold=0.0)

    def test_invalid_drift_threshold_above_one(self):
        with pytest.raises(ValueError):
            GradAccumMonitor(drift_threshold=1.5)


# ---------------------------------------------------------------------------
# record_grad_norms_dict + finalize_step_from_norm
# ---------------------------------------------------------------------------


class TestRecordAndFinalize:
    def _make_monitor(self, steps=4) -> GradAccumMonitor:
        return GradAccumMonitor(accumulation_steps=steps, drift_threshold=0.5)

    def test_record_returns_micro_batch_record(self):
        m = self._make_monitor()
        rec = m.record_grad_norms_dict(step=0, micro_batch_idx=0, grad_norms={"w": 1.0})
        assert isinstance(rec, MicroBatchRecord)
        assert rec.step == 0
        assert rec.micro_batch_idx == 0
        assert rec.running_norm == pytest.approx(1.0)

    def test_finalize_returns_step_report(self):
        m = self._make_monitor()
        for i in range(4):
            m.record_grad_norms_dict(0, i, {"w": 1.0})
        report = m.finalize_step_from_norm(0, final_norm=2.0)
        assert isinstance(report, AccumStepReport)
        assert report.step == 0
        assert report.num_micro_batches == 4

    def test_records_cleared_after_finalize(self):
        m = self._make_monitor()
        m.record_grad_norms_dict(0, 0, {"w": 1.0})
        m.finalize_step_from_norm(0, 1.0)
        # After finalize, internal state reset
        assert len(m._current_records) == 0
        assert len(m._micro_batch_norms) == 0

    def test_no_drift_when_norms_consistent(self):
        m = GradAccumMonitor(accumulation_steps=4, drift_threshold=0.5)
        # micro norms all = 1.0 → expected final = sqrt(4)*1.0 = 2.0
        for i in range(4):
            m.record_grad_norms_dict(0, i, {"w": 1.0})
        report = m.finalize_step_from_norm(0, final_norm=2.0)
        assert not report.has_drift

    def test_drift_detected_when_final_norm_diverges(self):
        m = GradAccumMonitor(accumulation_steps=4, drift_threshold=0.1)
        for i in range(4):
            m.record_grad_norms_dict(0, i, {"w": 1.0})
        # expected ~2.0, but we pass 10.0 → huge drift
        report = m.finalize_step_from_norm(0, final_norm=10.0)
        assert report.has_drift
        assert report.drift_ratio > 0.1

    def test_nan_detection(self):
        m = self._make_monitor()
        m.record_grad_norms_dict(0, 0, {"w": 1.0})
        report = m.finalize_step_from_norm(0, final_norm=float("nan"))
        assert report.has_nan
        assert report.drift_detected

    def test_inf_detection(self):
        m = self._make_monitor()
        m.record_grad_norms_dict(0, 0, {"w": 1.0})
        report = m.finalize_step_from_norm(0, final_norm=float("inf"))
        assert report.has_inf
        assert report.drift_detected

    def test_history_accumulates(self):
        m = GradAccumMonitor(accumulation_steps=2, history_size=10)
        for step in range(3):
            for i in range(2):
                m.record_grad_norms_dict(step, i, {"w": 1.0})
            m.finalize_step_from_norm(step, 1.5)
        assert len(m.get_history()) == 3

    def test_history_capped_at_history_size(self):
        m = GradAccumMonitor(accumulation_steps=1, history_size=3)
        for step in range(10):
            m.record_grad_norms_dict(step, 0, {"w": 1.0})
            m.finalize_step_from_norm(step, 1.0)
        assert len(m.get_history()) == 3

    def test_summary_keys(self):
        m = GradAccumMonitor(accumulation_steps=2)
        for step in range(2):
            for i in range(2):
                m.record_grad_norms_dict(step, i, {"w": 1.0})
            m.finalize_step_from_norm(step, 1.5)
        s = m.summary()
        assert "steps_tracked" in s
        assert "drift_rate" in s
        assert "avg_grad_norm" in s

    def test_reset_clears_everything(self):
        m = GradAccumMonitor(accumulation_steps=2)
        m.record_grad_norms_dict(0, 0, {"w": 1.0})
        m.finalize_step_from_norm(0, 1.0)
        m.reset()
        assert len(m.get_history()) == 0
        assert m.drift_rate() == 0.0


# ---------------------------------------------------------------------------
# check_accumulation convenience function
# ---------------------------------------------------------------------------


class TestCheckAccumulation:
    def test_no_drift_expected(self):
        # micro norms = [1, 1, 1, 1], expected final ≈ 2.0
        has_drift, ratio = check_accumulation([1.0, 1.0, 1.0, 1.0], 2.0)
        assert not has_drift

    def test_drift_detected(self):
        has_drift, ratio = check_accumulation([1.0, 1.0, 1.0, 1.0], 100.0)
        assert has_drift
        assert ratio > 0.15

    def test_empty_micro_norms(self):
        has_drift, ratio = check_accumulation([], 0.0)
        # Should not raise
        assert isinstance(has_drift, bool)
        assert math.isfinite(ratio)
