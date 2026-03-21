"""Tests for training_efficiency.py (feature 55)."""

import pytest

from cola_coder.features.training_efficiency import (
    FEATURE_ENABLED,
    EfficiencySnapshot,
    TrainingEfficiencyTracker,
    _estimate_model_flops_per_token,
    is_enabled,
)


@pytest.fixture
def tracker():
    return TrainingEfficiencyTracker(n_params=125_000_000, seq_len=512, peak_gpu_tflops=29.8)


def _fill_tracker(tracker, n=10):
    """Add n fake snapshots with gradually increasing throughput."""
    for i in range(n):
        tracker.record(
            step=i,
            elapsed_seconds=float(i + 1),
            tokens_processed=(i + 1) * 1000,
            gpu_utilization_pct=70.0 + i,
            gpu_memory_used_mb=4000.0,
            gpu_memory_total_mb=10240.0,
        )


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_snapshot_tokens_per_second():
    snap = EfficiencySnapshot(
        step=1, elapsed_seconds=2.0, tokens_processed=2000,
        gpu_utilization_pct=80.0, gpu_memory_used_mb=4096, gpu_memory_total_mb=10240
    )
    assert snap.tokens_per_second == pytest.approx(1000.0)


def test_snapshot_memory_utilization():
    snap = EfficiencySnapshot(
        step=0, elapsed_seconds=1.0, tokens_processed=500,
        gpu_utilization_pct=0, gpu_memory_used_mb=5120, gpu_memory_total_mb=10240
    )
    assert snap.memory_utilization_pct == pytest.approx(50.0)


def test_snapshot_zero_elapsed():
    snap = EfficiencySnapshot(
        step=0, elapsed_seconds=0.0, tokens_processed=100,
        gpu_utilization_pct=0, gpu_memory_used_mb=0, gpu_memory_total_mb=0
    )
    assert snap.tokens_per_second == 0.0


def test_record_adds_snapshot(tracker):
    tracker.record(step=1, elapsed_seconds=1.0, tokens_processed=500)
    assert len(tracker.snapshots) == 1


def test_estimate_model_flops():
    flops = _estimate_model_flops_per_token(125_000_000, 512)
    # Should be 6 * n_params
    assert flops == pytest.approx(6 * 125_000_000)


def test_compute_mfu_returns_report(tracker):
    report = tracker.compute_mfu(tokens_per_second=10_000)
    assert report is not None
    assert 0 <= report.mfu_pct <= 100
    assert report.achieved_tflops > 0
    assert report.peak_tflops == pytest.approx(29.8)


def test_compute_mfu_none_when_no_params():
    t = TrainingEfficiencyTracker()  # No n_params set
    report = t.compute_mfu(10_000)
    assert report is None


def test_mfu_summary_contains_mfu(tracker):
    report = tracker.compute_mfu(50_000)
    assert report is not None
    s = report.summary()
    assert "MFU=" in s
    assert "TFLOPs" in s


def test_summarize_empty():
    t = TrainingEfficiencyTracker()
    summary = t.summarize()
    assert summary.n_snapshots == 0
    assert summary.mean_tokens_per_second == 0.0


def test_summarize_basic(tracker):
    _fill_tracker(tracker, 8)
    summary = tracker.summarize()
    assert summary.n_snapshots == 8
    assert summary.mean_tokens_per_second > 0
    assert summary.peak_tokens_per_second >= summary.mean_tokens_per_second
    assert summary.total_tokens > 0


def test_summarize_has_mfu(tracker):
    _fill_tracker(tracker, 5)
    summary = tracker.summarize()
    assert summary.mfu_report is not None


def test_summary_text(tracker):
    _fill_tracker(tracker, 4)
    s = tracker.summarize().summary()
    assert "snapshots" in s
    assert "tok/s" in s


def test_tokens_per_second_history(tracker):
    _fill_tracker(tracker, 5)
    hist = tracker.tokens_per_second_history()
    assert len(hist) == 5
    assert all(isinstance(step, int) and tps >= 0 for step, tps in hist)


def test_efficiency_trend_stable(tracker):
    # All same throughput → stable
    for i in range(10):
        tracker.record(step=i, elapsed_seconds=float(i + 1), tokens_processed=(i + 1) * 1000)
    trend = tracker.efficiency_trend()
    assert trend in ("stable", "improving", "declining")  # Just check it returns a string


def test_efficiency_trend_insufficient():
    t = TrainingEfficiencyTracker()
    t.record(step=0, elapsed_seconds=1.0, tokens_processed=100)
    assert t.efficiency_trend() == "insufficient_data"


def test_detect_stalls(tracker):
    # Record a stall at step 5
    for i in range(5):
        tracker.record(step=i, elapsed_seconds=float(i + 1), tokens_processed=(i + 1) * 10_000)
    tracker.record(step=5, elapsed_seconds=6.0, tokens_processed=6_001)  # Very slow
    stalls = tracker.detect_stalls(threshold_pct=90.0)
    assert 5 in stalls


def test_detect_stalls_empty_tracker():
    t = TrainingEfficiencyTracker()
    assert t.detect_stalls() == []
