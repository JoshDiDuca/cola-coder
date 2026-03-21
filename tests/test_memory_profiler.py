"""Tests for MemoryProfiler (features/memory_profiler.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.memory_profiler import (
    FEATURE_ENABLED,
    LayerMemRecord,
    MemoryProfileReport,
    MemoryProfiler,
    is_enabled,
    profile_layers,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Manual record() + report()
# ---------------------------------------------------------------------------


class TestManualRecord:
    def test_record_returns_layer_record(self):
        p = MemoryProfiler(use_tracemalloc=False)
        rec = p.record("attn", 100.0, 150.0, duration_ms=5.0)
        assert isinstance(rec, LayerMemRecord)
        assert rec.name == "attn"

    def test_ram_delta(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("attn", 100.0, 150.0)
        r = p.report()
        assert r.records[0].ram_delta_mb == pytest.approx(50.0)

    def test_report_peak_ram(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("layer1", 100.0, 200.0, ram_peak_mb=200.0)
        p.record("layer2", 200.0, 180.0, ram_peak_mb=200.0)
        r = p.report()
        assert r.peak_ram_mb == pytest.approx(200.0)

    def test_bottleneck_layer(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("small", 100.0, 110.0, ram_peak_mb=110.0)
        p.record("big", 100.0, 500.0, ram_peak_mb=500.0)
        r = p.report()
        assert r.bottleneck_layer == "big"

    def test_total_duration(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("a", 0.0, 0.0, duration_ms=10.0)
        p.record("b", 0.0, 0.0, duration_ms=20.0)
        r = p.report()
        assert r.total_duration_ms == pytest.approx(30.0)

    def test_empty_report(self):
        p = MemoryProfiler(use_tracemalloc=False)
        r = p.report()
        assert r.num_samples if hasattr(r, "num_samples") else r.records == []

    def test_reset_clears_records(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("a", 0.0, 1.0)
        p.reset()
        r = p.report()
        assert r.records == []


# ---------------------------------------------------------------------------
# Context manager track()
# ---------------------------------------------------------------------------


class TestTrackContextManager:
    def test_track_records_layer(self):
        p = MemoryProfiler(use_tracemalloc=False)
        with p.track("embedding"):
            _ = [i**2 for i in range(1000)]  # some work
        r = p.report()
        assert len(r.records) == 1
        assert r.records[0].name == "embedding"

    def test_track_duration_positive(self):
        p = MemoryProfiler(use_tracemalloc=False)
        with p.track("work"):
            _ = sum(range(10000))
        r = p.report()
        assert r.records[0].duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_high_delta_generates_suggestion(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("ffn", 0.0, 600.0, ram_peak_mb=600.0)
        r = p.report()
        assert any("offloading" in s for s in r.suggestions)

    def test_no_suggestion_for_small_layers(self):
        p = MemoryProfiler(use_tracemalloc=False)
        p.record("tiny", 0.0, 10.0, ram_peak_mb=10.0)
        r = p.report()
        assert not any("offloading" in s for s in r.suggestions)


# ---------------------------------------------------------------------------
# profile_layers convenience
# ---------------------------------------------------------------------------


class TestProfileLayers:
    def test_profile_layers_returns_report(self):
        r = profile_layers(
            {
                "embed": {"before": 100.0, "after": 200.0, "duration_ms": 5.0},
                "attn": {"before": 200.0, "after": 250.0, "duration_ms": 10.0},
            }
        )
        assert isinstance(r, MemoryProfileReport)
        assert len(r.records) == 2

    def test_layer_summary(self):
        r = profile_layers({"a": {"before": 10.0, "after": 30.0}})
        summary = r.layer_summary()
        assert "a" in summary
        assert summary["a"] == pytest.approx(20.0)
