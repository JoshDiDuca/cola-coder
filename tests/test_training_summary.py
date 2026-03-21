"""Tests for features/training_summary.py — Feature 100.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import math

import pytest

from cola_coder.features.training_summary import (
    FEATURE_ENABLED,
    CheckpointInfo,
    CurveStats,
    HardwareStats,
    TrainingSummary,
    TrainingSummaryGenerator,
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


@pytest.fixture
def gen():
    g = TrainingSummaryGenerator(run_name="test_run")
    g.set_duration(3600.0)  # 1 hour
    g.set_total_tokens(1_000_000)
    g.set_hardware(HardwareStats(gpu_name="RTX 4080", gpu_count=1))
    for i in range(10):
        g.record_metrics({"train_loss": 3.0 - i * 0.1, "val_loss": 3.1 - i * 0.1})
    g.add_checkpoint(CheckpointInfo(path="ckpt/step_0", step=0, val_loss=3.1))
    g.add_checkpoint(CheckpointInfo(path="ckpt/step_5", step=5, val_loss=2.6))
    g.add_checkpoint(CheckpointInfo(path="ckpt/step_9", step=9, val_loss=2.2))
    return g


# ---------------------------------------------------------------------------
# Builder API
# ---------------------------------------------------------------------------


def test_build_returns_summary(gen):
    summary = gen.build()
    assert isinstance(summary, TrainingSummary)


def test_run_name(gen):
    summary = gen.build()
    assert summary.run_name == "test_run"


def test_total_steps(gen):
    summary = gen.build()
    assert summary.total_steps == 10


def test_total_tokens(gen):
    summary = gen.build()
    assert summary.total_tokens == 1_000_000


def test_duration_hours(gen):
    summary = gen.build()
    assert summary.duration_hours == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Best checkpoint selection
# ---------------------------------------------------------------------------


def test_best_checkpoint_selected(gen):
    summary = gen.build()
    assert summary.best_checkpoint is not None
    assert summary.best_checkpoint.path == "ckpt/step_9"


def test_no_checkpoints_none(gen):
    g = TrainingSummaryGenerator("empty")
    g.set_duration(100.0)
    g.set_total_tokens(0)
    g.record_metric("train_loss", 2.0)
    summary = g.build()
    assert summary.best_checkpoint is None


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_best_loss(gen):
    summary = gen.build()
    assert summary.best_loss is not None
    assert summary.best_loss < 3.1


def test_best_perplexity(gen):
    summary = gen.build()
    ppl = summary.best_perplexity
    assert ppl is not None
    assert ppl == pytest.approx(math.exp(summary.best_loss))


def test_tokens_per_second(gen):
    summary = gen.build()
    tps = summary.tokens_per_second
    assert tps is not None
    assert tps == pytest.approx(1_000_000 / 3600.0)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def test_hardware_gpu_name(gen):
    summary = gen.build()
    assert summary.hardware.gpu_name == "RTX 4080"


def test_gpu_hours_computed(gen):
    summary = gen.build()
    # 1 hour × 1 GPU = 1 GPU-hour
    assert summary.hardware.total_gpu_hours == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CurveStats
# ---------------------------------------------------------------------------


def test_curve_min(gen):
    summary = gen.build()
    curve = summary.curves["train_loss"]
    assert curve.min_value == pytest.approx(min(3.0 - i * 0.1 for i in range(10)))


def test_curve_final(gen):
    summary = gen.build()
    curve = summary.curves["train_loss"]
    assert curve.final == pytest.approx(3.0 - 9 * 0.1)


def test_curve_trend_improving():
    c = CurveStats(name="val_loss", values=[3.0, 2.5, 2.0, 1.8, 1.6])
    assert c.trend == "improving"


def test_curve_trend_stable():
    c = CurveStats(name="val_loss", values=[2.0, 2.0, 2.0, 2.0])
    assert c.trend == "stable"


def test_curve_best_step_loss():
    c = CurveStats(name="val_loss", values=[3.0, 2.0, 1.5, 2.5])
    assert c.best_step == 2  # index of minimum


def test_curve_as_dict():
    c = CurveStats(name="train_loss", values=[2.0, 1.9])
    d = c.as_dict()
    assert d["name"] == "train_loss"
    assert "trend" in d


# ---------------------------------------------------------------------------
# as_dict / text_report
# ---------------------------------------------------------------------------


def test_as_dict_keys(gen):
    d = gen.build().as_dict()
    assert "run_name" in d
    assert "best_loss" in d
    assert "hardware" in d
    assert "curves" in d


def test_text_report_contains_run_name(gen):
    report = gen.build().text_report()
    assert "test_run" in report


def test_text_report_contains_loss(gen):
    report = gen.build().text_report()
    assert "Best loss" in report


# ---------------------------------------------------------------------------
# Config and notes
# ---------------------------------------------------------------------------


def test_config_stored(gen):
    gen.set_config({"lr": 1e-4, "batch_size": 16})
    summary = gen.build()
    assert summary.config["lr"] == pytest.approx(1e-4)


def test_notes_stored(gen):
    gen.add_note("Training completed normally")
    summary = gen.build()
    assert "Training completed normally" in summary.notes


# ---------------------------------------------------------------------------
# CheckpointInfo
# ---------------------------------------------------------------------------


def test_checkpoint_score_uses_val_loss():
    c = CheckpointInfo(path="ckpt/0", step=0, loss=3.0, val_loss=2.5)
    assert c.score == pytest.approx(2.5)


def test_checkpoint_score_fallback_to_loss():
    c = CheckpointInfo(path="ckpt/0", step=0, loss=3.0)
    assert c.score == pytest.approx(3.0)


def test_checkpoint_score_inf_when_no_loss():
    c = CheckpointInfo(path="ckpt/0", step=0)
    assert math.isinf(c.score)
