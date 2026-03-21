"""Tests for model_comparison_dashboard.py (feature 58)."""

import json
import pytest

from cola_coder.features.model_comparison_dashboard import (
    FEATURE_ENABLED,
    CheckpointMetrics,
    ModelComparisonDashboard,
    is_enabled,
)


@pytest.fixture
def dashboard():
    return ModelComparisonDashboard()


@pytest.fixture
def checkpoints():
    return [
        CheckpointMetrics(name="step_500", step=500, loss=2.5, perplexity=12.2, accuracy=0.45),
        CheckpointMetrics(name="step_1000", step=1000, loss=2.0, perplexity=7.4, accuracy=0.60),
        CheckpointMetrics(name="step_2000", step=2000, loss=1.5, perplexity=4.5, accuracy=0.72),
    ]


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_build_empty(dashboard):
    data = dashboard.build([])
    assert data.checkpoints == []
    assert data.metrics_table == []
    assert "No checkpoints" in data.summary


def test_build_returns_data(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    assert len(data.checkpoints) == 3
    assert len(data.metrics_table) == 3


def test_chart_data_labels(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    assert data.chart_data["labels"] == ["step_500", "step_1000", "step_2000"]


def test_chart_data_loss_series(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    series = data.chart_data["series"]
    assert "loss" in series
    assert series["loss"] == [2.5, 2.0, 1.5]


def test_best_per_metric_loss(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    # Lowest loss should be best
    assert data.best_per_metric.get("loss") == "step_2000"


def test_worst_per_metric_loss(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    assert data.worst_per_metric.get("loss") == "step_500"


def test_best_per_metric_accuracy(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    # Highest accuracy should be best
    assert data.best_per_metric.get("accuracy") == "step_2000"


def test_summary_contains_metrics(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    assert "loss" in data.summary
    assert "accuracy" in data.summary


def test_summary_delta_section(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    assert "→" in data.summary  # First → Last delta


def test_render_markdown_table(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    md = dashboard.render_markdown(data)
    assert "| name" in md or "name" in md
    assert "step_500" in md
    assert "|" in md


def test_render_html_contains_table(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    html = dashboard.render_html(data)
    assert "<html>" in html
    assert "step_500" in html
    assert "chart_data" in html.lower() or "Chart Data" in html


def test_render_html_valid_structure(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    html = dashboard.render_html(data)
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_render_json_parseable(dashboard, checkpoints):
    data = dashboard.build(checkpoints)
    j = dashboard.render_json(data)
    parsed = json.loads(j)
    assert "metrics_table" in parsed
    assert "chart_data" in parsed
    assert len(parsed["metrics_table"]) == 3


def test_extra_metrics(dashboard):
    cps = [
        CheckpointMetrics(name="a", loss=1.0, extra={"custom_metric": 0.9}),
        CheckpointMetrics(name="b", loss=0.8, extra={"custom_metric": 0.95}),
    ]
    data = dashboard.build(cps)
    series = data.chart_data["series"]
    assert "custom_metric" in series


def test_single_checkpoint(dashboard):
    cp = CheckpointMetrics(name="only", loss=1.5, accuracy=0.7)
    data = dashboard.build([cp])
    assert len(data.checkpoints) == 1
    assert data.best_per_metric.get("loss") == "only"


def test_checkpoint_to_dict():
    cp = CheckpointMetrics(name="x", step=100, loss=0.5, extra={"custom": 1.0})
    d = cp.to_dict()
    assert d["name"] == "x"
    assert d["step"] == 100
    assert d["loss"] == 0.5
    assert d["custom"] == 1.0
