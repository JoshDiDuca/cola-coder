"""Tests for ExperimentComparator (features/experiment_comparator.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.experiment_comparator import (
    FEATURE_ENABLED,
    ComparisonReport,
    ExperimentComparator,
    ExperimentRecord,
    MetricComparison,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_experiment(name: str, loss_values: dict[int, float]) -> ExperimentRecord:
    exp = ExperimentRecord(name=name)
    for step, val in loss_values.items():
        exp.add_metric("loss", step, val)
    return exp


BASELINE = make_experiment("baseline", {100: 3.0, 200: 2.5, 300: 2.2, 400: 2.0})
BETTER = make_experiment("better_lr", {100: 2.8, 200: 2.2, 300: 1.9, 400: 1.7})
WORSE = make_experiment("worse_lr", {100: 3.2, 200: 2.8, 300: 2.6, 400: 2.5})


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestExperimentRecord:
    def test_add_metric(self):
        exp = ExperimentRecord(name="test")
        exp.add_metric("loss", 100, 2.5)
        assert exp.metrics["loss"][100] == 2.5

    def test_get_value_at_exact_step(self):
        assert BASELINE.get_value_at("loss", 100) == 3.0

    def test_get_value_at_interpolated(self):
        val = BASELINE.get_value_at("loss", 150)
        assert val is not None
        # Should be between 3.0 and 2.5
        assert 2.5 <= val <= 3.0

    def test_get_value_at_extrapolation_start(self):
        val = BASELINE.get_value_at("loss", 50)
        # Before first step, returns first value
        assert val == 3.0

    def test_get_value_at_extrapolation_end(self):
        val = BASELINE.get_value_at("loss", 500)
        # After last step, returns last value
        assert val == 2.0

    def test_get_value_at_missing_metric(self):
        assert BASELINE.get_value_at("accuracy", 100) is None

    def test_best_value_lower_is_better(self):
        best = BASELINE.best_value("loss", lower_is_better=True)
        assert best == 2.0

    def test_final_value(self):
        final = BASELINE.final_value("loss")
        assert final == 2.0


class TestComparison:
    def test_compare_returns_report(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER, WORSE], "baseline", ["loss"])
        assert isinstance(report, ComparisonReport)

    def test_rankings_ordered_lower_is_better(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER, WORSE], "baseline", ["loss"])
        assert "loss" in report.rankings
        names = [name for name, _ in report.rankings["loss"]]
        assert names[0] == "better_lr"  # best (lowest) loss

    def test_worst_experiment_ranked_last(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER, WORSE], "baseline", ["loss"])
        names = [name for name, _ in report.rankings["loss"]]
        assert names[-1] == "worse_lr"

    def test_metric_comparisons_populated(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER], "baseline", ["loss"])
        assert "loss" in report.metric_comparisons
        assert len(report.metric_comparisons["loss"]) > 0


class TestMetricComparison:
    def test_absolute_delta(self):
        mc = MetricComparison(metric="loss", step=100, baseline_value=3.0, experiment_value=2.5)
        assert mc.absolute_delta == pytest.approx(-0.5)

    def test_relative_improvement_positive(self):
        mc = MetricComparison(metric="loss", step=100, baseline_value=3.0, experiment_value=2.5)
        # improvement = (3.0 - 2.5) / 3.0 ≈ 0.167
        assert mc.relative_improvement > 0

    def test_relative_improvement_negative_when_worse(self):
        mc = MetricComparison(metric="loss", step=100, baseline_value=2.5, experiment_value=3.0)
        assert mc.relative_improvement < 0

    def test_relative_improvement_zero_baseline(self):
        mc = MetricComparison(metric="loss", step=100, baseline_value=0.0, experiment_value=0.5)
        assert mc.relative_improvement == 0.0


class TestAlignMetrics:
    def test_align_returns_all_experiments(self):
        comparator = ExperimentComparator()
        aligned = comparator.align_metrics([BASELINE, BETTER], "loss", [100, 200, 300])
        assert "baseline" in aligned
        assert "better_lr" in aligned

    def test_aligned_values_length(self):
        comparator = ExperimentComparator()
        steps = [100, 200, 300]
        aligned = comparator.align_metrics([BASELINE], "loss", steps)
        assert len(aligned["baseline"]) == 3


class TestSummaryTable:
    def test_summary_table_is_string(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER, WORSE], "baseline", ["loss"])
        table = report.summary_table("loss")
        assert isinstance(table, str)
        assert "better_lr" in table
        assert "worse_lr" in table

    def test_summary_table_marks_baseline(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE, BETTER, WORSE], "baseline", ["loss"])
        table = report.summary_table("loss")
        assert "*" in table  # baseline is marked

    def test_summary_table_unknown_metric(self):
        comparator = ExperimentComparator()
        report = comparator.compare([BASELINE], "baseline", ["loss"])
        table = report.summary_table("accuracy")
        assert "No data" in table
