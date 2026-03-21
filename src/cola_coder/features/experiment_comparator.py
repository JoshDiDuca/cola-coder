"""Experiment Comparator: compare training experiments across runs.

Provides:
  - Metric alignment by step: interpolate / sample metrics at common steps
  - Relative improvement computation vs a baseline experiment
  - Ranked experiment table (best to worst on a chosen metric)
  - Text comparison report

For a TS dev: like comparing several Lighthouse runs and getting a diff table
showing which configuration improved which metrics by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """A single training experiment with step-indexed metrics."""

    name: str
    # {metric_name: {step: value}}
    metrics: dict[str, dict[int, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: str, step: int, value: float) -> None:
        if metric not in self.metrics:
            self.metrics[metric] = {}
        self.metrics[metric][step] = value

    def get_value_at(self, metric: str, step: int) -> float | None:
        """Return the value for *metric* at *step*, interpolating if needed."""
        if metric not in self.metrics:
            return None
        series = self.metrics[metric]
        if step in series:
            return series[step]
        # Linear interpolation between nearest neighbours
        steps = sorted(series.keys())
        if not steps:
            return None
        if step < steps[0]:
            return series[steps[0]]
        if step > steps[-1]:
            return series[steps[-1]]
        # find bracket
        lo = max(s for s in steps if s <= step)
        hi = min(s for s in steps if s >= step)
        if lo == hi:
            return series[lo]
        t = (step - lo) / (hi - lo)
        return series[lo] + t * (series[hi] - series[lo])

    def best_value(self, metric: str, lower_is_better: bool = True) -> float | None:
        """Return the best (min or max) value seen for *metric*."""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        values = list(self.metrics[metric].values())
        return min(values) if lower_is_better else max(values)

    def final_value(self, metric: str) -> float | None:
        """Return the value at the last recorded step for *metric*."""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        last_step = max(self.metrics[metric].keys())
        return self.metrics[metric][last_step]


@dataclass
class MetricComparison:
    """Comparison of one metric between an experiment and a baseline."""

    metric: str
    step: int
    baseline_value: float
    experiment_value: float

    @property
    def absolute_delta(self) -> float:
        return self.experiment_value - self.baseline_value

    @property
    def relative_improvement(self) -> float:
        """Positive means experiment improved vs baseline (lower-is-better assumed)."""
        if self.baseline_value == 0:
            return 0.0
        return (self.baseline_value - self.experiment_value) / abs(self.baseline_value)


@dataclass
class ComparisonReport:
    """Full comparison of multiple experiments."""

    baseline_name: str
    experiment_names: list[str] = field(default_factory=list)
    metric_comparisons: dict[str, list[MetricComparison]] = field(default_factory=dict)
    # Experiment rankings per metric: {metric: [(name, best_value), ...]}
    rankings: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def summary_table(self, metric: str) -> str:
        """Return a simple ASCII table for *metric* rankings."""
        if metric not in self.rankings:
            return f"No data for metric '{metric}'"
        lines = [f"{'Rank':<6} {'Experiment':<30} {metric}"]
        lines.append("-" * 50)
        for rank, (name, value) in enumerate(self.rankings[metric], start=1):
            marker = " *" if name == self.baseline_name else ""
            lines.append(f"{rank:<6} {name:<30} {value:.4f}{marker}")
        return "\n".join(lines)

    def relative_improvements(self, metric: str) -> dict[str, float]:
        """Return {experiment_name: relative_improvement_vs_baseline} for *metric*."""
        if metric not in self.metric_comparisons:
            return {}
        result: dict[str, float] = {}
        for mc in self.metric_comparisons[metric]:
            # Use experiment name as key
            result[mc.metric] = mc.relative_improvement  # will be overwritten in loop below
        # Rebuild keyed by experiment
        out: dict[str, float] = {}
        for mc in self.metric_comparisons[metric]:
            out[f"step_{mc.step}"] = mc.relative_improvement
        return out


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class ExperimentComparator:
    """Compare a collection of training experiments."""

    def compare(
        self,
        experiments: list[ExperimentRecord],
        baseline_name: str,
        metrics: list[str],
        comparison_steps: list[int] | None = None,
        lower_is_better: bool = True,
    ) -> ComparisonReport:
        """Generate a :class:`ComparisonReport` for *experiments*.

        Parameters
        ----------
        experiments:
            List of experiment records.
        baseline_name:
            Name of the baseline experiment to compare against.
        metrics:
            Which metrics to include in the comparison.
        comparison_steps:
            Steps at which to compare (defaults to all steps in baseline).
        lower_is_better:
            True for loss/perplexity, False for accuracy/score metrics.
        """
        exp_by_name = {e.name: e for e in experiments}
        baseline = exp_by_name.get(baseline_name)

        report = ComparisonReport(
            baseline_name=baseline_name,
            experiment_names=[e.name for e in experiments],
        )

        for metric in metrics:
            # Determine comparison steps
            if comparison_steps:
                steps = comparison_steps
            elif baseline and metric in baseline.metrics:
                steps = sorted(baseline.metrics[metric].keys())
            else:
                # Union of all steps across all experiments
                all_steps: set[int] = set()
                for exp in experiments:
                    all_steps.update(exp.metrics.get(metric, {}).keys())
                steps = sorted(all_steps)

            if not steps:
                continue

            # Build metric comparisons vs baseline
            comparisons: list[MetricComparison] = []
            if baseline:
                for exp in experiments:
                    if exp.name == baseline_name:
                        continue
                    for step in steps:
                        bv = baseline.get_value_at(metric, step)
                        ev = exp.get_value_at(metric, step)
                        if bv is not None and ev is not None:
                            comparisons.append(
                                MetricComparison(
                                    metric=metric,
                                    step=step,
                                    baseline_value=bv,
                                    experiment_value=ev,
                                )
                            )
            report.metric_comparisons[metric] = comparisons

            # Rankings by best value
            ranked: list[tuple[str, float]] = []
            for exp in experiments:
                best = exp.best_value(metric, lower_is_better=lower_is_better)
                if best is not None:
                    ranked.append((exp.name, best))

            ranked.sort(key=lambda x: x[1], reverse=not lower_is_better)
            report.rankings[metric] = ranked

        return report

    @staticmethod
    def align_metrics(
        experiments: list[ExperimentRecord],
        metric: str,
        steps: list[int],
    ) -> dict[str, list[float | None]]:
        """Return {experiment_name: [value_at_step, ...]} aligned to *steps*."""
        return {
            exp.name: [exp.get_value_at(metric, s) for s in steps]
            for exp in experiments
        }
