"""Continuous Eval: run evaluations at configurable intervals during training.

Tracks eval metrics over time, detects regressions, and reports best steps.
Useful for monitoring model quality throughout a training run without waiting
until the end.
"""

from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ContinuousEvalConfig:
    """Configuration for continuous evaluation."""
    eval_interval_steps: int = 500
    metrics: list = field(default_factory=list)
    patience: int = 5


class ContinuousEvaluator:
    """Runs evaluations continuously during training and tracks metric history."""

    def __init__(self, config: Optional[ContinuousEvalConfig] = None):
        self.config = config or ContinuousEvalConfig()
        # history[metric_name] = list of (step, value)
        self._history: dict[str, list[tuple[int, float]]] = {}

    def should_eval(self, step: int) -> bool:
        """Return True if an evaluation should run at this step."""
        return step % self.config.eval_interval_steps == 0

    def record_result(self, step: int, metrics: dict) -> None:
        """Record evaluation metrics at the given step."""
        for name, value in metrics.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append((step, float(value)))

    def _is_lower_better(self, metric_name: str) -> bool:
        """Heuristic: loss-like metrics are lower-is-better; others are higher-is-better."""
        lower_better_keywords = ("loss", "perplexity", "error", "ppl")
        return any(kw in metric_name.lower() for kw in lower_better_keywords)

    def check_regression(self, metric_name: str) -> bool:
        """Return True if the most recent value is worse than the best seen so far.

        Uses patience: the metric must have been recorded at least twice and the
        latest value must be strictly worse than the historical best.
        """
        history = self._history.get(metric_name, [])
        if len(history) < 2:
            return False

        values = [v for _, v in history]
        latest = values[-1]
        historical_best = min(values[:-1]) if self._is_lower_better(metric_name) else max(values[:-1])

        if self._is_lower_better(metric_name):
            return latest > historical_best
        else:
            return latest < historical_best

    def best_step(self, metric_name: str) -> int:
        """Return the step at which the metric had its best (optimal) value."""
        history = self._history.get(metric_name, [])
        if not history:
            raise ValueError(f"No history for metric '{metric_name}'")

        if self._is_lower_better(metric_name):
            best_step, _ = min(history, key=lambda x: x[1])
        else:
            best_step, _ = max(history, key=lambda x: x[1])
        return best_step

    def get_history(self, metric_name: str) -> list[tuple[int, float]]:
        """Return list of (step, value) pairs for the given metric."""
        return list(self._history.get(metric_name, []))

    def is_improving(self, metric_name: str, window: int = 3) -> bool:
        """Return True if the metric has been improving over the last `window` records.

        For lower-is-better metrics, improving means the values are decreasing.
        For higher-is-better metrics, improving means the values are increasing.
        Requires at least 2 data points in the window to make a determination.
        """
        history = self._history.get(metric_name, [])
        if len(history) < 2:
            return False

        recent = history[-window:]
        values = [v for _, v in recent]

        if self._is_lower_better(metric_name):
            # Each step should be <= previous (trending down)
            return all(values[i] <= values[i - 1] for i in range(1, len(values)))
        else:
            # Each step should be >= previous (trending up)
            return all(values[i] >= values[i - 1] for i in range(1, len(values)))

    def summary(self) -> dict:
        """Return a summary dict with best step and latest value per tracked metric."""
        result: dict = {}
        for metric_name, history in self._history.items():
            if not history:
                continue
            latest_step, latest_value = history[-1]
            try:
                b_step = self.best_step(metric_name)
                b_value = dict(history)[b_step]
            except (ValueError, KeyError):
                b_step = latest_step
                b_value = latest_value

            result[metric_name] = {
                "best_step": b_step,
                "best_value": b_value,
                "latest_step": latest_step,
                "latest_value": latest_value,
                "num_evals": len(history),
                "regression": self.check_regression(metric_name),
                "improving": self.is_improving(metric_name),
            }
        return result
