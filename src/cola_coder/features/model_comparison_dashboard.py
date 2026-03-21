"""Model Comparison Dashboard: compare multiple model checkpoints.

Generates an HTML or Markdown dashboard that shows:
- A metrics table comparing checkpoints across key performance indicators
- Chart data (JSON-serializable) for loss curves, accuracy, perplexity, etc.
- A summary section with best/worst performers per metric
- Color-coded diff highlighting improvements vs regressions

No external dependencies (no matplotlib, pandas, etc.) — outputs are
plain strings (HTML/Markdown) plus JSON-serializable data structures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the model comparison dashboard feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMetrics:
    """Metrics for a single model checkpoint."""

    name: str  # e.g. "step_1000", "epoch_3"
    step: Optional[int] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    bleu: Optional[float] = None
    tokens_per_second: Optional[float] = None
    params_million: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.step is not None:
            d["step"] = self.step
        if self.loss is not None:
            d["loss"] = self.loss
        if self.perplexity is not None:
            d["perplexity"] = self.perplexity
        if self.accuracy is not None:
            d["accuracy"] = self.accuracy
        if self.bleu is not None:
            d["bleu"] = self.bleu
        if self.tokens_per_second is not None:
            d["tokens_per_second"] = self.tokens_per_second
        if self.params_million is not None:
            d["params_million"] = self.params_million
        d.update(self.extra)
        return d


@dataclass
class DashboardData:
    """All data needed to render the dashboard."""

    checkpoints: list[CheckpointMetrics]
    metrics_table: list[dict[str, Any]]  # Row-per-checkpoint dicts
    chart_data: dict[str, Any]  # JSON-serializable chart series
    best_per_metric: dict[str, str]  # metric → checkpoint name
    worst_per_metric: dict[str, str]
    summary: str


# ---------------------------------------------------------------------------
# Dashboard generator
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = {"loss", "perplexity"}
_HIGHER_IS_BETTER = {"accuracy", "bleu", "tokens_per_second"}


class ModelComparisonDashboard:
    """Build comparison dashboards for multiple model checkpoints."""

    def __init__(self, metric_format: dict[str, str] | None = None) -> None:
        """
        Parameters
        ----------
        metric_format:
            Optional dict mapping metric name → Python format spec,
            e.g. {"loss": ".4f", "accuracy": ".2%"}.
        """
        self.metric_format = metric_format or {}

    def _fmt(self, metric: str, value: Any) -> str:
        fmt = self.metric_format.get(metric, ".4g")
        try:
            return format(float(value), fmt)
        except (TypeError, ValueError):
            return str(value)

    def _collect_numeric_metrics(
        self, checkpoints: list[CheckpointMetrics]
    ) -> dict[str, list[Optional[float]]]:
        """Collect values for each numeric metric across checkpoints."""
        keys = ["loss", "perplexity", "accuracy", "bleu", "tokens_per_second", "params_million"]
        # Add extra keys
        for cp in checkpoints:
            for k, v in cp.extra.items():
                if isinstance(v, (int, float)) and k not in keys:
                    keys.append(k)

        result: dict[str, list[Optional[float]]] = {}
        for k in keys:
            values: list[Optional[float]] = []
            for cp in checkpoints:
                val = getattr(cp, k, None)
                if val is None:
                    val = cp.extra.get(k)
                values.append(float(val) if val is not None else None)
            # Only include metrics with at least one non-None value
            if any(v is not None for v in values):
                result[k] = values
        return result

    def _best_worst(
        self,
        metric: str,
        values: list[Optional[float]],
        names: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (best_name, worst_name) for a metric."""
        pairs = [(n, v) for n, v in zip(names, values) if v is not None]
        if not pairs:
            return None, None
        lower_better = metric in _LOWER_IS_BETTER
        pairs_sorted = sorted(pairs, key=lambda x: x[1])
        best = pairs_sorted[0][0] if lower_better else pairs_sorted[-1][0]
        worst = pairs_sorted[-1][0] if lower_better else pairs_sorted[0][0]
        return best, worst

    def build(self, checkpoints: list[CheckpointMetrics]) -> DashboardData:
        """Build dashboard data from a list of checkpoints.

        Parameters
        ----------
        checkpoints:
            Ordered list of CheckpointMetrics (typically oldest to newest).
        """
        if not checkpoints:
            return DashboardData(
                checkpoints=[],
                metrics_table=[],
                chart_data={},
                best_per_metric={},
                worst_per_metric={},
                summary="No checkpoints provided.",
            )

        names = [cp.name for cp in checkpoints]
        numeric = self._collect_numeric_metrics(checkpoints)

        # Build metrics table
        metrics_table = []
        for cp in checkpoints:
            row = cp.to_dict()
            metrics_table.append(row)

        # Chart data: each metric becomes a time series keyed by step or name
        chart_data: dict[str, Any] = {"labels": names, "series": {}}
        for metric, values in numeric.items():
            chart_data["series"][metric] = [v for v in values]

        # Best / worst
        best_per: dict[str, str] = {}
        worst_per: dict[str, str] = {}
        for metric, values in numeric.items():
            best, worst = self._best_worst(metric, values, names)
            if best:
                best_per[metric] = best
            if worst:
                worst_per[metric] = worst

        summary = self._build_summary(checkpoints, numeric, best_per, worst_per)

        return DashboardData(
            checkpoints=checkpoints,
            metrics_table=metrics_table,
            chart_data=chart_data,
            best_per_metric=best_per,
            worst_per_metric=worst_per,
            summary=summary,
        )

    def _build_summary(
        self,
        checkpoints: list[CheckpointMetrics],
        numeric: dict[str, list[Optional[float]]],
        best_per: dict[str, str],
        worst_per: dict[str, str],
    ) -> str:
        lines = [f"# Model Comparison Dashboard ({len(checkpoints)} checkpoints)\n"]
        lines.append("## Best performers per metric")
        for metric, name in sorted(best_per.items()):
            lines.append(f"- **{metric}**: {name}")
        lines.append("\n## Worst performers per metric")
        for metric, name in sorted(worst_per.items()):
            lines.append(f"- **{metric}**: {name}")

        # Delta table: first vs last
        if len(checkpoints) >= 2:
            lines.append("\n## First → Last delta")
            for metric, values in numeric.items():
                v0 = values[0]
                vn = values[-1]
                if v0 is not None and vn is not None:
                    delta = vn - v0
                    sign = "+" if delta >= 0 else ""
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                    lines.append(f"- {metric}: {self._fmt(metric, v0)} → {self._fmt(metric, vn)} ({sign}{delta:.4g}) {arrow}")

        return "\n".join(lines)

    def render_markdown(self, data: DashboardData) -> str:
        """Render the metrics table as a Markdown string."""
        if not data.metrics_table:
            return "No data."

        # Collect all keys
        all_keys: list[str] = []
        for row in data.metrics_table:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)

        lines = []
        # Header
        lines.append("| " + " | ".join(all_keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(all_keys)) + " |")
        # Rows
        for row in data.metrics_table:
            cells = []
            for k in all_keys:
                val = row.get(k, "")
                if isinstance(val, float):
                    cells.append(self._fmt(k, val))
                else:
                    cells.append(str(val) if val != "" else "-")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def render_html(self, data: DashboardData) -> str:
        """Render the dashboard as a minimal self-contained HTML page."""
        md_table = self.render_markdown(data)
        chart_json = json.dumps(data.chart_data, indent=2)
        summary_html = data.summary.replace("\n", "<br>\n")

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Model Comparison Dashboard</title>
<style>
body {{ font-family: monospace; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
th {{ background: #f0f0f0; text-align: center; }}
pre {{ background: #f8f8f8; padding: 1em; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Model Comparison Dashboard</h1>
<h2>Summary</h2>
<p>{summary_html}</p>
<h2>Metrics Table</h2>
<pre>{md_table}</pre>
<h2>Chart Data (JSON)</h2>
<pre>{chart_json}</pre>
</body>
</html>"""
        return html

    def render_json(self, data: DashboardData) -> str:
        """Serialize the full dashboard data to JSON."""
        payload = {
            "metrics_table": data.metrics_table,
            "chart_data": data.chart_data,
            "best_per_metric": data.best_per_metric,
            "worst_per_metric": data.worst_per_metric,
            "summary": data.summary,
        }
        return json.dumps(payload, indent=2)
