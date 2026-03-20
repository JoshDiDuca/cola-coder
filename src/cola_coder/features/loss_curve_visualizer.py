"""Loss Curve Visualizer: plot training loss curves from checkpoint metadata or training logs.

Supports:
- Loss over steps (raw)
- Smoothed loss (EMA)
- Multiple run comparison
- Loss rate of change
- Learning rate overlay

Data can be loaded from a dict, a JSON file, or a checkpoint directory (metrics.jsonl
or checkpoint metadata JSON files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """A single training metrics record."""
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    lr: float = 0.0


@dataclass
class RunData:
    """All records for a named training run."""
    name: str
    records: List[StepRecord] = field(default_factory=list)

    @property
    def steps(self) -> List[int]:
        return [r.step for r in self.records]

    @property
    def train_losses(self) -> List[float]:
        return [r.train_loss for r in self.records]

    @property
    def val_losses(self) -> List[Optional[float]]:
        return [r.val_loss for r in self.records]

    @property
    def lrs(self) -> List[float]:
        return [r.lr for r in self.records]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _parse_loss_dict(data: Dict[str, float]) -> List[StepRecord]:
    """Parse a dict of {step_key: loss_value} into sorted StepRecords.

    Accepts keys like "step_100", "100", or integer keys.
    """
    records: List[StepRecord] = []
    for key, value in data.items():
        try:
            if isinstance(key, int):
                step = key
            else:
                # strip any non-numeric prefix (e.g. "step_100" -> 100)
                numeric = key.replace("step_", "").strip()
                step = int(numeric)
            records.append(StepRecord(step=step, train_loss=float(value)))
        except (ValueError, AttributeError):
            continue
    return sorted(records, key=lambda r: r.step)


def _load_from_jsonl(path: Path) -> List[StepRecord]:
    """Load records from a metrics.jsonl file."""
    records: List[StepRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(StepRecord(
                step=int(d.get("step", 0)),
                train_loss=float(d.get("train_loss", float("nan"))),
                val_loss=float(d["val_loss"]) if d.get("val_loss") is not None else None,
                lr=float(d.get("lr", 0.0)),
            ))
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    return sorted(records, key=lambda r: r.step)


def _load_from_checkpoint_json(path: Path) -> List[StepRecord]:
    """Load records from checkpoint metadata JSON file(s).

    Handles both a single metadata JSON and a directory of them.
    """
    records: List[StepRecord] = []
    paths_to_check: List[Path] = []

    if path.is_dir():
        # collect all *metadata*.json or ckpt_*.json files
        for pattern in ("*metadata*.json", "ckpt_*.json", "checkpoint*.json"):
            paths_to_check.extend(sorted(path.glob(pattern)))
    elif path.is_file() and path.suffix == ".json":
        paths_to_check = [path]

    for p in paths_to_check:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            # Flat record with step + loss
            if "step" in d and ("loss" in d or "train_loss" in d):
                records.append(StepRecord(
                    step=int(d.get("step", 0)),
                    train_loss=float(d.get("train_loss", d.get("loss", float("nan")))),
                    val_loss=float(d["val_loss"]) if d.get("val_loss") is not None else None,
                    lr=float(d.get("lr", 0.0)),
                ))
            # loss_history dict embedded in checkpoint metadata
            if "loss_history" in d:
                sub = d["loss_history"]
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        try:
                            step = int(str(k).replace("step_", "").strip())
                            records.append(StepRecord(step=step, train_loss=float(v)))
                        except (ValueError, AttributeError):
                            continue
                elif isinstance(sub, list):
                    for i, v in enumerate(sub):
                        try:
                            records.append(StepRecord(step=i, train_loss=float(v)))
                        except (ValueError, TypeError):
                            continue
        except (json.JSONDecodeError, OSError):
            continue

    return sorted(records, key=lambda r: r.step)


def load_run_from_directory(run_dir: Union[str, Path], name: Optional[str] = None) -> RunData:
    """Load a RunData from a training run directory.

    Tries in order:
    1. metrics.jsonl
    2. Checkpoint metadata JSON files
    """
    run_dir = Path(run_dir)
    run_name = name or run_dir.name

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        return RunData(name=run_name, records=_load_from_jsonl(metrics_jsonl))

    records = _load_from_checkpoint_json(run_dir)
    return RunData(name=run_name, records=records)


def load_run_from_json(json_path: Union[str, Path], name: Optional[str] = None) -> RunData:
    """Load a RunData from a JSON file.

    Supports:
    - metrics.jsonl (newline-delimited)
    - flat JSON with loss_history key
    - flat dict of {step: loss}
    - list of metric dicts
    """
    path = Path(json_path)
    run_name = name or path.stem

    content = path.read_text(encoding="utf-8")

    # Try JSONL first
    if "\n" in content.strip():
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        try:
            first = json.loads(lines[0])
            if isinstance(first, dict) and ("step" in first or "train_loss" in first):
                return RunData(name=run_name, records=_load_from_jsonl(path))
        except (json.JSONDecodeError, IndexError):
            pass

    # Full JSON parse
    data = json.loads(content)

    if isinstance(data, list):
        records: List[StepRecord] = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                records.append(StepRecord(
                    step=int(item.get("step", i)),
                    train_loss=float(item.get("train_loss", item.get("loss", float("nan")))),
                    val_loss=float(item["val_loss"]) if item.get("val_loss") is not None else None,
                    lr=float(item.get("lr", 0.0)),
                ))
            else:
                records.append(StepRecord(step=i, train_loss=float(item)))
        return RunData(name=run_name, records=sorted(records, key=lambda r: r.step))

    if isinstance(data, dict):
        if "loss_history" in data:
            sub = data["loss_history"]
            return RunData(name=run_name, records=_parse_loss_dict(sub) if isinstance(sub, dict)
                           else [StepRecord(step=i, train_loss=float(v)) for i, v in enumerate(sub)])
        # plain {step: loss} dict
        return RunData(name=run_name, records=_parse_loss_dict(data))

    raise ValueError(f"Cannot parse loss data from {json_path}")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _ema_smooth(values: List[float], alpha: float = 0.1) -> List[float]:
    """Exponential moving average smoothing.

    Args:
        values: Input loss values.
        alpha: Smoothing factor 0-1. Higher = less smoothing (more responsive).
                Common usage: 0.1 for heavy smoothing, 0.9 for light.

    Returns:
        EMA-smoothed list of the same length.
    """
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _rate_of_change(steps: List[int], values: List[float]) -> Tuple[List[int], List[float]]:
    """Finite difference approximation of d(loss)/d(step)."""
    if len(steps) < 2:
        return [], []
    roc_steps: List[int] = []
    roc_vals: List[float] = []
    for i in range(1, len(steps)):
        ds = steps[i] - steps[i - 1]
        if ds == 0:
            continue
        roc_steps.append(steps[i])
        roc_vals.append((values[i] - values[i - 1]) / ds)
    return roc_steps, roc_vals


def _downsample(steps: List[int], values: List[float], max_points: int = 2000
                ) -> Tuple[List[int], List[float]]:
    """Evenly downsample to at most max_points data points."""
    n = len(steps)
    if n <= max_points:
        return steps, values
    indices = [int(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    return [steps[i] for i in indices], [values[i] for i in indices]


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------

class LossCurveVisualizer:
    """Plot training loss curves to PNG files.

    Usage::

        viz = LossCurveVisualizer()

        # Add runs from various sources
        viz.add_run("run1", {"step_0": 10.0, "step_100": 9.5, ...})
        viz.add_run_from_directory("run2", Path("checkpoints/run2"))
        viz.add_run_from_json("run3", Path("metrics.json"))

        # Plot
        viz.plot_loss(output_path="loss.png")
        viz.plot_smoothed(output_path="smooth.png", alpha=0.9)
        viz.plot_comparison(output_path="compare.png")
        viz.plot_learning_rate(output_path="lr.png")
        viz.plot_rate_of_change(output_path="roc.png")
    """

    def __init__(self, style: str = "dark_background", figsize: Tuple[int, int] = (12, 5), dpi: int = 150):
        self._runs: Dict[str, RunData] = {}
        self._style = style
        self._figsize = figsize
        self._dpi = dpi

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_run(self, name: str, data: Union[Dict, List, RunData]) -> "LossCurveVisualizer":
        """Add a training run.

        Args:
            name: Identifier for this run (used in legends).
            data: One of:
                - Dict[str, float] mapping step keys to loss values
                - List[float] of losses (steps inferred as 0, 1, 2, ...)
                - List[dict] of metric records
                - RunData instance

        Returns:
            self, for chaining.
        """
        if isinstance(data, RunData):
            self._runs[name] = data
        elif isinstance(data, dict):
            self._runs[name] = RunData(name=name, records=_parse_loss_dict(data))
        elif isinstance(data, list):
            records: List[StepRecord] = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    records.append(StepRecord(
                        step=int(item.get("step", i)),
                        train_loss=float(item.get("train_loss", item.get("loss", float("nan")))),
                        val_loss=float(item["val_loss"]) if item.get("val_loss") is not None else None,
                        lr=float(item.get("lr", 0.0)),
                    ))
                else:
                    records.append(StepRecord(step=i, train_loss=float(item)))
            self._runs[name] = RunData(name=name, records=sorted(records, key=lambda r: r.step))
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return self

    def add_run_from_directory(self, name: str, run_dir: Union[str, Path]) -> "LossCurveVisualizer":
        """Add a run loaded from a checkpoint directory."""
        self._runs[name] = load_run_from_directory(run_dir, name=name)
        return self

    def add_run_from_json(self, name: str, json_path: Union[str, Path]) -> "LossCurveVisualizer":
        """Add a run loaded from a JSON or JSONL file."""
        self._runs[name] = load_run_from_json(json_path, name=name)
        return self

    def clear_runs(self) -> "LossCurveVisualizer":
        """Remove all loaded runs."""
        self._runs.clear()
        return self

    # ------------------------------------------------------------------
    # Internal matplotlib setup
    # ------------------------------------------------------------------

    def _get_matplotlib(self):
        """Import matplotlib with Agg backend (non-interactive, file output only)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.style as mplstyle
        try:
            mplstyle.use(self._style)
        except OSError:
            pass  # style not found, use default
        return plt

    def _save_and_close(self, plt, output_path: Union[str, Path]) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(out), dpi=self._dpi, bbox_inches="tight")
        plt.close("all")
        return out

    def _require_runs(self) -> None:
        if not self._runs:
            raise ValueError("No runs loaded. Call add_run() first.")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_loss(
        self,
        output_path: Union[str, Path] = "loss.png",
        title: str = "Training Loss",
        run_names: Optional[List[str]] = None,
        show_val: bool = True,
        max_points: int = 2000,
    ) -> Path:
        """Plot raw training (and optionally validation) loss curves.

        Args:
            output_path: Destination PNG file path.
            title: Plot title.
            run_names: Subset of run names to plot. None = all runs.
            show_val: Whether to overlay validation loss when available.
            max_points: Downsample to this many points per curve.

        Returns:
            Path to the saved PNG.
        """
        self._require_runs()
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        runs = [self._runs[n] for n in (run_names or list(self._runs.keys()))]
        for run in runs:
            if not run.records:
                continue
            steps, losses = _downsample(run.steps, run.train_losses, max_points)
            ax.plot(steps, losses, linewidth=1.5, label=f"{run.name} (train)")

            if show_val:
                val_pairs = [(r.step, r.val_loss) for r in run.records if r.val_loss is not None]
                if val_pairs:
                    v_steps, v_losses = zip(*val_pairs)
                    vs, vl = _downsample(list(v_steps), list(v_losses), max_points)
                    ax.plot(vs, vl, linewidth=1.5, linestyle="--", label=f"{run.name} (val)")

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return self._save_and_close(plt, output_path)

    def plot_smoothed(
        self,
        output_path: Union[str, Path] = "loss_smoothed.png",
        title: str = "Smoothed Training Loss (EMA)",
        alpha: float = 0.1,
        show_raw: bool = True,
        run_names: Optional[List[str]] = None,
        max_points: int = 2000,
    ) -> Path:
        """Plot EMA-smoothed loss curves.

        Args:
            output_path: Destination PNG file path.
            title: Plot title.
            alpha: EMA smoothing factor (0-1). Higher = less smoothing.
            show_raw: Overlay raw loss at low opacity for reference.
            run_names: Subset of run names to plot. None = all runs.
            max_points: Downsample to this many points per curve.

        Returns:
            Path to the saved PNG.
        """
        self._require_runs()
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.set_title(f"{title}  (α={alpha})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        runs = [self._runs[n] for n in (run_names or list(self._runs.keys()))]
        for run in runs:
            if not run.records:
                continue
            steps, losses = _downsample(run.steps, run.train_losses, max_points)
            smoothed = _ema_smooth(losses, alpha=alpha)

            if show_raw:
                ax.plot(steps, losses, linewidth=0.8, alpha=0.3, label=f"{run.name} (raw)")
            ax.plot(steps, smoothed, linewidth=2.0, label=f"{run.name} (EMA)")

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return self._save_and_close(plt, output_path)

    def plot_comparison(
        self,
        output_path: Union[str, Path] = "loss_comparison.png",
        title: str = "Loss Comparison",
        smooth: bool = True,
        alpha: float = 0.1,
        max_points: int = 2000,
    ) -> Path:
        """Compare multiple runs on the same axes.

        Args:
            output_path: Destination PNG file path.
            title: Plot title.
            smooth: Apply EMA smoothing to each run's curve.
            alpha: EMA alpha (only used if smooth=True).
            max_points: Downsample to this many points per curve.

        Returns:
            Path to the saved PNG.
        """
        self._require_runs()
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        for run in self._runs.values():
            if not run.records:
                continue
            steps, losses = _downsample(run.steps, run.train_losses, max_points)
            if smooth:
                losses = _ema_smooth(losses, alpha=alpha)
            ax.plot(steps, losses, linewidth=1.8, label=run.name)

            # Annotate final loss
            if steps and losses:
                ax.annotate(
                    f"{losses[-1]:.3f}",
                    xy=(steps[-1], losses[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                )

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return self._save_and_close(plt, output_path)

    def plot_learning_rate(
        self,
        output_path: Union[str, Path] = "learning_rate.png",
        title: str = "Learning Rate Schedule",
        run_names: Optional[List[str]] = None,
        overlay_loss: bool = True,
        max_points: int = 2000,
    ) -> Path:
        """Plot the learning rate schedule, optionally with loss on a secondary axis.

        Args:
            output_path: Destination PNG file path.
            title: Plot title.
            run_names: Subset of run names to include. None = all runs.
            overlay_loss: Show smoothed loss on a secondary y-axis.
            max_points: Downsample to this many points per curve.

        Returns:
            Path to the saved PNG.
        """
        self._require_runs()
        plt = self._get_matplotlib()

        fig, ax1 = plt.subplots(figsize=self._figsize)
        ax1.set_title(title)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Learning Rate")

        ax2 = ax1.twinx() if overlay_loss else None
        if ax2:
            ax2.set_ylabel("Loss (smoothed)", alpha=0.7)

        runs = [self._runs[n] for n in (run_names or list(self._runs.keys()))]
        has_lr_data = False
        for run in runs:
            lr_pairs = [(r.step, r.lr) for r in run.records if r.lr > 0]
            if lr_pairs:
                lr_steps, lrs = zip(*lr_pairs)
                ds, dl = _downsample(list(lr_steps), list(lrs), max_points)
                ax1.plot(ds, dl, linewidth=1.5, label=f"{run.name} LR")
                has_lr_data = True

            if ax2:
                steps, losses = _downsample(run.steps, run.train_losses, max_points)
                smoothed = _ema_smooth(losses, alpha=0.05)
                ax2.plot(steps, smoothed, linewidth=1.0, alpha=0.5, linestyle="--",
                         label=f"{run.name} loss")

        if not has_lr_data:
            ax1.text(0.5, 0.5, "No learning rate data found",
                     ha="center", va="center", transform=ax1.transAxes, fontsize=12)

        lines1, labels1 = ax1.get_legend_handles_labels()
        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        ax1.grid(True, alpha=0.3)
        return self._save_and_close(plt, output_path)

    def plot_rate_of_change(
        self,
        output_path: Union[str, Path] = "loss_roc.png",
        title: str = "Loss Rate of Change (d_loss/d_step)",
        run_names: Optional[List[str]] = None,
        smooth_alpha: float = 0.05,
        max_points: int = 2000,
    ) -> Path:
        """Plot the rate of change of loss (finite difference derivative).

        This reveals where the model is learning fastest vs. where it has plateaued.

        Args:
            output_path: Destination PNG file path.
            title: Plot title.
            run_names: Subset of run names to include. None = all runs.
            smooth_alpha: EMA alpha applied to the derivative for readability.
            max_points: Downsample to this many points per curve.

        Returns:
            Path to the saved PNG.
        """
        self._require_runs()
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("d(loss) / d(step)")
        ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.4)

        runs = [self._runs[n] for n in (run_names or list(self._runs.keys()))]
        for run in runs:
            if len(run.records) < 2:
                continue
            steps, losses = _downsample(run.steps, run.train_losses, max_points)
            roc_steps, roc_vals = _rate_of_change(steps, losses)
            if roc_vals:
                smoothed_roc = _ema_smooth(roc_vals, alpha=smooth_alpha)
                ax.plot(roc_steps, smoothed_roc, linewidth=1.5, label=run.name)

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return self._save_and_close(plt, output_path)

    def plot_all(
        self,
        output_dir: Union[str, Path] = ".",
        prefix: str = "",
        smooth_alpha: float = 0.1,
    ) -> Dict[str, Path]:
        """Generate all standard plots and save to output_dir.

        Returns:
            Dict mapping plot name to file path.
        """
        out = Path(output_dir)
        p = f"{prefix}_" if prefix else ""
        results: Dict[str, Path] = {}

        results["loss"] = self.plot_loss(out / f"{p}loss.png")
        results["smoothed"] = self.plot_smoothed(out / f"{p}loss_smoothed.png", alpha=smooth_alpha)
        results["comparison"] = self.plot_comparison(out / f"{p}loss_comparison.png", alpha=smooth_alpha)
        results["learning_rate"] = self.plot_learning_rate(out / f"{p}learning_rate.png")
        results["rate_of_change"] = self.plot_rate_of_change(out / f"{p}loss_roc.png")
        return results

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(cls, run_dir: Union[str, Path], name: Optional[str] = None, **kwargs) -> "LossCurveVisualizer":
        """Create a visualizer pre-loaded from a training run directory."""
        viz = cls(**kwargs)
        run_dir = Path(run_dir)
        viz.add_run_from_directory(name or run_dir.name, run_dir)
        return viz

    @classmethod
    def from_json(cls, json_path: Union[str, Path], name: Optional[str] = None, **kwargs) -> "LossCurveVisualizer":
        """Create a visualizer pre-loaded from a JSON/JSONL metrics file."""
        viz = cls(**kwargs)
        path = Path(json_path)
        viz.add_run_from_json(name or path.stem, path)
        return viz
