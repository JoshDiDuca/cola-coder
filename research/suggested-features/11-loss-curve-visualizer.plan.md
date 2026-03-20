# 11 - Loss Curve Visualizer

## Overview

Render a loss curve in the terminal after training or on demand, using `plotext` or `asciichartpy` for ASCII line charts. Displays train loss, validation loss (if available), and learning rate on separate axes. Supports filtering to the last N steps, full history, or a custom range. Optionally saves the chart as a PNG using matplotlib. Rendered inside a Rich panel.

---

## Motivation

The training log produces a JSON file of losses by step, but reading a list of numbers gives no insight into the training dynamics. A visual curve immediately reveals:
- Did loss plateau too early? (LR too low or data exhausted)
- Is there a spike at step X? (LR warm-up behavior or bad batch)
- When did validation loss diverge from train loss? (Overfitting onset)
- Is the curve still descending or has it flattened? (Is more training worthwhile?)

A terminal-based chart means the user never leaves their terminal session to get this insight.

---

## Architecture / Design

### Data Source

The visualizer reads from the training metrics log saved by the trainer:

```json
// runs/run_001/metrics.jsonl  (newline-delimited JSON)
{"step": 100, "train_loss": 3.241, "val_loss": null, "lr": 3e-5, "tokens_per_sec": 1200}
{"step": 200, "train_loss": 2.891, "val_loss": 2.943, "lr": 6e-5, "tokens_per_sec": 1234}
...
{"step": 10000, "train_loss": 1.234, "val_loss": 1.289, "lr": 3e-5, "tokens_per_sec": 1247}
```

### Display Layout

```
╔══════════════════════════════════════════════════════════════╗
║  Loss Curve: run_001_small  |  steps 0-10000                 ║
╠══════════════════════════════════════════════════════════════╣
║  3.5 ┤                                                        ║
║  3.0 ┤╲                                                       ║
║  2.5 ┤  ╲                                                     ║
║  2.0 ┤    ╲━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                ║
║  1.5 ┤      ╲╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌                      ║
║       └────────────────────────────────────                   ║
║        0    2000    4000    6000    8000   10000              ║
║  ━━ train_loss    ╌╌ val_loss                                 ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Implementation Steps

### Step 1: Metrics log reader

```python
# src/visualization/loss_curve.py
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class MetricsRecord:
    step: int
    train_loss: float
    val_loss: Optional[float]
    lr: float

def load_metrics(run_dir: Path) -> list[MetricsRecord]:
    """Load metrics from a run's metrics.jsonl file."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        # Fall back to scanning checkpoint files for embedded loss
        return _load_from_checkpoints(run_dir)

    records = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            records.append(MetricsRecord(
                step=d.get("step", 0),
                train_loss=d.get("train_loss", float("nan")),
                val_loss=d.get("val_loss"),
                lr=d.get("lr", 0.0),
            ))
        except json.JSONDecodeError:
            continue
    return sorted(records, key=lambda r: r.step)


def _load_from_checkpoints(run_dir: Path) -> list[MetricsRecord]:
    """If no metrics.jsonl, extract loss from checkpoint metadata."""
    import torch
    records = []
    for ckpt_path in sorted(run_dir.glob("ckpt_*.pt")):
        if "CRASH" in ckpt_path.name:
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            records.append(MetricsRecord(
                step=ckpt.get("step", 0),
                train_loss=ckpt.get("loss", float("nan")),
                val_loss=None,
                lr=0.0,
            ))
        except Exception:
            continue
    return sorted(records, key=lambda r: r.step)


def filter_metrics(
    records: list[MetricsRecord],
    last_n: Optional[int] = None,
    step_range: Optional[tuple[int, int]] = None,
) -> list[MetricsRecord]:
    if step_range:
        records = [r for r in records if step_range[0] <= r.step <= step_range[1]]
    if last_n:
        records = records[-last_n:]
    return records
```

### Step 2: ASCII chart rendering with plotext

```python
# src/visualization/loss_curve.py (continued)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Literal
import io

console = Console()

def render_ascii_chart(
    records: list[MetricsRecord],
    width: int = 80,
    height: int = 20,
    show_lr: bool = False,
) -> str:
    """
    Render an ASCII loss curve using plotext.
    Returns the chart as a string.
    """
    try:
        import plotext as plt
    except ImportError:
        return _render_fallback_sparkline(records, width)

    steps = [r.step for r in records]
    train_losses = [r.train_loss for r in records]
    val_losses = [r.val_loss for r in records if r.val_loss is not None]
    val_steps = [r.step for r in records if r.val_loss is not None]

    plt.clear_figure()
    plt.plot_size(width, height)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.plot(steps, train_losses, label="train_loss", color="green")
    if val_losses:
        plt.plot(val_steps, val_losses, label="val_loss", color="yellow")

    if show_lr:
        # Normalize LR to loss scale for overlay
        lrs = [r.lr for r in records]
        max_loss = max(train_losses)
        max_lr = max(lrs) if lrs else 1
        normalized_lrs = [lr / max_lr * max_loss for lr in lrs]
        plt.plot(steps, normalized_lrs, label="lr (scaled)", color="blue")

    # Capture output
    buf = io.StringIO()
    old_stdout = __import__("sys").stdout
    __import__("sys").stdout = buf
    plt.show()
    __import__("sys").stdout = old_stdout
    return buf.getvalue()


def _render_fallback_sparkline(records: list[MetricsRecord], width: int = 60) -> str:
    """Fallback if plotext is not installed."""
    from src.training.monitor.sparkline import sparkline
    losses = [r.train_loss for r in records]
    spark = sparkline(losses, width=width)
    min_l = min(losses)
    max_l = max(losses)
    return (
        f"Train Loss Sparkline (install plotext for full chart):\n"
        f"{max_l:.3f} ┤{spark}\n"
        f"{min_l:.3f} ┘\n"
        f"       Steps: {records[0].step} → {records[-1].step}"
    )


def render_to_terminal(
    run_dir: Path,
    last_n: Optional[int] = None,
    step_range: Optional[tuple[int, int]] = None,
    show_lr: bool = False,
) -> None:
    """Main entry: load metrics, filter, render in a Rich panel."""
    records = load_metrics(run_dir)
    if not records:
        console.print(Panel("[red]No metrics data found.[/red]", title="Loss Curve"))
        return

    filtered = filter_metrics(records, last_n=last_n, step_range=step_range)
    chart_str = render_ascii_chart(filtered, show_lr=show_lr)

    run_name = run_dir.name
    step_info = f"steps {filtered[0].step}–{filtered[-1].step}"
    final_train = filtered[-1].train_loss
    final_val_records = [r for r in filtered if r.val_loss is not None]
    final_val = final_val_records[-1].val_loss if final_val_records else None
    val_str = f"  |  val={final_val:.4f}" if final_val else ""

    console.print(Panel(
        chart_str,
        title=f"Loss Curve: {run_name}  |  {step_info}  |  final train={final_train:.4f}{val_str}",
        border_style="cyan",
    ))
```

### Step 3: Optional PNG export with matplotlib

```python
# src/visualization/loss_curve.py (continued)

def save_as_png(
    records: list[MetricsRecord],
    output_path: Path,
    title: str = "Training Loss",
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """Save a publication-quality loss curve PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping PNG export.[/yellow]")
        return

    fig, ax1 = plt.subplots(figsize=figsize)

    steps = [r.step for r in records]
    train_losses = [r.train_loss for r in records]
    ax1.plot(steps, train_losses, color="steelblue", label="Train Loss", linewidth=1.5)

    val_records = [(r.step, r.val_loss) for r in records if r.val_loss is not None]
    if val_records:
        v_steps, v_losses = zip(*val_records)
        ax1.plot(v_steps, v_losses, color="orange", label="Val Loss",
                 linestyle="--", linewidth=1.5)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Optional LR on secondary axis
    lrs = [r.lr for r in records if r.lr > 0]
    if lrs:
        ax2 = ax1.twinx()
        lr_steps = [r.step for r in records if r.lr > 0]
        ax2.plot(lr_steps, lrs, color="green", alpha=0.5, label="LR", linewidth=1)
        ax2.set_ylabel("Learning Rate", color="green")
        ax2.tick_params(axis='y', labelcolor='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    console.print(f"[green]Loss curve saved to {output_path}[/green]")
```

### Step 4: CLI entry point

```python
# src/visualization/loss_curve_cli.py
import typer
from pathlib import Path
from typing import Optional
from src.visualization.loss_curve import (
    load_metrics, filter_metrics,
    render_to_terminal, save_as_png
)

app = typer.Typer()

@app.command()
def main(
    run_dir: Path = typer.Argument(..., help="Path to a training run directory"),
    last_n: Optional[int] = typer.Option(None, "--last", help="Show only last N steps"),
    from_step: Optional[int] = typer.Option(None, help="Start step for range"),
    to_step: Optional[int] = typer.Option(None, help="End step for range"),
    show_lr: bool = typer.Option(False, "--lr", help="Overlay learning rate"),
    save_png: Optional[Path] = typer.Option(None, "--save-png", help="Save chart as PNG"),
):
    step_range = (from_step, to_step) if (from_step and to_step) else None
    render_to_terminal(run_dir, last_n=last_n, step_range=step_range, show_lr=show_lr)

    if save_png:
        records = load_metrics(run_dir)
        filtered = filter_metrics(records, last_n=last_n, step_range=step_range)
        save_as_png(filtered, save_png, title=f"Loss: {run_dir.name}")

if __name__ == "__main__":
    app()
```

### Step 5: Ensure trainer writes metrics.jsonl

```python
# src/trainer.py (addition)
import json
from pathlib import Path

class MetricsLogger:
    def __init__(self, run_dir: Path):
        self._path = run_dir / "metrics.jsonl"
        self._file = self._path.open("a")

    def log(self, data: dict) -> None:
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

# In Trainer:
# self._metrics_logger = MetricsLogger(run_dir)
# After each step:
# self._metrics_logger.log({"step": step, "train_loss": loss, "val_loss": val_loss, "lr": lr})
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/visualization/loss_curve.py` | New: metrics loading, ASCII chart, PNG export |
| `src/visualization/loss_curve_cli.py` | New: CLI entry point |
| `src/trainer.py` | Add `MetricsLogger` to write `metrics.jsonl` per step |
| `src/menu/menus/tools_menu.py` | Add "View Loss Curve" item |
| `requirements.txt` | Add `plotext` (optional), `matplotlib` (optional) |

---

## Testing Strategy

- **Metrics loader test**: Write a sample `metrics.jsonl`, load it, verify records parsed correctly
- **Filter test**: `filter_metrics(records, last_n=50)` returns last 50 records
- **Sparkline fallback test**: With plotext not installed, `render_ascii_chart()` falls back gracefully
- **PNG export test**: With matplotlib installed, call `save_as_png()`, verify file created
- **Empty metrics test**: Empty `metrics.jsonl` → "No metrics data found" panel
- **Checkpoint fallback test**: No `metrics.jsonl` but checkpoints exist → loads loss from checkpoint metadata

---

## Performance Considerations

- `plotext` renders to terminal using text characters — instantaneous
- `matplotlib` PNG export takes ~1-2 seconds for 10K data points
- Loading `metrics.jsonl` with 10K lines (each ~100 bytes) = 1MB file, parsed in <100ms
- Large runs (100K steps) may have 10MB metrics files; consider downsampling before plotting

---

## Dependencies

| Package | Use | Optional? | Install |
|---------|-----|-----------|---------|
| `plotext` | ASCII charts | Yes | `pip install plotext` |
| `matplotlib` | PNG export | Yes | `pip install matplotlib` |
| `rich` | Panel display | No | Already installed |

Both plotext and matplotlib are optional. The sparkline fallback works without either.

---

## Estimated Complexity

**Low** — 1 day.

- Metrics loader: 1.5 hours
- plotext chart rendering: 2 hours
- matplotlib PNG export: 1.5 hours
- CLI + filtering: 1.5 hours
- MetricsLogger in trainer: 1 hour
- Testing: 2 hours

Total: ~9.5 hours

---

## 2026 Best Practices

- **JSONL for streaming metrics**: Newline-delimited JSON (`.jsonl`) is the right format for metrics logs. Each step appends one line; partial writes don't corrupt the file; grep and jq work natively on it.
- **plotext for terminal, matplotlib for files**: plotext is purpose-built for terminal output. matplotlib is purpose-built for publication-quality images. Don't try to use matplotlib for terminal rendering.
- **Downsample for large runs**: When displaying a 100K-step run in a 80-column terminal, only 80 points are visible. Downsample to max(n_steps, terminal_width) before rendering.
- **Secondary LR axis**: Learning rate on the same chart as loss (using a secondary y-axis) immediately reveals whether loss plateaus coincide with LR schedule changes — invaluable for debugging.
- **Real-time metrics file**: Keep `metrics.jsonl` open during training and flush after every write. This allows `tail -f metrics.jsonl` monitoring from another terminal.
