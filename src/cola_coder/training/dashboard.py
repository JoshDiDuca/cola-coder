"""Real-time training dashboard using Rich Live display.

Provides a terminal UI that shows training progress in real-time:
- ASCII loss curve
- GPU memory and utilization
- Throughput (tokens/sec)
- ETA to completion
- Recent step log

Usage in trainer:
    dashboard = TrainingDashboard(config={"model_params": 350e6, ...}, total_steps=200000)
    dashboard.start()
    for step in range(total_steps):
        # ... training step ...
        dashboard.update(step=step, loss=loss, lr=lr, throughput=tok_per_s, gpu_mem_gb=mem)
    dashboard.stop()

The dashboard is OPTIONAL — if not created/started, training runs fine without it.
Never blocks the training loop (updates are non-blocking).

For a TS dev: this is like a React dashboard component that subscribes to
a stream of training events and re-renders at 2 FPS.
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# GPU stats helper
# ---------------------------------------------------------------------------

def get_gpu_stats() -> dict:
    """Get GPU stats using torch.cuda (no nvidia-smi subprocess).

    Returns a dict with GPU memory and device info.
    Gracefully handles CPU-only environments.
    """
    if not _HAS_TORCH:
        return {"available": False}
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_mem", None) or getattr(props, "total_memory", 0)
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_used_gb": torch.cuda.memory_allocated(0) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "memory_total_gb": total_mem / 1e9,
            # torch.cuda doesn't expose utilization directly without pynvml
            "utilization": "N/A",
        }
    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# ASCII chart
# ---------------------------------------------------------------------------

def ascii_chart(values: list[float], width: int = 50, height: int = 8) -> str:
    """Create an ASCII bar chart of values (e.g. loss curve).

    Each row represents a y-level; bars grow upward from the bottom.
    Columns represent time steps (last `width` values shown).

    Example output (height=4, width=12):
        3.2 │▓▓▓
        2.8 │▓▓▓▓▓
        2.4 │   ▓▓▓▓▓▓▓
        2.0 │         ▓▓▓
            └────────────
    """
    if not values:
        return "(no data yet)"

    # Take the last `width` values
    data = list(values[-width:])
    n = len(data)

    if n == 0:
        return "(no data yet)"

    v_min = min(data)
    v_max = max(data)

    # Avoid divide-by-zero when all values are identical
    v_range = v_max - v_min
    if v_range < 1e-10:
        v_range = 1.0
        v_min = v_min - 0.5
        v_max = v_max + 0.5

    # Build height x n grid of filled/empty cells
    # Row 0 = top (highest loss), row height-1 = bottom (lowest loss)
    rows: list[list[str]] = [["  " for _ in range(n)] for _ in range(height)]

    for col, val in enumerate(data):
        # Fraction from 0 (min) to 1 (max)
        frac = (val - v_min) / v_range
        # How many rows should be filled (column height in the chart)
        filled = max(1, round(frac * height))
        # Fill from the bottom up
        for row_idx in range(height - filled, height):
            rows[row_idx][col] = "\u2593\u2593"  # ▓▓

    # Y-axis labels: 4 evenly spaced ticks
    tick_count = min(height, 4)
    tick_positions: set[int] = set()
    for i in range(tick_count):
        tick_positions.add(round(i * (height - 1) / max(tick_count - 1, 1)))

    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        # y value for this row (top row = v_max, bottom row = v_min)
        y_val = v_max - (row_idx / (height - 1)) * v_range if height > 1 else v_max
        if row_idx in tick_positions:
            label = f"{y_val:5.2f} \u2502"  # │
        else:
            label = "       \u2502"
        lines.append(label + "".join(row))

    # Bottom axis
    lines.append("       \u2514" + "\u2500\u2500" * n)  # └──...

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds <= 0 or not math.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    if h < 24:
        return f"{h}h {m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d {h:02d}h {m:02d}m"


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------

class TrainingDashboard:
    """Real-time terminal dashboard for training monitoring.

    Drop-in monitoring layer for the Trainer — call update() after each step.
    The dashboard never raises exceptions; all errors are swallowed so training
    is never blocked by UI issues.

    Args:
        config: Dict with model/training metadata to display.
                Recognised keys: model_params, batch_size, effective_batch_size,
                learning_rate, seq_len, precision, model_size_name.
        total_steps: Total training steps (for progress bar and ETA).
    """

    MAX_HISTORY = 1000  # keep last N metric points in memory

    def __init__(self, config: dict, total_steps: int):
        self.config = config
        self.total_steps = max(total_steps, 1)
        self.metrics_history: dict[str, deque] = {
            "loss": deque(maxlen=self.MAX_HISTORY),
            "lr": deque(maxlen=self.MAX_HISTORY),
            "throughput": deque(maxlen=self.MAX_HISTORY),
            "gpu_mem": deque(maxlen=self.MAX_HISTORY),
            "grad_norm": deque(maxlen=self.MAX_HISTORY),
        }
        # Recent step log (step, loss, lr, grad_norm, throughput)
        self._recent_steps: deque[dict] = deque(maxlen=20)

        self._current_step = 0
        self._start_time = time.time()
        self._live: Live | None = None

        # Progress bar widget (reused across updates)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            expand=False,
        ) if _HAS_RICH else None

        self._task_id = None
        if self._progress is not None:
            self._task_id = self._progress.add_task("Training", total=self.total_steps)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the live dashboard. Call before the training loop begins."""
        if not _HAS_RICH:
            return
        try:
            layout = self._build_layout()
            self._live = Live(
                layout,
                refresh_per_second=2,
                screen=False,  # don't take over the whole terminal
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None  # graceful fallback — training continues without UI

    def stop(self) -> None:
        """Stop the live dashboard. Call after training ends."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def update(
        self,
        step: int,
        loss: float,
        lr: float,
        throughput: float,
        gpu_mem_gb: float = 0.0,
        grad_norm: float = 0.0,
        **extra: Any,
    ) -> None:
        """Update dashboard with metrics from the latest training step.

        This is non-blocking — if the live display is not running (e.g. no
        Rich, or Rich not started), the call is a quick dict append and returns.

        Args:
            step: Current training step number.
            loss: Training loss for this step.
            lr: Current learning rate.
            throughput: Tokens processed per second.
            gpu_mem_gb: GPU memory currently allocated (GB). 0 = not available.
            grad_norm: Gradient norm after clipping. 0 = not tracked.
            **extra: Additional metrics (ignored by display, stored for future use).
        """
        try:
            self._current_step = step

            # Store metric history
            self.metrics_history["loss"].append(loss)
            self.metrics_history["lr"].append(lr)
            self.metrics_history["throughput"].append(throughput)
            self.metrics_history["gpu_mem"].append(gpu_mem_gb)
            self.metrics_history["grad_norm"].append(grad_norm)

            # Append to recent steps log
            self._recent_steps.append({
                "step": step,
                "loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "throughput": throughput,
            })

            # Update progress bar
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, completed=step)

            # Rebuild and push new layout to the Live display
            if self._live is not None:
                try:
                    self._live.update(self._build_layout())
                except Exception:
                    pass  # never crash the training loop
        except Exception:
            pass  # safety net — dashboard errors must never affect training

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        # Three rows: top (chart + gpu), middle (progress + config), bottom (log)
        layout.split_column(
            Layout(name="top", ratio=4),
            Layout(name="middle", ratio=3),
            Layout(name="bottom", ratio=3),
        )

        layout["top"].split_row(
            Layout(name="loss_chart", ratio=3),
            Layout(name="gpu", ratio=2),
        )

        layout["middle"].split_row(
            Layout(name="progress", ratio=3),
            Layout(name="config", ratio=2),
        )

        layout["top"]["loss_chart"].update(self._loss_chart_panel())
        layout["top"]["gpu"].update(self._gpu_panel())
        layout["middle"]["progress"].update(self._progress_panel())
        layout["middle"]["config"].update(self._config_panel())
        layout["bottom"].update(self._recent_losses_table())

        return layout

    # ------------------------------------------------------------------
    # Individual panels
    # ------------------------------------------------------------------

    def _loss_chart_panel(self) -> Panel:
        """ASCII art loss curve panel."""
        losses = list(self.metrics_history["loss"])
        chart_str = ascii_chart(losses, width=50, height=7)

        current_loss = losses[-1] if losses else None
        min_loss = min(losses) if losses else None

        stats_parts = []
        if current_loss is not None:
            # Color-code current loss
            if current_loss < 2.0:
                color = "bold green"
            elif current_loss < 3.0:
                color = "green"
            elif current_loss < 4.0:
                color = "yellow"
            else:
                color = "red"
            stats_parts.append(f"[{color}]current: {current_loss:.4f}[/{color}]")
        if min_loss is not None:
            stats_parts.append(f"[dim]best: {min_loss:.4f}[/dim]")

        header = "  ".join(stats_parts) + "\n\n" if stats_parts else ""
        body = header + f"[dim]{chart_str}[/dim]"

        return Panel(
            body,
            title="[bold cyan]Loss Curve[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _gpu_panel(self) -> Panel:
        """GPU memory and utilization panel."""
        gpu = get_gpu_stats()

        if not gpu.get("available", False):
            body = "[dim]No GPU detected\n(CPU-only mode)[/dim]"
        else:
            used = gpu.get("memory_used_gb", 0.0)
            total = gpu.get("memory_total_gb", 0.0)
            reserved = gpu.get("memory_reserved_gb", 0.0)
            name = gpu.get("name", "GPU")

            # Shorten name: e.g. "NVIDIA GeForce RTX 4080" -> "RTX 4080"
            short_name = name
            for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD "):
                if name.startswith(prefix):
                    short_name = name[len(prefix):]
                    break

            # VRAM usage bar (10 chars)
            vram_pct = used / max(total, 1e-6)
            bar_len = round(vram_pct * 10)
            vram_bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]" + "░" * (10 - bar_len) + "[/dim]"

            # Use latest recorded gpu_mem if torch.cuda shows 0 (may not be updated yet)
            if used < 0.01 and self.metrics_history["gpu_mem"]:
                last_recorded = self.metrics_history["gpu_mem"][-1]
                if last_recorded > 0.01:
                    used = last_recorded
                    vram_pct = used / max(total, 1e-6)
                    bar_len = round(vram_pct * 10)
                    vram_bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]" + "░" * (10 - bar_len) + "[/dim]"

            body = (
                f"[bold]{short_name}[/bold]\n\n"
                f"[cyan]VRAM:[/cyan] {used:.1f} / {total:.1f} GB\n"
                f"  {vram_bar}  {vram_pct*100:.0f}%\n\n"
                f"[cyan]Reserved:[/cyan] {reserved:.1f} GB\n"
                f"[cyan]Util:[/cyan]    [dim]N/A[/dim] [dim](use nvidia-smi)[/dim]"
            )

        return Panel(
            body,
            title="[bold cyan]GPU[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _progress_panel(self) -> Panel:
        """Step counter, progress bar, throughput, and ETA."""
        step = self._current_step
        total = self.total_steps
        pct = step / total * 100 if total > 0 else 0.0

        # Progress bar (30 chars)
        filled = round(pct / 100 * 28)
        bar = "[cyan]" + "█" * filled + "[/cyan]" + "[dim]" + "░" * (28 - filled) + "[/dim]"

        # Throughput stats
        tps_hist = list(self.metrics_history["throughput"])
        if tps_hist:
            avg_tps = sum(tps_hist[-20:]) / len(tps_hist[-20:])
            tps_str = f"{avg_tps:,.0f} tok/s"
        else:
            avg_tps = 0.0
            tps_str = "—"

        # ETA
        elapsed = time.time() - self._start_time
        steps_done = max(step, 1)
        sec_per_step = elapsed / steps_done
        steps_left = total - step
        eta_secs = steps_left * sec_per_step
        eta_str = _format_eta(eta_secs)
        eta_abs = ""
        if eta_secs > 0 and math.isfinite(eta_secs):
            finish_time = datetime.now() + timedelta(seconds=eta_secs)
            eta_abs = f"  ({finish_time.strftime('%Y-%m-%d %H:%M')})"

        # Current LR
        lr_hist = list(self.metrics_history["lr"])
        lr_str = f"{lr_hist[-1]:.2e}" if lr_hist else "—"

        body = (
            f"[bold]Step:[/bold] [cyan]{step:,}[/cyan] / [dim]{total:,}[/dim]  "
            f"[dim]({pct:.1f}%)[/dim]\n\n"
            f"  {bar}\n\n"
            f"[cyan]ETA:[/cyan]        {eta_str}{eta_abs}\n"
            f"[cyan]Throughput:[/cyan] {tps_str}\n"
            f"[cyan]LR:[/cyan]         {lr_str}"
        )

        return Panel(
            body,
            title="[bold cyan]Progress[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _config_panel(self) -> Panel:
        """Model configuration summary panel."""
        cfg = self.config

        def _fmt_params(n: float | int | None) -> str:
            if n is None:
                return "?"
            n = float(n)
            if n >= 1e9:
                return f"{n/1e9:.1f}B"
            if n >= 1e6:
                return f"{n/1e6:.0f}M"
            return f"{n:.0f}"

        params = _fmt_params(cfg.get("model_params"))
        batch = cfg.get("effective_batch_size") or cfg.get("batch_size", "?")
        lr = cfg.get("learning_rate")
        lr_str = f"{lr:.2e}" if lr else "?"
        seq_len = cfg.get("seq_len") or cfg.get("max_seq_len", "?")
        precision = cfg.get("precision", "?")
        size_name = cfg.get("model_size_name", "?")

        body = (
            f"[cyan]Model:[/cyan]     {size_name} ({params} params)\n"
            f"[cyan]Batch:[/cyan]     {batch} (eff)\n"
            f"[cyan]LR:[/cyan]        {lr_str}\n"
            f"[cyan]Seq len:[/cyan]   {seq_len}\n"
            f"[cyan]Precision:[/cyan] {precision}"
        )

        return Panel(
            body,
            title="[bold cyan]Config[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _recent_losses_table(self) -> Panel:
        """Table of recent training steps."""
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Step", style="bold", justify="right", width=10)
        table.add_column("Loss", style="green", justify="right", width=8)
        table.add_column("LR", style="cyan", justify="right", width=10)
        table.add_column("Grad Norm", style="yellow", justify="right", width=10)
        table.add_column("Tok/s", style="white", justify="right", width=10)

        # Show last 5 entries, most recent first
        recent = list(self._recent_steps)[-5:]
        for entry in reversed(recent):
            step = entry["step"]
            loss = entry["loss"]
            lr = entry["lr"]
            grad_norm = entry["grad_norm"]
            tps = entry["throughput"]

            # Color-code loss
            if loss < 2.0:
                loss_style = "bold green"
            elif loss < 3.0:
                loss_style = "green"
            elif loss < 4.0:
                loss_style = "yellow"
            else:
                loss_style = "red"

            table.add_row(
                f"{step:,}",
                f"[{loss_style}]{loss:.4f}[/{loss_style}]",
                f"{lr:.2e}",
                f"{grad_norm:.3f}" if grad_norm else "—",
                f"{tps:,.0f}",
            )

        return Panel(
            table,
            title="[bold cyan]Recent Steps[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrainingDashboard":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
