# 06 - Training Monitor Mode

## Overview

A live-updating CLI dashboard that runs during training, displaying real-time metrics in a Rich `Live` layout. Panels include: ASCII loss sparkline, current learning rate, step progress, tokens/second throughput, VRAM usage from nvidia-smi, ETA to completion, and the last 3 generated samples. Updated every N steps in a non-blocking background thread.

---

## Motivation

The current training loop outputs a single log line per step:

```
step 1200/10000 | loss 2.341 | lr 3.00e-4 | 1247 tok/s
```

This is functional but provides no visual trend, no VRAM awareness, and requires scrolling up to see earlier steps. The training monitor replaces this with a persistent dashboard that stays on-screen and updates in place, giving a complete picture of training health at a glance.

---

## Architecture / Design

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Cola-Coder Training Monitor   cola-small @ step 1200/10000  │
├───────────────────┬─────────────────────────────────────────┤
│  PROGRESS         │  METRICS                                │
│  ████████░░ 12%   │  Train Loss:  2.341   Val Loss: 2.517   │
│  ETA: 1h 23m      │  LR:          3.00e-4                   │
│  Elapsed: 12m     │  Tokens/sec:  1,247                     │
├───────────────────┴─────────────────────────────────────────┤
│  LOSS CURVE (last 50 steps)                                  │
│  3.0 ▄▄▃▂▂▁▁▁▁▂▁▁▁▁▁▁▁▂▁▁▁▂▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  │
│  2.0                                                         │
├──────────────────────────────────────────────────────────────┤
│  VRAM: RTX 3080  ████████░░░░░░░ 6.2/10GB                   │
├──────────────────────────────────────────────────────────────┤
│  RECENT SAMPLES                                              │
│  [1] "function add" → ": number { return a + b; }"          │
│  [2] "interface User" → "{ id: number; name: string; }"     │
│  [3] "const fetch" → "= async (url) => { const res = ..."   │
└──────────────────────────────────────────────────────────────┘
```

### Threading Model

```
Main thread: Training loop (GPU-bound)
        │
        ├─ Every N steps: pushes metrics to thread-safe queue
        │
Monitor thread: Reads from queue, updates Rich Live display
        │
        └─ Every M seconds: queries nvidia-smi for VRAM info
```

The monitor thread never touches the model or GPU. It only reads metrics that the training loop pushes to a `queue.Queue`.

---

## Implementation Steps

### Step 1: Metrics data structure (thread-safe)

```python
# src/training/monitor/metrics_buffer.py
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

@dataclass
class StepMetrics:
    step: int
    total_steps: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    tokens_per_sec: float
    elapsed_seconds: float
    recent_samples: list[tuple[str, str]] = field(default_factory=list)  # (prompt, completion)

class MetricsBuffer:
    """
    Thread-safe buffer between the training loop and the monitor display.
    Training loop pushes metrics; monitor thread consumes them.
    """
    def __init__(self, history_size: int = 100):
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._lock = threading.Lock()
        self._latest: Optional[StepMetrics] = None
        self._loss_history: deque = deque(maxlen=history_size)
        self._val_loss_history: deque = deque(maxlen=history_size)

    def push(self, metrics: StepMetrics) -> None:
        """Called by training thread. Non-blocking: drops if full."""
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass  # Monitor is falling behind; drop the oldest, not a concern

    def consume(self) -> list[StepMetrics]:
        """Called by monitor thread. Drains queue."""
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if items:
            with self._lock:
                self._latest = items[-1]
                for m in items:
                    self._loss_history.append(m.train_loss)
                    if m.val_loss is not None:
                        self._val_loss_history.append(m.val_loss)
        return items

    @property
    def latest(self) -> Optional[StepMetrics]:
        with self._lock:
            return self._latest

    @property
    def loss_history(self) -> list[float]:
        with self._lock:
            return list(self._loss_history)

    @property
    def val_loss_history(self) -> list[float]:
        with self._lock:
            return list(self._val_loss_history)
```

### Step 2: ASCII sparkline renderer

```python
# src/training/monitor/sparkline.py
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 50) -> str:
    """
    Generate an ASCII sparkline string from a list of float values.
    Normalizes to the range [min, max] and maps to 8 block chars.
    """
    if not values:
        return "─" * width

    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return SPARK_CHARS[3] * len(values)

    normalized = [(v - min_v) / (max_v - min_v) for v in values]
    chars = [SPARK_CHARS[int(n * (len(SPARK_CHARS) - 1))] for n in normalized]
    return "".join(chars)

def format_loss_panel_content(
    train_history: list[float],
    val_history: list[float],
    width: int = 50,
) -> str:
    lines = []
    if train_history:
        spark = sparkline(train_history, width)
        min_l = min(train_history)
        max_l = max(train_history)
        lines.append(f"[green]Train:[/green] {spark}  [{min_l:.3f} → {train_history[-1]:.3f}]")
    if val_history:
        spark = sparkline(val_history, width)
        lines.append(f"[yellow]Val:  [/yellow] {spark}  [val={val_history[-1]:.3f}]")
    if not lines:
        lines.append("[dim]No data yet...[/dim]")
    return "\n".join(lines)
```

### Step 3: VRAM poller (separate background thread)

```python
# src/training/monitor/vram_poller.py
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class VRAMInfo:
    gpu_name: str
    used_mb: int
    total_mb: int

    @property
    def utilization(self) -> float:
        return self.used_mb / self.total_mb if self.total_mb else 0.0

    @property
    def bar(self) -> str:
        filled = int(self.utilization * 20)
        return "█" * filled + "░" * (20 - filled)

class VRAMPoller:
    """Polls nvidia-smi every poll_interval seconds in a background thread."""
    def __init__(self, poll_interval: float = 2.0):
        self._interval = poll_interval
        self._latest: Optional[VRAMInfo] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def latest(self) -> Optional[VRAMInfo]:
        with self._lock:
            return self._latest

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._interval):
            info = self._query_nvidia_smi()
            if info:
                with self._lock:
                    self._latest = info

    def _query_nvidia_smi(self) -> Optional[VRAMInfo]:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return VRAMInfo(
                gpu_name=parts[0],
                used_mb=int(parts[1]),
                total_mb=int(parts[2]),
            )
        except Exception:
            return None
```

### Step 4: Rich Live display builder

```python
# src/training/monitor/display.py
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich import box
from rich.text import Text
from src.training.monitor.metrics_buffer import MetricsBuffer, StepMetrics
from src.training.monitor.sparkline import format_loss_panel_content
from src.training.monitor.vram_poller import VRAMPoller
from typing import Optional
import time

class TrainingMonitorDisplay:
    def __init__(self, buffer: MetricsBuffer, vram_poller: VRAMPoller):
        self.buffer = buffer
        self.vram = vram_poller
        self._start_time = time.time()

    def build_layout(self) -> Layout:
        metrics = self.buffer.latest
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top_row", size=6),
            Layout(name="loss_panel", size=4),
            Layout(name="vram_panel", size=3),
            Layout(name="samples_panel"),
        )
        layout["top_row"].split_row(
            Layout(name="progress"),
            Layout(name="metrics"),
        )

        layout["header"].update(self._build_header(metrics))
        layout["progress"].update(self._build_progress(metrics))
        layout["metrics"].update(self._build_metrics(metrics))
        layout["loss_panel"].update(self._build_loss_curve())
        layout["vram_panel"].update(self._build_vram())
        layout["samples_panel"].update(self._build_samples(metrics))
        return layout

    def _build_header(self, m: Optional[StepMetrics]) -> Panel:
        if m:
            title = f"Cola-Coder Training Monitor   step {m.step}/{m.total_steps}"
        else:
            title = "Cola-Coder Training Monitor   (waiting for data...)"
        return Panel(Text(title, justify="center", style="bold cyan"), height=3)

    def _build_progress(self, m: Optional[StepMetrics]) -> Panel:
        if not m:
            return Panel("[dim]Waiting...[/dim]", title="Progress")
        pct = m.step / m.total_steps
        bar_width = 20
        filled = int(pct * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.time() - self._start_time
        if pct > 0:
            eta_s = elapsed / pct * (1 - pct)
            eta_str = f"{int(eta_s // 3600)}h {int((eta_s % 3600) // 60)}m"
        else:
            eta_str = "?"
        content = (
            f"{bar} {pct:.0%}\n"
            f"ETA:     {eta_str}\n"
            f"Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s"
        )
        return Panel(content, title="Progress")

    def _build_metrics(self, m: Optional[StepMetrics]) -> Panel:
        if not m:
            return Panel("[dim]No data yet[/dim]", title="Metrics")
        val_str = f"{m.val_loss:.4f}" if m.val_loss is not None else "N/A"
        content = (
            f"Train Loss:  [green]{m.train_loss:.4f}[/green]\n"
            f"Val Loss:    [yellow]{val_str}[/yellow]\n"
            f"LR:          {m.learning_rate:.2e}\n"
            f"Tokens/sec:  {m.tokens_per_sec:,.0f}"
        )
        return Panel(content, title="Metrics")

    def _build_loss_curve(self) -> Panel:
        content = format_loss_panel_content(
            self.buffer.loss_history,
            self.buffer.val_loss_history,
        )
        return Panel(content, title="Loss Curve (last 100 steps)")

    def _build_vram(self) -> Panel:
        vram = self.vram.latest
        if not vram:
            return Panel("[dim]nvidia-smi not available[/dim]", title="VRAM")
        content = f"{vram.gpu_name}  {vram.bar}  {vram.used_mb}/{vram.total_mb}MB  ({vram.utilization:.0%})"
        color = "red" if vram.utilization > 0.9 else "yellow" if vram.utilization > 0.75 else "green"
        return Panel(f"[{color}]{content}[/{color}]", title="VRAM")

    def _build_samples(self, m: Optional[StepMetrics]) -> Panel:
        if not m or not m.recent_samples:
            return Panel("[dim]No samples yet[/dim]", title="Recent Samples")
        lines = []
        for i, (prompt, completion) in enumerate(m.recent_samples[-3:], 1):
            prompt_short = prompt[:40].replace("\n", " ")
            completion_short = completion[:60].replace("\n", " ")
            lines.append(f"[{i}] [cyan]{prompt_short}[/cyan] → [white]{completion_short}[/white]")
        return Panel("\n".join(lines), title="Recent Samples")
```

### Step 5: Monitor thread controller

```python
# src/training/monitor/controller.py
import threading
import time
from rich.live import Live
from rich.console import Console
from src.training.monitor.metrics_buffer import MetricsBuffer
from src.training.monitor.display import TrainingMonitorDisplay
from src.training.monitor.vram_poller import VRAMPoller

class TrainingMonitor:
    """
    Start and stop the live monitor display.
    Usage:
        monitor = TrainingMonitor(config)
        monitor.start()
        # ... training loop ...
        monitor.push_metrics(step_metrics)
        # ...
        monitor.stop()
    """
    def __init__(self, refresh_per_second: int = 2):
        self.buffer = MetricsBuffer()
        self.vram_poller = VRAMPoller(poll_interval=2.0)
        self._display = TrainingMonitorDisplay(self.buffer, self.vram_poller)
        self._refresh_rate = refresh_per_second
        self._live: Live = None
        self._thread: threading.Thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.vram_poller.start()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.vram_poller.stop()
        if self._thread:
            self._thread.join(timeout=3)

    def push_metrics(self, metrics) -> None:
        self.buffer.push(metrics)

    def _render_loop(self) -> None:
        with Live(
            self._display.build_layout(),
            refresh_per_second=self._refresh_rate,
            screen=True,  # Use alternate screen buffer
        ) as live:
            while not self._stop.wait(1.0 / self._refresh_rate):
                self.buffer.consume()
                live.update(self._display.build_layout())
```

### Step 6: Integration in training loop

```python
# src/trainer.py (additions)
from src.training.monitor.controller import TrainingMonitor
from src.training.monitor.metrics_buffer import StepMetrics

class Trainer:
    def train(self, ...):
        monitor = None
        if self.config.get("monitor_enabled", False):
            monitor = TrainingMonitor()
            monitor.start()

        try:
            for step in range(total_steps):
                # ... training step ...

                if monitor and step % self.config.get("monitor_update_interval", 10) == 0:
                    monitor.push_metrics(StepMetrics(
                        step=step,
                        total_steps=total_steps,
                        train_loss=loss.item(),
                        val_loss=self._last_val_loss,
                        learning_rate=self.optimizer.param_groups[0]["lr"],
                        tokens_per_sec=self._compute_throughput(),
                        elapsed_seconds=time.time() - start_time,
                        recent_samples=self._recent_samples[-3:],
                    ))
        finally:
            if monitor:
                monitor.stop()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/training/monitor/__init__.py` | New package |
| `src/training/monitor/metrics_buffer.py` | Thread-safe metrics queue |
| `src/training/monitor/sparkline.py` | ASCII sparkline renderer |
| `src/training/monitor/vram_poller.py` | Background nvidia-smi poller |
| `src/training/monitor/display.py` | Rich Layout builder |
| `src/training/monitor/controller.py` | Monitor thread lifecycle |
| `src/trainer.py` | Push metrics to monitor, start/stop |
| `configs/*.yaml` | Add `monitor_enabled`, `monitor_update_interval` |

---

## Testing Strategy

- **Sparkline unit test**: `sparkline([3.0, 2.5, 2.0, 1.5], width=4)` should return 4 block chars decreasing in height
- **MetricsBuffer thread test**: Push 100 items from a producer thread, consume from another thread, verify no items lost
- **VRAMPoller mock test**: Mock subprocess, verify VRAMInfo parsed correctly from mock nvidia-smi output
- **Display test**: Create a `TrainingMonitorDisplay` with canned metrics, call `build_layout()`, verify no exceptions
- **Controller integration test**: Start monitor, push 5 metrics, stop monitor — verify clean exit
- **Screen restore test**: Verify `screen=True` on Rich Live returns terminal to normal state after `monitor.stop()`

---

## Performance Considerations

- Monitor thread uses `daemon=True` — it is automatically killed if the training process exits
- `screen=True` in Rich Live uses the terminal's alternate screen buffer, so monitor output does not pollute the main scroll buffer
- VRAM polling every 2 seconds is negligible overhead (subprocess call takes ~50ms)
- Metrics buffer has `maxsize=50` — if the monitor falls behind, the training loop drops metrics rather than blocking
- `refresh_per_second=2` means 2 redraws/second. At 1000 steps/sec training, only 1 in 500 steps actually updates the display

---

## Dependencies

| Package | Use | Install |
|---------|-----|---------|
| `rich` | Live, Layout, Panel, etc. | Already installed |
| `nvidia-smi` | VRAM info | System tool (included with NVIDIA drivers) |
| Python `queue`, `threading` | Thread communication | stdlib |

No new Python packages.

---

## Estimated Complexity

**Medium** — 2 days.

- MetricsBuffer + thread safety: 3 hours
- Sparkline + display panels: 3 hours
- VRAMPoller: 2 hours
- Controller lifecycle: 2 hours
- Trainer integration: 2 hours
- Testing + screen restore edge cases: 3 hours

Total: ~15 hours

---

## 2026 Best Practices

- **Daemon threads**: Always make monitoring threads daemon threads. If training crashes, they die automatically without blocking process exit.
- **Alternate screen buffer**: `screen=True` in Rich Live means the dashboard lives in an alternate terminal screen and the normal scroll buffer is untouched. This is essential for a good UX.
- **Non-blocking metrics push**: The training loop must never block waiting for the monitor. Use `put_nowait()` and drop metrics if the buffer is full — the monitor is optional instrumentation.
- **Separate VRAM polling thread**: VRAM polling has latency (subprocess call). Never do this in the training loop directly. Polling in a background thread at 2-second intervals is the right pattern.
- **Graceful degradation**: If Rich Live fails (e.g., not in a TTY), fall back to the line-by-line log output. The monitor is cosmetic; training must continue regardless.
