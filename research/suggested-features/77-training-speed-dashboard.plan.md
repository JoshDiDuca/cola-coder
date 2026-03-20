# 77 - Training Speed Dashboard

## Overview

A real-time CLI panel showing tokens/sec, estimated time remaining, GPU utilization, and VRAM usage in a single Rich Layout. Updates every training step. Shows a historical sparkline of throughput. Detects bottlenecks by comparing data loading time vs compute time vs memory bandwidth. Data sources: training loop counters, `nvidia-smi`, and `torch.cuda`.

**Feature flag:** `config.dashboard.enabled` (default: `true` when TTY detected)

---

## Motivation

Training a model without visibility into GPU utilization is like driving without a speedometer. Common failure modes this dashboard prevents:

- **Data loading bottleneck**: GPU utilization is 40% because DataLoader is slow. Invisible without monitoring.
- **VRAM fragmentation**: memory usage creeps up over steps until OOM. Detectable early with live VRAM display.
- **Thermal throttling**: GPU temperature hits 83°C, clock speed drops, tokens/sec falls. Obvious on dashboard, invisible otherwise.
- **Unexpected speed degradation**: a code change made the model 30% slower. Immediately visible in the throughput sparkline.

---

## Architecture / Design

### Layout (Rich)

```
╔═══════════════════════════════════════════════════════════════╗
║  Cola-Coder Training Dashboard   step 5,234 / 10,000          ║
╠═══════════════════╦═══════════════════╦═══════════════════════╣
║ Throughput        ║ GPU Compute       ║ GPU Memory            ║
║ 12,847 tok/s      ║ Util: 94%  ████▓  ║ 7.8 / 10.0 GB  ████░  ║
║ 384 steps/min     ║ Temp: 71°C [OK]   ║ Alloc: 6.9 GB         ║
║ ETA: 0:24:15      ║ Power: 247W       ║ Reserved: 7.8 GB      ║
╠═══════════════════╩═══════════════════╬═══════════════════════╣
║ Throughput History                    ║ Bottleneck Analysis   ║
║ ▃▄▅▆▇▇▇▆▇▇▇▇▆▆▇▇▇▇▇▇ 12.8k tok/s     ║ Data load:  8ms  ░░   ║
║                                       ║ Forward:   42ms  ████ ║
║                                       ║ Backward:  51ms  █████║
║                                       ║ Optimizer: 12ms  █░   ║
╚═══════════════════════════════════════╩═══════════════════════╝
```

### Data Collection Frequency

| Data source | Update frequency | Method |
|-------------|-----------------|--------|
| Tokens/sec, step count | Every step | Training loop counters |
| GPU utilization, temp | Every 5s | `nvidia-smi` subprocess |
| VRAM usage | Every step | `torch.cuda.memory_stats()` |
| Bottleneck timing | Every step | Python `time.perf_counter()` timers |

---

## Implementation Steps

### Step 1: GPU Monitor (`monitoring/gpu_monitor.py`)

```python
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from threading import Thread, Event
import time

@dataclass
class GpuStats:
    index: int
    name: str
    utilization_pct: int
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int
    power_draw_w: float
    fan_speed_pct: int | None = None
    clock_graphics_mhz: int = 0

class GpuMonitor:
    """Background thread that polls nvidia-smi for GPU stats."""

    def __init__(self, poll_interval_sec: float = 5.0):
        self.poll_interval = poll_interval_sec
        self._stats: list[GpuStats] = []
        self._stop = Event()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while not self._stop.wait(timeout=self.poll_interval):
            self._stats = self._query_nvidia_smi()

    def _query_nvidia_smi(self) -> list[GpuStats]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q", "-x"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []
            return self._parse_xml(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def _parse_xml(self, xml_str: str) -> list[GpuStats]:
        root = ET.fromstring(xml_str)
        stats = []
        for i, gpu in enumerate(root.findall("gpu")):
            def get(tag: str, default="0") -> str:
                el = gpu.find(tag)
                return el.text.strip() if el is not None and el.text else default

            def get_nested(path: str, default="0") -> str:
                parts = path.split("/")
                el = gpu
                for p in parts:
                    el = el.find(p) if el is not None else None
                return el.text.strip() if el is not None and el.text else default

            util_str = get_nested("utilization/gpu_util", "0 %").replace(" %", "")
            temp_str = get_nested("temperature/gpu_temp", "0 C").replace(" C", "")
            mem_used = get_nested("fb_memory_usage/used", "0 MiB").replace(" MiB", "")
            mem_total = get_nested("fb_memory_usage/total", "0 MiB").replace(" MiB", "")
            power_str = get_nested("power_readings/power_draw", "0.00 W").replace(" W", "")
            fan_str = get("fan_speed", "N/A %").replace(" %", "")

            stats.append(GpuStats(
                index=i,
                name=get("product_name"),
                utilization_pct=int(util_str) if util_str.isdigit() else 0,
                memory_used_mb=int(mem_used) if mem_used.isdigit() else 0,
                memory_total_mb=int(mem_total) if mem_total.isdigit() else 0,
                temperature_c=int(temp_str) if temp_str.isdigit() else 0,
                power_draw_w=float(power_str) if power_str.replace(".", "").isdigit() else 0.0,
                fan_speed_pct=int(fan_str) if fan_str.isdigit() else None,
            ))
        return stats

    def get_stats(self, gpu_index: int = 0) -> GpuStats | None:
        if not self._stats:
            self._stats = self._query_nvidia_smi()
        return self._stats[gpu_index] if self._stats and gpu_index < len(self._stats) else None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
```

### Step 2: Throughput Tracker (`monitoring/throughput_tracker.py`)

```python
import time
from collections import deque

class ThroughputTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._step_times: deque[tuple[float, int]] = deque(maxlen=window_size)
        self._start_time = time.perf_counter()
        self._total_steps = 0
        self._total_tokens = 0
        self.sparkline_history: deque[float] = deque(maxlen=25)

    def record_step(self, tokens_in_batch: int):
        t = time.perf_counter()
        self._step_times.append((t, tokens_in_batch))
        self._total_steps += 1
        self._total_tokens += tokens_in_batch

    @property
    def tokens_per_sec(self) -> float:
        if len(self._step_times) < 2:
            return 0.0
        oldest_t, _ = self._step_times[0]
        newest_t, _ = self._step_times[-1]
        window_time = newest_t - oldest_t
        if window_time <= 0:
            return 0.0
        window_tokens = sum(t for _, t in list(self._step_times)[1:])
        tps = window_tokens / window_time
        self.sparkline_history.append(tps)
        return tps

    @property
    def steps_per_min(self) -> float:
        if len(self._step_times) < 2:
            return 0.0
        oldest_t, _ = self._step_times[0]
        newest_t, _ = self._step_times[-1]
        window_time = newest_t - oldest_t
        return (len(self._step_times) - 1) / window_time * 60 if window_time > 0 else 0.0

    def eta_sec(self, current_step: int, total_steps: int) -> float:
        spm = self.steps_per_min
        if spm <= 0:
            return float("inf")
        remaining = total_steps - current_step
        return remaining / (spm / 60)

    def sparkline(self) -> str:
        hist = list(self.sparkline_history)
        if not hist:
            return ""
        min_v, max_v = min(hist), max(hist)
        rng = max_v - min_v or 1
        blocks = "▁▂▃▄▅▆▇█"
        return "".join(blocks[min(7, int((v - min_v) / rng * 7))] for v in hist)
```

### Step 3: Bottleneck Timer (`monitoring/bottleneck_timer.py`)

```python
import time
from contextlib import contextmanager
from collections import defaultdict, deque

class BottleneckTimer:
    """Context-manager based profiler for training phases."""

    def __init__(self, window: int = 20):
        self._timings: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    @contextmanager
    def time(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[phase].append((time.perf_counter() - t0) * 1000)

    def get_avg_ms(self, phase: str) -> float:
        vals = list(self._timings[phase])
        return sum(vals) / len(vals) if vals else 0.0

    def get_all_avg(self) -> dict[str, float]:
        return {phase: self.get_avg_ms(phase) for phase in self._timings}
```

### Step 4: Dashboard Renderer (`monitoring/dashboard.py`)

```python
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.console import Console
from rich.text import Text
from rich import box
import torch

def _bar(value: float, width: int = 20, color: str = "blue") -> str:
    filled = int(min(value, 1.0) * width)
    return f"[{color}]" + "█" * filled + "[/]" + "░" * (width - filled)

def _format_eta(secs: float) -> str:
    if secs == float("inf"):
        return "calculating..."
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

class TrainingDashboard:
    def __init__(
        self,
        total_steps: int,
        gpu_monitor: "GpuMonitor",
        throughput: "ThroughputTracker",
        bottleneck: "BottleneckTimer",
    ):
        self.total_steps = total_steps
        self.gpu_monitor = gpu_monitor
        self.throughput = throughput
        self.bottleneck = bottleneck
        self._console = Console()
        self._live = None

    def __enter__(self):
        self._live = Live(
            self._build_layout(step=0, loss=0.0),
            console=self._console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._live.__exit__(*args)

    def update(self, step: int, loss: float):
        if self._live:
            self._live.update(self._build_layout(step, loss))

    def _build_layout(self, step: int, loss: float) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top_row", size=7),
            Layout(name="bottom_row"),
        )
        layout["top_row"].split_row(
            Layout(name="throughput"),
            Layout(name="gpu_compute"),
            Layout(name="gpu_memory"),
        )
        layout["bottom_row"].split_row(
            Layout(name="history"),
            Layout(name="bottleneck"),
        )

        # Header
        pct = step / max(self.total_steps, 1)
        layout["header"].update(Panel(
            f"[bold cyan]Cola-Coder Training Dashboard[/]  "
            f"step [bold]{step:,}[/] / {self.total_steps:,}  "
            f"[dim]{_bar(pct, 30, 'green')} {pct*100:.1f}%[/]  "
            f"loss: [yellow]{loss:.4f}[/]",
            box=box.HORIZONTALS,
        ))

        # Throughput panel
        tps = self.throughput.tokens_per_sec
        spm = self.throughput.steps_per_min
        eta = self.throughput.eta_sec(step, self.total_steps)
        layout["throughput"].update(Panel(
            Text.from_markup(
                f"[bold cyan]{tps:,.0f}[/] tok/s\n"
                f"[cyan]{spm:.0f}[/] steps/min\n"
                f"ETA: [bold]{_format_eta(eta)}[/]"
            ),
            title="Throughput",
        ))

        # GPU compute panel
        gpu = self.gpu_monitor.get_stats(0)
        if gpu:
            util = gpu.utilization_pct / 100
            temp = gpu.temperature_c
            temp_color = "red" if temp > 85 else "yellow" if temp > 75 else "green"
            layout["gpu_compute"].update(Panel(
                Text.from_markup(
                    f"Util: [bold]{gpu.utilization_pct}%[/] {_bar(util, 12)}\n"
                    f"Temp: [{temp_color}]{temp}°C[/]\n"
                    f"Power: {gpu.power_draw_w:.0f}W"
                ),
                title="GPU Compute",
            ))
        else:
            layout["gpu_compute"].update(Panel("[dim]nvidia-smi unavailable[/]", title="GPU Compute"))

        # GPU memory panel
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total_gb = (gpu.memory_total_mb / 1024) if gpu else 10.0
            mem_frac = reserved / total_gb
            mem_color = "red" if mem_frac > 0.9 else "yellow" if mem_frac > 0.75 else "green"
            layout["gpu_memory"].update(Panel(
                Text.from_markup(
                    f"[{mem_color}]{reserved:.1f}[/] / {total_gb:.1f} GB\n"
                    f"{_bar(mem_frac, 12, mem_color)}\n"
                    f"Alloc: {alloc:.1f} GB"
                ),
                title="VRAM",
            ))
        else:
            layout["gpu_memory"].update(Panel("[dim]CUDA unavailable[/]", title="VRAM"))

        # History sparkline
        spark = self.throughput.sparkline()
        tps_hist = list(self.throughput.sparkline_history)
        last = f"{tps_hist[-1]:,.0f}" if tps_hist else "—"
        layout["history"].update(Panel(
            Text.from_markup(
                f"[dim]{spark}[/]\n"
                f"Latest: [cyan]{last}[/] tok/s"
            ),
            title="Throughput History",
        ))

        # Bottleneck analysis
        phases = self.bottleneck.get_all_avg()
        total_phase_time = sum(phases.values()) or 1
        bottleneck_lines = []
        for phase, ms in sorted(phases.items(), key=lambda x: -x[1]):
            frac = ms / total_phase_time
            bottleneck_lines.append(
                f"[dim]{phase:<12}[/] {ms:5.0f}ms {_bar(frac, 10)}"
            )
        layout["bottleneck"].update(Panel(
            Text.from_markup("\n".join(bottleneck_lines) or "[dim]no data[/]"),
            title="Bottleneck",
        ))

        return layout
```

### Step 5: Trainer Integration

```python
# training/trainer.py

from monitoring.gpu_monitor import GpuMonitor
from monitoring.throughput_tracker import ThroughputTracker
from monitoring.bottleneck_timer import BottleneckTimer
from monitoring.dashboard import TrainingDashboard

gpu_monitor = GpuMonitor(poll_interval_sec=5.0)
throughput = ThroughputTracker()
bottleneck = BottleneckTimer()

ctx = TrainingDashboard(total_steps, gpu_monitor, throughput, bottleneck) if use_dashboard else nullcontext()

with ctx as dashboard:
    for step in range(total_steps):
        with bottleneck.time("data_load"):
            batch = next(data_iter)

        with bottleneck.time("forward"):
            loss = model(batch)

        with bottleneck.time("backward"):
            loss.backward()

        with bottleneck.time("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

        throughput.record_step(batch_tokens)
        if dashboard:
            dashboard.update(step, loss.item())

gpu_monitor.stop()
```

---

## Key Files to Modify

- `monitoring/gpu_monitor.py` - New file
- `monitoring/throughput_tracker.py` - New file
- `monitoring/bottleneck_timer.py` - New file
- `monitoring/dashboard.py` - New file
- `training/trainer.py` - Integrate all monitoring
- `config/training.yaml` - Add `dashboard` section
- `cli/train_cmd.py` - Add `--no-dashboard` flag

---

## Testing Strategy

1. **GPU monitor parse test**: feed known nvidia-smi XML fixture, assert parsed stats match expected values.
2. **Throughput calculation test**: record 10 steps with known batch sizes and timing, assert tokens/sec is within 5% of expected.
3. **ETA test**: with 100/1000 steps done at 10 steps/sec, assert ETA is ~900s.
4. **Sparkline test**: record 20 throughput samples, assert sparkline string has length 20.
5. **Bottleneck timer test**: time a sleep(0.05) phase, assert `get_avg_ms("phase") ≈ 50 ± 5ms`.
6. **Dashboard render test**: instantiate with mock data, call `_build_layout`, assert no exceptions.
7. **nvidia-smi unavailable test**: mock subprocess to raise FileNotFoundError, assert `get_stats()` returns None (no crash).

---

## Performance Considerations

- `nvidia-smi -q -x` takes ~50-100ms and runs in a background thread every 5s. No impact on training throughput.
- `torch.cuda.memory_allocated()` is near-instant (reads a counter). Safe to call every step.
- Rich `Live.update()` with a complex layout takes ~5-20ms. At 10 steps/sec this would add 5-20% overhead. Reduce update frequency to every 5 steps: `if step % 5 == 0: dashboard.update(...)`.
- The bottleneck timer adds 2 `time.perf_counter()` calls per phase (4 phases = 8 calls/step). Each is ~100ns. Negligible at any training speed.
- On Windows, `nvidia-smi` may be at a different PATH. Document the requirement and test on both platforms.

---

## Dependencies

No new Python dependencies. Uses `subprocess`, `xml.etree.ElementTree`, `threading`, `torch.cuda`, `rich`.

---

## Estimated Complexity

**Medium.** The individual components are clean. The main complexity is the Rich Layout construction (nested splits require careful ordering) and the background thread management for GPU polling. Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Background thread for polling, not blocking**: nvidia-smi takes 50-100ms. Always poll it in a background daemon thread, not in the training loop. The main loop reads the last known value.
- **Deque-based windowing for throughput**: use a fixed-size deque (window of last 50 steps) for tokens/sec calculation. This gives a stable, non-noisy reading that responds to recent changes without being affected by startup artifacts.
- **Bottleneck visibility is actionable**: the bottleneck panel directly tells you what to optimize. If "data_load" dominates, increase DataLoader workers. If "forward" dominates, reduce model size or batch size. Make these suggestions appear when the bottleneck exceeds a threshold.
- **VRAM headroom warning**: warn when VRAM usage exceeds 90%. An OOM error mid-training is worse than a proactive warning. Display `[WARNING: VRAM > 90%]` in red when threshold is crossed.
- **Non-TTY fallback**: when stdout is not a TTY (CI, remote SSH without terminal), fall back to periodic single-line log messages: `step=500 tps=12847 loss=2.341 vram=7.8GB`.
