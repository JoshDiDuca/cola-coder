# 79 - GPU Status Panel

## Overview

A standalone Rich panel showing live GPU metrics from `nvidia-smi`: VRAM usage, temperature, utilization, fan speed, and power draw. Displays warning thresholds (temp > 80°C, VRAM > 90%). Shows a historical chart of utilization during training. Supports multi-GPU setups. Updates live using Rich `Live`.

**Feature flag:** `--gpu-status` (standalone CLI command); also embedded in training dashboard (plan 77)

---

## Motivation

When training on an RTX 3080 (10GB) or RTX 4080 (16GB), VRAM pressure is the most common cause of unexpected crashes. Temperature monitoring prevents thermal damage during long overnight runs. A dedicated GPU status panel provides:

- **VRAM headroom visibility**: know if you're at 95% VRAM before trying a larger batch size
- **Thermal safety**: alert when temperature exceeds 80°C before throttling begins at 83°C
- **Power monitoring**: verify the GPU is actually drawing full power (if not, it may be power-limited)
- **Fan speed**: on some cards, fan failure is the first sign of cooling issues

This can run standalone (`cola-coder gpu-status`) or be embedded in the training dashboard (plan 77).

---

## Architecture / Design

### Display

```
╔═════════════════════════════════════════════════════════════╗
║ GPU 0: NVIDIA GeForce RTX 3080              Live GPU Status ║
╠═══════════════════╦═════════════════════╦═══════════════════╣
║ VRAM              ║ Temperature         ║ Utilization       ║
║ 7.8 / 10.0 GB     ║ 71°C  [OK]          ║ 94%               ║
║ ████████░░  78%   ║ ▃▄▅▆▆▇▇▇▇▇ trend    ║ ████████████████░ ║
╠═══════════════════╬═════════════════════╬═══════════════════╣
║ Power             ║ Fan Speed           ║ Clock Speed       ║
║ 247 / 320 W       ║ 58%   [normal]      ║ GFX: 1710 MHz     ║
║ ████████░░  77%   ║ ████████░░          ║ MEM: 9501 MHz     ║
╠═══════════════════╩═════════════════════╩═══════════════════╣
║ Utilization History                                         ║
║ ▁▁▂▃▄▅▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ avg: 94%          ║
╚═════════════════════════════════════════════════════════════╝
```

### Warning Thresholds

```python
THRESHOLDS = {
    "temperature_warn": 80,    # °C - yellow warning
    "temperature_critical": 85, # °C - red critical
    "vram_warn": 0.85,          # 85% - yellow
    "vram_critical": 0.92,      # 92% - red
    "utilization_low": 0.50,    # below 50% suggests data bottleneck
}
```

---

## Implementation Steps

### Step 1: nvidia-smi XML Parser (`monitoring/nvidia_smi.py`)

```python
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import deque
from threading import Thread, Event
import time

@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    driver_version: str
    cuda_version: str
    utilization_gpu_pct: int
    utilization_mem_pct: int
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int
    temperature_gpu_c: int
    temperature_gpu_slowdown_c: int
    temperature_gpu_max_c: int
    power_draw_w: float
    power_limit_w: float
    fan_speed_pct: int | None
    clock_graphics_mhz: int
    clock_mem_mhz: int
    pcie_tx_mbps: int
    pcie_rx_mbps: int

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mib / 1024

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mib / 1024

    @property
    def memory_fraction(self) -> float:
        return self.memory_used_mib / max(self.memory_total_mib, 1)

def parse_nvidia_smi_xml(xml_str: str) -> list[GpuInfo]:
    root = ET.fromstring(xml_str)

    def safe_int(el, default=0) -> int:
        if el is None or el.text is None:
            return default
        text = el.text.strip().split()[0]  # "1710 MHz" → "1710"
        return int(text) if text.lstrip('-').isdigit() else default

    def safe_float(el, default=0.0) -> float:
        if el is None or el.text is None:
            return default
        text = el.text.strip().split()[0]
        try:
            return float(text)
        except ValueError:
            return default

    gpus = []
    for i, gpu in enumerate(root.findall("gpu")):
        def g(path: str):
            parts = path.split("/")
            el = gpu
            for p in parts:
                el = el.find(p) if el is not None else None
            return el

        fan_el = g("fan_speed")
        fan_text = fan_el.text.strip() if fan_el is not None and fan_el.text else "N/A"
        fan_speed = int(fan_text.replace(" %", "")) if fan_text.replace(" %", "").isdigit() else None

        gpus.append(GpuInfo(
            index=i,
            uuid=gpu.find("uuid").text.strip() if gpu.find("uuid") is not None else "",
            name=gpu.find("product_name").text.strip() if gpu.find("product_name") is not None else f"GPU {i}",
            driver_version=root.find("driver_version").text.strip() if root.find("driver_version") is not None else "",
            cuda_version=root.find("cuda_version").text.strip() if root.find("cuda_version") is not None else "",
            utilization_gpu_pct=safe_int(g("utilization/gpu_util")),
            utilization_mem_pct=safe_int(g("utilization/memory_util")),
            memory_used_mib=safe_int(g("fb_memory_usage/used")),
            memory_free_mib=safe_int(g("fb_memory_usage/free")),
            memory_total_mib=safe_int(g("fb_memory_usage/total")),
            temperature_gpu_c=safe_int(g("temperature/gpu_temp")),
            temperature_gpu_slowdown_c=safe_int(g("temperature/gpu_temp_slow_threshold"), 83),
            temperature_gpu_max_c=safe_int(g("temperature/gpu_temp_max_threshold"), 95),
            power_draw_w=safe_float(g("power_readings/power_draw")),
            power_limit_w=safe_float(g("power_readings/power_limit"), 350.0),
            fan_speed_pct=fan_speed,
            clock_graphics_mhz=safe_int(g("clocks/graphics_clock")),
            clock_mem_mhz=safe_int(g("clocks/mem_clock")),
            pcie_tx_mbps=safe_int(g("pcie/tx_util")),
            pcie_rx_mbps=safe_int(g("pcie/rx_util")),
        ))
    return gpus

class NvidiaSmiPoller:
    """Background thread that polls nvidia-smi on a configurable interval."""

    def __init__(self, interval_sec: float = 2.0, history_len: int = 60):
        self.interval = interval_sec
        self._gpus: list[GpuInfo] = []
        self._history: dict[int, dict[str, deque]] = {}  # gpu_idx → metric → deque
        self._stop = Event()
        self._history_len = history_len
        self._thread = Thread(target=self._loop, daemon=True)

    def start(self):
        self._gpus = self._query()
        for gpu in self._gpus:
            self._history[gpu.index] = {
                "utilization": deque(maxlen=self._history_len),
                "temperature": deque(maxlen=self._history_len),
                "vram_gb": deque(maxlen=self._history_len),
            }
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(timeout=self.interval):
            self._gpus = self._query()
            for gpu in self._gpus:
                h = self._history.get(gpu.index, {})
                h.get("utilization", deque()).append(gpu.utilization_gpu_pct)
                h.get("temperature", deque()).append(gpu.temperature_gpu_c)
                h.get("vram_gb", deque()).append(gpu.memory_used_gb)

    def _query(self) -> list[GpuInfo]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q", "-x"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return parse_nvidia_smi_xml(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, ET.ParseError):
            pass
        return []

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    @property
    def gpus(self) -> list[GpuInfo]:
        return self._gpus

    def get_history(self, gpu_index: int, metric: str) -> list[float]:
        return list(self._history.get(gpu_index, {}).get(metric, []))
```

### Step 2: GPU Panel Renderer (`monitoring/gpu_panel.py`)

```python
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

THRESHOLDS = {
    "temperature_warn": 80,
    "temperature_critical": 85,
    "vram_warn": 0.85,
    "vram_critical": 0.92,
}

def _bar(frac: float, width: int = 18, warn: float = 0.85, crit: float = 0.92) -> str:
    filled = int(min(frac, 1.0) * width)
    color = "red" if frac >= crit else "yellow" if frac >= warn else "green"
    return f"[{color}]" + "█" * filled + "[/]" + "[dim]░[/]" * (width - filled)

def _sparkline(values: list[float], high_is_bad: bool = False) -> str:
    if not values:
        return ""
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v or 1
    blocks = "▁▂▃▄▅▆▇█"
    line = "".join(blocks[int((v - min_v) / rng * 7)] for v in values[-40:])
    return line

def build_gpu_panel(gpu: "GpuInfo", history: dict) -> Panel:
    vram_frac = gpu.memory_fraction
    vram_color = "red" if vram_frac >= THRESHOLDS["vram_critical"] else \
                 "yellow" if vram_frac >= THRESHOLDS["vram_warn"] else "green"

    temp_color = "red" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_critical"] else \
                 "yellow" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_warn"] else "green"

    temp_status = (
        "[red][CRITICAL][/]" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_critical"] else
        "[yellow][WARNING][/]" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_warn"] else
        "[green][OK][/]"
    )

    util_spark = _sparkline(list(history.get("utilization", [])))
    util_avg = (sum(history.get("utilization", [0])) / max(len(history.get("utilization", [1])), 1))

    fan_str = f"{gpu.fan_speed_pct}%" if gpu.fan_speed_pct is not None else "N/A"
    fan_bar = _bar(gpu.fan_speed_pct / 100, width=12) if gpu.fan_speed_pct is not None else "[dim]—[/]"

    content = Text.from_markup(
        f"[bold cyan]VRAM[/]\n"
        f"  {gpu.memory_used_gb:.1f} / {gpu.memory_total_gb:.1f} GB\n"
        f"  {_bar(vram_frac)} [{vram_color}]{vram_frac*100:.0f}%[/]\n\n"
        f"[bold cyan]Temperature[/]\n"
        f"  [{temp_color}]{gpu.temperature_gpu_c}°C[/]  {temp_status}\n"
        f"  (throttle at {gpu.temperature_gpu_slowdown_c}°C)\n\n"
        f"[bold cyan]Utilization[/]\n"
        f"  {_bar(gpu.utilization_gpu_pct/100)} {gpu.utilization_gpu_pct}%\n"
        f"  Mem: {gpu.utilization_mem_pct}%\n\n"
        f"[bold cyan]Power[/]\n"
        f"  {gpu.power_draw_w:.0f} / {gpu.power_limit_w:.0f} W\n"
        f"  {_bar(gpu.power_draw_w/max(gpu.power_limit_w,1), width=12)}\n\n"
        f"[bold cyan]Fan[/] {fan_str}  {fan_bar}\n\n"
        f"[bold cyan]Clocks[/]\n"
        f"  GFX: {gpu.clock_graphics_mhz} MHz\n"
        f"  MEM: {gpu.clock_mem_mhz} MHz\n\n"
        f"[bold cyan]Utilization History[/]\n"
        f"  [dim]{util_spark}[/]\n"
        f"  avg: [cyan]{util_avg:.0f}%[/]"
    )

    return Panel(
        content,
        title=f"[bold]{gpu.name}[/]  [dim]GPU {gpu.index}[/]",
        border_style="red" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_critical"] else
                     "yellow" if gpu.temperature_gpu_c >= THRESHOLDS["temperature_warn"] else "blue",
        box=box.ROUNDED,
    )

class GpuStatusDisplay:
    def __init__(self, poller: NvidiaSmiPoller):
        self.poller = poller
        self._console = Console()

    def run_live(self):
        """Run interactive live display until Ctrl+C."""
        with Live(self._render(), console=self._console, refresh_per_second=2) as live:
            try:
                while True:
                    live.update(self._render())
                    import time
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def _render(self):
        from rich.columns import Columns
        if not self.poller.gpus:
            return Panel("[yellow]No NVIDIA GPUs detected (nvidia-smi not available)[/]")
        panels = []
        for gpu in self.poller.gpus:
            hist = self.poller._history.get(gpu.index, {})
            panels.append(build_gpu_panel(gpu, hist))
        return Columns(panels, equal=True, expand=True) if len(panels) > 1 else panels[0]
```

### Step 3: CLI Command

```python
# cola-coder gpu-status
# cola-coder gpu-status --once    # print once and exit
# cola-coder gpu-status --json    # output JSON for scripting

@click.command("gpu-status")
@click.option("--once", is_flag=True, help="Print once and exit")
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.option("--gpu", default=0, help="GPU index to focus on")
@click.option("--interval", default=2.0, help="Polling interval in seconds")
def cmd_gpu_status(once, as_json, gpu, interval):
    poller = NvidiaSmiPoller(interval_sec=interval).start()
    import time
    time.sleep(0.5)  # Wait for first poll

    if as_json:
        import json
        gpu_data = [
            {
                "index": g.index, "name": g.name,
                "utilization_pct": g.utilization_gpu_pct,
                "memory_used_gb": g.memory_used_gb,
                "memory_total_gb": g.memory_total_gb,
                "temperature_c": g.temperature_gpu_c,
                "power_draw_w": g.power_draw_w,
                "fan_speed_pct": g.fan_speed_pct,
            }
            for g in poller.gpus
        ]
        print(json.dumps(gpu_data, indent=2))
        poller.stop()
        return

    display = GpuStatusDisplay(poller)
    if once:
        Console().print(display._render())
        poller.stop()
    else:
        display.run_live()
        poller.stop()
```

---

## Key Files to Modify

- `monitoring/nvidia_smi.py` - New file: XML parser and poller
- `monitoring/gpu_panel.py` - New file: Rich panel renderer
- `cli/gpu_cmd.py` - New file: CLI command
- `cli/main.py` - Register `gpu-status` command
- `monitoring/dashboard.py` (plan 77) - Import and reuse `NvidiaSmiPoller`

---

## Testing Strategy

1. **XML parser test**: feed a known nvidia-smi XML fixture (from real output), assert all fields parse correctly.
2. **Missing field test**: feed XML with missing `fan_speed` element, assert `fan_speed_pct=None` (not a crash).
3. **Threshold test**: create GpuInfo with `temperature_gpu_c=86`, assert `temp_color == "red"`.
4. **Poller background thread test**: start poller, sleep 3s, assert `gpus` list is populated.
5. **History recording test**: run poller for 6s with 2s interval, assert history deques have ~3 entries.
6. **nvidia-smi unavailable test**: mock subprocess to raise FileNotFoundError, assert poller returns empty list (no crash).
7. **Multi-GPU test**: feed XML with 2 GPUs, assert `parse_nvidia_smi_xml` returns list of length 2.

---

## Performance Considerations

- nvidia-smi XML query takes 50-100ms and runs in a background thread every 2s. Negligible.
- ET.fromstring on a typical nvidia-smi XML (~10KB) takes <1ms.
- History deques with `maxlen=60` store 60 data points per metric per GPU. At 2s polling interval, this is 2 minutes of history. Memory: 60 × 4 metrics × 2 GPUs × 8 bytes = ~4KB. Negligible.
- Rich panel rendering at 2 updates/sec: ~10-20ms/render. No training impact.
- For the standalone `gpu-status` command, a 2s polling interval is fine. When embedded in the training dashboard (plan 77), use the shared `NvidiaSmiPoller` instance with 5s interval.

---

## Dependencies

No new dependencies. Uses `subprocess`, `xml.etree.ElementTree` (stdlib), `rich` (already required).

Requires `nvidia-smi` in PATH (comes with NVIDIA drivers). On systems without NVIDIA GPU, all commands gracefully show "No NVIDIA GPUs detected."

---

## Estimated Complexity

**Low.** The XML parsing is straightforward (nvidia-smi XML is well-documented). The Rich panel layout is standard. The background thread pattern is the same as plan 77. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **XML output from nvidia-smi, not text**: `nvidia-smi -q -x` produces structured XML. Never parse the plain text output—it's inconsistent across driver versions and formatting differs between Linux and Windows. XML is stable.
- **Graceful failure on non-NVIDIA systems**: always wrap nvidia-smi calls in try/except for FileNotFoundError and TimeoutExpired. AMD GPU users or CPU-only machines should get a friendly "not available" message, not a traceback.
- **Temperature alert is the most important feature**: if only one thing works, make it the temperature alert. Thermal throttling is invisible and insidious. An alert at 80°C gives you time to improve airflow before throttling begins at 83°C.
- **JSON output for scripting**: `--json` output mode makes the panel scriptable. Users can wrap it in a monitoring script that sends alerts via Discord or Telegram when temperature exceeds a threshold.
- **Multi-GPU column layout**: for 2+ GPUs, use `rich.Columns` to display each GPU side-by-side. For 4+ GPUs, switch to a table layout since columns become too narrow.
