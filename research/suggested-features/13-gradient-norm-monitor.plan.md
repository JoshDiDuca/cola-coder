# 13 - Gradient Norm Monitor

## Overview

Display gradient norms per layer during training to detect exploding or vanishing gradients. Computes the L2 norm of gradients for each named parameter group after the backward pass but before the optimizer step and gradient clipping. Color-codes the display (green = healthy, yellow = warning, red = critical). Logs to training metrics and optionally displays in a Rich panel.

---

## Motivation

Gradient problems are invisible without explicit monitoring. Symptoms that should trigger investigation:
- **Exploding gradients**: Loss spikes upward suddenly; training becomes unstable
- **Vanishing gradients**: Early layers stop learning while later layers appear fine; the model learns shallow patterns only
- **Layer-specific issues**: One layer's gradients are 1000x larger than others, dominating all learning

The gradient norm monitor provides a per-layer view of gradient health that the global gradient norm (commonly logged) does not provide.

**When is this most useful?**
- After a loss spike: identify which layer caused it
- When a model stops improving: check for vanishing gradients in early embedding/attention layers
- When adding a new architectural component: verify its gradients are in the same range as existing layers

---

## Architecture / Design

### Norm Categories

| Norm Value | Category | Color | Action |
|------------|----------|-------|--------|
| < 1e-7 | Vanishing | Red | Alert |
| 1e-7 to 1e-4 | Weak | Yellow | Warn |
| 1e-4 to 10.0 | Healthy | Green | None |
| 10.0 to 100.0 | Elevated | Yellow | Warn |
| > 100.0 | Exploding | Red | Alert |

These thresholds are soft guidelines. The important signal is *relative* differences between layers and *trends over time*.

### Display Modes

1. **Summary mode** (default, per step): Single line showing mean norm, max norm, and any critical layers
2. **Full table mode** (on demand, or every N steps): Per-layer breakdown table
3. **Training monitor integration** (feature 06): Add a gradient health indicator to the dashboard

---

## Implementation Steps

### Step 1: Gradient norm computation

```python
# src/training/gradient_monitor.py
import torch
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

@dataclass
class LayerGradStats:
    name: str
    norm: float
    num_params: int
    has_grad: bool
    is_frozen: bool

    @property
    def norm_per_param(self) -> float:
        return self.norm / math.sqrt(self.num_params) if self.num_params > 0 else 0.0

    @property
    def category(self) -> str:
        if not self.has_grad or self.is_frozen:
            return "frozen"
        if self.norm < 1e-7:
            return "vanishing"
        if self.norm < 1e-4:
            return "weak"
        if self.norm <= 10.0:
            return "healthy"
        if self.norm <= 100.0:
            return "elevated"
        return "exploding"

    @property
    def color(self) -> str:
        return {
            "frozen": "dim",
            "vanishing": "bold red",
            "weak": "yellow",
            "healthy": "green",
            "elevated": "yellow",
            "exploding": "bold red",
        }[self.category]


def compute_gradient_norms(model: torch.nn.Module) -> list[LayerGradStats]:
    """
    Compute per-parameter gradient norms after backward pass.
    Groups by logical layer name (first two name components).
    """
    # Per-parameter stats
    param_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            norm = param.grad.norm(2).item() if has_grad else 0.0
            param_stats[name] = LayerGradStats(
                name=name,
                norm=norm,
                num_params=param.numel(),
                has_grad=has_grad,
                is_frozen=not param.requires_grad,
            )

    return list(param_stats.values())


def aggregate_by_layer(param_stats: list[LayerGradStats]) -> list[LayerGradStats]:
    """
    Aggregate individual parameter norms into logical layer norms.
    e.g., "layers.0.attn.q_proj.weight" + "layers.0.attn.q_proj.bias"
    → "layers.0.attn"
    """
    layer_sums: dict[str, dict] = defaultdict(lambda: {"sq_sum": 0.0, "n_params": 0, "any_grad": False})

    for stat in param_stats:
        # Group by first 3 name components
        parts = stat.name.split(".")
        layer_key = ".".join(parts[:3]) if len(parts) >= 3 else stat.name

        layer_sums[layer_key]["sq_sum"] += stat.norm ** 2
        layer_sums[layer_key]["n_params"] += stat.num_params
        if stat.has_grad:
            layer_sums[layer_key]["any_grad"] = True

    results = []
    for layer_name, agg in layer_sums.items():
        results.append(LayerGradStats(
            name=layer_name,
            norm=math.sqrt(agg["sq_sum"]),
            num_params=agg["n_params"],
            has_grad=agg["any_grad"],
            is_frozen=False,
        ))

    return sorted(results, key=lambda s: s.name)
```

### Step 2: Summary line (per-step)

```python
# src/training/gradient_monitor.py (continued)
from rich.console import Console
from rich.text import Text

console = Console()

def format_summary_line(layer_stats: list[LayerGradStats]) -> str:
    """
    One-line summary: global norm, per-group critical flags.
    Example: "grad_norm: 0.432 | max: 1.23 (layers.5.attn) | ⚠ layers.0: 1e-9 (vanishing)"
    """
    healthy = [s for s in layer_stats if s.category == "healthy"]
    weak = [s for s in layer_stats if s.category == "weak"]
    vanishing = [s for s in layer_stats if s.category == "vanishing"]
    exploding = [s for s in layer_stats if s.category == "exploding"]

    all_norms = [s.norm for s in layer_stats if s.has_grad]
    if not all_norms:
        return "grad_norm: no gradients"

    global_norm = math.sqrt(sum(n**2 for n in all_norms))
    max_norm = max(all_norms)
    max_layer = max(layer_stats, key=lambda s: s.norm if s.has_grad else 0).name

    parts = [f"global={global_norm:.3f}", f"max={max_norm:.3f}({max_layer.split('.')[-2]})"]

    if exploding:
        names = ", ".join(s.name for s in exploding[:2])
        parts.append(f"[bold red]EXPLODING: {names}[/bold red]")
    if vanishing:
        names = ", ".join(s.name for s in vanishing[:2])
        parts.append(f"[bold red]VANISHING: {names}[/bold red]")
    if weak:
        parts.append(f"[yellow]weak: {len(weak)} layers[/yellow]")

    return "  ".join(parts)
```

### Step 3: Full layer breakdown table

```python
# src/training/gradient_monitor.py (continued)
from rich.table import Table
from rich.panel import Panel
from rich import box

def render_layer_table(layer_stats: list[LayerGradStats], step: int) -> Panel:
    """Rich table showing per-layer gradient norms."""
    table = Table(
        title=f"Gradient Norms @ Step {step}",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("Layer", style="cyan", no_wrap=True)
    table.add_column("Norm", justify="right", width=10)
    table.add_column("Norm/√params", justify="right", width=14)
    table.add_column("Params", justify="right", width=10)
    table.add_column("Status", width=12)

    for stat in layer_stats:
        if stat.is_frozen or not stat.has_grad:
            continue
        norm_str = f"{stat.norm:.4e}"
        npp_str = f"{stat.norm_per_param:.4e}"
        param_str = f"{stat.num_params/1000:.1f}K"
        status = f"[{stat.color}]{stat.category}[/{stat.color}]"
        table.add_row(
            stat.name,
            f"[{stat.color}]{norm_str}[/{stat.color}]",
            npp_str,
            param_str,
            status,
        )

    return Panel(table)
```

### Step 4: History tracking and logging

```python
# src/training/gradient_monitor.py (continued)
from pathlib import Path
import json

@dataclass
class GradMonitor:
    run_dir: Path
    check_interval: int = 100           # Compute every N steps
    full_display_interval: int = 1000   # Full table every M steps
    explode_threshold: float = 100.0
    vanish_threshold: float = 1e-7

    _history: list[dict] = field(default_factory=list)
    _alert_count: int = 0

    def check(self, model: torch.nn.Module, step: int) -> Optional[str]:
        """
        Call after backward pass, before optimizer step.
        Returns summary string or None.
        """
        if step % self.check_interval != 0:
            return None

        param_stats = compute_gradient_norms(model)
        layer_stats = aggregate_by_layer(param_stats)

        # Check for critical issues
        critical = [s for s in layer_stats if s.category in ("exploding", "vanishing")]

        # Log
        entry = {
            "step": step,
            "global_norm": math.sqrt(sum(s.norm**2 for s in layer_stats if s.has_grad)),
            "max_norm": max((s.norm for s in layer_stats if s.has_grad), default=0),
            "critical_layers": [{"name": s.name, "norm": s.norm, "category": s.category}
                                 for s in critical],
        }
        self._history.append(entry)

        if step % 1000 == 0:
            self._save_history()

        if step % self.full_display_interval == 0:
            console.print(render_layer_table(layer_stats, step))
        else:
            summary = format_summary_line(layer_stats)
            return summary  # Caller prints this inline

        return None

    def _save_history(self) -> None:
        path = self.run_dir / "gradient_log.json"
        path.write_text(json.dumps(self._history[-1000:], indent=2))  # Keep last 1000 entries
```

### Step 5: Training loop integration

```python
# src/trainer.py (additions)
from src.training.gradient_monitor import GradMonitor

class Trainer:
    def __init__(self, ...):
        # ...
        self._grad_monitor = GradMonitor(
            run_dir=run_dir,
            check_interval=config.get("grad_monitor_interval", 100),
            full_display_interval=config.get("grad_monitor_full_interval", 1000),
        ) if config.get("grad_monitor_enabled", False) else None

    def _train_step(self, batch, step: int):
        # ... forward pass ...
        loss.backward()

        # Check gradients BEFORE clipping
        if self._grad_monitor:
            summary = self._grad_monitor.check(self.model, step)
            if summary:
                # Append to step log line
                console.print(f"  grad: {summary}")

        # Gradient clipping (existing)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])

        self.optimizer.step()
        self.optimizer.zero_grad()
```

### Step 6: Config additions

```yaml
training:
  grad_monitor_enabled: false        # Enable gradient norm monitoring
  grad_monitor_interval: 100         # Check every N steps
  grad_monitor_full_interval: 1000   # Show full table every N steps
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/training/gradient_monitor.py` | New: LayerGradStats, compute norms, display |
| `src/trainer.py` | Call `grad_monitor.check()` after backward pass |
| `configs/*.yaml` | Add `training.grad_monitor_enabled` |
| `runs/<run>/gradient_log.json` | New: per-step gradient norm history |
| `src/menu/menus/tools_menu.py` | Add "Gradient Norm Check" item |

---

## Testing Strategy

- **Norm computation test**: Create a model with known parameters, set `.grad` manually, verify `compute_gradient_norms()` returns the correct L2 norm
- **Aggregation test**: Two params "layers.0.attn.q_proj.weight" and "layers.0.attn.q_proj.bias" → aggregated to "layers.0.attn" with combined norm
- **Category classification test**: norm=0.5 → "healthy", norm=1e-9 → "vanishing", norm=200 → "exploding"
- **Summary line test**: All healthy layers → no alert tokens in summary
- **Full table render test**: Call `render_layer_table()` on canned stats, verify no exceptions
- **Interval test**: `check()` called at step 99 with interval=100 → returns None, no computation

---

## Performance Considerations

- Iterating `model.named_parameters()` for a 50M parameter model: O(n_layers × n_params_per_layer) = ~200 parameter tensors. Fast.
- `param.grad.norm(2)` for each parameter: O(param.numel()) = total ~50M operations per check. At CUDA speed, this takes ~5-10ms.
- With `check_interval=100`, this adds 5ms overhead every 100 steps — ~0.005% overhead for typical step times.
- The full table display (`full_display_interval=1000`) takes ~50ms due to Rich rendering — acceptable.

---

## Dependencies

No new packages. Uses `torch`, `rich`, `math`, `json` (all existing).

---

## Estimated Complexity

**Low** — 1 day.

- Norm computation + aggregation: 2.5 hours
- Summary line formatter: 1 hour
- Full table renderer: 1.5 hours
- GradMonitor class + history: 1.5 hours
- Trainer integration: 1 hour
- Testing: 2 hours

Total: ~9.5 hours

---

## 2026 Best Practices

- **Check before clipping, not after**: Gradient clipping modifies the gradients. Check norms on the raw gradients to see the true training signal. Clipped norms are always <= threshold, which makes the monitor useless.
- **Norm per √params**: A layer with 10M parameters will naturally have a higher raw norm than a layer with 10K parameters. Normalize by `√params` to compare layers fairly.
- **Track trends, not absolute values**: A norm of 0.1 might be "healthy" or "weak" depending on the architecture and training stage. More important: is it stable, increasing, or decreasing over steps?
- **Global norm vs per-layer**: PyTorch's `clip_grad_norm_()` computes the global norm. Per-layer monitoring reveals which layers are dominating. Both are useful.
- **Don't check every step in production**: Norm computation adds overhead. Default to every 100 steps; enable every-step monitoring only when debugging a specific problem.
