# 14 - Dead Neuron Detection

## Overview

Identify neurons that never activate across a validation batch. Runs a forward pass on N validation batches with activation hooks, tracks which neurons have zero activation throughout, and reports percentage of dead neurons per layer. Displays results in a Rich table. Suggests remedies: reduce dropout, adjust initialization, or prune the dead neurons. Useful diagnostic for model health after training.

---

## Motivation

Dead neurons are a known problem in neural networks using ReLU-family activations. For Cola-Coder's SwiGLU activation (a gated linear unit variant), neurons can effectively die when the gate consistently outputs near-zero, causing the corresponding dimension to carry no information.

A model with 20% dead neurons in layer 3 is effectively a smaller, less capable model. This problem:
- Often goes undetected (loss looks "fine" but model capacity is wasted)
- Gets worse with high learning rates or poor initialization
- Can be fixed without retraining if caught early enough

The detector runs as a diagnostic tool after training, not during training, to avoid overhead.

---

## Architecture / Design

### What Counts as "Dead"?

For SwiGLU activations `f(x) = x * sigmoid(gate(x))`, a neuron is "dead" if its output is below a threshold `ε` across all samples in the evaluation batch.

We track three statistics per neuron:
1. **Mean activation**: Average absolute value across all tokens and batches
2. **Max activation**: Maximum absolute value seen
3. **Activation rate**: Fraction of tokens where |activation| > ε

A neuron is dead if: `max_activation < ε` (never activated at all during eval)
A neuron is weak if: `activation_rate < 0.01` (activates less than 1% of the time)

### Layers to Monitor

For a decoder-only transformer with SwiGLU:
- **FFN gate activations**: The gate in `SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))` — most likely to produce dead neurons
- **Attention output activations**: After the output projection, before residual add
- **Optional**: Embedding layer activation statistics

---

## Implementation Steps

### Step 1: Activation capture hooks

```python
# src/diagnostics/dead_neuron_detector.py
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

@dataclass
class ActivationStats:
    layer_name: str
    n_neurons: int
    mean_activation: float
    max_activation: float
    activation_rate: float    # Fraction of tokens with |act| > threshold
    dead_neurons: int         # Count with max_activation == 0
    weak_neurons: int         # Count with activation_rate < 0.01

    @property
    def dead_pct(self) -> float:
        return self.dead_neurons / self.n_neurons * 100 if self.n_neurons > 0 else 0.0

    @property
    def weak_pct(self) -> float:
        return self.weak_neurons / self.n_neurons * 100 if self.n_neurons > 0 else 0.0

    @property
    def health_color(self) -> str:
        if self.dead_pct > 20:
            return "red"
        if self.dead_pct > 5 or self.weak_pct > 20:
            return "yellow"
        return "green"


class ActivationCapture:
    """
    Context manager that attaches forward hooks to capture intermediate activations.
    Accumulates activation statistics across multiple batches.
    """

    def __init__(self, model: nn.Module, target_modules: list[str]):
        self._model = model
        self._target_names = set(target_modules)
        self._hooks = []
        self._activation_data: dict[str, list[torch.Tensor]] = defaultdict(list)

    def __enter__(self):
        for name, module in self._model.named_modules():
            if any(name.endswith(t) or name == t for t in self._target_names):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
        return self

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Store on CPU to avoid VRAM pressure
            if isinstance(output, tuple):
                output = output[0]
            self._activation_data[name].append(output.detach().cpu())
        return hook

    def __exit__(self, *args):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_stats(self, threshold: float = 1e-6) -> list[ActivationStats]:
        """Compute dead neuron statistics from captured activations."""
        stats = []
        for layer_name, tensors in self._activation_data.items():
            # Stack all batches: shape (total_tokens, n_neurons)
            all_acts = torch.cat([t.view(-1, t.shape[-1]) for t in tensors], dim=0)
            abs_acts = all_acts.abs()

            n_neurons = all_acts.shape[-1]
            mean_per_neuron = abs_acts.mean(dim=0)   # (n_neurons,)
            max_per_neuron = abs_acts.max(dim=0).values  # (n_neurons,)
            active_per_neuron = (abs_acts > threshold).float().mean(dim=0)  # (n_neurons,)

            dead = (max_per_neuron < threshold).sum().item()
            weak = ((active_per_neuron < 0.01) & (max_per_neuron >= threshold)).sum().item()

            stats.append(ActivationStats(
                layer_name=layer_name,
                n_neurons=n_neurons,
                mean_activation=mean_per_neuron.mean().item(),
                max_activation=max_per_neuron.max().item(),
                activation_rate=active_per_neuron.mean().item(),
                dead_neurons=int(dead),
                weak_neurons=int(weak),
            ))

        return sorted(stats, key=lambda s: s.dead_pct, reverse=True)
```

### Step 2: Automatic target module discovery for SwiGLU

```python
# src/diagnostics/dead_neuron_detector.py (continued)

def find_activation_layers(model: nn.Module) -> list[str]:
    """
    Auto-discover layer names to monitor.
    For SwiGLU transformers: look for gate activation outputs and FFN linear layers.
    """
    targets = []
    for name, module in model.named_modules():
        # Target FFN layers (the intermediate activation, not input/output projections)
        if any(kw in name for kw in ["ffn", "mlp", "feed_forward"]):
            # Look for activation-like modules
            if isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh)):
                targets.append(name)
            # Or the full FFN block if no activation submodule
            elif hasattr(module, "gate_proj") or hasattr(module, "w1"):
                targets.append(name)
        # Attention output projections
        if "out_proj" in name or "o_proj" in name:
            if isinstance(module, nn.Linear):
                targets.append(name)
    return targets
```

### Step 3: Main detector runner

```python
# src/diagnostics/dead_neuron_detector.py (continued)
import numpy as np

def run_dead_neuron_detection(
    model: nn.Module,
    val_data: np.ndarray,
    device: str = "cuda",
    n_batches: int = 20,
    batch_size: int = 4,
    threshold: float = 1e-6,
    seed: int = 42,
) -> list[ActivationStats]:
    """
    Run forward pass on validation data with activation capture.
    Returns per-layer activation statistics.
    """
    import random
    rng = random.Random(seed)

    # Discover layers to monitor
    target_layers = find_activation_layers(model)
    if not target_layers:
        # Fallback: monitor all Linear layers in the model
        target_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "proj" in name
        ][:20]  # Limit to first 20

    model.eval()
    with torch.no_grad():
        with ActivationCapture(model, target_layers) as capture:
            for _ in range(n_batches):
                indices = [rng.randint(0, len(val_data) - 1) for _ in range(batch_size)]
                batch = torch.tensor(val_data[indices], dtype=torch.long, device=device)
                x = batch[:, :-1]
                model(x)  # Forward pass only — no loss needed

    return capture.compute_stats(threshold=threshold)
```

### Step 4: Rich report display

```python
# src/diagnostics/dead_neuron_detector.py (continued)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def render_dead_neuron_report(
    stats: list[ActivationStats],
    checkpoint_name: str = "",
) -> None:
    """Render a Rich table report of dead neuron statistics."""
    table = Table(
        title=f"Dead Neuron Detection{' — ' + checkpoint_name if checkpoint_name else ''}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Layer", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Neurons", justify="right", width=8)
    table.add_column("Dead %", justify="right", width=8)
    table.add_column("Weak %", justify="right", width=8)
    table.add_column("Mean Act", justify="right", width=10)
    table.add_column("Act Rate", justify="right", width=10)
    table.add_column("Health", width=10)

    total_neurons = sum(s.n_neurons for s in stats)
    total_dead = sum(s.dead_neurons for s in stats)
    total_weak = sum(s.weak_neurons for s in stats)

    for s in stats:
        color = s.health_color
        dead_cell = f"[{color}]{s.dead_pct:.1f}%[/{color}]"
        weak_cell = f"{s.weak_pct:.1f}%"
        health_cell = f"[{color}]{'DEAD' if s.dead_pct > 20 else 'WEAK' if s.dead_pct > 5 else 'OK'}[/{color}]"
        table.add_row(
            s.layer_name,
            str(s.n_neurons),
            dead_cell,
            weak_cell,
            f"{s.mean_activation:.4f}",
            f"{s.activation_rate:.2%}",
            health_cell,
        )

    table.add_section()
    overall_dead_pct = total_dead / total_neurons * 100 if total_neurons > 0 else 0
    overall_color = "red" if overall_dead_pct > 20 else "yellow" if overall_dead_pct > 5 else "green"
    table.add_row(
        "[bold]TOTAL[/bold]",
        str(total_neurons),
        f"[bold {overall_color}]{overall_dead_pct:.1f}%[/bold {overall_color}]",
        f"{total_weak/total_neurons*100:.1f}%",
        "", "", "",
    )

    console.print(table)
    _render_suggestions(stats, overall_dead_pct)


def _render_suggestions(stats: list[ActivationStats], overall_dead_pct: float) -> None:
    if overall_dead_pct < 2.0:
        console.print(Panel("[green]Model health: good. No action needed.[/green]", style="green"))
        return

    suggestions = []
    worst = max(stats, key=lambda s: s.dead_pct)

    if overall_dead_pct > 30:
        suggestions.append("[bold]Severe dead neuron problem. Consider:[/bold]")
        suggestions.append("  - Reduce learning rate (current LR may be too high)")
        suggestions.append("  - Change initialization: use Kaiming uniform instead of normal")
        suggestions.append("  - Reduce dropout rate or disable it for initial training")
        suggestions.append("  - Check for zero-initialization bugs in layer norm scales")
    elif overall_dead_pct > 10:
        suggestions.append("[bold]Moderate dead neuron problem:[/bold]")
        suggestions.append(f"  - Worst layer: {worst.layer_name} ({worst.dead_pct:.1f}% dead)")
        suggestions.append("  - Consider reducing dropout")
        suggestions.append("  - Try training for more steps (neurons may recover)")
    else:
        suggestions.append("[bold]Minor dead neuron issue (< 10%):[/bold]")
        suggestions.append("  - Usually resolves with continued training")
        suggestions.append("  - Monitor across checkpoints for trends")

    console.print(Panel(
        "\n".join(suggestions),
        title="Suggestions",
        style="yellow" if overall_dead_pct > 10 else "cyan",
    ))
```

### Step 5: CLI entry point

```python
# src/diagnostics/dead_neuron_cli.py
import typer
from pathlib import Path
import numpy as np
from src.diagnostics.dead_neuron_detector import run_dead_neuron_detection, render_dead_neuron_report

app = typer.Typer()

@app.command()
def main(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint"),
    val_data: Path = typer.Argument(..., help="Path to val_data.npy"),
    n_batches: int = typer.Option(20, help="Number of batches to analyze"),
    threshold: float = typer.Option(1e-6, help="Activation threshold for dead detection"),
    device: str = typer.Option("cuda"),
):
    """Detect dead neurons in a trained model."""
    from src.inference import load_model_for_inference
    model, _ = load_model_for_inference(checkpoint, device)
    val_array = np.load(val_data)
    stats = run_dead_neuron_detection(model, val_array, device=device,
                                      n_batches=n_batches, threshold=threshold)
    render_dead_neuron_report(stats, checkpoint_name=checkpoint.name)

if __name__ == "__main__":
    app()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/diagnostics/__init__.py` | New package |
| `src/diagnostics/dead_neuron_detector.py` | New: full detection + display |
| `src/diagnostics/dead_neuron_cli.py` | New: CLI entry point |
| `src/menu/menus/tools_menu.py` | Add "Dead Neuron Detection" item |
| `src/model/swiglu.py` (or wherever SwiGLU is defined) | Verify activation outputs are hookable |

---

## Testing Strategy

- **Truly dead neuron test**: Manually zero out a weight matrix to create guaranteed dead neurons; verify detector finds them
- **Activation capture test**: Run 1 batch through a small model with hooks; verify `_activation_data` has entries for each target layer
- **Stats computation test**: Create a tensor where 10/100 neurons are always zero; verify `dead_neurons=10, dead_pct=10.0`
- **Auto-discover test**: For a Cola-Coder model, verify `find_activation_layers()` finds at least one layer per model block
- **No val data test**: When val_data is empty array, verify graceful error
- **Report render test**: Call `render_dead_neuron_report()` with canned stats; verify no exceptions

---

## Performance Considerations

- The activation capture accumulates tensors on CPU (`.detach().cpu()`). For 20 batches × 4 sequences × 512 tokens × hidden_dim=512, this is 20 × 4 × 512 × 512 × 4 bytes ≈ 80MB. Manageable.
- For larger models (dim=2048), reduce n_batches or batch_size to control memory.
- The detection run is a one-time diagnostic, not a continuous monitor. Running time: ~30 seconds for a 50M model with 20 batches.
- Hook overhead: each hooked layer adds one extra `.cpu()` transfer. For 10-20 layers this is acceptable.

---

## Dependencies

No new packages. Uses `torch`, `numpy`, `rich` (all existing).

---

## Estimated Complexity

**Medium** — 1.5 days.

- ActivationCapture with hooks: 3 hours
- Stats computation: 2 hours
- Auto-discovery of target layers: 1.5 hours
- Rich report + suggestions: 2 hours
- CLI + integration: 1.5 hours
- Testing (especially hook correctness): 3 hours

Total: ~13 hours

---

## 2026 Best Practices

- **CPU offload for accumulated activations**: During diagnostic runs, accumulate activation tensors on CPU to avoid VRAM pressure. The detection is not latency-sensitive.
- **Use context manager for hooks**: `register_forward_hook()` hooks must be removed after use. Using a context manager guarantees cleanup even on exceptions.
- **Threshold selection**: A threshold of 1e-6 is reasonable for bf16 training (minimum representable value is ~6e-8). Adjust if using fp16 (minimum ~6e-5) or fp32 (minimum ~1e-45).
- **Compare across checkpoints**: Run dead neuron detection on checkpoint at step 1000 and step 10000. If dead neuron % is increasing, the training dynamics are unhealthy.
- **SwiGLU and dead neurons**: SwiGLU is less prone to dying than ReLU because the sigmoid gate has a non-zero gradient everywhere. However, if the gate consistently saturates to 0, the neuron effectively dies. Monitor the gate output statistics specifically.
