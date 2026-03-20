# 12 - Perplexity Tracker

## Overview

Track and display model perplexity on the held-out validation set every N steps during training. Perplexity is computed as `exp(cross_entropy_loss)`, making it a human-interpretable measure: a model with perplexity 50 is predicting each token with roughly `1/50` probability on average. Displays perplexity history as an ASCII chart in the training monitor, saves values to the metrics log, and supports comparing perplexity across checkpoints.

---

## Motivation

Cross-entropy loss is the natural training objective but is not intuitive to interpret. A loss of 1.8 vs 2.1 — is that good? How much better?

Perplexity provides a more interpretable scale:
- **Perplexity 500**: Barely better than random (50K vocabulary → random = perplexity 50K)
- **Perplexity 50**: The model roughly narrows each next token to 50 candidates
- **Perplexity 20**: The model is confident — narrows to ~20 candidates
- **Perplexity < 10**: The model has memorized or learned the corpus well

For TypeScript code generation:
- After 1000 steps: expect perplexity 80-200
- After 10000 steps: expect perplexity 20-60 (good small model)
- Below 10: likely overfitting to repetitive boilerplate

---

## Architecture / Design

### Formula

```
perplexity = exp(cross_entropy_loss)
```

For a sequence of tokens `t_1, ..., t_N`:
```
cross_entropy = -1/N * Σ log P(t_i | t_1, ..., t_{i-1})
perplexity    = exp(cross_entropy)
```

We compute this over multiple validation batches and average the per-batch losses before exponentiating (equivalent to computing perplexity over the entire validation set).

### Perplexity Scale Reference

| Perplexity | Interpretation |
|-----------|----------------|
| > 1000 | Model is barely learning |
| 100-1000 | Early training / small model |
| 30-100 | Decent small model |
| 10-30 | Good small model on familiar data |
| < 10 | Either memorization or very good model |

---

## Implementation Steps

### Step 1: Perplexity computation

```python
# src/evaluation/perplexity.py
import torch
import math
import numpy as np
from typing import Optional
from rich.console import Console

console = Console()

def compute_perplexity(
    model,
    val_data: np.ndarray,
    device: str = "cuda",
    n_batches: int = 50,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute perplexity on validation data.
    Returns (perplexity, std_dev) over sampled batches.
    """
    import random
    rng = random.Random(seed)
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            # Sample a random batch of sequences
            indices = [rng.randint(0, len(val_data) - 1) for _ in range(batch_size)]
            batch = torch.tensor(
                val_data[indices],
                dtype=torch.long,
                device=device
            )
            x = batch[:, :-1]
            y = batch[:, 1:]

            _, loss = model(x, targets=y)
            losses.append(loss.item())

    model.train()
    mean_loss = sum(losses) / len(losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5
    perplexity = math.exp(mean_loss)
    perplexity_std = math.exp(std_loss)  # Approximate std of perplexity
    return perplexity, perplexity_std


def perplexity_to_level(ppl: float) -> tuple[str, str]:
    """Return (description, color) for a given perplexity value."""
    if ppl > 1000:
        return "barely learning", "red"
    elif ppl > 200:
        return "early training", "red"
    elif ppl > 100:
        return "below average", "yellow"
    elif ppl > 50:
        return "reasonable", "yellow"
    elif ppl > 20:
        return "good", "green"
    elif ppl > 10:
        return "very good", "green"
    else:
        return "excellent / check for overfitting", "cyan"
```

### Step 2: Perplexity history tracker

```python
# src/evaluation/perplexity_tracker.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

@dataclass
class PerplexityRecord:
    step: int
    perplexity: float
    std_dev: float
    val_loss: float

@dataclass
class PerplexityTracker:
    history: list[PerplexityRecord] = field(default_factory=list)
    _best_ppl: float = float("inf")
    _best_step: int = 0

    def record(self, step: int, perplexity: float, std_dev: float, val_loss: float) -> None:
        self.history.append(PerplexityRecord(step, perplexity, std_dev, val_loss))
        if perplexity < self._best_ppl:
            self._best_ppl = perplexity
            self._best_step = step

    @property
    def best_perplexity(self) -> float:
        return self._best_ppl

    @property
    def best_step(self) -> int:
        return self._best_step

    @property
    def current_perplexity(self) -> Optional[float]:
        return self.history[-1].perplexity if self.history else None

    def ppl_values(self) -> list[float]:
        return [r.perplexity for r in self.history]

    def step_values(self) -> list[int]:
        return [r.step for r in self.history]

    def save(self, path: Path) -> None:
        data = [{"step": r.step, "perplexity": r.perplexity, "std": r.std_dev, "val_loss": r.val_loss}
                for r in self.history]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PerplexityTracker":
        tracker = cls()
        if not path.exists():
            return tracker
        data = json.loads(path.read_text())
        for d in data:
            tracker.record(d["step"], d["perplexity"], d.get("std", 0.0), d.get("val_loss", 0.0))
        return tracker
```

### Step 3: Rich display panel

```python
# src/evaluation/perplexity_tracker.py (continued)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.training.monitor.sparkline import sparkline
from src.evaluation.perplexity import perplexity_to_level

console = Console()

def render_perplexity_panel(tracker: PerplexityTracker, n_recent: int = 10) -> Panel:
    """Build a Rich Panel showing perplexity history."""
    if not tracker.history:
        return Panel("[dim]No perplexity data yet[/dim]", title="Perplexity")

    ppl_values = tracker.ppl_values()
    spark = sparkline(ppl_values[-50:], width=40)  # Recent 50 measurements

    current = tracker.current_perplexity
    desc, color = perplexity_to_level(current)

    recent = tracker.history[-n_recent:]
    table = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    table.add_column("Step", justify="right", width=8)
    table.add_column("Perplexity", justify="right", width=12)
    table.add_column("±", justify="right", width=8)
    table.add_column("Level", width=20)

    for r in reversed(recent):
        r_desc, r_color = perplexity_to_level(r.perplexity)
        is_best = (r.step == tracker.best_step)
        star = " ★" if is_best else ""
        table.add_row(
            str(r.step),
            f"[{r_color}]{r.perplexity:.1f}[/{r_color}]{star}",
            f"{r.std_dev:.1f}",
            f"[dim]{r_desc}[/dim]",
        )

    content = (
        f"[{color}]Current: {current:.1f}[/{color}]  "
        f"[dim]({desc})[/dim]  "
        f"|  Best: {tracker.best_perplexity:.1f} @ step {tracker.best_step}\n\n"
        f"History (recent): {spark}\n\n"
    )

    return Panel(
        content + str(table),
        title="Perplexity Tracker",
        border_style=color,
    )
```

### Step 4: Integration into training loop

```python
# src/trainer.py (additions)
from src.evaluation.perplexity import compute_perplexity
from src.evaluation.perplexity_tracker import PerplexityTracker, render_perplexity_panel

class Trainer:
    def __init__(self, ...):
        # ...
        self._ppl_tracker = PerplexityTracker()
        self._ppl_tracker_path = run_dir / "perplexity_log.json"

    def _eval_step(self, step: int, train_loss: float, val_data) -> None:
        if val_data is None:
            return
        val_loss = self.compute_val_loss(val_data)
        ppl, ppl_std = compute_perplexity(
            self.model, val_data, device=self.device,
            n_batches=self.config.get("ppl_eval_batches", 50),
        )
        self._ppl_tracker.record(step, ppl, ppl_std, val_loss)
        self._ppl_tracker.save(self._ppl_tracker_path)

        # Display in terminal
        from rich.console import Console
        Console().print(render_perplexity_panel(self._ppl_tracker))

        # Push to training monitor (feature 06) if running
        if self._monitor:
            self._monitor.push_perplexity(step, ppl)

        # Log to metrics.jsonl
        self._metrics_logger.log({
            "step": step,
            "val_loss": val_loss,
            "perplexity": ppl,
            "perplexity_std": ppl_std,
        })
```

### Step 5: Standalone perplexity evaluation CLI

```python
# src/evaluation/perplexity_cli.py
import typer
from pathlib import Path
import numpy as np
from rich.console import Console

console = Console()
app = typer.Typer()

@app.command()
def main(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint"),
    val_data: Path = typer.Argument(..., help="Path to val_data.npy"),
    n_batches: int = typer.Option(100, help="Number of batches to evaluate"),
    device: str = typer.Option("cuda", help="Device"),
    compare: list[Path] = typer.Option([], "--compare", help="Additional checkpoints to compare"),
):
    """Compute perplexity for one or more checkpoints."""
    from src.inference import load_model_for_inference
    from src.evaluation.perplexity import compute_perplexity, perplexity_to_level

    checkpoints = [checkpoint] + list(compare)
    val_array = np.load(val_data)

    for ckpt_path in checkpoints:
        model, _ = load_model_for_inference(ckpt_path, device)
        ppl, ppl_std = compute_perplexity(model, val_array, device=device, n_batches=n_batches)
        desc, color = perplexity_to_level(ppl)
        console.print(
            f"[cyan]{ckpt_path.name}[/cyan]  "
            f"Perplexity: [{color}]{ppl:.2f}[/{color}] ± {ppl_std:.2f}  "
            f"[dim]({desc})[/dim]"
        )

if __name__ == "__main__":
    app()
```

### Step 6: Perplexity in training monitor (feature 06 integration)

Add a perplexity line to the metrics panel in `TrainingMonitorDisplay._build_metrics()`:

```python
# src/training/monitor/display.py (addition to _build_metrics)
ppl_str = f"{m.perplexity:.1f}" if hasattr(m, "perplexity") and m.perplexity else "N/A"
content += f"\nPerplexity:  [magenta]{ppl_str}[/magenta]"
```

And add `perplexity: Optional[float]` to `StepMetrics` in `metrics_buffer.py`.

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/evaluation/perplexity.py` | New: compute_perplexity, perplexity_to_level |
| `src/evaluation/perplexity_tracker.py` | New: PerplexityTracker, render_perplexity_panel |
| `src/evaluation/perplexity_cli.py` | New: standalone evaluation CLI |
| `src/trainer.py` | Add perplexity computation in `_eval_step()` |
| `src/training/monitor/metrics_buffer.py` | Add `perplexity` field to StepMetrics |
| `src/training/monitor/display.py` | Show perplexity in metrics panel |
| `runs/<run>/perplexity_log.json` | New: per-step perplexity history |

---

## Testing Strategy

- **Perplexity formula test**: Feed a model with known fixed output (uniform distribution over vocab V) → perplexity should equal V
- **Compute test**: Use a small random model + tiny val data, verify `compute_perplexity()` returns a finite positive float
- **Tracker record test**: Record 5 values, verify `best_perplexity` tracks the minimum
- **Sparkline test**: `ppl_values = [200, 150, 100, 80, 60]` → sparkline should trend downward
- **Save/load test**: Save tracker to JSON, reload, verify all records preserved
- **Level color test**: ppl=500 → "red", ppl=25 → "green", ppl=8 → "cyan"

---

## Performance Considerations

- `compute_perplexity()` runs 50 batches × batch_size=4 sequences: 200 forward passes
- For a 50M model at batch_size=4, seq_len=512: ~200ms total — acceptable at every 500-step interval
- Perplexity is computed in `no_grad()` mode — no gradient storage, minimal VRAM overhead
- `perplexity_log.json` is rewritten entirely each time (small file, <100KB even for 10K evals)
- For efficiency, use the same batches as `compute_val_loss()` — avoid running two separate validation passes

---

## Dependencies

| Package | Use | Already? |
|---------|-----|----------|
| `math` | `math.exp()` | stdlib |
| `numpy` | Val data loading | Already installed |
| `torch` | Model evaluation | Already installed |
| `rich` | Display | Already installed |

No new packages required.

---

## Estimated Complexity

**Low** — 1 day.

- `compute_perplexity()`: 1.5 hours
- `PerplexityTracker` + save/load: 2 hours
- Rich panel display: 1.5 hours
- Trainer integration: 1.5 hours
- Standalone CLI: 1 hour
- Testing: 2 hours

Total: ~9.5 hours

---

## 2026 Best Practices

- **Compute perplexity on same eval as val_loss**: Don't add a second forward pass for perplexity — it's just `exp(val_loss)`. Call `compute_val_loss()` once, then `math.exp(val_loss)` to get perplexity.
- **Report std dev**: Single-point perplexity estimates are noisy. Report ± std dev to communicate uncertainty. With 50 batches, std dev is usually < 5% of the mean for stable models.
- **Log-scale is natural for perplexity**: Perplexity is exponential in loss, so improvements look small at low perplexity and large at high perplexity. Consider plotting on log-scale for full training history.
- **Bits-per-character alternative**: For byte-level models, report bits-per-character (BPC) = log2(perplexity). This is common in NLP literature and scale-agnostic.
- **Calibration check**: If val perplexity is lower than train perplexity, something is wrong (reversed split, data leakage, or the val set is much simpler). Flag this immediately.
