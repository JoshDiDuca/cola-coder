# 65 - Continuous Evaluation During Training

## Overview

Automatically run evaluation benchmarks at regular training step intervals, plot improvement over time, save results to `eval_history.json`, and support early stopping when no improvement is detected. Evaluation hooks into the training loop with minimal overhead.

**Feature flag:** `config.continuous_eval.enabled` (default: `true` when benchmarks are configured)

---

## Motivation

Training loss is a necessary but insufficient signal. A model can have decreasing loss while generating syntactically broken code. Conversely, a model might plateau in loss but still be improving on the metrics that matter (syntax validity, type correctness).

Running eval every N steps provides:
- **Early regression detection**: catch if a code change broke something before the full run completes
- **Optimal checkpoint selection**: the best checkpoint by eval metric is often not the final one
- **Training insights**: see which metrics improve early (syntax) vs late (type correctness)
- **Early stopping**: terminate training if eval hasn't improved for K consecutive evaluations, saving compute

---

## Architecture / Design

### Evaluation Suite

At each eval step, run a configurable set of evaluators:

| Evaluator           | Speed       | Signal                        |
|---------------------|-------------|-------------------------------|
| `NanoBenchmark`     | ~5s         | pass@1 on 10 micro-prompts    |
| `SyntaxValidator`   | <1s         | % of samples that parse       |
| `TypeChecker`       | ~30s        | % of samples that pass tsc    |
| `PerplexityCalc`    | ~2s         | avg NLL on val set            |

All evaluators are optional and configurable. A minimal continuous eval might run only `NanoBenchmark` and `SyntaxValidator` (total: ~6s overhead every N steps).

### History Format (`eval_history.json`)

```json
{
  "model_name": "cola-coder-small",
  "eval_interval": 500,
  "history": [
    {
      "step": 500,
      "timestamp": "2026-03-15T10:22:11Z",
      "training_loss": 3.421,
      "metrics": {
        "nano_benchmark_pass_at_1": 0.12,
        "syntax_validity_rate": 0.43,
        "type_correctness_rate": 0.08,
        "val_perplexity": 28.4
      }
    },
    {
      "step": 1000,
      "timestamp": "2026-03-15T11:05:33Z",
      "training_loss": 2.891,
      "metrics": {
        "nano_benchmark_pass_at_1": 0.28,
        "syntax_validity_rate": 0.67,
        "type_correctness_rate": 0.19,
        "val_perplexity": 21.7
      }
    }
  ],
  "best": {
    "metric": "nano_benchmark_pass_at_1",
    "step": 1000,
    "value": 0.28
  }
}
```

### Early Stopping Logic

```
patience = config.continuous_eval.early_stopping_patience  # e.g., 5
primary_metric = "nano_benchmark_pass_at_1"

if eval_history has >= patience consecutive evals with no improvement:
    log warning: "No improvement in {primary_metric} for {patience} evals"
    if config.continuous_eval.early_stopping_enabled:
        save best checkpoint
        stop training
```

---

## Implementation Steps

### Step 1: Evaluator Base Class (`eval/base_evaluator.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EvalResult:
    name: str
    value: float          # primary scalar (for early stopping comparison)
    details: dict         # arbitrary additional data

class BaseEvaluator(ABC):
    name: str

    @abstractmethod
    def run(self, model, tokenizer, config: dict) -> EvalResult:
        """Run the evaluation and return results."""
        ...

    def is_available(self) -> bool:
        """Return False if this evaluator's dependencies are missing."""
        return True
```

### Step 2: Perplexity Evaluator (`eval/perplexity_eval.py`)

```python
import torch
import numpy as np

class PerplexityEvaluator(BaseEvaluator):
    name = "val_perplexity"

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        val_data = np.load(config["val_data_path"], mmap_mode="r")
        # Sample up to 50 sequences for speed
        n_samples = min(50, len(val_data))
        indices = np.random.choice(len(val_data), n_samples, replace=False)

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for idx in indices:
                seq = torch.tensor(val_data[idx]).unsqueeze(0).to(model.device)
                if seq.shape[1] < 2:
                    continue
                logits = model(seq[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    seq[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += seq.shape[1] - 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = float(np.exp(avg_loss))

        return EvalResult(
            name=self.name,
            value=perplexity,    # lower is better (handled by comparator)
            details={"avg_loss": avg_loss, "n_samples": n_samples}
        )
```

### Step 3: Continuous Eval Manager (`eval/continuous_eval.py`)

```python
import json
import time
from pathlib import Path
from datetime import datetime, timezone

class ContinuousEvalManager:
    def __init__(self, evaluators: list[BaseEvaluator], config: dict):
        self.evaluators = evaluators
        self.config = config
        self.eval_interval = config.get("eval_interval", 500)
        self.patience = config.get("early_stopping_patience", 5)
        self.primary_metric = config.get("primary_metric", "val_perplexity")
        self.primary_higher_is_better = config.get("primary_higher_is_better", False)
        self.history_path = Path(config.get("history_path", "eval_history.json"))
        self._history = self._load_history()
        self._no_improve_count = 0

    def _load_history(self) -> dict:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return {
            "model_name": self.config.get("model_name", "unknown"),
            "eval_interval": self.eval_interval,
            "history": [],
            "best": None,
        }

    def _save_history(self):
        self.history_path.write_text(json.dumps(self._history, indent=2))

    def should_eval(self, step: int) -> bool:
        return step % self.eval_interval == 0

    def run_eval(self, step: int, model, tokenizer, training_loss: float) -> dict:
        """Run all evaluators and record results."""
        t0 = time.time()
        metrics = {}
        for evaluator in self.evaluators:
            if not evaluator.is_available():
                continue
            try:
                result = evaluator.run(model, tokenizer, self.config)
                metrics[result.name] = result.value
                metrics.update({
                    f"{result.name}__{k}": v
                    for k, v in result.details.items()
                })
            except Exception as e:
                print(f"[continuous-eval] Evaluator {evaluator.name} failed: {e}")

        entry = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_loss": training_loss,
            "metrics": {k: v for k, v in metrics.items() if "__" not in k},
            "eval_time_sec": round(time.time() - t0, 1),
        }
        self._history["history"].append(entry)
        self._update_best(entry)
        self._save_history()
        return entry

    def _update_best(self, entry: dict):
        metric_val = entry["metrics"].get(self.primary_metric)
        if metric_val is None:
            return

        current_best = self._history.get("best")
        if current_best is None:
            self._history["best"] = {
                "metric": self.primary_metric,
                "step": entry["step"],
                "value": metric_val,
            }
            self._no_improve_count = 0
            return

        improved = (
            metric_val > current_best["value"]
            if self.primary_higher_is_better
            else metric_val < current_best["value"]
        )
        if improved:
            self._history["best"] = {
                "metric": self.primary_metric,
                "step": entry["step"],
                "value": metric_val,
            }
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

    def should_stop_early(self) -> bool:
        if not self.config.get("early_stopping_enabled", False):
            return False
        return self._no_improve_count >= self.patience

    def print_summary(self, entry: dict):
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(
            title=f"[bold]Eval @ Step {entry['step']}[/]",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Best", justify="right", style="green")

        best_metrics = {
            e["metrics"].get(k): e["step"]
            for e in self._history["history"]
            for k in e["metrics"]
        }

        for metric, value in entry["metrics"].items():
            # Compare to best known value
            all_vals = [
                e["metrics"].get(metric)
                for e in self._history["history"]
                if metric in e["metrics"]
            ]
            is_best = bool(all_vals and (
                value == max(all_vals) if self.primary_higher_is_better
                else value == min(all_vals)
            ))
            best_str = "[bold green]NEW BEST[/]" if is_best else f"{min(all_vals) if not self.primary_higher_is_better else max(all_vals):.4f}"
            table.add_row(
                metric,
                f"{value:.4f}",
                best_str,
            )
        console.print(table)
        if self._no_improve_count > 0:
            console.print(f"[yellow]No improvement for {self._no_improve_count} evals[/]")
```

### Step 4: Trainer Integration (`training/trainer.py`)

```python
# In the training loop:

if continuous_eval_manager and continuous_eval_manager.should_eval(step):
    eval_entry = continuous_eval_manager.run_eval(
        step=step,
        model=model,
        tokenizer=tokenizer,
        training_loss=current_loss,
    )
    continuous_eval_manager.print_summary(eval_entry)

    # Save best checkpoint
    best_step = continuous_eval_manager._history["best"]["step"]
    if eval_entry["step"] == best_step:
        save_checkpoint(model, f"checkpoints/best.safetensors")

    if continuous_eval_manager.should_stop_early():
        print("[continuous-eval] Early stopping triggered.")
        break
```

### Step 5: CLI Trend Viewer (`cli/eval_trend_cmd.py`)

```python
# cola-coder eval trend --metric val_perplexity --last 20

def cmd_eval_trend(args):
    history = json.loads(Path(args.history_file).read_text())
    entries = history["history"][-args.last:]

    from rich.console import Console
    from rich.table import Table
    console = Console()

    # ASCII sparkline
    values = [e["metrics"].get(args.metric, 0) for e in entries]
    steps = [e["step"] for e in entries]

    console.print(f"\n[bold]{args.metric}[/] trend (last {len(entries)} evals)\n")

    # Normalize to 0-8 for block chars
    min_v, max_v = min(values), max(values)
    blocks = "▁▂▃▄▅▆▇█"
    sparkline = ""
    for v in values:
        norm = (v - min_v) / (max_v - min_v + 1e-9)
        sparkline += blocks[int(norm * 7)]

    console.print(f"  {sparkline}")
    console.print(f"  min={min_v:.4f}  max={max_v:.4f}  last={values[-1]:.4f}\n")

    # Table of last 10
    table = Table(show_header=True)
    table.add_column("Step", justify="right")
    table.add_column(args.metric, justify="right")
    table.add_column("Train Loss", justify="right")
    for e in entries[-10:]:
        val = e["metrics"].get(args.metric, "—")
        table.add_row(
            str(e["step"]),
            f"{val:.4f}" if isinstance(val, float) else str(val),
            f"{e.get('training_loss', 0):.4f}",
        )
    console.print(table)
```

### Step 6: Config

```yaml
continuous_eval:
  enabled: true
  eval_interval: 500          # run eval every 500 steps
  primary_metric: val_perplexity
  primary_higher_is_better: false
  early_stopping_enabled: false
  early_stopping_patience: 5
  history_path: eval_history.json
  evaluators:
    - perplexity
    - syntax_validity      # requires tree-sitter
    - nano_benchmark       # requires benchmark fixtures
    # - type_correctness   # slow, enable at milestone steps only
  save_best_checkpoint: true
```

---

## Key Files to Modify

- `training/trainer.py` - Add eval hook in training loop
- `eval/continuous_eval.py` - New file: eval manager
- `eval/base_evaluator.py` - New file: abstract base class
- `eval/perplexity_eval.py` - New file: perplexity evaluator
- `cli/eval_trend_cmd.py` - New file: trend viewer CLI
- `config/training.yaml` - Add `continuous_eval` section
- `eval/syntax_eval.py` - Extend existing syntax check to implement `BaseEvaluator`

---

## Testing Strategy

1. **Eval manager unit test**: mock evaluators, call `run_eval` at steps 500/1000/1500, verify `eval_history.json` has 3 entries with correct step numbers.
2. **Best tracking test**: feed sequence of improving then worsening values, verify `best` entry updates only when improved and `no_improve_count` increments correctly.
3. **Early stopping test**: feed K+1 non-improving evals, assert `should_stop_early()` returns `True` after K+1.
4. **History persistence test**: create manager, run 3 evals, destroy object, recreate manager from same history file, assert history has 3 entries.
5. **Trainer integration test**: run 1500 training steps with `eval_interval=500`, assert 3 eval entries in history, assert `best.safetensors` exists.

---

## Performance Considerations

- Perplexity evaluation on 50 sequences adds ~2s to each eval step. Acceptable for 500-step intervals.
- `NanoBenchmark` with 10 prompts adds ~5s. Keep the benchmark tiny.
- Type correctness check adds ~30s. Gate this behind a separate `milestone_eval_interval` (e.g., every 2000 steps).
- Evaluations run synchronously in the training loop. For minimal disruption, run evaluations on a separate thread and allow the training loop to continue (with the caveat that the model weights are being updated). A safer option: pause gradient accumulation during eval.
- On RTX 3080/4080, move model to eval mode (disabling dropout) and use `torch.inference_mode()` to reduce VRAM during eval forward passes.

---

## Dependencies

No new dependencies beyond what evaluators require individually (tree-sitter for syntax, tsc for type check).

---

## Estimated Complexity

**Medium.** The eval manager and trainer integration are well-scoped. The main complexity is ensuring the eval doesn't interfere with training state (model mode, RNG state, gradient accumulation counter). Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Restore model state after eval**: always call `model.train()` after `model.eval()`. Consider saving and restoring RNG state if reproducibility across eval runs matters.
- **Non-blocking eval option**: for large models where even 6s eval overhead is significant, run eval in a subprocess with a checkpoint copy. More complex but zero impact on training throughput.
- **Multiple metrics, one primary**: track many metrics but define exactly one as the primary for early stopping and best checkpoint selection. Avoid multi-metric optimization in early stopping (it's hard to reason about).
- **Eval history as a first-class artifact**: commit `eval_history.json` to version control alongside model configs. It's a compact, human-readable record of the entire training run's progress.
- **Alert on regression**: if any metric drops more than 10% from best, log a prominent warning. This often indicates a training instability or a bug introduced mid-run.
