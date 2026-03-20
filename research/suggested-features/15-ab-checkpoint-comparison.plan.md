# 15 - A/B Checkpoint Comparison

## Overview

Load two checkpoints side-by-side, run the same set of prompts through both, and display the outputs in a Rich comparison table. Includes quantitative metrics (perplexity, generation length, syntax validity) alongside qualitative side-by-side output comparison. Supports comparing any two checkpoints by path, making it easy to measure the impact of training for N additional steps or any architectural change.

---

## Motivation

Visual comparison of two checkpoints is the fastest way to answer "did this training improvement actually matter?" Without it, the only signal is the numeric loss difference, which doesn't tell you:
- Did the model get better at specific patterns (async functions, type definitions)?
- Did it improve on structured code (classes, interfaces) vs simple code?
- Did one checkpoint produce shorter, less complete responses?
- At what point in training did the model "click" for TypeScript idioms?

The A/B comparison turns these questions into 30 seconds of terminal output.

---

## Architecture / Design

### Display Layout

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  A/B Checkpoint Comparison                                                   ║
║  A: runs/run_001/ckpt_005000.pt (step 5000, loss 1.876)                      ║
║  B: runs/run_001/ckpt_010000.pt (step 10000, loss 1.521)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Prompt: function add(a: number, b: number)                                  ║
╠════════════════════════════╦════════════════════════════════════════════════╣
║  CHECKPOINT A               ║  CHECKPOINT B                                  ║
║  : number {                 ║  : number {                                    ║
║    return a;                ║    return a + b;                               ║
║  }                          ║  }                                             ║
╠════════════════════════════╩════════════════════════════════════════════════╣
║  A: syntax=PASS  ppl=45.2  len=8      B: syntax=PASS  ppl=28.1  len=9      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Metrics Summary                                                             ║
║  Avg Perplexity:  A=62.4   B=31.7   Δ=-30.7 (-49%)  [B wins]               ║
║  Syntax Valid:    A=6/10   B=9/10   Δ=+3               [B wins]             ║
║  Avg Length:      A=45     B=67     Δ=+22 tokens      [B wins]              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Default Prompts

By default, uses the same 5 prompts from the smoke test (feature 03) plus 5 additional TypeScript patterns for broader coverage:

| # | Prompt | Pattern |
|---|--------|---------|
| 1 | `function add(a: number, b: number)` | Basic function |
| 2 | `interface User {` | Interface |
| 3 | `const fetchData = async` | Async |
| 4 | `class Logger {` | Class |
| 5 | `export function validateEmail` | Export |
| 6 | `type ApiResponse<T> = {` | Generic type |
| 7 | `const useUserStore = create<` | Zustand store pattern |
| 8 | `describe('UserService',` | Jest test |
| 9 | `const router = express.Router()` | Express route |
| 10 | `enum Status {` | Enum definition |

---

## Implementation Steps

### Step 1: Checkpoint loader

```python
# src/evaluation/ab_comparison/loader.py
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class CheckpointBundle:
    path: Path
    label: str         # "A" or "B"
    step: int
    loss: float
    model: Any         # Loaded model
    tokenizer: Any     # Loaded tokenizer
    config: dict

def load_bundle(path: Path, label: str, device: str = "cuda") -> CheckpointBundle:
    from src.inference import load_model_for_inference
    model, tokenizer = load_model_for_inference(path, device)

    # Load metadata
    raw = torch.load(path, map_location="cpu", weights_only=False)
    step = raw.get("step", 0)
    loss = raw.get("loss", 0.0)
    config = raw.get("config", {})

    return CheckpointBundle(
        path=path,
        label=label,
        step=step,
        loss=loss,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
```

### Step 2: Generation for both checkpoints

```python
# src/evaluation/ab_comparison/generator.py
from dataclasses import dataclass
from typing import Optional
from src.evaluation.smoke_test.detectors import run_all_detectors
from src.evaluation.smoke_test.syntax_check import quick_syntax_check
import math

@dataclass
class GenerationResult:
    prompt: str
    completion: str
    full_code: str
    syntax_valid: Optional[bool]
    perplexity: Optional[float]
    token_count: int
    failures: list

    @property
    def has_failure(self) -> bool:
        return bool(self.failures)


def generate_for_prompt(
    bundle,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.2,
    device: str = "cuda",
) -> GenerationResult:
    from src.inference import generate_text
    import torch

    # Generate
    completion = generate_text(
        bundle.model,
        bundle.tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )
    full_code = prompt + completion

    # Compute metrics
    syntax_valid, _ = quick_syntax_check(full_code)
    failures = run_all_detectors(completion)

    # Compute perplexity of the completion given the prompt
    ppl = _compute_completion_perplexity(bundle.model, bundle.tokenizer, prompt, completion, device)

    token_ids = bundle.tokenizer.encode(completion)

    return GenerationResult(
        prompt=prompt,
        completion=completion,
        full_code=full_code,
        syntax_valid=syntax_valid,
        perplexity=ppl,
        token_count=len(token_ids),
        failures=failures,
    )


def _compute_completion_perplexity(
    model, tokenizer, prompt: str, completion: str, device: str
) -> Optional[float]:
    """
    Compute perplexity of the completion tokens given the prompt context.
    Only scores the completion, not the prompt.
    """
    import torch
    full = prompt + completion
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(full)
    completion_ids = full_ids[len(prompt_ids):]

    if len(completion_ids) < 2:
        return None

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        _, loss = model(
            input_tensor[:, :-1],
            targets=input_tensor[:, 1:],
        )
    return math.exp(loss.item()) if loss else None
```

### Step 3: Side-by-side comparison for a single prompt

```python
# src/evaluation/ab_comparison/display.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from src.evaluation.ab_comparison.generator import GenerationResult

console = Console()

def render_prompt_comparison(
    prompt: str,
    result_a: GenerationResult,
    result_b: GenerationResult,
) -> None:
    """Render side-by-side comparison for a single prompt."""
    console.print(f"\n[bold cyan]Prompt:[/bold cyan] [yellow]{prompt}[/yellow]")
    console.rule()

    # Code panels side by side using Columns
    code_a = Syntax(result_a.full_code, "typescript", theme="monokai", line_numbers=False)
    code_b = Syntax(result_b.full_code, "typescript", theme="monokai", line_numbers=False)

    panel_a = Panel(code_a, title="[cyan]Checkpoint A[/cyan]", border_style="cyan")
    panel_b = Panel(code_b, title="[green]Checkpoint B[/green]", border_style="green")

    console.print(Columns([panel_a, panel_b], equal=True))

    # Metrics comparison line
    def fmt_syntax(r: GenerationResult) -> str:
        if r.syntax_valid is True:
            return "[green]PASS[/green]"
        elif r.syntax_valid is False:
            return "[red]FAIL[/red]"
        return "[dim]?[/dim]"

    ppl_a = f"{result_a.perplexity:.1f}" if result_a.perplexity else "?"
    ppl_b = f"{result_b.perplexity:.1f}" if result_b.perplexity else "?"

    # Determine winner per metric
    def ppl_color(a, b, target):
        """Green if target is lower (better perplexity)."""
        if a is None or b is None:
            return "white"
        return "green" if target < (b if target is a else a) else "red"

    a_color = ppl_color(result_a.perplexity, result_b.perplexity, result_a.perplexity)
    b_color = ppl_color(result_a.perplexity, result_b.perplexity, result_b.perplexity)

    console.print(
        f"A: syntax={fmt_syntax(result_a)}  "
        f"ppl=[{a_color}]{ppl_a}[/{a_color}]  "
        f"len={result_a.token_count}      "
        f"B: syntax={fmt_syntax(result_b)}  "
        f"ppl=[{b_color}]{ppl_b}[/{b_color}]  "
        f"len={result_b.token_count}"
    )
```

### Step 4: Summary metrics table

```python
# src/evaluation/ab_comparison/display.py (continued)
from typing import Optional
import statistics

@dataclass
class ComparisonSummary:
    prompts: list[str]
    results_a: list[GenerationResult]
    results_b: list[GenerationResult]

    @property
    def avg_ppl_a(self) -> Optional[float]:
        ppls = [r.perplexity for r in self.results_a if r.perplexity]
        return statistics.mean(ppls) if ppls else None

    @property
    def avg_ppl_b(self) -> Optional[float]:
        ppls = [r.perplexity for r in self.results_b if r.perplexity]
        return statistics.mean(ppls) if ppls else None

    @property
    def syntax_valid_a(self) -> int:
        return sum(1 for r in self.results_a if r.syntax_valid)

    @property
    def syntax_valid_b(self) -> int:
        return sum(1 for r in self.results_b if r.syntax_valid)

    @property
    def avg_len_a(self) -> float:
        return statistics.mean(r.token_count for r in self.results_a)

    @property
    def avg_len_b(self) -> float:
        return statistics.mean(r.token_count for r in self.results_b)


def render_summary_table(
    summary: ComparisonSummary,
    bundle_a_label: str,
    bundle_b_label: str,
) -> None:
    n = len(summary.prompts)
    table = Table(title="Comparison Summary", box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column(f"A ({bundle_a_label})", justify="right")
    table.add_column(f"B ({bundle_b_label})", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Winner", justify="center")

    def add_metric(name, val_a, val_b, lower_is_better=True, fmt=".2f"):
        if val_a is None or val_b is None:
            table.add_row(name, "N/A", "N/A", "N/A", "?")
            return
        delta = val_b - val_a
        delta_str = f"{delta:+.2f}"
        if lower_is_better:
            winner = "B" if val_b < val_a else "A" if val_a < val_b else "="
        else:
            winner = "B" if val_b > val_a else "A" if val_a > val_b else "="
        winner_color = "green" if winner == "B" else ("yellow" if winner == "A" else "dim")
        table.add_row(
            name,
            f"{val_a:{fmt}}",
            f"{val_b:{fmt}}",
            delta_str,
            f"[{winner_color}]{winner}[/{winner_color}]",
        )

    add_metric("Avg Perplexity", summary.avg_ppl_a, summary.avg_ppl_b, lower_is_better=True)
    add_metric("Syntax Valid", summary.syntax_valid_a, summary.syntax_valid_b,
               lower_is_better=False, fmt="d")
    add_metric("Avg Completion Length", summary.avg_len_a, summary.avg_len_b,
               lower_is_better=False, fmt=".1f")

    # Overall winner: best 2/3 of metrics
    scores = {"A": 0, "B": 0}
    for r_a, r_b in zip(summary.results_a, summary.results_b):
        if r_a.perplexity and r_b.perplexity:
            if r_b.perplexity < r_a.perplexity:
                scores["B"] += 1
            elif r_a.perplexity < r_b.perplexity:
                scores["A"] += 1
        if r_a.syntax_valid and not r_b.syntax_valid:
            scores["A"] += 1
        elif r_b.syntax_valid and not r_a.syntax_valid:
            scores["B"] += 1

    overall_winner = "B" if scores["B"] > scores["A"] else "A" if scores["A"] > scores["B"] else "Tie"
    console.print(table)
    overall_color = "green" if overall_winner == "B" else "yellow" if overall_winner == "A" else "dim"
    console.print(Panel(
        f"Overall: [{overall_color}]{overall_winner} wins[/{overall_color}]  "
        f"(A: {scores['A']} pts, B: {scores['B']} pts)",
        style=overall_color,
    ))
```

### Step 5: Main comparison runner

```python
# src/evaluation/ab_comparison/runner.py
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from src.evaluation.ab_comparison.loader import load_bundle
from src.evaluation.ab_comparison.generator import generate_for_prompt
from src.evaluation.ab_comparison.display import (
    render_prompt_comparison, render_summary_table, ComparisonSummary
)
from src.evaluation.smoke_test.prompts import SMOKE_PROMPTS

console = Console()

EXTRA_PROMPTS = [
    "type ApiResponse<T> = {",
    "describe('UserService',",
    "const router = express.Router()",
    "enum Status {",
    "const useUserStore = create<",
]

def run_ab_comparison(
    checkpoint_a: Path,
    checkpoint_b: Path,
    custom_prompts: list[str] = None,
    max_new_tokens: int = 150,
    temperature: float = 0.2,
    device: str = "cuda",
) -> None:
    """Run full A/B comparison between two checkpoints."""

    console.print(Panel(
        f"[bold]A/B Checkpoint Comparison[/bold]\n\n"
        f"A: {checkpoint_a}\n"
        f"B: {checkpoint_b}",
        style="cyan",
    ))

    console.print("[cyan]Loading checkpoint A...[/cyan]")
    bundle_a = load_bundle(checkpoint_a, "A", device)
    console.print(f"[cyan]Loading checkpoint B...[/cyan]")
    bundle_b = load_bundle(checkpoint_b, "B", device)

    label_a = f"step {bundle_a.step:,} loss={bundle_a.loss:.3f}"
    label_b = f"step {bundle_b.step:,} loss={bundle_b.loss:.3f}"

    prompts = custom_prompts or [p.text for p in SMOKE_PROMPTS] + EXTRA_PROMPTS

    results_a, results_b = [], []
    for i, prompt in enumerate(prompts, 1):
        console.print(f"\n[dim][{i}/{len(prompts)}][/dim]")
        r_a = generate_for_prompt(bundle_a, prompt, max_new_tokens, temperature, device)
        r_b = generate_for_prompt(bundle_b, prompt, max_new_tokens, temperature, device)
        render_prompt_comparison(prompt, r_a, r_b)
        results_a.append(r_a)
        results_b.append(r_b)

    summary = ComparisonSummary(prompts=prompts, results_a=results_a, results_b=results_b)
    console.print("\n")
    render_summary_table(summary, label_a, label_b)
```

### Step 6: CLI entry point

```python
# src/evaluation/ab_comparison/__main__.py
import typer
from pathlib import Path
from typing import Optional
from src.evaluation.ab_comparison.runner import run_ab_comparison

app = typer.Typer()

@app.command()
def main(
    checkpoint_a: Path = typer.Argument(..., help="First checkpoint path"),
    checkpoint_b: Path = typer.Argument(..., help="Second checkpoint path"),
    prompts_file: Optional[Path] = typer.Option(None, help="Text file with custom prompts (one per line)"),
    max_tokens: int = typer.Option(150, help="Max tokens to generate"),
    temperature: float = typer.Option(0.2),
    device: str = typer.Option("cuda"),
):
    custom_prompts = None
    if prompts_file and prompts_file.exists():
        custom_prompts = [l.strip() for l in prompts_file.read_text().splitlines() if l.strip()]

    run_ab_comparison(
        checkpoint_a, checkpoint_b,
        custom_prompts=custom_prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=device,
    )

if __name__ == "__main__":
    app()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/evaluation/ab_comparison/__init__.py` | New package |
| `src/evaluation/ab_comparison/loader.py` | Checkpoint bundle loader |
| `src/evaluation/ab_comparison/generator.py` | Generation + per-prompt metrics |
| `src/evaluation/ab_comparison/display.py` | Side-by-side table, summary |
| `src/evaluation/ab_comparison/runner.py` | Main comparison orchestrator |
| `src/evaluation/ab_comparison/__main__.py` | CLI entry point |
| `src/menu/menus/evaluate_menu.py` | Add "A/B Compare Checkpoints" item |

---

## Testing Strategy

- **Loader test**: Load a real checkpoint with known step/loss; verify fields populated correctly
- **Generation test**: Use a tiny model, run `generate_for_prompt()` with a short prompt, verify result is a `GenerationResult` with all fields populated
- **Perplexity computation test**: Known prompt + completion → verify perplexity is finite and positive
- **Side-by-side render test**: Create two canned `GenerationResult` objects, call `render_prompt_comparison()`, verify no exceptions
- **Summary metrics test**: 10 result pairs where B has consistently lower perplexity → B wins overall
- **Winner edge case**: All metrics tied → "Tie" output
- **Custom prompts test**: Provide a prompts file, verify those prompts are used instead of defaults

---

## Performance Considerations

- Loading two checkpoints simultaneously uses 2× model VRAM (2 × ~200MB for bf16 50M model = ~400MB). Well within 10GB capacity.
- Alternatively, load one checkpoint, generate all completions, save to memory, unload, load second checkpoint. This halves VRAM at the cost of 2× load time.
- For 10 prompts at 150 tokens each, 2 models: ~3000 generated tokens total. At 1000 tok/s: ~3 seconds of generation.
- Total comparison time: ~30 seconds (including loads and tsc checks).
- If comparing very large models, add a `--sequential` flag to load and unload each model separately.

---

## Dependencies

| Package | Use | Already? |
|---------|-----|----------|
| `rich` | All display | Yes |
| `torch` | Model loading + generation | Yes |
| `statistics` | Mean computation | stdlib |
| `tsc` | Syntax checking (optional) | System |

No new Python packages.

---

## Estimated Complexity

**Medium** — 2 days.

- Loader + generation: 3 hours
- Side-by-side Rich display: 3 hours
- Metrics computation + summary table: 2.5 hours
- Runner orchestrator: 1.5 hours
- CLI + custom prompts: 1 hour
- Testing: 3 hours

Total: ~14 hours

---

## 2026 Best Practices

- **Low temperature for comparison**: Use temperature=0.2 for A/B comparisons. Higher temperatures introduce randomness that makes the comparison noisy — you'd need multiple samples to average. At 0.2, outputs are mostly deterministic and directly comparable.
- **Same random seed for both models**: If using any stochastic sampling, fix the seed before generating from each checkpoint. This ensures differences are due to the model, not luck.
- **Perplexity of completion, not full sequence**: When comparing, compute perplexity only on the generated tokens (conditioned on the prompt). This measures how confident each model is about its own output.
- **Present qualitative before quantitative**: Show the actual generated code first, then the metrics. Numbers alone can mislead — a model that generates 200 tokens of garbage has "higher length" but is not better.
- **Support custom prompt files**: Power users often want to compare on their own test prompts (their specific codebase patterns). The `--prompts-file` option is essential for this.
- **Document the "winner" algorithm**: The winner is determined by majority vote across metrics. Be explicit about this so users understand what "B wins" means.
