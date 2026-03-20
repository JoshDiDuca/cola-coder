# 69 - Token Efficiency Metric

## Overview

Measure how concisely the model solves problems relative to a reference solution. The metric is `tokens_generated / minimum_tokens_needed`, where minimum is derived from a curated reference solution. Penalize redundancy (unnecessary type annotations, dead code, excessive comments) and reward concise, idiomatic TypeScript. Track over training steps.

**Feature flag:** `config.metrics.token_efficiency.enabled` (default: `false`; requires reference benchmark suite)

---

## Motivation

A model that generates 400 tokens to solve a problem solvable in 80 tokens is less useful than one that generates 90 tokens. Verbosity is a real quality problem in code generation:

- Redundant type annotations that TypeScript infers automatically
- Unnecessary intermediate variables
- Copy-paste boilerplate that doesn't contribute to the solution
- Excessive inline comments explaining obvious code
- Dead code branches that are never reached

Token efficiency captures a dimension of quality that neither syntax validity nor type correctness measures. A perfectly type-correct but verbose solution is worse than a concise equivalent.

**Important**: this metric only makes sense on a benchmark where a reference solution exists. It should not be used on open-ended generation.

---

## Architecture / Design

### Metric Formula

```
efficiency = ref_tokens / model_tokens

efficiency = 1.0   → model matches reference length exactly
efficiency > 1.0   → model is MORE concise than reference (rare, usually truncation)
efficiency < 1.0   → model is MORE verbose than reference (penalized)
```

Range [0.0, 1.0] for display purposes; capped at 1.0 (being more concise than reference counts as 1.0).

```python
def compute_efficiency(model_tokens: int, ref_tokens: int) -> float:
    if model_tokens == 0:
        return 0.0
    raw = ref_tokens / model_tokens
    return min(raw, 1.0)  # cap at 1.0
```

### Verbosity Analysis

Beyond the scalar metric, analyze *what kinds* of verbosity the model produces:

```python
VerbositySignal = Literal[
    "redundant_type_annotation",    # const x: string = "hello" → const x = "hello"
    "unnecessary_variable",          # const temp = fn(); return temp; → return fn();
    "dead_code",                     # unreachable branches, unused imports
    "excessive_comments",            # >1 comment line per 3 code lines
    "explicit_return_type_on_trivial", # (a: number, b: number): number → inferred
]
```

---

## Implementation Steps

### Step 1: Reference Benchmark Suite (`eval/benchmarks/efficiency_suite.yaml`)

```yaml
# Each item: a prompt and a reference solution
problems:
  - id: add-two-numbers
    prompt: "// Add two numbers\nfunction add("
    reference: |
      function add(a: number, b: number): number {
        return a + b;
      }
    ref_tokens: 18   # pre-computed

  - id: filter-evens
    prompt: "// Filter even numbers from an array\nfunction filterEvens("
    reference: |
      function filterEvens(nums: number[]): number[] {
        return nums.filter(n => n % 2 === 0);
      }
    ref_tokens: 22

  - id: safe-get
    prompt: "// Get a value from an object safely, returning null if not found\nfunction safeGet<T>("
    reference: |
      function safeGet<T>(obj: Record<string, T>, key: string): T | null {
        return obj[key] ?? null;
      }
    ref_tokens: 28

  - id: debounce
    prompt: "// Create a debounce function\nfunction debounce("
    reference: |
      function debounce<T extends (...args: unknown[]) => void>(
        fn: T, delay: number
      ): (...args: Parameters<T>) => void {
        let timer: ReturnType<typeof setTimeout>;
        return (...args) => {
          clearTimeout(timer);
          timer = setTimeout(() => fn(...args), delay);
        };
      }
    ref_tokens: 58

  - id: group-by
    prompt: "// Group array items by a key function\nfunction groupBy<T, K extends string | number>("
    reference: |
      function groupBy<T, K extends string | number>(
        items: T[],
        keyFn: (item: T) => K
      ): Record<K, T[]> {
        return items.reduce((acc, item) => {
          const key = keyFn(item);
          (acc[key] ??= []).push(item);
          return acc;
        }, {} as Record<K, T[]>);
      }
    ref_tokens: 68
```

### Step 2: Verbosity Analyzer (`eval/verbosity_analyzer.py`)

```python
import re
from dataclasses import dataclass, field

@dataclass
class VerbosityReport:
    redundant_annotations: int = 0
    unnecessary_variables: int = 0
    dead_code_indicators: int = 0
    excessive_comments: int = 0
    total_tokens: int = 0
    code_lines: int = 0
    comment_lines: int = 0

class VerbosityAnalyzer:
    """Static analysis for TypeScript verbosity patterns."""

    # Patterns that often indicate redundant annotations
    REDUNDANT_ANNOTATION_PATTERNS = [
        # const x: string = "literal" → string is inferred
        r'const\s+\w+:\s*string\s*=\s*"',
        r"const\s+\w+:\s*string\s*=\s*'",
        # const x: number = 0 → number is inferred
        r'const\s+\w+:\s*number\s*=\s*\d',
        # const x: boolean = true/false → boolean is inferred
        r'const\s+\w+:\s*boolean\s*=\s*(true|false)',
        # const arr: any[] = [] → any is too broad; but also: string[] = []
        r'const\s+\w+:\s*\w+\[\]\s*=\s*\[\]',
    ]

    INTERMEDIATE_VARIABLE_PATTERN = re.compile(
        r'const\s+(\w+)\s*=\s*[^;]+;\s*\n\s*return\s+\1\s*;',
        re.MULTILINE,
    )

    def analyze(self, code: str) -> VerbosityReport:
        report = VerbosityReport()

        lines = code.split("\n")
        report.total_tokens = len(code.split())  # rough proxy

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
                report.comment_lines += 1
            elif stripped:
                report.code_lines += 1

        # Check redundant annotations
        for pattern in self.REDUNDANT_ANNOTATION_PATTERNS:
            report.redundant_annotations += len(re.findall(pattern, code))

        # Check intermediate variables
        report.unnecessary_variables = len(
            self.INTERMEDIATE_VARIABLE_PATTERN.findall(code)
        )

        # Excessive comments: more than 1 comment per 3 code lines
        if report.code_lines > 0:
            comment_ratio = report.comment_lines / report.code_lines
            if comment_ratio > 0.33:
                report.excessive_comments = int(
                    report.comment_lines - report.code_lines * 0.33
                )

        return report
```

### Step 3: Token Efficiency Evaluator (`eval/token_efficiency_eval.py`)

```python
from eval.base_evaluator import BaseEvaluator, EvalResult
from eval.verbosity_analyzer import VerbosityAnalyzer
import yaml
import torch
from pathlib import Path

class TokenEfficiencyEvaluator(BaseEvaluator):
    name = "token_efficiency"

    def __init__(self, benchmark_path: str = "eval/benchmarks/efficiency_suite.yaml"):
        self.benchmark_path = Path(benchmark_path)
        self.analyzer = VerbosityAnalyzer()

    def is_available(self) -> bool:
        return self.benchmark_path.exists()

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        with open(self.benchmark_path) as f:
            benchmark = yaml.safe_load(f)

        problems = benchmark["problems"]
        model.eval()

        efficiency_scores = []
        verbosity_totals = {
            "redundant_annotations": 0,
            "unnecessary_variables": 0,
            "excessive_comments": 0,
        }

        with torch.inference_mode():
            for problem in problems:
                prompt = problem["prompt"]
                ref_tokens = problem["ref_tokens"]

                # Generate completion
                tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens]).to(model.device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=300,
                    temperature=0.3,   # low temp for efficiency eval
                    do_sample=True,
                )
                generated = tokenizer.decode(output_ids[0][len(tokens):].tolist())

                # Count model tokens
                model_token_count = len(output_ids[0]) - len(tokens)
                efficiency = min(ref_tokens / max(model_token_count, 1), 1.0)
                efficiency_scores.append(efficiency)

                # Verbosity analysis
                report = self.analyzer.analyze(prompt + generated)
                verbosity_totals["redundant_annotations"] += report.redundant_annotations
                verbosity_totals["unnecessary_variables"] += report.unnecessary_variables
                verbosity_totals["excessive_comments"] += report.excessive_comments

        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)

        # Per-problem breakdown
        per_problem = [
            {
                "id": p["id"],
                "ref_tokens": p["ref_tokens"],
                "efficiency": round(e, 3),
            }
            for p, e in zip(problems, efficiency_scores)
        ]

        return EvalResult(
            name=self.name,
            value=avg_efficiency,
            details={
                "per_problem": per_problem,
                "avg_efficiency": round(avg_efficiency, 3),
                "verbosity": {
                    k: v // len(problems)   # average per problem
                    for k, v in verbosity_totals.items()
                },
                "n_problems": len(problems),
            }
        )
```

### Step 4: CLI Report

```python
# cola-coder eval efficiency \
#   --checkpoint checkpoints/step-5000.safetensors \
#   --benchmark eval/benchmarks/efficiency_suite.yaml

def cmd_efficiency_eval(args):
    from rich.table import Table
    from rich.console import Console

    model, tokenizer = load_checkpoint(args.checkpoint)
    evaluator = TokenEfficiencyEvaluator(benchmark_path=args.benchmark)
    result = evaluator.run(model, tokenizer, config={})

    console = Console()
    console.rule("[bold]Token Efficiency Report[/]")

    avg = result.value
    color = "green" if avg > 0.7 else "yellow" if avg > 0.5 else "red"
    console.print(f"\n  Average efficiency: [{color}]{avg:.1%}[/]\n")

    # Per-problem table
    table = Table(title="Per-Problem Efficiency")
    table.add_column("Problem ID")
    table.add_column("Ref Tokens", justify="right")
    table.add_column("Efficiency", justify="right")
    for item in result.details["per_problem"]:
        eff = item["efficiency"]
        c = "green" if eff > 0.7 else "yellow" if eff > 0.5 else "red"
        bar = "█" * int(eff * 10) + "░" * (10 - int(eff * 10))
        table.add_row(
            item["id"],
            str(item["ref_tokens"]),
            f"[{c}]{bar} {eff:.0%}[/]",
        )
    console.print(table)

    # Verbosity summary
    console.print("\n[bold]Verbosity Signals:[/]")
    v = result.details["verbosity"]
    console.print(f"  Redundant annotations (avg/problem): {v['redundant_annotations']}")
    console.print(f"  Unnecessary variables (avg/problem): {v['unnecessary_variables']}")
    console.print(f"  Excessive comment lines (avg/problem): {v['excessive_comments']}")
```

### Step 5: Reference Token Count Pre-computation

```python
# Utility to pre-compute ref_tokens for the benchmark YAML
# cola-coder eval efficiency update-refs --tokenizer tokenizer/

def update_ref_tokens(benchmark_path: Path, tokenizer):
    with open(benchmark_path) as f:
        benchmark = yaml.safe_load(f)

    for problem in benchmark["problems"]:
        ref_code = problem.get("reference", "")
        ref_tokens = len(tokenizer.encode(ref_code))
        problem["ref_tokens"] = ref_tokens
        print(f"  {problem['id']}: {ref_tokens} tokens")

    with open(benchmark_path, "w") as f:
        yaml.dump(benchmark, f, default_flow_style=False, allow_unicode=True)
```

---

## Key Files to Modify

- `eval/token_efficiency_eval.py` - New file: evaluator
- `eval/verbosity_analyzer.py` - New file: static verbosity analysis
- `eval/benchmarks/efficiency_suite.yaml` - New file: reference problem set
- `cli/eval_cmd.py` - Add `efficiency` subcommand
- `config/eval.yaml` - Add `token_efficiency` section
- `eval/continuous_eval.py` - Register evaluator by name

---

## Testing Strategy

1. **Efficiency formula test**: `compute_efficiency(model_tokens=100, ref_tokens=50)` → `0.5`; `compute_efficiency(50, 50)` → `1.0`; `compute_efficiency(40, 50)` → `1.0` (capped).
2. **Verbosity analyzer test**: check a code snippet with known `const x: string = "hello"` pattern, assert `redundant_annotations == 1`.
3. **Intermediate variable detection**: check `const result = fn(); return result;` pattern, assert `unnecessary_variables == 1`.
4. **Benchmark YAML validation**: load `efficiency_suite.yaml`, assert all problems have `id`, `prompt`, `reference`, `ref_tokens` fields.
5. **Evaluator integration test**: run with a model that outputs only the reference solution, assert average efficiency == 1.0.
6. **Zero-length guard**: pass `model_tokens=0` to efficiency formula, assert no ZeroDivisionError.

---

## Performance Considerations

- Generation at temperature=0.3 is deterministic enough for reliable benchmarking. Use `do_sample=False` (greedy) for perfectly reproducible numbers, though this may not reflect typical generation quality.
- Reference token counts are pre-computed and stored in YAML. No tokenizer call at eval time for the reference side.
- Verbosity analysis is pure regex (~0.1ms per snippet). Negligible.
- The benchmark suite should have 20-50 problems. More problems improve statistical reliability but increase eval time. 20 problems × 300 tokens = ~6000 tokens of generation, which takes ~5s on a GPU.

---

## Dependencies

No new Python dependencies. Uses `yaml`, `re`, `torch` (already required).

---

## Estimated Complexity

**Medium.** The efficiency metric itself is trivial. The main effort is in curating a high-quality benchmark suite with accurate reference solutions and correct reference token counts, and implementing the verbosity analyzer with enough signal fidelity to be useful. Estimated implementation time: 2-3 days (1 day code, 1-2 days benchmark curation).

---

## 2026 Best Practices

- **Reference tokenizer must match training tokenizer**: pre-compute `ref_tokens` using the same tokenizer the model uses. If you change tokenizers, re-run `update-refs`.
- **Multiple reference solutions**: some problems have multiple valid concise solutions. Consider storing several references and taking the shortest as the target. This prevents penalizing valid alternative idioms.
- **Separate efficiency from correctness**: a model that generates 10 tokens by truncating early will have high efficiency but obviously wrong outputs. Always pair this metric with a type correctness or functional correctness check. Only count efficiency for samples that pass some minimum correctness bar.
- **Don't train directly on this metric**: using token efficiency as a training reward will cause the model to learn to truncate early, not to be genuinely concise. Use it as an evaluation metric only, or apply it very carefully as an RLHF signal with correctness guards.
- **Benchmark versioning**: version the `efficiency_suite.yaml` file. Changing reference solutions or adding problems breaks comparability. Tie benchmark version to model eval reports.
