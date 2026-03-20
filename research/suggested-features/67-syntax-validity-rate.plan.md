# 67 - Syntax Validity Rate

## Overview

Measure the percentage of model-generated TypeScript samples that parse as syntactically valid. Use `tree-sitter-typescript` for fast (~1ms/parse) evaluation. Track this metric over training steps as an early-signal indicator of learning progress, and break down common syntax error types.

**Feature flag:** `config.metrics.syntax_validity_rate.enabled` (default: `true` when tree-sitter is available)

---

## Motivation

Syntax validity rate is the fastest and most actionable training metric available:

- **Speed**: tree-sitter parses TypeScript in ~0.5-2ms per file. Running 100 samples takes under 1 second. Compare this to `tsc` (~500ms per invocation) or execution (~1s+).
- **Early signal**: syntax validity should improve rapidly in the first few thousand steps. If it doesn't, there's likely a tokenization bug or training data problem.
- **Clear improvement curve**: goes from ~5% (random weights) to ~80%+ (well-trained model). The curve is smooth and interpretable.
- **Error taxonomy**: tree-sitter error nodes tell you *where* parsing fails and what was expected. This helps debug training data quality issues.

This metric complements perplexity (which measures distribution matching) and type correctness (which is expensive to compute). Syntax validity is the cheap, fast, early-stopping-friendly signal.

---

## Architecture / Design

### Measurement Protocol

1. Sample N prompts from a fixed benchmark prompt library (same prompts every evaluation for comparability)
2. Generate one completion per prompt at `temperature=0.7`
3. Concatenate prompt + completion into a full TypeScript snippet
4. Parse with tree-sitter, check `root_node.has_error`
5. Report: `valid / N` as the syntax validity rate
6. For invalid samples: collect error node positions and error messages for breakdown

### Error Classification

```python
# tree-sitter marks parse errors with error nodes.
# We classify based on the sibling tokens around the error node:

ERROR_PATTERNS = {
    "missing_semicolon": re.compile(r'expected ";"'),
    "missing_closing_brace": re.compile(r'expected "}"'),
    "missing_closing_paren": re.compile(r'expected "\\)"'),
    "unexpected_token": re.compile(r'unexpected token'),
    "unexpected_eof": re.compile(r'unexpected end'),
    "invalid_expression": re.compile(r'expected expression'),
}
```

---

## Implementation Steps

### Step 1: Parser Wrapper (`eval/syntax_eval.py`)

```python
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tree_sitter_typescript as ts_ts
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

@dataclass
class SyntaxResult:
    valid: bool
    error_count: int
    errors: list[dict] = field(default_factory=list)
    parse_time_ms: float = 0.0

class TypeScriptSyntaxChecker:
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter-typescript is required for syntax checking. "
                "Install with: pip install tree-sitter-typescript"
            )
        language = Language(ts_ts.language())
        self.parser = Parser(language)

    def check(self, code: str) -> SyntaxResult:
        import time
        t0 = time.perf_counter()
        tree = self.parser.parse(code.encode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        errors = []
        self._collect_errors(tree.root_node, errors)

        return SyntaxResult(
            valid=not tree.root_node.has_error,
            error_count=len(errors),
            errors=errors[:5],  # cap at 5 for storage efficiency
            parse_time_ms=round(elapsed_ms, 2),
        )

    def _collect_errors(self, node, errors: list, depth: int = 0):
        if depth > 20:  # prevent runaway recursion on malformed input
            return
        if node.type == "ERROR" or node.is_missing:
            errors.append({
                "type": node.type,
                "start": node.start_point,
                "end": node.end_point,
                "text": node.text.decode("utf-8")[:50] if node.text else "",
                "is_missing": node.is_missing,
            })
        for child in node.children:
            self._collect_errors(child, errors, depth + 1)

    def check_batch(self, codes: list[str]) -> list[SyntaxResult]:
        return [self.check(code) for code in codes]
```

### Step 2: Syntax Evaluator (implements `BaseEvaluator`) (`eval/syntax_eval.py`)

```python
from eval.base_evaluator import BaseEvaluator, EvalResult
from collections import Counter
import torch

class SyntaxValidityEvaluator(BaseEvaluator):
    name = "syntax_validity_rate"

    EVAL_PROMPTS = [
        "function greet(name: string): string {",
        "interface User {\n  id: string;\n  name: string;\n}",
        "const numbers: number[] = [1, 2, 3];\nconst doubled = numbers.map(",
        "async function fetchData<T>(url: string): Promise<T> {",
        "class Stack<T> {\n  private items: T[] = [];\n  push(item: T) {",
        "type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };",
        "export const add = (a: number, b: number) =>",
        "enum Direction { Up, Down, Left, Right }",
        "const obj = { name: 'Alice', age: 30 } as const;\ntype Name = typeof obj[",
        "try {\n  const data = JSON.parse(input);",
    ]

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        self.checker = TypeScriptSyntaxChecker() if TREE_SITTER_AVAILABLE else None

    def is_available(self) -> bool:
        return TREE_SITTER_AVAILABLE

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        if not self.checker:
            return EvalResult(name=self.name, value=0.0, details={"error": "tree-sitter unavailable"})

        prompts = (self.EVAL_PROMPTS * (self.n_samples // len(self.EVAL_PROMPTS) + 1))[:self.n_samples]
        model.eval()

        results = []
        all_error_types = Counter()

        with torch.inference_mode():
            for prompt in prompts:
                tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens]).to(model.device)

                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                )
                generated = tokenizer.decode(output_ids[0][len(tokens):].tolist())
                full_code = prompt + generated

                result = self.checker.check(full_code)
                results.append(result)

                for err in result.errors:
                    # Classify error node type
                    if err.get("is_missing"):
                        all_error_types["missing_token"] += 1
                    else:
                        all_error_types["parse_error"] += 1

        valid_count = sum(1 for r in results if r.valid)
        validity_rate = valid_count / len(results)
        avg_parse_ms = sum(r.parse_time_ms for r in results) / len(results)

        return EvalResult(
            name=self.name,
            value=validity_rate,
            details={
                "valid": valid_count,
                "total": len(results),
                "avg_parse_ms": round(avg_parse_ms, 2),
                "top_errors": dict(all_error_types.most_common(5)),
            }
        )
```

### Step 3: Standalone Batch Evaluation (`eval/batch_syntax_check.py`)

```python
# cola-coder eval syntax \
#   --checkpoint checkpoints/step-5000.safetensors \
#   --samples 200 \
#   --output-json syntax_results.json

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from collections import Counter

def run_batch_syntax_eval(checkpoint_path: Path, n_samples: int = 100) -> dict:
    checker = TypeScriptSyntaxChecker()
    model, tokenizer = load_checkpoint(checkpoint_path)

    evaluator = SyntaxValidityEvaluator(n_samples=n_samples)
    result = evaluator.run(model, tokenizer, config={})

    return {
        "checkpoint": str(checkpoint_path),
        "n_samples": n_samples,
        "validity_rate": result.value,
        "details": result.details,
    }

def print_syntax_report(report: dict):
    console = Console()
    console.rule("[bold]Syntax Validity Report[/]")
    console.print(f"\n  Checkpoint: [cyan]{report['checkpoint']}[/]")
    console.print(f"  Samples:    {report['n_samples']}")
    pct = report["validity_rate"] * 100
    color = "green" if pct > 70 else "yellow" if pct > 40 else "red"
    console.print(f"  Valid:      [{color}]{pct:.1f}%[/]  ({report['details']['valid']}/{report['details']['total']})\n")

    if report["details"].get("top_errors"):
        table = Table(title="Top Error Types", show_header=True)
        table.add_column("Error Type", style="dim")
        table.add_column("Count", justify="right")
        for err_type, count in report["details"]["top_errors"].items():
            table.add_row(err_type, str(count))
        console.print(table)
```

### Step 4: Training Step Tracking

In `eval/continuous_eval.py`, `SyntaxValidityEvaluator` is already a `BaseEvaluator` and integrates automatically when listed in the config:

```yaml
continuous_eval:
  evaluators:
    - syntax_validity    # runs at every eval_interval
```

The history entry looks like:
```json
{
  "step": 1000,
  "metrics": {
    "syntax_validity_rate": 0.62,
    "syntax_validity_rate__valid": 31,
    "syntax_validity_rate__total": 50,
    "syntax_validity_rate__avg_parse_ms": 0.84
  }
}
```

### Step 5: Error Breakdown Tracker (`eval/error_tracker.py`)

```python
def aggregate_error_trends(history: list[dict]) -> dict:
    """Extract syntax error trends across training steps."""
    steps = []
    rates = []
    error_breakdowns = []

    for entry in history:
        metrics = entry.get("metrics", {})
        rate = metrics.get("syntax_validity_rate")
        if rate is not None:
            steps.append(entry["step"])
            rates.append(rate)

            # Look for error detail keys
            errors = {
                k.replace("syntax_validity_rate__", ""): v
                for k, v in metrics.items()
                if k.startswith("syntax_validity_rate__top_errors")
            }
            error_breakdowns.append(errors)

    return {
        "steps": steps,
        "validity_rates": rates,
        "error_breakdowns": error_breakdowns,
    }
```

---

## Key Files to Modify

- `eval/syntax_eval.py` - New file: checker and evaluator
- `eval/batch_syntax_check.py` - New file: standalone batch CLI
- `eval/continuous_eval.py` - Register `SyntaxValidityEvaluator` by name
- `cli/eval_cmd.py` - Add `syntax` subcommand
- `config/eval.yaml` - Add `syntax_validity` section
- `requirements.txt` - Add `tree-sitter-typescript` as optional dependency

---

## Testing Strategy

1. **Parser unit tests**: check a known valid TypeScript snippet (e.g., `const x: number = 1;`) returns `valid=True`, error_count=0.
2. **Error detection test**: check a deliberately broken snippet (e.g., `const x: number = ;`) returns `valid=False`, error_count > 0.
3. **Batch consistency test**: run `check_batch` on 10 snippets, verify results match individual `check()` calls.
4. **Performance test**: parse 1000 snippets, assert total time < 5 seconds (1000 * 5ms budget).
5. **Evaluator integration test**: run `SyntaxValidityEvaluator.run()` with a dummy model that always outputs `return 0;`, verify validity rate > 0.5.
6. **Missing token detection**: parse `function foo(` (unclosed paren), verify `is_missing` error is detected.

---

## Performance Considerations

- tree-sitter in Python: ~0.5-2ms per parse depending on code length. 100 samples = <200ms.
- The `Parser` object should be reused (not recreated per parse). `TypeScriptSyntaxChecker` creates it once in `__init__`.
- For very high throughput (1000+ samples), consider `concurrent.futures.ThreadPoolExecutor` since tree-sitter releases the GIL.
- The `_collect_errors` recursive traversal can be slow on deeply nested ASTs. The `depth > 20` guard prevents pathological cases.
- Model generation (not parsing) is the bottleneck. At 50 samples × 150 tokens each, generation dominates. Consider reducing `n_samples` to 20 for frequent eval intervals.

---

## Dependencies

```
tree-sitter>=0.21.0
tree-sitter-typescript>=0.21.0
```

These are lightweight wheels with no system dependencies. Available on PyPI for all major platforms.

---

## Estimated Complexity

**Low.** Tree-sitter has an excellent Python API. The checker is ~50 lines. The evaluator is ~80 lines. Integration with the continuous eval system via `BaseEvaluator` is straightforward. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **Fixed prompt library for eval**: always use the same prompts across training runs to make metrics comparable. Store prompts in a versioned YAML file, not hardcoded in the evaluator.
- **tree-sitter, not regex**: never use regex to validate TypeScript syntax. Regex will give wrong answers on edge cases and provide no error location information. tree-sitter is the right tool.
- **Error rate as complement**: report *error rate* (1 - validity_rate) as well as validity rate. "12% error rate" is more interpretable than "88% valid" when debugging.
- **Version tree-sitter grammar**: pin `tree-sitter-typescript` to a specific version. Grammar updates can change what parses as valid, breaking metric comparability across training runs.
- **Don't use as sole gating metric**: syntax validity alone doesn't mean the code is correct or useful. It's an early-training signal and a minimum bar, not a measure of quality. Always pair with semantic metrics (type correctness, nano-benchmark pass rate).
