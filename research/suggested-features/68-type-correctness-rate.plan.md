# 68 - Type Correctness Rate

## Overview

Measure the percentage of model-generated TypeScript that passes `tsc --strict --noEmit`. Extends the existing `type_check` reward infrastructure into a batch evaluation metric. Generates N samples, writes each to a temp `.ts` file, runs `tsc`, counts passes, and tracks the rate over training steps.

**Feature flag:** `config.metrics.type_correctness_rate.enabled` (default: `false`; requires tsc in PATH and is slow)

---

## Motivation

Syntax validity (plan 67) tells you if the model knows TypeScript grammar. Type correctness tells you if the model understands TypeScript's type system—a much higher bar. A snippet like `const x: number = "hello"` is syntactically valid but type-incorrect.

Type correctness rate is the metric most aligned with real-world usefulness: code that passes `tsc --strict` will generally work correctly in a TypeScript project. It directly measures the quality of the model's type reasoning.

Cola-Coder already uses TypeScript compilation as a reward signal during training (the `type_check` reward). This plan reuses that infrastructure for systematic batch evaluation rather than just per-sample rewards.

**Expected trajectory**: starts near 0%, improves slowly through mid-training, accelerates after type-aware pretraining or RLHF stages. A trained model should reach 40-70% on standard prompts.

---

## Architecture / Design

### Evaluation Protocol

```
1. Sample N prompts from a fixed prompt library (same as syntax eval or extended)
2. Generate completions at temperature=0.7
3. For each (prompt, completion) pair:
   a. Write full_code = prompt + completion to /tmp/cola_eval_{i}.ts
   b. Write minimal tsconfig.json to same temp dir
   c. Run: tsc --project tsconfig.json
   d. Check return code (0 = pass, non-zero = fail)
   e. Parse error output for error codes (TS2xxx)
4. Report: passes / N
5. Aggregate error code frequency
```

### Parallelization

Running 50 `tsc` invocations sequentially takes ~50s (1s each). Running 10 in parallel via `subprocess` pool reduces this to ~8s. The parallelization is safe because each invocation uses an isolated temp directory.

---

## Implementation Steps

### Step 1: Type Check Runner (`eval/type_check_runner.py`)

```python
import subprocess
import tempfile
import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

TSC_TIMEOUT = 20  # seconds

TSCONFIG_STRICT = {
    "compilerOptions": {
        "strict": True,
        "noEmit": True,
        "target": "ES2020",
        "module": "commonjs",
        "skipLibCheck": True,
    }
}

@dataclass
class TypeCheckResult:
    passed: bool
    return_code: int
    error_lines: list[str]
    error_codes: list[str]       # e.g., ["TS2322", "TS2345"]
    stderr_preview: str

def check_single(code: str, timeout: int = TSC_TIMEOUT) -> TypeCheckResult:
    """Run tsc on a single code snippet in an isolated temp directory."""
    with tempfile.TemporaryDirectory(prefix="cola_typecheck_") as tmpdir:
        ts_file = Path(tmpdir) / "snippet.ts"
        tsconfig = Path(tmpdir) / "tsconfig.json"

        ts_file.write_text(code, encoding="utf-8")
        tsconfig.write_text(json.dumps(TSCONFIG_STRICT, indent=2))

        try:
            result = subprocess.run(
                ["tsc", "--project", str(tsconfig)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return TypeCheckResult(
                passed=False,
                return_code=-1,
                error_lines=[],
                error_codes=["TIMEOUT"],
                stderr_preview="tsc timed out",
            )
        except FileNotFoundError:
            return TypeCheckResult(
                passed=False,
                return_code=-2,
                error_lines=[],
                error_codes=["TSC_NOT_FOUND"],
                stderr_preview="tsc not found in PATH",
            )

        output = result.stdout + result.stderr
        error_lines = [l for l in output.split("\n") if l.strip()]

        # Extract TS error codes: e.g., "error TS2322:"
        error_codes = re.findall(r'error (TS\d+)', output)

        return TypeCheckResult(
            passed=result.returncode == 0,
            return_code=result.returncode,
            error_lines=error_lines[:10],
            error_codes=list(set(error_codes)),
            stderr_preview=output[:200],
        )

def check_batch_parallel(
    codes: list[str],
    max_workers: int = 8,
    timeout: int = TSC_TIMEOUT,
) -> list[TypeCheckResult]:
    """Run type checks in parallel."""
    results = [None] * len(codes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_single, code, timeout): i
            for i, code in enumerate(codes)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = TypeCheckResult(
                    passed=False, return_code=-3,
                    error_lines=[], error_codes=["INTERNAL_ERROR"],
                    stderr_preview=str(e)
                )

    return results
```

### Step 2: Type Correctness Evaluator (`eval/type_correctness_eval.py`)

```python
from eval.base_evaluator import BaseEvaluator, EvalResult
from eval.type_check_runner import check_batch_parallel
from collections import Counter
import torch
import shutil

# Extended prompt library for type-focused evaluation
TYPE_CHECK_PROMPTS = [
    # Type annotations
    "function identity<T>(value: T): T {\n  return",
    "const map = new Map<string, number>();\nmap.set('key',",
    # Object types
    "interface Config { host: string; port: number; }\nconst cfg: Config = {",
    # Union types
    "type StringOrNumber = string | number;\nfunction format(val: StringOrNumber): string {",
    # Generics
    "function first<T>(arr: T[]): T | undefined {\n  return",
    # Async/await
    "async function fetchUser(id: string): Promise<{ name: string }> {",
    # Type narrowing
    "function process(val: string | number): string {\n  if (typeof val === 'string') {",
    # Class with types
    "class Queue<T> {\n  private items: T[] = [];\n  enqueue(item: T): void {",
    # Readonly
    "function freeze<T>(obj: T): Readonly<T> {\n  return Object.freeze(",
    # Optional chaining return
    "interface User { address?: { city: string } }\nfunction getCity(u: User): string | undefined {",
    # Record type
    "const scores: Record<string, number> = {};\nscores['alice'] =",
    # Partial utility
    "function updateUser(user: User, updates: Partial<User>): User {\n  return",
    # Type assertion
    "function asString(val: unknown): string {\n  return",
    # Enum usage
    "enum Status { Active = 'active', Inactive = 'inactive' }\nconst s: Status =",
    # Tuple
    "function split(s: string): [string, string] {\n  const idx = s.indexOf(':');\n  return",
]

class TypeCorrectnessEvaluator(BaseEvaluator):
    name = "type_correctness_rate"

    def __init__(self, n_samples: int = 30, max_workers: int = 8):
        self.n_samples = n_samples
        self.max_workers = max_workers

    def is_available(self) -> bool:
        return shutil.which("tsc") is not None

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        prompts = (TYPE_CHECK_PROMPTS * (self.n_samples // len(TYPE_CHECK_PROMPTS) + 1))[:self.n_samples]

        model.eval()
        generated_codes = []

        with torch.inference_mode():
            for prompt in prompts:
                tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens]).to(model.device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                )
                completion = tokenizer.decode(output_ids[0][len(tokens):].tolist())
                generated_codes.append(prompt + completion)

        # Run type checks in parallel
        results = check_batch_parallel(
            generated_codes,
            max_workers=self.max_workers,
        )

        passed = sum(1 for r in results if r.passed)
        type_correctness_rate = passed / len(results)

        # Aggregate error codes
        all_error_codes = Counter()
        for r in results:
            for code in r.error_codes:
                all_error_codes[code] += 1

        # Error count distribution
        error_counts = Counter(len(r.error_codes) for r in results if not r.passed)

        return EvalResult(
            name=self.name,
            value=type_correctness_rate,
            details={
                "passed": passed,
                "total": len(results),
                "top_error_codes": dict(all_error_codes.most_common(10)),
                "error_count_distribution": dict(error_counts),
                "timeout_count": all_error_codes.get("TIMEOUT", 0),
            }
        )
```

### Step 3: Error Code Explainer (`eval/ts_error_codes.py`)

```python
# Map common TypeScript error codes to human-readable descriptions
TS_ERROR_DESCRIPTIONS = {
    "TS2322": "Type mismatch (e.g., string assigned to number)",
    "TS2339": "Property does not exist on type",
    "TS2345": "Argument type mismatch in function call",
    "TS2304": "Cannot find name (undefined variable/type)",
    "TS2305": "Module has no exported member",
    "TS2307": "Cannot find module",
    "TS7006": "Parameter implicitly has 'any' type",
    "TS7017": "Element implicitly has 'any' type",
    "TS2532": "Object is possibly undefined",
    "TS2531": "Object is possibly null",
    "TS2366": "Function lacks ending return statement",
    "TS2349": "Object is not callable",
    "TS2554": "Expected N arguments but got M",
    "TS1005": "Expected token (syntax-adjacent type error)",
}

def explain_errors(error_codes: list[str]) -> list[str]:
    return [
        f"{code}: {TS_ERROR_DESCRIPTIONS.get(code, 'Unknown error')}"
        for code in error_codes
    ]
```

### Step 4: CLI Report Command

```python
# cola-coder eval type-check \
#   --checkpoint checkpoints/step-5000.safetensors \
#   --samples 50 \
#   --output eval_results/typecheck_step5000.json

def cmd_type_check_eval(args):
    from rich.console import Console
    from rich.table import Table

    model, tokenizer = load_checkpoint(args.checkpoint)
    evaluator = TypeCorrectnessEvaluator(
        n_samples=args.samples,
        max_workers=args.workers,
    )
    result = evaluator.run(model, tokenizer, config={})

    console = Console()
    console.rule("[bold]Type Correctness Report[/]")

    pct = result.value * 100
    color = "green" if pct > 50 else "yellow" if pct > 25 else "red"
    console.print(f"\n  Checkpoint: [cyan]{args.checkpoint}[/]")
    console.print(f"  Pass rate:  [{color}]{pct:.1f}%[/]  "
                  f"({result.details['passed']}/{result.details['total']})\n")

    if result.details["top_error_codes"]:
        table = Table(title="Most Common Type Errors")
        table.add_column("Error Code", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Description", style="dim")
        for code, count in sorted(
            result.details["top_error_codes"].items(),
            key=lambda x: -x[1]
        ):
            table.add_row(
                code, str(count),
                TS_ERROR_DESCRIPTIONS.get(code, "—")
            )
        console.print(table)

    # Save JSON
    output = {
        "checkpoint": args.checkpoint,
        "n_samples": args.samples,
        "type_correctness_rate": result.value,
        "details": result.details,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
```

### Step 5: Cross-Checkpoint Comparison

```python
# cola-coder eval type-check compare \
#   --results eval_results/typecheck_step*.json

def cmd_compare_type_checks(result_files: list[str]):
    from rich.table import Table
    results = []
    for f in sorted(result_files):
        data = json.loads(Path(f).read_text())
        results.append(data)

    console = Console()
    table = Table(title="Type Correctness Across Checkpoints")
    table.add_column("Checkpoint")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Top Error", style="dim")

    for r in results:
        pct = r["type_correctness_rate"] * 100
        color = "green" if pct > 50 else "yellow" if pct > 25 else "red"
        top_err = next(iter(r["details"]["top_error_codes"]), "—")
        table.add_row(
            Path(r["checkpoint"]).stem,
            f"[{color}]{pct:.1f}%[/]",
            top_err,
        )
    console.print(table)
```

---

## Key Files to Modify

- `eval/type_check_runner.py` - New file: subprocess runner with parallelization
- `eval/type_correctness_eval.py` - New file: evaluator implementing `BaseEvaluator`
- `eval/ts_error_codes.py` - New file: error code taxonomy
- `cli/eval_cmd.py` - Add `type-check` subcommand
- `config/eval.yaml` - Add `type_correctness_rate` section
- `training/rewards.py` - Optionally unify with existing `type_check` reward

---

## Testing Strategy

1. **Standalone compile test**: write `const x: number = 1;` to temp file, run `check_single`, assert `passed=True`.
2. **Type error detection**: write `const x: number = "hello";`, assert `passed=False` and `error_codes` contains `"TS2322"`.
3. **Parallel consistency**: run `check_batch_parallel` on 10 snippets, compare to sequential results, assert identical outcomes.
4. **Timeout test**: mock subprocess to take 30s, assert `TypeCheckResult` with `error_codes=["TIMEOUT"]`.
5. **No tsc test**: mock `shutil.which("tsc")` to return `None`, assert `is_available()` returns `False`.
6. **Evaluator integration test**: run with a small model, assert result has `value` in [0.0, 1.0].

---

## Performance Considerations

- Each `tsc` invocation: ~500ms-1s including JVM-like Node.js startup overhead.
- 30 samples × 1s / 8 workers = ~4s total for the parallel batch. Acceptable for eval intervals ≥ 1000 steps.
- Node.js starts fresh for each `tsc` call. Investigate `ts-server` or `typescript-language-server` for persistent compilation sessions (~10x speedup), but this adds significant complexity.
- Temp directories are cleaned up by `tempfile.TemporaryDirectory` context managers. No disk leak.
- On Windows, `tsc` startup is slower (~2s). Reduce default `max_workers` to 4 to avoid overwhelming the process pool.
- Consider running type check eval only at milestone steps (e.g., every 5000 steps) rather than every `eval_interval` to minimize training disruption.

---

## Dependencies

- `tsc` (TypeScript compiler) must be in PATH: `npm install -g typescript`
- No new Python dependencies

---

## Estimated Complexity

**Low-Medium.** The subprocess runner reuses patterns from the existing `type_check` reward. The main additions are parallelization, error code aggregation, and the `BaseEvaluator` wrapper. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **Reuse existing `type_check` infrastructure**: the training-time `type_check` reward already runs tsc on generated code. Factor out the subprocess invocation into a shared utility (`type_check_runner.py`) used by both the reward and the evaluator.
- **Isolate each invocation**: always use separate temp directories per invocation. Shared directories cause false failures when files from one generation interfere with another.
- **Error code taxonomy as training signal**: if evaluation shows TS2339 (property doesn't exist) dominates, this suggests the model is hallucinating method names. This can guide training data curation toward examples with correct method usage.
- **skipLibCheck in tsconfig**: always include `skipLibCheck: true` in the eval tsconfig. Without it, tsc checks all referenced `.d.ts` files, which can cause failures unrelated to the generated code.
- **Pin TypeScript version**: use a pinned `tsc` version in your `devDependencies`. Changing TypeScript versions can change what compiles successfully, breaking metric comparability.
