# 02 - Nano Benchmark

## Overview

A curated set of 10 hand-written TypeScript problems specifically calibrated for a 50M-parameter code generation model. Unlike HumanEval (which targets GPT-3+ scale models), the Nano Benchmark uses elementary problems that a small model can realistically solve after adequate training. The benchmark produces a "report card" showing pass rates across syntax validity, TypeScript type correctness, and functional test execution.

---

## Motivation

HumanEval is the de-facto benchmark for code generation, but its problems are too difficult for a 50M model — most require multi-step reasoning, library knowledge, or algorithmic sophistication. Running HumanEval on a small model produces 0-2% pass@1 rates that are meaningless for tracking training progress.

The Nano Benchmark fills the gap:
- **Calibrated difficulty**: Problems solvable by a well-trained 50M model
- **Fast feedback**: 10 problems × 1 sample = ~30 seconds on local GPU
- **TypeScript-native**: Tests `tsc` type checking in addition to runtime correctness
- **Progress tracking**: Run after each major checkpoint to track capability growth

The benchmark also serves as a **sanity check**: if a model can't pass any nano problems, something is fundamentally wrong (bad tokenizer, collapsed training, etc.).

---

## Architecture / Design

### Problem Format

Each problem is defined as a Python dataclass:

```python
@dataclass
class NanoProblem:
    id: str                        # e.g. "nano_01_hello_world"
    prompt: str                    # The TypeScript stub shown to the model
    canonical_solution: str        # Reference solution (not shown to model)
    test_cases: list[TestCase]     # Input/output pairs for evaluation
    type_annotations: bool = True  # Whether to check with tsc
    difficulty: str = "easy"       # easy / medium
```

### Scoring Dimensions

1. **Syntax Validity** (0 or 1): Can the generated code be parsed as valid TypeScript? Check with `tsc --noEmit`.
2. **Type Correctness** (0 or 1): Does TypeScript type-check cleanly with strict mode? (`tsc --strict --noEmit`)
3. **Test Pass Rate** (0.0 to 1.0): Fraction of test cases that produce correct output when executed via `ts-node` or compiled JavaScript.

**Overall score** = mean of (syntax + type + test_pass_rate) / 3 per problem, averaged across all 10 problems.

### The 10 Problems

#### Problem 1: Hello World Function
```typescript
// nano_01_hello_world
// Returns a greeting string for the given name.
function greet(name: string): string {
    // your implementation here
}

// Tests:
// greet("Alice") === "Hello, Alice!"
// greet("World") === "Hello, World!"
```

#### Problem 2: Array Sum
```typescript
// nano_02_array_sum
// Returns the sum of all numbers in the array.
function arraySum(nums: number[]): number {
    // your implementation here
}

// Tests:
// arraySum([1, 2, 3]) === 6
// arraySum([]) === 0
// arraySum([-1, 1]) === 0
```

#### Problem 3: String Reverse
```typescript
// nano_03_string_reverse
// Returns the input string reversed.
function reverseString(s: string): string {
    // your implementation here
}

// Tests:
// reverseString("hello") === "olleh"
// reverseString("") === ""
// reverseString("a") === "a"
```

#### Problem 4: Fibonacci
```typescript
// nano_04_fibonacci
// Returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.
function fibonacci(n: number): number {
    // your implementation here
}

// Tests:
// fibonacci(0) === 0
// fibonacci(1) === 1
// fibonacci(7) === 13
// fibonacci(10) === 55
```

#### Problem 5: isPrime
```typescript
// nano_05_is_prime
// Returns true if n is a prime number, false otherwise.
function isPrime(n: number): boolean {
    // your implementation here
}

// Tests:
// isPrime(2) === true
// isPrime(4) === false
// isPrime(17) === true
// isPrime(1) === false
```

#### Problem 6: FizzBuzz
```typescript
// nano_06_fizzbuzz
// Returns an array of strings for 1..n applying FizzBuzz rules.
function fizzBuzz(n: number): string[] {
    // your implementation here
}

// Tests:
// fizzBuzz(5) === ["1", "2", "Fizz", "4", "Buzz"]
// fizzBuzz(15)[14] === "FizzBuzz"
```

#### Problem 7: Max of Array
```typescript
// nano_07_max_array
// Returns the maximum value in a non-empty number array.
function maxArray(nums: number[]): number {
    // your implementation here
}

// Tests:
// maxArray([3, 1, 4, 1, 5, 9]) === 9
// maxArray([-1, -5, -3]) === -1
// maxArray([42]) === 42
```

#### Problem 8: Palindrome Check
```typescript
// nano_08_palindrome
// Returns true if the string is a palindrome (case-insensitive, ignore spaces).
function isPalindrome(s: string): boolean {
    // your implementation here
}

// Tests:
// isPalindrome("racecar") === true
// isPalindrome("hello") === false
// isPalindrome("A man a plan a canal Panama") === true
```

#### Problem 9: Capitalize Words
```typescript
// nano_09_capitalize
// Returns the string with the first letter of each word capitalized.
function capitalizeWords(s: string): string {
    // your implementation here
}

// Tests:
// capitalizeWords("hello world") === "Hello World"
// capitalizeWords("the quick brown fox") === "The Quick Brown Fox"
// capitalizeWords("") === ""
```

#### Problem 10: Flatten Array
```typescript
// nano_10_flatten
// Flattens a nested array one level deep.
function flattenOnce(arr: (number | number[])[]): number[] {
    // your implementation here
}

// Tests:
// flattenOnce([1, [2, 3], 4]) -> [1, 2, 3, 4]
// flattenOnce([[1, 2], [3, 4]]) -> [1, 2, 3, 4]
// flattenOnce([1, 2, 3]) -> [1, 2, 3]
```

---

## Implementation Steps

### Step 1: Define problem registry

```python
# src/evaluation/nano_benchmark/problems.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TestCase:
    args: list[Any]
    expected: Any
    description: str = ""

@dataclass
class NanoProblem:
    id: str
    name: str
    prompt: str
    canonical_solution: str
    test_harness: str       # TypeScript test runner code
    test_cases: list[TestCase]
    difficulty: str = "easy"

PROBLEMS: list[NanoProblem] = [
    NanoProblem(
        id="nano_01",
        name="Hello World Function",
        prompt='function greet(name: string): string {\n',
        canonical_solution='    return `Hello, ${name}!`;\n}\n',
        test_harness="""
const assert = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };
assert(greet("Alice") === "Hello, Alice!", "test 1");
assert(greet("World") === "Hello, World!", "test 2");
console.log("PASS");
""",
        test_cases=[
            TestCase(args=["Alice"], expected="Hello, Alice!"),
            TestCase(args=["World"], expected="Hello, World!"),
        ],
    ),
    # ... remaining 9 problems
]
```

### Step 2: Runner that executes TypeScript via subprocess

```python
# src/evaluation/nano_benchmark/runner.py
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProblemResult:
    problem_id: str
    generated: str
    syntax_valid: bool
    type_correct: bool
    tests_passed: int
    tests_total: int

    @property
    def test_pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0

    @property
    def score(self) -> float:
        return (int(self.syntax_valid) + int(self.type_correct) + self.test_pass_rate) / 3.0

def check_syntax(code: str) -> bool:
    """Check if TypeScript parses without syntax errors using tsc."""
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--allowJs", fname],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False  # tsc not available; skip type check
    finally:
        os.unlink(fname)

def check_types(code: str) -> bool:
    """Strict TypeScript type check."""
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", fname],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(fname)

def run_tests(generated_fn: str, test_harness: str) -> tuple[int, int]:
    """Execute the generated function + test harness, count passes."""
    full_code = generated_fn + "\n" + test_harness
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(full_code)
        fname = f.name
    try:
        result = subprocess.run(
            ["ts-node", "--transpileOnly", fname],
            capture_output=True, text=True, timeout=15
        )
        if "PASS" in result.stdout:
            return (result.stdout.count("PASS"), result.stdout.count("PASS"))
        # Count individual assertion lines
        passed = result.stdout.count("ok")
        failed = result.stdout.count("FAIL") + result.stdout.count("Error")
        total = passed + failed
        return (passed, max(total, 1))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return (0, 1)
    finally:
        os.unlink(fname)
```

### Step 3: Model generation for each problem

```python
# src/evaluation/nano_benchmark/benchmark.py
import torch
from pathlib import Path
from src.model import ColaCoderModel
from src.tokenizer import load_tokenizer
from src.evaluation.nano_benchmark.problems import PROBLEMS
from src.evaluation.nano_benchmark.runner import ProblemResult, check_syntax, check_types, run_tests

def generate_completion(
    model: ColaCoderModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.2,  # low temp for benchmarking
    device: str = "cuda",
) -> str:
    """Generate a completion for the given prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    # Decode only the newly generated tokens
    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def run_nano_benchmark(
    checkpoint_path: Path,
    n_samples: int = 1,       # pass@1 by default
    temperature: float = 0.2,
    device: str = "cuda",
) -> list[ProblemResult]:
    model, tokenizer = load_checkpoint(checkpoint_path, device)
    results = []

    for problem in PROBLEMS:
        full_prompt = problem.prompt
        generated = generate_completion(model, tokenizer, full_prompt, device=device)
        full_code = full_prompt + generated

        # Evaluate
        syntax_ok = check_syntax(full_code)
        type_ok = check_types(full_code) if syntax_ok else False
        tests_passed, tests_total = run_tests(full_code, problem.test_harness) if syntax_ok else (0, len(problem.test_cases))

        results.append(ProblemResult(
            problem_id=problem.id,
            generated=generated,
            syntax_valid=syntax_ok,
            type_correct=type_ok,
            tests_passed=tests_passed,
            tests_total=tests_total,
        ))

    return results
```

### Step 4: Report card renderer

```python
# src/evaluation/nano_benchmark/report.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from src.evaluation.nano_benchmark.runner import ProblemResult
from src.evaluation.nano_benchmark.problems import PROBLEMS

console = Console()

PROBLEM_NAMES = {p.id: p.name for p in PROBLEMS}

def render_report_card(results: list[ProblemResult]) -> None:
    """Render a Rich report card table to the terminal."""
    table = Table(
        title="Nano Benchmark Report Card",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Problem", style="bold cyan", width=28)
    table.add_column("Syntax", justify="center", width=8)
    table.add_column("Types", justify="center", width=8)
    table.add_column("Tests", justify="center", width=10)
    table.add_column("Score", justify="center", width=8)

    total_score = 0.0
    for r in results:
        name = PROBLEM_NAMES.get(r.problem_id, r.problem_id)
        syntax_cell = "[green]PASS[/green]" if r.syntax_valid else "[red]FAIL[/red]"
        type_cell = "[green]PASS[/green]" if r.type_correct else "[red]FAIL[/red]"
        test_cell = f"{r.tests_passed}/{r.tests_total}"
        test_color = "green" if r.test_pass_rate == 1.0 else ("yellow" if r.test_pass_rate > 0 else "red")
        score_pct = r.score * 100
        score_color = "green" if score_pct >= 80 else ("yellow" if score_pct >= 40 else "red")
        table.add_row(
            name,
            syntax_cell,
            type_cell,
            f"[{test_color}]{test_cell}[/{test_color}]",
            f"[{score_color}]{score_pct:.0f}%[/{score_color}]",
        )
        total_score += r.score

    overall = total_score / len(results) * 100
    overall_color = "green" if overall >= 70 else ("yellow" if overall >= 40 else "red")
    table.add_section()
    table.add_row(
        "[bold]OVERALL[/bold]", "", "", "",
        f"[bold {overall_color}]{overall:.1f}%[/bold {overall_color}]"
    )

    console.print(table)
    grade = "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D" if overall >= 50 else "F"
    console.print(Panel(
        f"[bold]Grade: {grade}[/bold]  —  Overall score: {overall:.1f}%  |  {sum(r.syntax_valid for r in results)}/10 syntax valid  |  {sum(r.tests_passed == r.tests_total for r in results)}/10 all tests pass",
        title="Summary",
        style=overall_color,
    ))
```

### Step 5: CLI entry point

```python
# src/evaluation/nano_benchmark/__main__.py
import typer
from pathlib import Path
from src.evaluation.nano_benchmark.benchmark import run_nano_benchmark
from src.evaluation.nano_benchmark.report import render_report_card

app = typer.Typer()

@app.command()
def main(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory or .pt file"),
    temperature: float = typer.Option(0.2, help="Sampling temperature (lower = more deterministic)"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    save_json: bool = typer.Option(False, help="Save results to benchmark_results.json"),
):
    results = run_nano_benchmark(checkpoint, temperature=temperature, device=device)
    render_report_card(results)
    if save_json:
        import json
        out = [vars(r) for r in results]
        Path("benchmark_results.json").write_text(json.dumps(out, indent=2))
        typer.echo("Results saved to benchmark_results.json")

if __name__ == "__main__":
    app()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/evaluation/nano_benchmark/__init__.py` | New package |
| `src/evaluation/nano_benchmark/problems.py` | All 10 problem definitions |
| `src/evaluation/nano_benchmark/runner.py` | TypeScript execution harness |
| `src/evaluation/nano_benchmark/benchmark.py` | Model generation + scoring |
| `src/evaluation/nano_benchmark/report.py` | Rich report card |
| `src/menu/menus/evaluate_menu.py` | Wire benchmark as a menu item |
| `requirements.txt` | No new Python deps; require `tsc` and `ts-node` as system deps |

---

## Testing Strategy

- **Unit tests for problems.py**: Verify each canonical solution passes all test cases using the test harness
- **Syntax checker test**: Feed known-valid and known-invalid TypeScript; verify `check_syntax()` returns correct results
- **Type checker test**: Feed code with a type error (e.g., `return 42` where `string` expected); verify failure
- **Runner test**: Execute a trivial harness with `ts-node`, verify PASS output is detected
- **End-to-end test**: Load a small saved checkpoint, generate on problem 1, score it
- **CI**: Run `check_syntax(canonical_solution)` for all 10 problems in CI to catch broken references

---

## Performance Considerations

- Each problem generates at most 200 new tokens; at ~1000 tok/s on RTX 3080, this is ~2 seconds per problem, ~20 seconds total
- `tsc` startup is ~0.5 seconds; for 10 problems this adds ~5 seconds — acceptable
- `ts-node --transpileOnly` skips full type checking at runtime, reducing execution time to ~1 second per test
- If TypeScript tools are not installed, the benchmark gracefully skips type checking and execution, reporting only what it can
- Cache the tokenizer and model in memory — do not reload between problems

---

## Dependencies

| Tool | Use | Install |
|------|-----|---------|
| `tsc` (TypeScript compiler) | Syntax + type checking | `npm install -g typescript` |
| `ts-node` | Executing TypeScript | `npm install -g ts-node` |
| Python `rich` | Report card | Already installed |
| Python `typer` | CLI entry | Add to requirements.txt |

No new Python packages beyond typer.

---

## Estimated Complexity

**Low-Medium** — 1.5 days.

- Writing all 10 problems with test harnesses: 3 hours
- TypeScript execution harness (runner.py): 3 hours
- Model generation integration: 2 hours
- Report card rendering: 2 hours
- Testing and edge cases (tsc not installed, etc.): 2 hours

Total: ~12 hours

---

## 2026 Best Practices

- **pass@k evaluation**: The industry standard is to sample k completions and check if any passes. Start with k=1 for speed; add k=5 option for more reliable scores.
- **Temperature calibration for benchmarks**: Use temperature=0.2 for benchmarks (not 0.0 / greedy) — greedy decoding often gets stuck in degenerate patterns with small models.
- **Isolated test execution**: Never `eval()` or `exec()` generated Python/JavaScript in the same process. Always use subprocess with a timeout.
- **Canonical solution validation**: All 10 canonical solutions should be validated against their test cases in CI. If the reference solution fails, the benchmark is broken.
- **Difficulty ladder**: As the model improves, add more problems before moving to full HumanEval. Consider nano (10) -> micro (30) -> HumanEval (164) progression.
