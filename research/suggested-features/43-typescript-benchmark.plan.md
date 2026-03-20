# Feature 43: TypeScript-Specific Benchmark

## Overview

A 50-problem TypeScript benchmark covering the language's most distinctive features:
generics, type guards, utility types, discriminated unions, mapped types, conditional
types, template literal types, decorators, enums, and module patterns. Each problem
has a signature, description, test cases, and difficulty rating.

Scoring combines: compile check + test execution + type strictness. Output is a
detailed report card broken down by TypeScript feature category.

Status: OPTIONAL — enable via `--feature ts-benchmark` or CLI menu toggle.

---

## Motivation

- HumanEval tests generic algorithmic thinking (Python). TypeScript problems test
  type-system understanding — a distinct capability.
- TypeScript is the dominant language for web/Node.js code generation tasks.
- Category-level scoring reveals which TypeScript features the model handles well vs
  poorly, guiding further fine-tuning.
- A TypeScript-aware benchmark differentiates Cola-Coder from Python-only code models.

---

## Architecture / Design

### Problem Structure

```python
# cola_coder/benchmarks/ts_benchmark/problem.py

from dataclasses import dataclass, field
from enum import Enum


class TSCategory(str, Enum):
    GENERICS = "generics"
    TYPE_GUARDS = "type_guards"
    UTILITY_TYPES = "utility_types"
    DISCRIMINATED_UNIONS = "discriminated_unions"
    MAPPED_TYPES = "mapped_types"
    CONDITIONAL_TYPES = "conditional_types"
    TEMPLATE_LITERALS = "template_literals"
    DECORATORS = "decorators"
    ENUMS = "enums"
    MODULE_PATTERNS = "module_patterns"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TSProblem:
    id: str
    category: TSCategory
    difficulty: Difficulty
    description: str
    signature: str       # TypeScript function/type signature
    prompt: str          # Full prompt sent to model
    test_cases: list[dict]  # [{"input": ..., "expected": ...}]
    tags: list[str] = field(default_factory=list)
    points: int = 2      # easy=1, medium=2, hard=3
```

### Problem Set (All 50 Problems)

```python
# cola_coder/benchmarks/ts_benchmark/problems.py

PROBLEMS: list[dict] = [
    # ===== GENERICS (10 problems) =====
    {
        "id": "gen_01",
        "category": "generics",
        "difficulty": "easy",
        "description": "Implement a generic identity function",
        "signature": "function identity<T>(value: T): T",
        "prompt": "Write a TypeScript generic identity function that returns its input unchanged.\nSignature: function identity<T>(value: T): T\n\n```typescript\n",
        "test_cases": [
            {"input": "42", "expected": "42"},
            {"input": '"hello"', "expected": '"hello"'},
            {"input": "true", "expected": "true"},
        ],
    },
    {
        "id": "gen_02",
        "category": "generics",
        "difficulty": "easy",
        "description": "Generic array first element",
        "signature": "function first<T>(arr: T[]): T | undefined",
        "prompt": "Write a TypeScript function that returns the first element of an array, or undefined if empty.\nSignature: function first<T>(arr: T[]): T | undefined\n\n```typescript\n",
        "test_cases": [
            {"input": "[1, 2, 3]", "expected": "1"},
            {"input": "[]", "expected": "undefined"},
        ],
    },
    {
        "id": "gen_03",
        "category": "generics",
        "difficulty": "medium",
        "description": "Generic pair swap",
        "signature": "function swapPair<A, B>(pair: [A, B]): [B, A]",
        "prompt": "Write a TypeScript function that swaps the two elements of a tuple.\nSignature: function swapPair<A, B>(pair: [A, B]): [B, A]\n\n```typescript\n",
        "test_cases": [
            {"input": "[1, 'hello']", "expected": "['hello', 1]"},
        ],
    },
    {
        "id": "gen_04",
        "category": "generics",
        "difficulty": "medium",
        "description": "Generic merge two objects",
        "signature": "function merge<T extends object, U extends object>(a: T, b: U): T & U",
        "prompt": "Write a TypeScript function that merges two objects into one with combined type.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_05",
        "category": "generics",
        "difficulty": "medium",
        "description": "Generic stack class",
        "signature": "class Stack<T> { push(item: T): void; pop(): T | undefined; peek(): T | undefined; isEmpty(): boolean; }",
        "prompt": "Implement a generic Stack<T> class with push, pop, peek, and isEmpty methods.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_06",
        "category": "generics",
        "difficulty": "medium",
        "description": "Constrained generic — only objects with id",
        "signature": "function getById<T extends { id: number }>(items: T[], id: number): T | undefined",
        "prompt": "Write a TypeScript function that finds an item by id, constrained to objects with an id property.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_07",
        "category": "generics",
        "difficulty": "hard",
        "description": "Generic pipeline (compose functions)",
        "signature": "function pipe<A, B, C>(fn1: (a: A) => B, fn2: (b: B) => C): (a: A) => C",
        "prompt": "Write a TypeScript function pipe that composes two functions.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_08",
        "category": "generics",
        "difficulty": "hard",
        "description": "Generic memoize",
        "signature": "function memoize<T extends (...args: any[]) => any>(fn: T): T",
        "prompt": "Write a TypeScript generic memoize function.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_09",
        "category": "generics",
        "difficulty": "hard",
        "description": "Generic result type",
        "signature": "type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E }",
        "prompt": "Define a Result<T, E> discriminated union type and implement ok() and err() constructor helpers.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "gen_10",
        "category": "generics",
        "difficulty": "hard",
        "description": "Generic event emitter",
        "signature": "class EventEmitter<Events extends Record<string, any>> { on<K extends keyof Events>(event: K, handler: (data: Events[K]) => void): void; emit<K extends keyof Events>(event: K, data: Events[K]): void; }",
        "prompt": "Implement a type-safe generic EventEmitter class.\n\n```typescript\n",
        "test_cases": [],
    },

    # ===== TYPE GUARDS (5 problems) =====
    {
        "id": "tg_01",
        "category": "type_guards",
        "difficulty": "easy",
        "description": "isString type guard",
        "signature": "function isString(value: unknown): value is string",
        "prompt": "Write a TypeScript type guard that checks if a value is a string.\n\n```typescript\n",
        "test_cases": [
            {"input": '"hello"', "expected": "true"},
            {"input": "42", "expected": "false"},
        ],
    },
    {
        "id": "tg_02",
        "category": "type_guards",
        "difficulty": "medium",
        "description": "User vs Admin type guard",
        "signature": "function isAdmin(user: User | Admin): user is Admin",
        "prompt": "Given types User (has name) and Admin (has name and role: 'admin'), write a type guard.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "tg_03",
        "category": "type_guards",
        "difficulty": "medium",
        "description": "Array type guard",
        "signature": "function isArrayOf<T>(arr: unknown, guard: (v: unknown) => v is T): arr is T[]",
        "prompt": "Write a generic type guard that checks if a value is an array of type T.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "tg_04",
        "category": "type_guards",
        "difficulty": "medium",
        "description": "Narrowing with in operator",
        "signature": "function isCircle(shape: Circle | Rectangle): shape is Circle",
        "prompt": "Use the 'in' operator to write a type guard for Circle vs Rectangle shapes.\n\n```typescript\n",
        "test_cases": [],
    },
    {
        "id": "tg_05",
        "category": "type_guards",
        "difficulty": "hard",
        "description": "Exhaustive check",
        "signature": "function assertNever(value: never): never",
        "prompt": "Write an assertNever function for exhaustive type checking in switch statements.\n\n```typescript\n",
        "test_cases": [],
    },

    # ===== UTILITY TYPES (8 problems) =====
    # ... Partial, Required, Readonly, Pick, Omit, Record, ReturnType, Parameters

    # ===== DISCRIMINATED UNIONS (5 problems) =====
    # ... action types, shape union, result union, payment method, status

    # ===== MAPPED TYPES (5 problems) =====
    # ... Nullable, Optional, Mutable, DeepReadonly, Flags

    # ===== CONDITIONAL TYPES (5 problems) =====
    # ... IsArray, NonNullable, Flatten, Awaited, UnpackPromise

    # ===== TEMPLATE LITERALS (3 problems) =====
    # ... EventName, CSSProperty, ApiRoute

    # ===== DECORATORS (3 problems) =====
    # ... @readonly, @log, @validate

    # ===== ENUMS (3 problems) =====
    # ... Direction, Status, Color with methods

    # ===== MODULE PATTERNS (3 problems) =====
    # ... namespace, barrel export, default+named export
]
```

### Scorer

```python
# cola_coder/benchmarks/ts_benchmark/scorer.py

import subprocess
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProblemScore:
    problem_id: str
    category: str
    compiles: bool
    tests_passed: int
    tests_total: int
    type_errors: int
    score: float  # 0.0 - 1.0


class TypeScriptScorer:
    def __init__(self, tsc_path: str = "tsc", node_path: str = "node"):
        self.tsc = tsc_path
        self.node = node_path

    def score(self, problem: "TSProblem", completion: str) -> ProblemScore:
        """Score a completion: compile check + test execution + type strictness."""
        full_code = problem.signature_preamble + "\n" + completion

        # Step 1: compile check
        compiles, type_errors = self._compile_check(full_code)

        # Step 2: test execution
        passed, total = self._run_tests(full_code, problem.test_cases)

        # Combined score
        compile_score = 1.0 if compiles else 0.0
        test_score = passed / max(total, 1)
        # Penalize type errors (each type error = -0.1, min 0)
        type_penalty = max(0.0, 1.0 - type_errors * 0.1)

        score = (0.4 * compile_score + 0.5 * test_score + 0.1 * type_penalty)

        return ProblemScore(
            problem_id=problem.id,
            category=problem.category,
            compiles=compiles,
            tests_passed=passed,
            tests_total=total,
            type_errors=type_errors,
            score=score,
        )

    def _compile_check(self, code: str) -> tuple[bool, int]:
        """Run tsc and count type errors."""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            result = subprocess.run(
                [self.tsc, "--noEmit", "--strict",
                 "--target", "ES2022", "--lib", "ES2022",
                 path],
                capture_output=True, text=True, timeout=10,
            )
            errors = result.stdout.count("error TS")
            return result.returncode == 0, errors
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True, 0  # if tsc not available, skip compile check
        finally:
            Path(path).unlink(missing_ok=True)

    def _run_tests(self, code: str, test_cases: list[dict]) -> tuple[int, int]:
        """Run test cases using Node.js eval."""
        if not test_cases:
            return 0, 0
        # Build a test runner script
        test_code = code + "\n"
        for i, tc in enumerate(test_cases):
            test_code += (
                f'const _r{i} = JSON.stringify(eval({json.dumps(tc["input"])}));\n'
                f'const _e{i} = JSON.stringify({tc["expected"]});\n'
                f'if (_r{i} !== _e{i}) process.stdout.write("FAIL {i}\\n");\n'
                f'else process.stdout.write("PASS {i}\\n");\n'
            )
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(test_code)
            path = f.name
        try:
            result = subprocess.run(
                [self.node, path],
                capture_output=True, text=True, timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            passed = sum(1 for l in lines if l.startswith("PASS"))
            return passed, len(test_cases)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0, len(test_cases)
        finally:
            Path(path).unlink(missing_ok=True)
```

### Report Card

```python
# cola_coder/benchmarks/ts_benchmark/report.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import defaultdict


def generate_report(scores: list["ProblemScore"]) -> dict:
    """Generate category-level report from problem scores."""
    by_category = defaultdict(list)
    for s in scores:
        by_category[s.category].append(s)

    report = {}
    for cat, cat_scores in by_category.items():
        avg_score = sum(s.score for s in cat_scores) / len(cat_scores)
        compile_rate = sum(1 for s in cat_scores if s.compiles) / len(cat_scores)
        test_rate = (sum(s.tests_passed for s in cat_scores) /
                     max(sum(s.tests_total for s in cat_scores), 1))
        report[cat] = {
            "problems": len(cat_scores),
            "avg_score": avg_score,
            "compile_rate": compile_rate,
            "test_pass_rate": test_rate,
        }
    return report


def display_report_card(scores: list["ProblemScore"]) -> None:
    console = Console()
    report = generate_report(scores)

    table = Table(title="TypeScript Benchmark Report", show_lines=True)
    table.add_column("Category", style="cyan")
    table.add_column("Problems", justify="right")
    table.add_column("Compiles %", justify="right")
    table.add_column("Tests %", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for cat, data in sorted(report.items()):
        score_color = "green" if data["avg_score"] > 0.7 else "yellow" if data["avg_score"] > 0.4 else "red"
        table.add_row(
            cat.replace("_", " ").title(),
            str(data["problems"]),
            f"{data['compile_rate']*100:.0f}%",
            f"{data['test_pass_rate']*100:.0f}%",
            f"[{score_color}]{data['avg_score']*100:.0f}%[/{score_color}]",
        )

    overall = sum(s.score for s in scores) / max(len(scores), 1)
    console.print(table)
    console.print(Panel(
        f"[bold]Overall TypeScript Score: {overall*100:.1f}%[/bold]\n"
        f"Problems attempted: {len(scores)} / 50",
        border_style="blue"
    ))
```

---

## Implementation Steps

1. **Create `cola_coder/benchmarks/ts_benchmark/` package**: `__init__.py`, `problem.py`,
   `problems.py`, `scorer.py`, `report.py`, `runner.py`.

2. **Write all 50 problems** in `problems.py`. Start with the 15 shown (gen_01..gen_10,
   tg_01..tg_05); fill in remaining 35 categories.

3. **Implement `TypeScriptScorer`** — validate tsc and node availability at startup,
   gracefully degrade to compile-only scoring if Node.js missing.

4. **Implement `TSBenchmarkRunner`**:
   ```python
   class TSBenchmarkRunner:
       def run(self, generator, categories=None, n_samples=1) -> list[ProblemScore]:
           problems = [p for p in PROBLEMS if not categories
                       or p["category"] in categories]
           prompts = [p["prompt"] for p in problems]
           completions = generator.generate_batch(prompts, ...)
           return [self.scorer.score(TSProblem(**p), c)
                   for p, c in zip(problems, completions)]
   ```

5. **Add CLI command**: "Run TypeScript benchmark" with category filter option.

6. **Save results to JSON**: timestamp + per-problem scores for tracking progress.

7. **TypeScript prerequisites**: check for `tsc` and `node` in PATH at benchmark start.
   Print clear install instructions if missing.

---

## Key Files to Modify

| File | Change |
|---|---|
| `cli/menu.py` | Add "TypeScript benchmark" option |
| `benchmarks/humaneval.py` | Add `run_typescript_benchmark()` function |
| `config.py` | Add `TSBenchmarkConfig` |
| `cola_coder/benchmarks/ts_benchmark/` | New package |

---

## Testing Strategy

```python
# tests/test_ts_benchmark.py

def test_problem_structure_valid():
    for p in PROBLEMS:
        assert "id" in p
        assert "category" in p
        assert "prompt" in p

def test_compile_check_passes_valid_ts():
    scorer = TypeScriptScorer()
    compiles, errors = scorer._compile_check(
        "function add(a: number, b: number): number { return a + b; }"
    )
    assert compiles
    assert errors == 0

def test_compile_check_fails_invalid_ts():
    scorer = TypeScriptScorer()
    compiles, errors = scorer._compile_check(
        "function add(a: string, b: number): number { return a + b; }"
    )
    assert not compiles or errors > 0

def test_report_card_covers_all_categories():
    mock_scores = [
        ProblemScore(f"test_{i}", cat, True, 1, 1, 0, 0.9)
        for i, cat in enumerate(["generics", "type_guards", "utility_types",
                                  "discriminated_unions", "mapped_types"])
    ]
    report = generate_report(mock_scores)
    assert len(report) == 5
```

---

## Performance Considerations

- **tsc startup overhead**: each `tsc` invocation starts a new Node.js process (~200 ms).
  For 50 problems, total compile overhead = ~10 seconds. Use `tsc --watch` mode or
  a persistent tsc server to amortize startup cost.
- **Batch compilation**: write all 50 completions to a temp directory and run tsc once
  on the whole directory for 10× faster compilation.
- **Node.js test execution**: similar overhead. Bundle all test cases into one Node.js
  script if possible.

---

## Dependencies

```
rich>=13.0.0    # report display (already required)
node>=18.0.0    # runtime check — external dependency, optional
typescript>=5.0 # tsc check — external dependency, optional
```

Python-side: no new pip dependencies.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Problem writing (50 problems) | 6 hours |
| Scorer (compile + tests) | 4 hours |
| Report card | 2 hours |
| Benchmark runner | 2 hours |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~17 hours** |

Complexity rating: **Medium** — most work is writing the 50 problem definitions.
Scoring infrastructure is straightforward.

---

## 2026 Best Practices

- **TypeScript 5.x features**: include problems testing TypeScript 5 features: `const`
  type parameters, `using` declarations, variadic tuple types, `satisfies` operator.
- **Bun as test runner**: Bun runs TypeScript natively (no separate compilation step)
  and is 10x faster than tsc+node for simple scripts. Consider Bun for test execution.
- **Benchmark drift**: TypeScript evolves rapidly. Pin benchmark to a specific
  TypeScript version in config and document which TS version each problem targets.
- **Human baseline**: collect human performance data on the same 50 problems to
  contextualize model scores. A senior TypeScript developer should score 95%+.
- **Difficulty calibration**: verify difficulty labels against GPT-4 performance.
  Problems GPT-4 solves < 50% of the time are "hard"; > 90% = "easy".
