# Feature 44: Fill-in-the-Middle (FIM) Completion Benchmark

## Overview

Fill-in-the-middle (FIM) evaluates the model's ability to generate a code segment that
fits between a known prefix and suffix. Unlike left-to-right generation, FIM requires
understanding bidirectional context — the generated code must be consistent with what
comes before AND after.

This benchmark creates test cases by removing known sections from real code, then
measures how accurately the model can restore them.

Status: OPTIONAL — enable via `--feature fim-benchmark` or CLI menu toggle.

---

## Motivation

- FIM is the dominant mode for IDE autocomplete (Copilot-style in-editor completion
  fills the current cursor position, not just appends to the end).
- Standard HumanEval only tests prefix→completion. FIM tests a qualitatively different
  capability.
- Comparing FIM vs left-to-right on the same problems reveals whether the model
  genuinely uses the suffix context or ignores it.
- Models trained with FIM objectives (SPM/PSM format) should significantly outperform
  standard models on this benchmark.

---

## Architecture / Design

### FIM Prompt Format

The standard FIM format uses three special tokens:
- `<|fim_prefix|>` — marks start of prefix
- `<|fim_suffix|>` — marks start of suffix
- `<|fim_middle|>` — marks the position to fill (model generates from here)

```
<|fim_prefix|>def calculate_area(shape):
    if shape.type == "circle":
        <|fim_suffix|>
    elif shape.type == "rectangle":
        return shape.width * shape.height
<|fim_middle|>
```

Expected completion: `return math.pi * shape.radius ** 2`

### Problem Categories

```python
# cola_coder/benchmarks/fim_benchmark/categories.py

from enum import Enum


class FIMCategory(str, Enum):
    FUNCTION_BODY = "function_body"       # complete the entire function body
    EXPRESSION = "expression"             # fill in a single expression
    TYPE_ANNOTATION = "type_annotation"   # fill in a type hint
    IMPORT = "import"                     # complete an import statement
    VARIABLE_INIT = "variable_init"       # fill in variable initialization
    CONDITION = "condition"               # fill in an if-condition
    RETURN_VALUE = "return_value"         # fill in a return value
```

### Problem Generator

```python
# cola_coder/benchmarks/fim_benchmark/generator.py

import ast
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FIMProblem:
    id: str
    category: str
    prefix: str        # code before the hole
    suffix: str        # code after the hole
    middle: str        # the correct fill-in (ground truth)
    prompt: str        # formatted FIM prompt
    language: str = "python"
    difficulty: str = "medium"
    source_file: str = ""


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def create_fim_prompt(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


class FIMProblemGenerator:
    """Generate FIM problems by cutting known sections out of real code."""

    def __init__(self, source_files: list[Path], max_problems: int = 200):
        self.source_files = source_files
        self.max_problems = max_problems

    def generate_function_body_problems(self) -> list[FIMProblem]:
        """Remove function bodies, leaving signature + docstring as prefix and next function as suffix."""
        problems = []
        for path in self.source_files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            lines = source.splitlines(keepends=True)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if len(node.body) < 2:
                    continue  # skip trivially short functions

                # Extract function signature (first line) as prefix
                func_start = node.lineno - 1
                body_start = node.body[0].lineno - 1
                body_end = node.end_lineno

                signature = "".join(lines[func_start:body_start])
                body = "".join(lines[body_start:body_end])
                suffix = "".join(lines[body_end:body_end + 3])  # next 3 lines

                if len(body.split("\n")) < 2 or len(body) > 500:
                    continue

                problem = FIMProblem(
                    id=f"fb_{path.stem}_{node.lineno}",
                    category="function_body",
                    prefix=signature,
                    suffix=suffix,
                    middle=body,
                    prompt=create_fim_prompt(signature, suffix),
                    source_file=str(path),
                    difficulty=self._estimate_difficulty(body),
                )
                problems.append(problem)
                if len(problems) >= self.max_problems:
                    return problems

        return problems

    def generate_expression_problems(self, source: str) -> list[FIMProblem]:
        """Remove single expressions (right-hand side of assignments)."""
        problems = []
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "=" in line and not line.strip().startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    prefix = "\n".join(lines[:i]) + "\n" + parts[0] + "="
                    middle = " " + parts[1].strip()
                    suffix = "\n".join(lines[i + 1:i + 4])
                    problems.append(FIMProblem(
                        id=f"expr_{i}",
                        category="expression",
                        prefix=prefix,
                        suffix=suffix,
                        middle=middle,
                        prompt=create_fim_prompt(prefix, suffix),
                    ))
        return problems[:20]

    def generate_type_annotation_problems(self, source: str) -> list[FIMProblem]:
        """Remove type annotations from function signatures."""
        import re
        problems = []
        lines = source.splitlines()
        annotation_pattern = re.compile(r"(def \w+\([^)]*\))(\s*->\s*[^:]+)(:)")
        for i, line in enumerate(lines):
            m = annotation_pattern.match(line.strip())
            if m:
                indent = len(line) - len(line.lstrip())
                prefix = "\n".join(lines[:i]) + "\n" + " " * indent + m.group(1)
                middle = m.group(2)
                suffix = m.group(3) + "\n" + "\n".join(lines[i + 1:i + 3])
                problems.append(FIMProblem(
                    id=f"type_{i}",
                    category="type_annotation",
                    prefix=prefix,
                    suffix=suffix,
                    middle=middle,
                    prompt=create_fim_prompt(prefix, suffix),
                ))
        return problems[:15]

    def _estimate_difficulty(self, body: str) -> str:
        lines = body.strip().split("\n")
        if len(lines) <= 3:
            return "easy"
        elif len(lines) <= 10:
            return "medium"
        else:
            return "hard"
```

### Metrics

```python
# cola_coder/benchmarks/fim_benchmark/metrics.py

import re
import ast
from difflib import SequenceMatcher


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact string match after normalizing whitespace."""
    return prediction.strip() == ground_truth.strip()


def normalized_exact_match(prediction: str, ground_truth: str) -> bool:
    """Match after normalizing whitespace, indentation, and trailing newlines."""
    def normalize(s: str) -> str:
        return re.sub(r'\s+', ' ', s.strip())
    return normalize(prediction) == normalize(ground_truth)


def bleu_score(prediction: str, ground_truth: str) -> float:
    """Compute BLEU-4 score (token-level)."""
    try:
        from sacrebleu import sentence_bleu
        return sentence_bleu(prediction, [ground_truth]).score / 100.0
    except ImportError:
        # Fallback: simple unigram overlap
        pred_tokens = set(prediction.split())
        ref_tokens = set(ground_truth.split())
        if not ref_tokens:
            return 0.0
        return len(pred_tokens & ref_tokens) / len(ref_tokens)


def edit_similarity(prediction: str, ground_truth: str) -> float:
    """Character-level edit similarity (1 - normalized edit distance)."""
    return SequenceMatcher(None, prediction, ground_truth).ratio()


def syntax_valid_python(code_with_fill: str) -> bool:
    """Check if the completed code is syntactically valid Python."""
    try:
        ast.parse(code_with_fill)
        return True
    except SyntaxError:
        return False


def type_correct_python(code: str) -> bool:
    """Run mypy on completed code (optional, slow)."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["mypy", "--ignore-missing-imports", path],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # mypy not available — skip
    finally:
        Path(path).unlink(missing_ok=True)


def score_completion(
    prediction: str,
    ground_truth: str,
    full_code: str,
) -> dict:
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "normalized_match": normalized_exact_match(prediction, ground_truth),
        "bleu": bleu_score(prediction, ground_truth),
        "edit_similarity": edit_similarity(prediction, ground_truth),
        "syntax_valid": syntax_valid_python(full_code),
    }
```

### Benchmark Runner

```python
# cola_coder/benchmarks/fim_benchmark/runner.py

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class FIMBenchmarkResult:
    problems: list[dict] = field(default_factory=list)

    def exact_match_rate(self) -> float:
        if not self.problems:
            return 0.0
        return sum(1 for p in self.problems if p["exact_match"]) / len(self.problems)

    def avg_bleu(self) -> float:
        if not self.problems:
            return 0.0
        return sum(p["bleu"] for p in self.problems) / len(self.problems)

    def syntax_validity_rate(self) -> float:
        if not self.problems:
            return 0.0
        return sum(1 for p in self.problems if p["syntax_valid"]) / len(self.problems)

    def by_category(self) -> dict[str, dict]:
        by_cat = {}
        for p in self.problems:
            cat = p["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(p)
        return {
            cat: {
                "count": len(probs),
                "exact_match": sum(1 for p in probs if p["exact_match"]) / len(probs),
                "avg_bleu": sum(p["bleu"] for p in probs) / len(probs),
                "syntax_valid": sum(1 for p in probs if p["syntax_valid"]) / len(probs),
            }
            for cat, probs in by_cat.items()
        }


class FIMBenchmarkRunner:
    def __init__(self, generator, tokenizer, config):
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config

    def run(self, problems: list[FIMProblem]) -> FIMBenchmarkResult:
        result = FIMBenchmarkResult()

        for problem in problems:
            # Generate FIM completion
            completion = self.generator.generate(
                problem.prompt,
                max_new_tokens=200,
                temperature=0.1,       # low temp for deterministic fill-in
                stop_tokens=["<|fim_suffix|>", "<|endoftext|>"],
            )

            # Reconstruct full code
            full_code = problem.prefix + completion + problem.suffix

            # Score
            scores = score_completion(completion, problem.middle, full_code)
            result.problems.append({
                "id": problem.id,
                "category": problem.category,
                "difficulty": problem.difficulty,
                "prediction": completion,
                "ground_truth": problem.middle,
                **scores,
            })

        return result

    def compare_fim_vs_ltr(
        self,
        problems: list[FIMProblem],
    ) -> dict:
        """Compare FIM vs left-to-right generation on the same problems."""
        fim_result = self.run(problems)

        # Left-to-right: only use prefix, ignore suffix
        ltr_problems = [
            FIMProblem(
                id=p.id + "_ltr",
                category=p.category,
                prefix=p.prefix,
                suffix="",
                middle=p.middle,
                prompt=p.prefix,  # no FIM tokens
            )
            for p in problems
        ]
        ltr_result = self.run(ltr_problems)

        return {
            "fim": {
                "exact_match": fim_result.exact_match_rate(),
                "avg_bleu": fim_result.avg_bleu(),
                "syntax_valid": fim_result.syntax_validity_rate(),
            },
            "ltr": {
                "exact_match": ltr_result.exact_match_rate(),
                "avg_bleu": ltr_result.avg_bleu(),
                "syntax_valid": ltr_result.syntax_validity_rate(),
            },
            "fim_improvement": {
                "exact_match": (fim_result.exact_match_rate() - ltr_result.exact_match_rate()),
                "bleu": (fim_result.avg_bleu() - ltr_result.avg_bleu()),
            },
        }
```

---

## Implementation Steps

1. **Create `cola_coder/benchmarks/fim_benchmark/` package**: `__init__.py`,
   `categories.py`, `generator.py`, `metrics.py`, `runner.py`, `report.py`.

2. **Add FIM special tokens** to the tokenizer and train with FIM objective
   (SPM: Suffix-Prefix-Middle format) for best results. Without FIM training,
   the model may ignore the suffix entirely.

3. **Build problem set**: either from a curated set of 100+ Python/TypeScript files,
   or generate dynamically from the project's own source files.

4. **Add CLI command**: "Run FIM benchmark" with options for category filter and
   FIM vs LTR comparison.

5. **BLEU scoring**: add `sacrebleu` as an optional dependency for standard BLEU.
   Fall back to simple token overlap if not installed.

6. **Report display**: table showing per-category exact match rate, BLEU, and syntax
   validity, with a side-by-side FIM vs LTR comparison.

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Ensure FIM special tokens handled as stop tokens |
| `cli/menu.py` | Add "FIM benchmark" option |
| `config.py` | Add `FIMBenchmarkConfig` |
| `cola_coder/benchmarks/fim_benchmark/` | New package |
| `requirements.txt` | Add `sacrebleu` (optional) |

---

## Testing Strategy

```python
# tests/test_fim_benchmark.py

def test_create_fim_prompt():
    prompt = create_fim_prompt("def f():\n    ", "\n    return result")
    assert FIM_PREFIX in prompt
    assert FIM_SUFFIX in prompt
    assert FIM_MIDDLE in prompt

def test_exact_match():
    assert exact_match("    return x * 2", "    return x * 2")
    assert not exact_match("    return x*2", "    return x * 2")

def test_normalized_match_ignores_whitespace():
    assert normalized_exact_match("return x * 2", "return  x  *  2")

def test_bleu_score_perfect():
    assert bleu_score("hello world", "hello world") > 0.99

def test_bleu_score_different():
    assert bleu_score("hello world", "goodbye moon") < 0.3

def test_edit_similarity_identical():
    assert abs(edit_similarity("foo", "foo") - 1.0) < 1e-6

def test_syntax_valid_python():
    assert syntax_valid_python("def f():\n    return 1\n")
    assert not syntax_valid_python("def f(:\n    return 1")

def test_problem_generator_creates_problems(tmp_path):
    py_file = tmp_path / "test.py"
    py_file.write_text(
        "def add(a, b):\n    result = a + b\n    return result\n\n"
        "def mul(a, b):\n    return a * b\n"
    )
    gen = FIMProblemGenerator([py_file])
    problems = gen.generate_function_body_problems()
    assert len(problems) >= 1
    assert problems[0].prefix != ""
    assert problems[0].middle != ""
```

---

## Performance Considerations

- **Problem count**: 200 FIM problems × inference time = ~5–10 minutes at 30 tok/s.
  Use batch inference (feature 36) to speed up to ~1–2 minutes.
- **Exact match rate** will be low (5–20% is normal for code FIM) because there are
  many valid completions. Focus on BLEU and edit similarity as primary metrics.
- **BLEU overhead**: sacrebleu is fast (< 1 ms per pair). Not a bottleneck.
- **mypy type checking**: slow (2–5 seconds per file). Make optional.

---

## Dependencies

```
sacrebleu>=2.3.0   # BLEU scoring (optional, falls back to token overlap)
mypy>=1.8.0        # type correctness check (optional, very slow)
torch>=2.2.0       # base requirement
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| FIM problem generator | 4 hours |
| Metrics (exact match, BLEU, edit sim) | 2 hours |
| Benchmark runner + FIM vs LTR compare | 3 hours |
| Report display | 2 hours |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~14 hours** |

Complexity rating: **Medium** — well-defined metrics; main work is generating a
good problem set from real code.

---

## 2026 Best Practices

- **HumanEval-FIM**: OpenAI has published a FIM extension to HumanEval. Using this
  as a gold standard enables direct comparison with published models.
- **Cursor Fill-in the Middle (CFIM) format**: Cursor uses a slightly different format
  with explicit `<|CURSOR|>` token. Support multiple FIM formats to test models
  fine-tuned on different conventions.
- **PSM vs SPM**: two FIM training formats. PSM (Prefix-Suffix-Middle) teaches the
  model to generate middle given both prefix and suffix. SPM shuffles the order.
  Test which format the model was trained with before running the benchmark.
- **Multi-line vs single-line FIM**: separate benchmarks for single-line fills
  (expression, annotation) vs multi-line fills (function body). Models often handle
  one better than the other.
- **CodeBLEU**: instead of standard BLEU, use CodeBLEU (weights = 0.25 n-gram + 0.25
  weighted n-gram + 0.25 syntax match + 0.25 data-flow match). Better correlation
  with human judgment for code.
