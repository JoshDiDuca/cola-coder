# Feature 60: Synthetic Bug Injection

**Status:** Proposed
**CLI Flag:** `--inject-bugs`
**Complexity:** Medium-High

---

## Overview

Takes working, tested TypeScript code and programmatically introduces targeted mutations to create buggy versions. Pairs (buggy code + "Fix the bug:") → (correct code) are used to train the model to find and fix bugs. Bug types include: off-by-one errors, wrong operator, missing null check, wrong variable name, missing return, type error, and swapped arguments. The original code must pass tests; the mutated version must fail tests (validated by the test runner).

---

## Motivation

Bug-fixing is one of the highest-value developer tasks, yet most code generation training data contains only correct code. Training on (buggy → fixed) pairs teaches the model to:

1. Recognize common bug patterns
2. Understand the difference between correct and incorrect implementations
3. Generate fixes that preserve surrounding code structure

Mutation testing (Jia & Harman, 2011) and bug injection for ML (Tufano et al., 2019 — "An Empirical Study on Learning Bug-Fixing Patches") both show that synthetic bugs are effective training signals. DeepSeek-Coder and WizardCoder use similar synthetic bug data.

---

## Architecture / Design

```
Working code (passes tests)
  │
  ▼
ASTMutator
  ├── OffByOneMutation    (i < n → i <= n, arr[i-1] → arr[i])
  ├── OperatorMutation    (+ → -, === → !==, && → ||)
  ├── NullCheckMutation   (remove null guard)
  ├── VariableNameMutation (swap two similar-named variables)
  ├── MissingReturnMutation (remove return statement)
  ├── TypeMutation         (number → string in type annotation)
  └── SwapArgsMutation     (swap two arguments in a call)
  │
  ▼
Mutated code
  │
  ▼
Validator (run tests)
  ├── original passes → OK
  ├── mutated fails → valid mutation
  └── mutated passes → discard (mutation was semantically equivalent)
  │
  ▼
Training pair:
  instruction = "[buggy code]\nFix the bug:"
  output = "[correct code]"
```

---

## Implementation Steps

### Step 1: Base Mutator Interface

```python
# src/data/bug_injector/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Mutation:
    mutator_name: str
    original_text: str
    mutated_text: str
    location: str       # description of where the mutation was applied
    line_number: int

class BaseMutator(ABC):
    name: str = "base"

    @abstractmethod
    def mutate(self, source: str) -> list[Mutation]:
        """Return all possible mutations of this type for the given source."""
        ...

    def first_mutation(self, source: str) -> Optional[Mutation]:
        mutations = self.mutate(source)
        return mutations[0] if mutations else None
```

### Step 2: Mutation Implementations

```python
# src/data/bug_injector/mutations.py
import re
from src.data.bug_injector.base import BaseMutator, Mutation

# --- Off-By-One ---
class OffByOneMutator(BaseMutator):
    name = "off_by_one"

    OBO_PATTERNS = [
        (re.compile(r'\b(\w+)\s*<\s*(\w+(?:\.\w+)?)\b'), r'\1 <= \2'),
        (re.compile(r'\b(\w+)\s*<=\s*(\w+(?:\.\w+)?)\b'), r'\1 < \2'),
        (re.compile(r'\b(\w+)\s*>\s*(\w+(?:\.\w+)?)\b'), r'\1 >= \2'),
        (re.compile(r'\[(\w+)\s*-\s*1\]'), r'[\1]'),
        (re.compile(r'\[(\w+)\](?=\s*[^=])'), r'[\1 - 1]'),
    ]

    def mutate(self, source: str) -> list[Mutation]:
        results = []
        for pattern, replacement in self.OBO_PATTERNS:
            for m in pattern.finditer(source):
                new_source = source[:m.start()] + m.expand(replacement) + source[m.end():]
                if new_source != source:
                    results.append(Mutation(
                        mutator_name=self.name,
                        original_text=source,
                        mutated_text=new_source,
                        location=f"position {m.start()}",
                        line_number=source[:m.start()].count("\n") + 1,
                    ))
        return results[:3]  # limit to 3 per file

# --- Operator Mutation ---
class OperatorMutator(BaseMutator):
    name = "operator"

    SWAPS = [
        (r'\+(?!=)', '-'), (r'-(?!=)', '+'),
        (r'===', '!=='),   (r'!==', '==='),
        (r'&&', '||'),     (r'\|\|', '&&'),
        (r'>=', '>'),      (r'<=', '<'),
    ]

    def mutate(self, source: str) -> list[Mutation]:
        results = []
        for pattern_str, replacement in self.SWAPS:
            pattern = re.compile(pattern_str)
            for m in pattern.finditer(source):
                new_source = source[:m.start()] + replacement + source[m.end():]
                if new_source != source:
                    results.append(Mutation(
                        mutator_name=self.name,
                        original_text=source,
                        mutated_text=new_source,
                        location=f"op at line {source[:m.start()].count(chr(10))+1}",
                        line_number=source[:m.start()].count("\n") + 1,
                    ))
        return results[:3]

# --- Null Check Removal ---
class NullCheckMutator(BaseMutator):
    name = "null_check"

    NULL_PATTERNS = [
        re.compile(r'if\s*\(\s*(\w+)\s*(?:===?\s*null|!==?\s*null|!== undefined|=== undefined)\s*\)\s*\{[^}]*\}\s*'),
        re.compile(r'(\w+)\s*\?\.\s*'),   # optional chaining: obj?.method → obj.method
        re.compile(r'(\w+)\s*\?\?\s*[^:]+'),  # nullish coalescing
    ]

    def mutate(self, source: str) -> list[Mutation]:
        results = []
        for pattern in self.NULL_PATTERNS:
            for m in pattern.finditer(source):
                if pattern == self.NULL_PATTERNS[1]:
                    replacement = m.group(1) + "."
                elif pattern == self.NULL_PATTERNS[2]:
                    replacement = m.group(1)
                else:
                    replacement = ""
                new_source = source[:m.start()] + replacement + source[m.end():]
                if new_source != source:
                    results.append(Mutation(
                        mutator_name=self.name,
                        original_text=source,
                        mutated_text=new_source,
                        location=f"null check at line {source[:m.start()].count(chr(10))+1}",
                        line_number=source[:m.start()].count("\n") + 1,
                    ))
        return results[:2]

# --- Variable Name Swap ---
class VariableNameMutator(BaseMutator):
    name = "variable_swap"

    def mutate(self, source: str) -> list[Mutation]:
        # Find variable declarations and swap two
        var_pattern = re.compile(r'\b(?:const|let|var)\s+(\w+)\b')
        names = [m.group(1) for m in var_pattern.finditer(source)]

        if len(names) < 2:
            return []

        # Swap first two unique variable names in the body
        a, b = names[0], names[1]
        if a == b:
            return []

        # Only swap in usage (not declaration), affecting code body
        declaration_end = source.index(a) + len(a) + 10
        body = source[declaration_end:]
        swapped = body.replace(a, "__SWAP_A__").replace(b, a).replace("__SWAP_A__", b)
        new_source = source[:declaration_end] + swapped

        if new_source == source:
            return []

        return [Mutation(
            mutator_name=self.name,
            original_text=source,
            mutated_text=new_source,
            location=f"swapped '{a}' and '{b}'",
            line_number=1,
        )]

# --- Missing Return ---
class MissingReturnMutator(BaseMutator):
    name = "missing_return"

    def mutate(self, source: str) -> list[Mutation]:
        pattern = re.compile(r'\breturn\s+([^;]+);')
        results = []
        for m in pattern.finditer(source):
            # Remove the return statement
            new_source = source[:m.start()] + source[m.end():]
            results.append(Mutation(
                mutator_name=self.name,
                original_text=source,
                mutated_text=new_source,
                location=f"removed return at line {source[:m.start()].count(chr(10))+1}",
                line_number=source[:m.start()].count("\n") + 1,
            ))
        return results[:1]

# --- Type Annotation Mutation ---
class TypeMutator(BaseMutator):
    name = "type_error"

    TYPE_SWAPS = [
        (re.compile(r':\s*number\b'), ': string'),
        (re.compile(r':\s*string\b'), ': number'),
        (re.compile(r':\s*boolean\b'), ': number'),
        (re.compile(r'\[\]\s*(?=[,);=])'), ''),  # remove array type
    ]

    def mutate(self, source: str) -> list[Mutation]:
        results = []
        for pattern, replacement in self.TYPE_SWAPS:
            for m in pattern.finditer(source):
                new_source = source[:m.start()] + replacement + source[m.end():]
                if new_source != source:
                    results.append(Mutation(
                        mutator_name=self.name,
                        original_text=source,
                        mutated_text=new_source,
                        location=f"type at line {source[:m.start()].count(chr(10))+1}",
                        line_number=source[:m.start()].count("\n") + 1,
                    ))
        return results[:2]

# --- Swapped Arguments ---
class SwapArgsMutator(BaseMutator):
    name = "swapped_args"

    def mutate(self, source: str) -> list[Mutation]:
        # Find function calls with exactly 2 arguments and swap them
        call_pattern = re.compile(r'(\w+)\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)')
        results = []
        for m in call_pattern.finditer(source):
            fn_name, arg1, arg2 = m.group(1), m.group(2), m.group(3)
            if fn_name in {"if", "for", "while", "function", "class"}:
                continue
            swapped = f"{fn_name}({arg2.strip()}, {arg1.strip()})"
            new_source = source[:m.start()] + swapped + source[m.end():]
            if new_source != source:
                results.append(Mutation(
                    mutator_name=self.name,
                    original_text=source,
                    mutated_text=new_source,
                    location=f"swapped args in {fn_name}() at line {source[:m.start()].count(chr(10))+1}",
                    line_number=source[:m.start()].count("\n") + 1,
                ))
        return results[:2]

ALL_MUTATORS = [
    OffByOneMutator(),
    OperatorMutator(),
    NullCheckMutator(),
    VariableNameMutator(),
    MissingReturnMutator(),
    TypeMutator(),
    SwapArgsMutator(),
]
```

### Step 3: Mutation Validator

```python
# src/data/bug_injector/validator.py
import subprocess
import tempfile
import os
from src.data.bug_injector.base import Mutation

def validate_mutation(
    original: str,
    mutation: Mutation,
    test_command: str,         # e.g. "npx jest --testPathPattern=foo.test.ts"
    timeout: int = 30,
) -> dict:
    """
    Returns:
      original_passes: bool
      mutated_fails: bool
      valid: bool (original_passes AND mutated_fails)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "solution.ts")

        # Test original
        with open(src_path, "w") as f:
            f.write(original)
        orig_result = subprocess.run(
            test_command.split(), capture_output=True, timeout=timeout, cwd=tmpdir
        )
        original_passes = orig_result.returncode == 0

        if not original_passes:
            return {"original_passes": False, "mutated_fails": False, "valid": False}

        # Test mutated
        with open(src_path, "w") as f:
            f.write(mutation.mutated_text)
        mut_result = subprocess.run(
            test_command.split(), capture_output=True, timeout=timeout, cwd=tmpdir
        )
        mutated_fails = mut_result.returncode != 0

        return {
            "original_passes": original_passes,
            "mutated_fails": mutated_fails,
            "valid": mutated_fails,
        }
```

### Step 4: Bug Injection Pipeline

```python
# src/data/bug_injector/pipeline.py
import json
import random
from pathlib import Path
from src.data.bug_injector.mutations import ALL_MUTATORS
from src.data.bug_injector.validator import validate_mutation
from src.data.bug_injector.base import Mutation

BUG_FIX_TEMPLATE = """The following TypeScript code contains a bug. Fix it.

Buggy code:
```typescript
{buggy_code}
```

Bug location hint: {location}

Fixed code:"""

def generate_bug_pairs(
    source_files: list[str],
    output_path: str,
    validate: bool = True,
    test_command: str = None,
    mutations_per_file: int = 2,
    max_pairs: int = 100_000,
    include_hint: bool = True,
) -> dict:
    pairs = []
    skipped = 0

    for file_path in source_files:
        if len(pairs) >= max_pairs:
            break

        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if len(source) < 50 or len(source) > 10_000:
            continue

        # Try all mutators, pick random subset
        all_mutations: list[Mutation] = []
        for mutator in ALL_MUTATORS:
            all_mutations.extend(mutator.mutate(source))

        if not all_mutations:
            continue

        random.shuffle(all_mutations)
        selected = all_mutations[:mutations_per_file]

        for mutation in selected:
            if validate and test_command:
                result = validate_mutation(source, mutation, test_command)
                if not result["valid"]:
                    skipped += 1
                    continue

            location_hint = mutation.location if include_hint else "somewhere in the code"
            instruction = BUG_FIX_TEMPLATE.format(
                buggy_code=mutation.mutated_text,
                location=location_hint,
            )

            pairs.append({
                "instruction": instruction,
                "input": "",
                "output": f"```typescript\n{source}\n```",
                "metadata": {
                    "mutator": mutation.mutator_name,
                    "file": file_path,
                    "line": mutation.line_number,
                },
            })

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return {
        "pairs_generated": len(pairs),
        "skipped_invalid": skipped,
        "output_path": output_path,
    }
```

### Step 5: CLI Tool

```python
# cli/inject_bugs.py
import argparse
from pathlib import Path
from src.data.bug_injector.pipeline import generate_bug_pairs

def main():
    parser = argparse.ArgumentParser(description="Inject synthetic bugs into code for training.")
    parser.add_argument("source_dir", help="Directory of TypeScript source files.")
    parser.add_argument("output", help="Output JSONL path.")
    parser.add_argument("--mutations-per-file", type=int, default=2,
        help="Bug mutations to generate per file (default: 2).")
    parser.add_argument("--max-pairs", type=int, default=100_000)
    parser.add_argument("--validate", action="store_true",
        help="Validate mutations by running tests (requires --test-command).")
    parser.add_argument("--test-command", type=str, default=None,
        help="Command to run tests, e.g. 'npx jest'. Used with --validate.")
    parser.add_argument("--no-hint", action="store_true",
        help="Omit bug location hint from instruction.")
    parser.add_argument("--mutators", nargs="+",
        choices=["off_by_one", "operator", "null_check", "variable_swap",
                 "missing_return", "type_error", "swapped_args"],
        default=None, help="Specific mutators to use (default: all).")
    args = parser.parse_args()

    files = [str(f) for f in Path(args.source_dir).rglob("*.ts")]
    print(f"Found {len(files)} TypeScript files")

    stats = generate_bug_pairs(
        files, args.output,
        validate=args.validate,
        test_command=args.test_command,
        mutations_per_file=args.mutations_per_file,
        max_pairs=args.max_pairs,
        include_hint=not args.no_hint,
    )
    print(f"Done: {stats}")

if __name__ == "__main__":
    main()
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/bug_injector/base.py` | New — base mutator |
| `src/data/bug_injector/mutations.py` | New — all mutation types |
| `src/data/bug_injector/validator.py` | New — test-based validation |
| `src/data/bug_injector/pipeline.py` | New — full pipeline |
| `cli/inject_bugs.py` | New CLI entry point |

---

## Testing Strategy

```python
# tests/test_bug_injector.py

SAMPLE = """
function binarySearch(arr: number[], target: number): number {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}
"""

def test_off_by_one_mutates():
    m = OffByOneMutator()
    mutations = m.mutate(SAMPLE)
    assert len(mutations) > 0
    assert mutations[0].mutated_text != SAMPLE

def test_operator_mutates_comparison():
    m = OperatorMutator()
    mutations = m.mutate(SAMPLE)
    assert any("!==" in mut.mutated_text for mut in mutations)

def test_missing_return_removes_return():
    m = MissingReturnMutator()
    mutations = m.mutate(SAMPLE)
    if mutations:
        assert "return mid" not in mutations[0].mutated_text

def test_mutation_produces_different_code():
    from src.data.bug_injector.mutations import ALL_MUTATORS
    for mutator in ALL_MUTATORS:
        mutations = mutator.mutate(SAMPLE)
        for mut in mutations:
            assert mut.mutated_text != mut.original_text, \
                f"{mutator.name} produced identity mutation"

def test_pipeline_produces_pairs(tmp_path):
    src_file = tmp_path / "search.ts"
    src_file.write_text(SAMPLE)
    out = str(tmp_path / "bugs.jsonl")
    import json
    from src.data.bug_injector.pipeline import generate_bug_pairs
    stats = generate_bug_pairs([str(src_file)], out, validate=False)
    assert stats["pairs_generated"] > 0
    with open(out) as f:
        pair = json.loads(f.readline())
    assert "Fix" in pair["instruction"]
    assert "typescript" in pair["output"]
```

---

## Performance Considerations

- Mutation generation is pure text processing — O(|source|) per file. 100k files can be processed in minutes.
- Validation (running tests) is the bottleneck: ~1-5 seconds per mutation × 2 (original + mutated) = 2-10 seconds per pair. With 100k pairs and parallelization across 16 workers: ~3-10 hours.
- For large-scale generation, use `--no-validate` for a fast first pass, then validate asynchronously.
- Limit mutations per file to 2-3 to avoid generating too many low-quality mutations from simple files.

---

## Dependencies

No new pip dependencies for mutation generation. Test validation requires existing Node.js/Jest setup.

---

## Estimated Complexity

**Development time:** 4-5 days
**Risk:** Medium. Mutation quality varies — some mutations will be semantically equivalent (tests still pass). Validation eliminates these but adds execution cost. Without validation, ~30-50% of mutations may be semantically neutral (depending on test coverage).
**Lines of new code:** ~500

---

## 2026 Best Practices

- **Validate with tests when possible:** Unvalidated mutations may be neutral or even make the code "more correct" in edge cases. The validation step is what makes this training data reliable.
- **Diverse mutation types:** Using 7 different mutation types ensures the model learns to recognize diverse bug patterns rather than specializing on one.
- **Bug type metadata:** Always store the mutator name in the training record. This enables post-hoc analysis of which bug types the model learns to fix most effectively.
- **Real bugs > synthetic bugs:** If the project has access to real git commit diffs that fix bugs, prefer those over synthetic mutations. Real bugs are harder and more valuable. Use synthetic bugs to fill gaps at scale.
- **Fuzzing integration:** For higher-confidence validation, use property-based tests (e.g., fast-check) rather than unit tests — they catch more semantic equivalences than hand-written tests.
