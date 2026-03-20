# Feature 58: Test-Code Pair Extractor

**Status:** Proposed
**CLI Flag:** `--extract-test-pairs`
**Complexity:** Medium

---

## Overview

Identifies test files and matches them to their corresponding source files using naming convention heuristics and tree-sitter analysis. Creates paired training examples in both directions: [test → source] (test-driven development) and [source → test] (generating tests for code). Filters to ensure the test actually tests the source via function name matching.

---

## Motivation

Most code generation models are trained on source files in isolation and struggle to:
1. Write code that passes a given test suite (TDD workflow)
2. Write tests for a given function (a common developer task)

Test-code pairs provide a unique training signal: the model learns the relationship between intent (tests) and implementation (source). This is the closest available approximation to training on (specification → implementation) pairs.

Research precedent: AlphaCode and CodeContests use problem statements + test cases as training pairs, showing 30-50% improvement on competitive programming vs source-only training.

---

## Architecture / Design

```
Repository scan
  │
  ├── Identify test files by pattern:
  │   foo.test.ts → foo.ts
  │   foo.spec.ts → foo.ts
  │   __tests__/foo.ts → ../foo.ts
  │   test/foo.ts → src/foo.ts (heuristic)
  │
  ├── Validate pair:
  │   - Source file exists
  │   - Test imports from source (or source path matches)
  │   - At least one function name in test matches source
  │
  └── Format:
      Direction A: [test file] → [source file]
      Direction B: [source file] → [test file]
```

---

## Implementation Steps

### Step 1: Test File Detector

```python
# src/data/test_file_detector.py
import re
from pathlib import Path
from typing import Optional

TEST_FILE_PATTERNS = [
    # Pattern, source resolution strategy
    (re.compile(r"^(.+)\.test\.(ts|tsx|js|jsx)$"), "same_dir"),
    (re.compile(r"^(.+)\.spec\.(ts|tsx|js|jsx)$"), "same_dir"),
    (re.compile(r"^(.+)-test\.(ts|tsx|js|jsx)$"),  "same_dir"),
    (re.compile(r"^(.+)_test\.(ts|tsx|js|jsx)$"),  "same_dir"),
]

TEST_DIR_PATTERNS = [
    ("__tests__", ".."),         # __tests__/foo.ts → ../foo.ts
    ("test",      "../src"),     # test/foo.ts → ../src/foo.ts
    ("tests",     "../src"),
    ("__test__",  ".."),
]

def is_test_file(path: Path) -> bool:
    name = path.name
    for pattern, _ in TEST_FILE_PATTERNS:
        if pattern.match(name):
            return True
    for test_dir, _ in TEST_DIR_PATTERNS:
        if test_dir in path.parts:
            return True
    return False

def resolve_source_path(test_path: Path, project_root: Path) -> Optional[Path]:
    name = test_path.name
    parent = test_path.parent

    # Strategy 1: remove .test/.spec suffix
    for pattern, strategy in TEST_FILE_PATTERNS:
        m = pattern.match(name)
        if m:
            base_name = m.group(1)
            ext = m.group(2)
            if strategy == "same_dir":
                candidate = parent / f"{base_name}.{ext}"
                if candidate.exists():
                    return candidate

    # Strategy 2: __tests__ directory
    for test_dir, relative_to in TEST_DIR_PATTERNS:
        if test_dir in test_path.parts:
            idx = list(test_path.parts).index(test_dir)
            # Build path without the test_dir component
            parts_before = test_path.parts[:idx]
            parts_after  = test_path.parts[idx+1:]
            candidate = Path(*parts_before, *parts_after) if parts_before else Path(*parts_after)
            for ext in [".ts", ".tsx", ".js", ".jsx"]:
                with_ext = candidate.with_suffix(ext)
                if with_ext.exists():
                    return with_ext

    return None
```

### Step 2: Test-Source Validator

```python
# src/data/test_pair_validator.py
import re
from pathlib import Path

def extract_function_names(source: str) -> set[str]:
    """Extract all function/method names from TypeScript source."""
    pattern = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)"
        r"|const\s+(\w+)\s*=\s*(?:async\s+)?\(?"
        r"|(?:public|private|protected)?\s+(?:async\s+)?(\w+)\s*\(",
        re.MULTILINE,
    )
    names = set()
    for m in pattern.finditer(source):
        for g in m.groups():
            if g and len(g) > 1 and g not in {"function", "const", "async"}:
                names.add(g)
    return names

def extract_test_references(test_source: str) -> set[str]:
    """Extract function names referenced in tests."""
    # Look for describe/it/test blocks and what they call
    names = set()

    # Functions called in test bodies
    call_pattern = re.compile(r"(\w{2,})\s*\(")
    for m in call_pattern.finditer(test_source):
        name = m.group(1)
        if name not in {"describe", "it", "test", "expect", "beforeEach", "afterEach",
                        "beforeAll", "afterAll", "jest", "vi", "assert", "should"}:
            names.add(name)

    # Import references
    import_pattern = re.compile(r"import\s*\{([^}]+)\}")
    for m in import_pattern.finditer(test_source):
        for name in m.group(1).split(","):
            names.add(name.strip().split(" as ")[0].strip())

    return names

def validate_pair(
    test_source: str,
    source_source: str,
    min_overlap: int = 1,
) -> dict:
    """
    Check that the test file references at least min_overlap functions from source.
    """
    source_fns = extract_function_names(source_source)
    test_refs  = extract_test_references(test_source)

    overlap = source_fns & test_refs
    return {
        "valid": len(overlap) >= min_overlap,
        "overlap": list(overlap),
        "source_functions": list(source_fns),
        "test_references": list(test_refs),
        "overlap_count": len(overlap),
    }
```

### Step 3: Pair Formatter

```python
# src/data/test_pair_formatter.py
import json
from dataclasses import dataclass

@dataclass
class TestCodePair:
    test_file: str
    source_file: str
    test_content: str
    source_content: str
    overlap_functions: list[str]
    direction: str    # "test_to_source" or "source_to_test"

# Prompt templates
TEST_TO_SOURCE_PROMPT = """Given the following test file, implement the source code that would make all tests pass.

Tests:
```typescript
{test_content}
```

Implement the source code:"""

SOURCE_TO_TEST_PROMPT = """Write comprehensive TypeScript tests for the following code:

Source:
```typescript
{source_content}
```

Tests:"""

def format_pair(pair: TestCodePair, format_style: str = "alpaca") -> dict:
    if pair.direction == "test_to_source":
        instruction = TEST_TO_SOURCE_PROMPT.format(test_content=pair.test_content[:2000])
        output = f"```typescript\n{pair.source_content[:3000]}\n```"
    else:
        instruction = SOURCE_TO_TEST_PROMPT.format(source_content=pair.source_content[:2000])
        output = f"```typescript\n{pair.test_content[:3000]}\n```"

    if format_style == "alpaca":
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "metadata": {
                "test_file": pair.test_file,
                "source_file": pair.source_file,
                "direction": pair.direction,
                "overlap_functions": pair.overlap_functions,
            },
        }
    return {
        "prompt": instruction,
        "completion": output,
    }
```

### Step 4: Repository Scanner

```python
# src/data/test_pair_scanner.py
from pathlib import Path
from src.data.test_file_detector import is_test_file, resolve_source_path
from src.data.test_pair_validator import validate_pair
from src.data.test_pair_formatter import TestCodePair, format_pair
import json

def scan_repository(
    repo_root: str,
    output_path: str,
    include_test_to_source: bool = True,
    include_source_to_test: bool = True,
    min_overlap: int = 1,
    max_source_chars: int = 8000,
    max_test_chars: int = 8000,
) -> dict:
    root = Path(repo_root)
    extensions = {".ts", ".tsx", ".js", ".jsx"}

    all_files = [f for f in root.rglob("*") if f.suffix in extensions]
    test_files = [f for f in all_files if is_test_file(f)]

    pairs = []
    unresolved = 0
    invalid = 0

    for test_file in test_files:
        source_file = resolve_source_path(test_file, root)
        if not source_file:
            unresolved += 1
            continue

        try:
            test_src   = test_file.read_text(encoding="utf-8", errors="ignore")
            source_src = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Size limits
        if len(test_src) > max_test_chars or len(source_src) > max_source_chars:
            continue
        if len(test_src) < 50 or len(source_src) < 50:
            continue

        validation = validate_pair(test_src, source_src, min_overlap)
        if not validation["valid"]:
            invalid += 1
            continue

        overlap = validation["overlap"]

        if include_test_to_source:
            pairs.append(TestCodePair(
                test_file=str(test_file.relative_to(root)),
                source_file=str(source_file.relative_to(root)),
                test_content=test_src,
                source_content=source_src,
                overlap_functions=overlap,
                direction="test_to_source",
            ))

        if include_source_to_test:
            pairs.append(TestCodePair(
                test_file=str(test_file.relative_to(root)),
                source_file=str(source_file.relative_to(root)),
                test_content=test_src,
                source_content=source_src,
                overlap_functions=overlap,
                direction="source_to_test",
            ))

    # Write output
    with open(output_path, "w") as f:
        for pair in pairs:
            record = format_pair(pair)
            f.write(json.dumps(record) + "\n")

    return {
        "test_files_found": len(test_files),
        "pairs_created": len(pairs),
        "unresolved": unresolved,
        "invalid_pairs": invalid,
        "output_path": output_path,
    }
```

### Step 5: CLI Integration

```python
# cli/extract_test_pairs.py
import argparse
from src.data.test_pair_scanner import scan_repository

def main():
    parser = argparse.ArgumentParser(description="Extract test-source pairs for training.")
    parser.add_argument("repo_root", help="Root of repository to scan.")
    parser.add_argument("output", help="Output JSONL path.")
    parser.add_argument("--no-test-to-source", action="store_true",
        help="Exclude [test → source] direction.")
    parser.add_argument("--no-source-to-test", action="store_true",
        help="Exclude [source → test] direction.")
    parser.add_argument("--min-overlap", type=int, default=1,
        help="Minimum overlapping function names required (default: 1).")
    parser.add_argument("--max-source-chars", type=int, default=8000)
    parser.add_argument("--max-test-chars", type=int, default=8000)
    args = parser.parse_args()

    stats = scan_repository(
        args.repo_root,
        args.output,
        include_test_to_source=not args.no_test_to_source,
        include_source_to_test=not args.no_source_to_test,
        min_overlap=args.min_overlap,
        max_source_chars=args.max_source_chars,
        max_test_chars=args.max_test_chars,
    )
    print(f"Results: {stats}")

if __name__ == "__main__":
    main()
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/test_file_detector.py` | New |
| `src/data/test_pair_validator.py` | New |
| `src/data/test_pair_formatter.py` | New |
| `src/data/test_pair_scanner.py` | New |
| `cli/extract_test_pairs.py` | New CLI entry point |

---

## Testing Strategy

```python
# tests/test_test_pair_extractor.py
from pathlib import Path
import tempfile

def test_detects_test_file():
    assert is_test_file(Path("src/utils/helpers.test.ts"))
    assert is_test_file(Path("src/__tests__/utils.ts"))
    assert not is_test_file(Path("src/utils/helpers.ts"))

def test_resolves_dot_test_suffix(tmp_path):
    source = tmp_path / "helpers.ts"
    test   = tmp_path / "helpers.test.ts"
    source.write_text("export function add() {}")
    test.write_text("import { add } from './helpers';\ntest('add', () => {});")
    resolved = resolve_source_path(test, tmp_path)
    assert resolved == source

def test_validates_pair_with_overlap():
    source = "export function add(a: number, b: number) { return a + b; }"
    test   = "import { add } from './add';\ntest('add', () => { expect(add(1, 2)).toBe(3); });"
    result = validate_pair(test, source, min_overlap=1)
    assert result["valid"] is True
    assert "add" in result["overlap"]

def test_rejects_unrelated_pair():
    source = "export function encrypt(data: string) { return data; }"
    test   = "import { formatDate } from './date';\ntest('formatDate', () => {});"
    result = validate_pair(test, source, min_overlap=1)
    assert result["valid"] is False

def test_both_directions_created(tmp_path):
    source_file = tmp_path / "math.ts"
    test_file   = tmp_path / "math.test.ts"
    source_file.write_text("export function add(a: number, b: number): number { return a + b; }")
    test_file.write_text("import { add } from './math';\ntest('add', () => { expect(add(1,2)).toBe(3); });")
    out = str(tmp_path / "pairs.jsonl")
    stats = scan_repository(str(tmp_path), out)
    assert stats["pairs_created"] == 2  # both directions
```

---

## Performance Considerations

- Scanning 10k files: ~5-10 seconds for file discovery + 20-50 seconds for content reading and validation. Acceptable as a one-time preprocessing step.
- Validation regex is O(|file_size|) — fast for normal-sized files.
- Use `min_overlap=1` (loose) initially and increase to 2-3 for higher quality pairs.
- Large test files (e.g., integration tests with 500+ test cases) may inflate context length. Apply the `max_test_chars` limit aggressively.

---

## Dependencies

No new pip dependencies.

---

## Estimated Complexity

**Development time:** 2-3 days
**Risk:** Low. The heuristics are straightforward; validation is simple regex. Main risk is false positives (test file matched to wrong source) — the overlap validation reduces this significantly.
**Lines of new code:** ~400

---

## 2026 Best Practices

- **Both directions matter equally:** Test→Source trains TDD ability; Source→Test trains test generation ability. Both are valuable tasks in modern software development.
- **Validate with actual test execution:** For the highest quality pairs, run the test file against the source file and only include pairs where tests pass. This adds execution overhead but eliminates false pairs.
- **Coverage as a quality signal:** Use `istanbul`/`c8` coverage data if available. Pairs where the test achieves >80% line coverage are higher quality training examples.
- **Repository-level deduplication:** Many repositories contain forks of other projects. Apply MinHash dedup at the (test_content, source_content) level to avoid training on near-duplicate pairs.
