# Feature 59: Code Complexity Scorer

**Status:** Proposed
**CLI Flag:** `--complexity-score`
**Complexity:** Low-Medium

---

## Overview

Rates each file in the training corpus by multiple complexity metrics (cyclomatic complexity, nesting depth, line count, function count, import count) and assigns a difficulty bucket (1-5). These scores are stored in a manifest file and consumed by the curriculum scheduler (Feature 51) and training sampler. Supports both Python-based analysis (`radon`) and custom TypeScript/JavaScript analysis via tree-sitter.

---

## Motivation

Before curriculum learning can work, we need accurate complexity scores for every file in the corpus. Manual labeling is infeasible at scale. Automated complexity scoring provides:

- Ground-truth difficulty labels for curriculum learning
- Analysis of dataset composition (what fraction of files are L1 vs L5?)
- Quality filtering: files with very low complexity are trivial; extremely high complexity may be unlearnable
- Debugging: when the model underperforms, check if the difficulty distribution of training data matches evaluation

Cyclomatic complexity is a well-validated proxy for code difficulty and bug density (McCabe, 1976; empirical studies consistently show correlation with defect rates).

---

## Architecture / Design

```
Source file (.ts/.py)
  │
  ├── TypeScriptComplexityAnalyzer (tree-sitter)
  │   ├── cyclomatic_complexity  (decision points)
  │   ├── max_nesting_depth      (deepest block)
  │   ├── line_count
  │   ├── function_count
  │   └── import_count
  │
  └── PythonComplexityAnalyzer (radon, for .py files)
  │
  ▼
ComplexityScore → difficulty_bucket: 1-5
  │
  ▼
Manifest: { file_path, scores, bucket } (JSONL)
```

### Bucket Thresholds

| Bucket | Cyclomatic | Nesting | Lines | Functions |
|---|---|---|---|---|
| 1 (trivial) | ≤ 2 | ≤ 1 | ≤ 20 | ≤ 2 |
| 2 (easy) | ≤ 5 | ≤ 2 | ≤ 50 | ≤ 5 |
| 3 (medium) | ≤ 10 | ≤ 3 | ≤ 150 | ≤ 10 |
| 4 (hard) | ≤ 20 | ≤ 4 | ≤ 400 | ≤ 20 |
| 5 (expert) | > 20 | > 4 | > 400 | > 20 |

---

## Implementation Steps

### Step 1: TypeScript Complexity Analyzer

```python
# src/data/ts_complexity.py
from dataclasses import dataclass
from typing import Optional
import re

try:
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language, Parser, Node
    TS_LANGUAGE = Language(ts_typescript.language_typescript())
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False

# Decision point keywords that increase cyclomatic complexity
DECISION_NODES = {
    "if_statement", "else_clause", "ternary_expression",
    "for_statement", "for_in_statement", "for_of_statement",
    "while_statement", "do_statement",
    "switch_case", "catch_clause",
    "logical_expression",   # && and || are branching
}

@dataclass
class ComplexityMetrics:
    cyclomatic: int = 1         # McCabe: starts at 1
    max_nesting: int = 0
    line_count: int = 0
    function_count: int = 0
    import_count: int = 0
    avg_function_length: float = 0.0
    source_file: str = ""

def analyze_typescript(source: str, file_path: str = "") -> Optional[ComplexityMetrics]:
    if HAS_TREESITTER:
        return _analyze_with_treesitter(source, file_path)
    else:
        return _analyze_with_regex(source, file_path)

def _analyze_with_treesitter(source: str, file_path: str) -> Optional[ComplexityMetrics]:
    parser = Parser(TS_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    if tree.root_node.has_error and len(source) < 100:
        return None

    metrics = ComplexityMetrics(source_file=file_path)
    metrics.line_count = source.count("\n") + 1
    metrics.import_count = source.count("import ")

    fn_lines = []
    _walk(tree.root_node, metrics, depth=0, fn_lines=fn_lines)

    if fn_lines:
        metrics.avg_function_length = sum(fn_lines) / len(fn_lines)

    return metrics

def _walk(node: "Node", metrics: ComplexityMetrics, depth: int, fn_lines: list):
    metrics.max_nesting = max(metrics.max_nesting, depth)

    if node.type in DECISION_NODES:
        metrics.cyclomatic += 1

    if node.type in {"function_declaration", "arrow_function", "method_definition",
                     "function_expression"}:
        metrics.function_count += 1
        fn_len = node.end_point[0] - node.start_point[0] + 1
        fn_lines.append(fn_len)
        new_depth = depth + 1
    else:
        new_depth = depth + (1 if node.type in {
            "statement_block", "class_body", "switch_body"
        } else 0)

    for child in node.children:
        _walk(child, metrics, new_depth, fn_lines)

def _analyze_with_regex(source: str, file_path: str) -> ComplexityMetrics:
    """Fallback regex-based analysis when tree-sitter is not available."""
    metrics = ComplexityMetrics(source_file=file_path)
    metrics.line_count = source.count("\n") + 1
    metrics.import_count = len(re.findall(r"^\s*import\b", source, re.MULTILINE))
    metrics.function_count = len(re.findall(r"\bfunction\b|\barrow\b|=>\s*\{", source))

    decision_keywords = r"\b(if|else\s+if|for|while|do|switch|catch|&&|\|\||\?[^?])\b"
    metrics.cyclomatic = 1 + len(re.findall(decision_keywords, source))

    # Estimate nesting by max consecutive open braces
    max_depth = 0
    current_depth = 0
    for ch in source:
        if ch == "{":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif ch == "}":
            current_depth = max(0, current_depth - 1)
    metrics.max_nesting = max_depth

    return metrics
```

### Step 2: Python Complexity Analyzer (radon)

```python
# src/data/py_complexity.py
from src.data.ts_complexity import ComplexityMetrics

def analyze_python(source: str, file_path: str = "") -> ComplexityMetrics:
    try:
        from radon.complexity import cc_rank, cc_visit
        from radon.metrics import h_visit, mi_visit
        from radon.raw import analyze

        raw = analyze(source)
        blocks = cc_visit(source)
        cyclomatic = max((b.complexity for b in blocks), default=1)
        function_count = len(blocks)

        return ComplexityMetrics(
            cyclomatic=cyclomatic,
            max_nesting=_estimate_nesting(source),
            line_count=raw.loc,
            function_count=function_count,
            import_count=source.count("import "),
            source_file=file_path,
        )
    except ImportError:
        # radon not installed: fall back
        from src.data.ts_complexity import _analyze_with_regex
        return _analyze_with_regex(source, file_path)

def _estimate_nesting(source: str) -> int:
    max_indent = 0
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent // 4)
    return max_indent
```

### Step 3: Difficulty Bucket Assignment

```python
# src/data/complexity_scorer.py
from dataclasses import dataclass, asdict
from src.data.ts_complexity import ComplexityMetrics

@dataclass
class ScoredFile:
    file_path: str
    cyclomatic: int
    max_nesting: int
    line_count: int
    function_count: int
    import_count: int
    bucket: int          # 1-5

BUCKET_RULES = [
    # (max_cyclomatic, max_nesting, max_lines, max_functions) → bucket
    (2,  1,  20,  2,  1),
    (5,  2,  50,  5,  2),
    (10, 3,  150, 10, 3),
    (20, 4,  400, 20, 4),
]

def assign_bucket(metrics: ComplexityMetrics) -> int:
    for max_cc, max_nest, max_lines, max_fns, bucket in BUCKET_RULES:
        if (metrics.cyclomatic  <= max_cc   and
            metrics.max_nesting <= max_nest  and
            metrics.line_count  <= max_lines and
            metrics.function_count <= max_fns):
            return bucket
    return 5

def score_file(source: str, file_path: str, extension: str = ".ts") -> ScoredFile:
    from src.data.ts_complexity import analyze_typescript
    from src.data.py_complexity import analyze_python

    if extension in {".py"}:
        metrics = analyze_python(source, file_path)
    else:
        metrics = analyze_typescript(source, file_path)

    if metrics is None:
        return ScoredFile(
            file_path=file_path,
            cyclomatic=0, max_nesting=0, line_count=0,
            function_count=0, import_count=0, bucket=1,
        )

    bucket = assign_bucket(metrics)
    return ScoredFile(
        file_path=file_path,
        cyclomatic=metrics.cyclomatic,
        max_nesting=metrics.max_nesting,
        line_count=metrics.line_count,
        function_count=metrics.function_count,
        import_count=metrics.import_count,
        bucket=bucket,
    )
```

### Step 4: Manifest Builder

```python
# src/data/complexity_manifest.py
import json
from pathlib import Path
from src.data.complexity_scorer import score_file, ScoredFile
from dataclasses import asdict
from collections import Counter

def build_complexity_manifest(
    source_dir: str,
    manifest_path: str,
    extensions: list[str] = None,
    max_files: int = None,
) -> dict:
    if extensions is None:
        extensions = [".ts", ".tsx", ".js", ".jsx", ".py"]

    files = [f for f in Path(source_dir).rglob("*") if f.suffix in extensions]
    if max_files:
        files = files[:max_files]

    bucket_counts = Counter()
    processed = 0

    with open(manifest_path, "w") as f:
        for file_path in files:
            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
                scored = score_file(source, str(file_path), file_path.suffix)
                f.write(json.dumps(asdict(scored)) + "\n")
                bucket_counts[scored.bucket] += 1
                processed += 1
            except Exception:
                continue

            if processed % 5000 == 0:
                print(f"Scored {processed}/{len(files)} files")

    print("\nDifficulty distribution:")
    for bucket in range(1, 6):
        count = bucket_counts[bucket]
        pct = count / processed * 100 if processed else 0
        bar = "█" * int(pct / 2)
        print(f"  L{bucket}: {count:6d} ({pct:5.1f}%) {bar}")

    return {"processed": processed, "distribution": dict(bucket_counts)}

def load_manifest(manifest_path: str) -> list[ScoredFile]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            data = json.loads(line)
            records.append(ScoredFile(**data))
    return records

def filter_by_bucket(manifest: list[ScoredFile], bucket: int) -> list[ScoredFile]:
    return [r for r in manifest if r.bucket == bucket]

def get_distribution_stats(manifest: list[ScoredFile]) -> dict:
    from collections import Counter
    import statistics
    buckets = [r.bucket for r in manifest]
    cyclomatics = [r.cyclomatic for r in manifest]
    return {
        "total": len(manifest),
        "bucket_distribution": dict(Counter(buckets)),
        "cyclomatic_mean": statistics.mean(cyclomatics) if cyclomatics else 0,
        "cyclomatic_median": statistics.median(cyclomatics) if cyclomatics else 0,
        "cyclomatic_p95": sorted(cyclomatics)[int(len(cyclomatics)*0.95)] if cyclomatics else 0,
    }
```

### Step 5: CLI Integration

```python
# cli/score_complexity.py
import argparse
from src.data.complexity_manifest import build_complexity_manifest, load_manifest, get_distribution_stats

def main():
    parser = argparse.ArgumentParser(description="Score training files by complexity.")
    parser.add_argument("source_dir", help="Directory of source files to score.")
    parser.add_argument("output", help="Output manifest JSONL path.")
    parser.add_argument("--extensions", nargs="+", default=[".ts", ".tsx", ".js"],
        help="File extensions to process.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stats-only", action="store_true",
        help="Load existing manifest and show distribution stats.")
    args = parser.parse_args()

    if args.stats_only:
        manifest = load_manifest(args.output)
        stats = get_distribution_stats(manifest)
        import json
        print(json.dumps(stats, indent=2))
    else:
        build_complexity_manifest(
            args.source_dir, args.output,
            extensions=args.extensions,
            max_files=args.max_files,
        )

if __name__ == "__main__":
    main()
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/ts_complexity.py` | New — tree-sitter TS analyzer |
| `src/data/py_complexity.py` | New — radon Python analyzer |
| `src/data/complexity_scorer.py` | New — bucket assignment |
| `src/data/complexity_manifest.py` | New — manifest I/O |
| `cli/score_complexity.py` | New CLI entry point |
| `src/curriculum/problem_pool.py` | Consume manifest for curriculum learning |

---

## Testing Strategy

```python
# tests/test_complexity_scorer.py

TRIVIAL_TS = "function add(a: number, b: number): number { return a + b; }"
COMPLEX_TS = """
function solve(n: number, graph: number[][]): number {
  const dp = Array(n).fill(Infinity);
  dp[0] = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (graph[i][j] > 0) {
        if (dp[i] < Infinity && dp[j] > dp[i] + graph[i][j]) {
          dp[j] = dp[i] + graph[i][j];
        }
      }
    }
  }
  return dp[n-1] === Infinity ? -1 : dp[n-1];
}
"""

def test_trivial_gets_low_bucket():
    scored = score_file(TRIVIAL_TS, "add.ts", ".ts")
    assert scored.bucket <= 2

def test_complex_gets_high_bucket():
    scored = score_file(COMPLEX_TS, "solve.ts", ".ts")
    assert scored.bucket >= 3

def test_cyclomatic_counts_branches():
    metrics = analyze_typescript(COMPLEX_TS, "x.ts")
    assert metrics.cyclomatic > 3   # has multiple if statements

def test_manifest_round_trip(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "add.ts").write_text(TRIVIAL_TS)
    (src_dir / "solve.ts").write_text(COMPLEX_TS)
    manifest_path = str(tmp_path / "manifest.jsonl")
    build_complexity_manifest(str(src_dir), manifest_path)
    loaded = load_manifest(manifest_path)
    assert len(loaded) == 2
    buckets = {r.file_path.split("/")[-1]: r.bucket for r in loaded}
    assert buckets.get("add.ts", 5) < buckets.get("solve.ts", 1)
```

---

## Performance Considerations

- tree-sitter parsing: ~50MB/s → 1M 1KB files = ~20 seconds.
- Regex fallback: ~100MB/s → even faster.
- For 100M file corpora (The Stack scale), parallelize across 16+ workers:

```python
from multiprocessing import Pool
from functools import partial

def _score_one(file_info: tuple[str, str]) -> dict:
    path, ext = file_info
    try:
        source = open(path, errors="ignore").read()
        return vars(score_file(source, path, ext))
    except Exception:
        return None

with Pool(16) as pool:
    file_infos = [(str(f), f.suffix) for f in all_files]
    results = [r for r in pool.map(_score_one, file_infos) if r]
```

- Manifest storage: 200 bytes per record × 10M files = 2GB JSONL. Consider binary format (parquet) for large corpora.

---

## Dependencies

```
radon>=6.0.0                    # optional, for Python files only
tree-sitter==0.23.0             # optional, from Feature 55
tree-sitter-typescript==0.23.0  # optional, from Feature 55
```

The regex fallback requires no dependencies.

---

## Estimated Complexity

**Development time:** 2-3 days
**Risk:** Low. Complexity analysis is well-understood; the main risk is bucket threshold calibration. Adjust thresholds based on the actual distribution of your corpus.
**Lines of new code:** ~350

---

## 2026 Best Practices

- **Calibrate buckets to your corpus:** The default thresholds assume a typical open-source TypeScript codebase. Run `--stats-only` on your full corpus and adjust bucket boundaries so roughly 20% of files fall in each bucket.
- **Multiple metrics beat single metric:** A file can have low cyclomatic complexity but deep nesting (e.g., deeply nested callbacks). Using multiple signals for bucket assignment is more robust.
- **Store in manifest, not in filenames:** Embedding complexity scores in a separate JSONL manifest is preferable to renaming files or embedding in directory structure. The manifest is queryable and doesn't affect file paths.
- **Update incrementally:** If you add new files to the corpus, only score the new files and append to the manifest rather than recomputing everything.
