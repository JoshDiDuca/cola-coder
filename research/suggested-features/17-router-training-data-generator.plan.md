# Feature 17: Router Training Data Generator

**Status:** Optional | **CLI Flag:** `--generate-router-data` | **Complexity:** Low-Medium

---

## Overview

An automated pipeline that scans existing code samples, applies rule-based domain detection heuristics (import analysis, keyword patterns, file patterns), and produces labeled `(code_snippet, domain_label)` pairs in JSONL format. This JSONL file is the training dataset for the learned router model (Feature 16).

The generator requires no manual labeling — it leverages deterministic signals present in TypeScript/JavaScript source code (import statements, framework-specific APIs, test runner keywords) to assign confident domain labels automatically.

---

## Motivation

Training the router model (Feature 16) requires labeled examples. Manual annotation at scale is expensive. TypeScript/JavaScript code has strong, consistent domain signals that can be extracted programmatically:

- `import React` or JSX syntax → React domain
- `from 'next'` or `getServerSideProps` → Next.js domain
- `gql\`` or `import { gql }` → GraphQL domain
- `@prisma/client` or `prisma.` method chains → Prisma domain
- `z.object(`, `z.string()` → Zod domain
- `describe(`, `it(`, `test(`, `expect(` → Testing domain
- Everything else → General TypeScript

A high-quality auto-labeling pipeline can produce thousands of training examples from existing corpora (GitHub TypeScript repos, cola-coder's own training data) with minimal human effort.

---

## Architecture / Design

### Pipeline Overview

```
Source: code files / JSONL corpus
          ↓
File Scanner (glob *.ts, *.tsx, *.js, *.mts)
          ↓
Import Extractor (AST or regex)
          ↓
Domain Classifier (rules engine)
          ↓
Multi-domain Resolver (primary by import count)
          ↓
Snippet Chunker (split large files into 128-512 token windows)
          ↓
JSONL Writer → router_train.jsonl, router_val.jsonl
```

### Domain Rules (Priority Ordered)

```python
DOMAIN_RULES = {
    "react": {
        "imports": ["react", "react-dom", "react-native", "@types/react"],
        "keywords": ["useState", "useEffect", "useRef", "JSX.Element", "<Component"],
        "file_patterns": [r"\.tsx$", r"\.jsx$"],
        "weight": 1.0,
    },
    "nextjs": {
        "imports": ["next", "next/router", "next/navigation", "next/image", "next/link"],
        "keywords": ["getServerSideProps", "getStaticProps", "NextPage", "NextApiHandler",
                     "useRouter", "app/", "pages/"],
        "file_patterns": [r"pages/.*\.tsx$", r"app/.*\.tsx$"],
        "weight": 1.5,  # Next.js implies React; give it priority
    },
    "graphql": {
        "imports": ["graphql", "@apollo/client", "apollo-server", "graphql-tag",
                    "urql", "@graphql-codegen"],
        "keywords": ["gql`", "useQuery", "useMutation", "GraphQLSchema", "resolvers:",
                     "typeDefs", "ApolloClient"],
        "file_patterns": [r"\.graphql$", r"\.gql$", r"schema\.ts$"],
        "weight": 1.0,
    },
    "prisma": {
        "imports": ["@prisma/client", "prisma"],
        "keywords": ["prisma.", "PrismaClient", "prisma.findMany", "prisma.create",
                     "$transaction", "Prisma."],
        "file_patterns": [r"schema\.prisma$", r"prisma/.*\.ts$"],
        "weight": 1.0,
    },
    "zod": {
        "imports": ["zod", "zod/lib"],
        "keywords": ["z.object(", "z.string()", "z.number()", "z.array(",
                     "z.infer<", ".parse(", ".safeParse(", "ZodSchema"],
        "file_patterns": [r"schema\.ts$", r"validation\.ts$", r"validators/"],
        "weight": 1.0,
    },
    "testing": {
        "imports": ["vitest", "jest", "@testing-library", "playwright", "cypress",
                    "supertest", "@jest/globals"],
        "keywords": ["describe(", "it(", "test(", "expect(", "beforeEach(",
                     "afterEach(", "vi.mock(", "jest.mock(", "render("],
        "file_patterns": [r"\.test\.[jt]sx?$", r"\.spec\.[jt]sx?$",
                          r"__tests__/", r"tests?/"],
        "weight": 1.0,
    },
    "general_ts": {
        "imports": [],
        "keywords": ["interface ", "type ", "enum ", "namespace ", "declare "],
        "file_patterns": [r"\.ts$", r"\.mts$"],
        "weight": 0.1,  # Lowest priority — catchall
    },
}
```

---

## Implementation Steps

### Step 1: Import Extractor

```python
# cola_coder/data/import_extractor.py
import re
from typing import List

IMPORT_PATTERN = re.compile(
    r"""(?:import\s+(?:type\s+)?(?:\*\s+as\s+\w+|[\w\s{},*]+)\s+from\s+['"]([^'"]+)['"]"""
    r"""|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
    re.MULTILINE,
)

def extract_imports(code: str) -> List[str]:
    """Extract all imported module names from a code snippet."""
    matches = IMPORT_PATTERN.findall(code)
    imports = []
    for m in matches:
        pkg = m[0] or m[1]
        # Normalize: take the top-level package name
        # e.g., "@prisma/client" → "@prisma/client", "next/router" → "next"
        parts = pkg.split("/")
        if pkg.startswith("@"):
            normalized = "/".join(parts[:2])  # scoped package
        else:
            normalized = parts[0]
        imports.append(normalized)
    return list(set(imports))

def count_domain_imports(code: str) -> dict[str, int]:
    """Return count of imports matching each domain."""
    imports = extract_imports(code)
    counts = {domain: 0 for domain in DOMAIN_RULES}
    for domain, rules in DOMAIN_RULES.items():
        for pkg in imports:
            if any(pkg == r or pkg.startswith(r) for r in rules["imports"]):
                counts[domain] += 1
    return counts
```

### Step 2: Rules Engine / Domain Classifier

```python
# cola_coder/data/domain_classifier.py
import re
from typing import Optional
from .import_extractor import count_domain_imports, DOMAIN_RULES

def classify_domain(
    code: str,
    filepath: str = "",
    min_confidence: float = 0.3,
) -> Optional[tuple[str, float]]:
    """
    Returns (domain_label, confidence) or None if below threshold.
    Confidence is a normalized score in [0, 1].
    """
    scores = {}

    # 1. Import scores (strongest signal)
    import_counts = count_domain_imports(code)
    for domain, count in import_counts.items():
        if count > 0:
            weight = DOMAIN_RULES[domain]["weight"]
            scores[domain] = scores.get(domain, 0) + count * weight * 3.0

    # 2. Keyword scores
    for domain, rules in DOMAIN_RULES.items():
        for kw in rules["keywords"]:
            occurrences = code.count(kw)
            if occurrences > 0:
                weight = DOMAIN_RULES[domain]["weight"]
                scores[domain] = scores.get(domain, 0) + occurrences * weight

    # 3. File pattern scores
    for domain, rules in DOMAIN_RULES.items():
        for pat in rules["file_patterns"]:
            if re.search(pat, filepath):
                scores[domain] = scores.get(domain, 0) + 5.0  # Strong signal

    if not scores:
        return ("general_ts", 0.5)

    total = sum(scores.values())
    best_domain = max(scores, key=scores.get)
    confidence = scores[best_domain] / total if total > 0 else 0.0

    if confidence < min_confidence:
        return None

    return (best_domain, confidence)


def classify_multi_domain(code: str, filepath: str = "") -> dict[str, float]:
    """Return normalized scores for ALL domains (for multi-label analysis)."""
    scores = {}
    import_counts = count_domain_imports(code)
    for domain, count in import_counts.items():
        if count > 0:
            scores[domain] = count * DOMAIN_RULES[domain]["weight"] * 3.0
    for domain, rules in DOMAIN_RULES.items():
        for kw in rules["keywords"]:
            occ = code.count(kw)
            if occ > 0:
                scores[domain] = scores.get(domain, 0) + occ * DOMAIN_RULES[domain]["weight"]
    total = sum(scores.values()) or 1.0
    return {d: s / total for d, s in scores.items()}
```

### Step 3: Snippet Chunker

```python
# cola_coder/data/snippet_chunker.py
from typing import Iterator

def chunk_code(
    code: str,
    tokenizer,
    chunk_size: int = 256,
    stride: int = 128,
    min_tokens: int = 32,
) -> Iterator[str]:
    """
    Yield overlapping chunks of code, each approximately chunk_size tokens.
    Prefers splitting at function/class boundaries when possible.
    """
    # Try to split at top-level boundaries first
    lines = code.split("\n")
    boundary_indices = [0]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("export function", "export const", "export class",
                                "function ", "class ", "const ", "async function")):
            boundary_indices.append(i)
    boundary_indices.append(len(lines))

    # Reconstruct chunks around boundaries
    for start_idx in range(0, len(boundary_indices) - 1):
        chunk_lines = []
        token_count = 0
        for j in range(start_idx, len(boundary_indices) - 1):
            segment = "\n".join(lines[boundary_indices[j]:boundary_indices[j+1]])
            segment_tokens = len(tokenizer.encode(segment))
            if token_count + segment_tokens > chunk_size and chunk_lines:
                break
            chunk_lines.append(segment)
            token_count += segment_tokens
        chunk = "\n".join(chunk_lines).strip()
        if len(tokenizer.encode(chunk)) >= min_tokens:
            yield chunk
```

### Step 4: Main Generator Pipeline

```python
# cola_coder/data/router_data_generator.py
import json
import random
import pathlib
from typing import Iterator
from .domain_classifier import classify_domain
from .snippet_chunker import chunk_code

def generate_router_dataset(
    source_dirs: list[str],
    output_path: str,
    tokenizer,
    chunk_size: int = 256,
    min_confidence: float = 0.5,
    val_split: float = 0.1,
    max_samples_per_domain: int = 10_000,
    seed: int = 42,
) -> dict[str, int]:
    """
    Scan source_dirs for .ts/.tsx/.js files, classify, chunk, and write JSONL.
    Returns domain count statistics.
    """
    random.seed(seed)
    domain_counts: dict[str, int] = {}
    records: list[dict] = []

    file_extensions = {".ts", ".tsx", ".js", ".mjs", ".jsx"}

    for source_dir in source_dirs:
        for path in pathlib.Path(source_dir).rglob("*"):
            if path.suffix not in file_extensions:
                continue
            if "node_modules" in path.parts:
                continue
            try:
                code = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            result = classify_domain(code, str(path), min_confidence=min_confidence)
            if result is None:
                continue
            domain, confidence = result

            current = domain_counts.get(domain, 0)
            if current >= max_samples_per_domain:
                continue

            for chunk in chunk_code(code, tokenizer, chunk_size=chunk_size):
                records.append({
                    "text": chunk,
                    "domain": domain,
                    "confidence": round(confidence, 4),
                    "source": str(path),
                })
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Shuffle and split
    random.shuffle(records)
    split_idx = int(len(records) * (1 - val_split))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    base = pathlib.Path(output_path)
    _write_jsonl(train_records, base.parent / (base.stem + "_train.jsonl"))
    _write_jsonl(val_records, base.parent / (base.stem + "_val.jsonl"))

    return domain_counts


def _write_jsonl(records: list[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {path}")
```

### Step 5: CLI Command

```python
# cli additions
@app.command()
def generate_router_data(
    source_dirs: List[str] = typer.Argument(...),
    output: str = typer.Option("data/router/router_data.jsonl", "--output", "-o"),
    chunk_size: int = typer.Option(256, "--chunk-size"),
    min_confidence: float = typer.Option(0.5, "--min-confidence"),
    max_per_domain: int = typer.Option(10000, "--max-per-domain"),
):
    """Auto-label code files and generate router training JSONL."""
    from cola_coder.data.router_data_generator import generate_router_dataset
    counts = generate_router_dataset(
        source_dirs, output, tokenizer,
        chunk_size=chunk_size,
        min_confidence=min_confidence,
        max_samples_per_domain=max_per_domain,
    )
    console.print("[bold]Domain distribution:[/bold]")
    for domain, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 100, 40)
        console.print(f"  {domain:15s} {count:6d}  {bar}")
```

---

## Key Files to Modify

- `cola_coder/data/import_extractor.py` — new file
- `cola_coder/data/domain_classifier.py` — new file
- `cola_coder/data/snippet_chunker.py` — new file
- `cola_coder/data/router_data_generator.py` — new file
- `cola_coder/cli.py` — add `generate-router-data` subcommand
- `tests/test_domain_classifier.py` — new test file
- `configs/router_data.yaml` — source dirs, output path, thresholds

---

## Testing Strategy

```python
# tests/test_domain_classifier.py

REACT_CODE = """
import React, { useState } from 'react';
import { Button } from '@/components/Button';

const Counter: React.FC = () => {
    const [count, setCount] = useState(0);
    return <Button onClick={() => setCount(c => c + 1)}>{count}</Button>;
};
export default Counter;
"""

PRISMA_CODE = """
import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();
export async function getUser(id: string) {
    return prisma.user.findUnique({ where: { id } });
}
"""

ZOD_CODE = """
import { z } from 'zod';
const UserSchema = z.object({
    name: z.string().min(1),
    email: z.string().email(),
    age: z.number().int().positive(),
});
type User = z.infer<typeof UserSchema>;
"""

def test_react_classification():
    domain, conf = classify_domain(REACT_CODE)
    assert domain == "react"
    assert conf > 0.5

def test_prisma_classification():
    domain, conf = classify_domain(PRISMA_CODE)
    assert domain == "prisma"

def test_zod_classification():
    domain, conf = classify_domain(ZOD_CODE)
    assert domain == "zod"

def test_nextjs_priority_over_react():
    code = "import { useRouter } from 'next/router';\nimport React from 'react';"
    domain, _ = classify_domain(code)
    assert domain == "nextjs"  # Next.js should win due to higher weight

def test_low_confidence_returns_none():
    code = "const x = 1;\nconst y = 2;"
    result = classify_domain(code, min_confidence=0.9)
    assert result is None

def test_import_extractor():
    from cola_coder.data.import_extractor import extract_imports
    code = "import { z } from 'zod';\nconst x = require('@prisma/client');"
    imports = extract_imports(code)
    assert "zod" in imports
    assert "@prisma/client" in imports
```

---

## Performance Considerations

- **Regex over AST:** Full TypeScript AST parsing (using `tree-sitter` via Python bindings) is more accurate but ~10x slower. For large corpora, regex-based import extraction is sufficient given the strong signal.
- **Parallel scanning:** Use `multiprocessing.Pool` or `concurrent.futures.ThreadPoolExecutor` for scanning large directories. I/O-bound, so threads are fine.
- **Deduplication:** Hash each snippet before writing; skip exact duplicates to avoid training on memorized code.
- **Class imbalance:** General TS will dominate. Enforce `max_samples_per_domain` and consider upsampling minority domains.
- **Memory:** Stream records to JSONL rather than accumulating all in memory for large corpora.

```python
# Streaming version for large corpora
def generate_streaming(source_dirs, output_path, tokenizer, **kwargs):
    with open(output_path, "w") as f:
        for record in _iter_records(source_dirs, tokenizer, **kwargs):
            f.write(json.dumps(record) + "\n")
```

---

## Dependencies

- Python standard library (`re`, `pathlib`, `json`, `random`)
- Cola-Coder tokenizer (for chunk_code token counting)
- Feature 24 (domain detection heuristic) — this feature IS feature 24's implementation; they share the rules engine
- Feature 16 (router model) — consumes the output JSONL

---

## Estimated Complexity

| Task                             | Effort  |
|----------------------------------|---------|
| Import extractor + rules         | 2h      |
| Domain classifier                | 2h      |
| Snippet chunker                  | 1h      |
| Pipeline + JSONL writer          | 2h      |
| CLI command                      | 0.5h    |
| Tests                            | 1.5h    |
| **Total**                        | **~9h** |

Overall complexity: **Low-Medium** (mostly string processing, no ML training required)

---

## 2026 Best Practices

- **Tree-sitter fallback:** For production use, integrate `tree-sitter-typescript` via Python bindings for accurate AST-based import extraction. Regex is fine for prototyping.
- **Confidence calibration:** Track classification confidence distribution. If >40% of samples land in "general_ts", lower the catchall weight or raise the min_confidence threshold.
- **Data versioning:** Store generator config (thresholds, rules version) in each JSONL record for reproducibility. Use DVC or simple hash-based versioning.
- **Deduplication at scale:** Use MinHash LSH (datasketch library) for near-duplicate detection across large GitHub corpora.
- **Domain co-occurrence tracking:** Log multi-domain files separately; these are valuable hard negatives for training a robust router.
- **Incremental updates:** Support `--since DATE` to only process files modified after a date, enabling cheap dataset refreshes.
