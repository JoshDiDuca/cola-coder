# Feature 24: Domain Detection Heuristic

**Status:** Optional | **CLI Flag:** `--heuristic-routing` | **Complexity:** Low

---

## Overview

A fast, pure-Python rules engine for domain classification that requires no ML model. Uses pattern matching on import statements, function/variable names, file extensions, and framework-specific code patterns to assign a domain label and confidence score to any TypeScript/JavaScript code snippet. Serves as: (1) a fast baseline for routing before the learned router is trained, (2) the primary source of training data labels for Feature 17, (3) a fallback when the router model is unavailable.

---

## Motivation

Before a learned router exists, the system needs some way to route prompts. Additionally:

- The heuristic generates the training labels for Feature 17 (bootstrap problem: need labels to train the router that will replace the heuristic)
- In production, a 0ms heuristic is always faster than a <5M param router
- The heuristic is fully transparent and debuggable; no black box
- It handles edge cases the router hasn't been trained on (new frameworks added to the rules without retraining)

The heuristic is also useful as an ensemble member in Feature 21 (evaluation suite) to benchmark against the learned router.

---

## Architecture / Design

### Rules Engine Structure

```
Input: code_snippet + optional filepath
         ↓
Stage 1: Import extraction (regex)
         ↓
Stage 2: Keyword scan (string matching)
         ↓
Stage 3: File pattern check (regex on filepath)
         ↓
Stage 4: Score aggregation per domain
         ↓
Stage 5: Priority resolution (Next.js > React, etc.)
         ↓
Output: (domain_label, confidence_score, evidence_list)
```

### Priority Ordering

Some domains subsume others. Priority (highest wins ties):

```
nextjs > react (Next.js is a superset framework of React)
testing > any  (test files are always testing domain regardless of framework)
graphql > general_ts
prisma > general_ts
zod > general_ts
react > general_ts
general_ts (catchall)
```

Exception: if confidence gap between nextjs and react is <0.1, prefer nextjs.

---

## Implementation Steps

### Step 1: Pattern Definitions

```python
# cola_coder/heuristic/patterns.py
import re
from dataclasses import dataclass, field
from typing import Pattern

@dataclass
class DomainPattern:
    name: str
    priority: int  # Higher = takes precedence
    import_packages: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)   # Regex for complex cases
    keyword_exact: list[str] = field(default_factory=list)     # Exact string match
    keyword_regex: list[str] = field(default_factory=list)     # Regex keyword patterns
    file_patterns: list[str] = field(default_factory=list)     # Filepath regex
    negative_signals: list[str] = field(default_factory=list)  # These reduce score
    import_weight: float = 3.0     # Imports are the strongest signal
    keyword_weight: float = 1.0
    file_weight: float = 5.0       # File pattern is very strong
    min_confidence: float = 0.15   # Below this, treat as noise


DOMAIN_PATTERNS: list[DomainPattern] = [
    DomainPattern(
        name="testing",
        priority=10,  # Highest priority — test files are always testing
        import_packages=["vitest", "jest", "@jest/globals", "@testing-library/react",
                         "@testing-library/user-event", "supertest", "playwright",
                         "cypress", "mocha", "chai", "sinon", "nock"],
        keyword_exact=["describe(", "it(", "test(", "expect(", "beforeEach(",
                       "afterEach(", "beforeAll(", "afterAll(", "vi.mock(",
                       "jest.mock(", "jest.fn(", "vi.fn(", "vi.spyOn(",
                       "jest.spyOn(", "assert.equal(", "assert.deepEqual("],
        keyword_regex=[r"expect\(.*\)\.to\w+\(", r"it\(['\"]"],
        file_patterns=[r"\.test\.[jt]sx?$", r"\.spec\.[jt]sx?$",
                       r"__tests__/.*\.[jt]sx?$", r"tests?/.*\.[jt]sx?$",
                       r"e2e/.*\.[jt]sx?$"],
    ),
    DomainPattern(
        name="nextjs",
        priority=8,
        import_packages=["next", "next/router", "next/navigation", "next/image",
                         "next/link", "next/head", "next/dynamic", "next/server",
                         "@next/font"],
        keyword_exact=["getServerSideProps", "getStaticProps", "getStaticPaths",
                       "NextPage", "NextApiHandler", "NextApiRequest",
                       "NextApiResponse", "useRouter", "usePathname",
                       "useSearchParams", "notFound()", "redirect("],
        keyword_regex=[r"export\s+default\s+function\s+\w+Page\s*\(",
                       r"app/.*route\.ts"],
        file_patterns=[r"pages/.*\.[jt]sx?$", r"app/.*\.(page|layout|route)\.[jt]sx?$",
                       r"next\.config\.[jt]s$"],
    ),
    DomainPattern(
        name="react",
        priority=6,
        import_packages=["react", "react-dom", "react-native", "@types/react",
                         "react-router-dom", "react-query", "@tanstack/react-query",
                         "zustand", "recoil", "jotai", "framer-motion",
                         "react-hook-form", "@radix-ui"],
        keyword_exact=["useState(", "useEffect(", "useRef(", "useContext(",
                       "useMemo(", "useCallback(", "useReducer(", "useLayoutEffect(",
                       "React.FC", "React.memo(", "React.createElement(",
                       "JSX.Element", "ReactNode", "ReactElement"],
        keyword_regex=[r"<[A-Z]\w+\s*/?>", r"return\s*\(\s*<"],
        file_patterns=[r"\.tsx$", r"\.jsx$", r"components/.*\.tsx$"],
        negative_signals=["getServerSideProps", "NextPage"],  # Reduce score if Next.js
    ),
    DomainPattern(
        name="graphql",
        priority=7,
        import_packages=["graphql", "@apollo/client", "apollo-server-express",
                         "apollo-server", "@apollo/server", "graphql-tag",
                         "urql", "@urql/core", "@graphql-codegen/cli",
                         "nexus", "type-graphql", "pothos"],
        keyword_exact=["gql`", "useQuery(", "useMutation(", "useSubscription(",
                       "GraphQLSchema", "GraphQLObjectType", "GraphQLString",
                       "ApolloClient", "ApolloProvider", "typeDefs", "resolvers:",
                       "fieldResolvers", "@ObjectType()", "@Field()"],
        keyword_regex=[r"gql`[\s\S]*?`", r"query\s+\w+\s*\{", r"mutation\s+\w+\s*\{"],
        file_patterns=[r"\.graphql$", r"\.gql$", r"schema\.ts$",
                       r"graphql/.*\.[jt]s$", r"resolvers/.*\.[jt]s$"],
    ),
    DomainPattern(
        name="prisma",
        priority=7,
        import_packages=["@prisma/client", "prisma"],
        keyword_exact=["PrismaClient", "prisma.findUnique(", "prisma.findMany(",
                       "prisma.create(", "prisma.update(", "prisma.delete(",
                       "prisma.upsert(", "prisma.$transaction(", "Prisma.",
                       "prisma.$connect(", "prisma.$disconnect("],
        keyword_regex=[r"prisma\.\w+\.\w+\(", r"@prisma/client"],
        file_patterns=[r"schema\.prisma$", r"prisma/.*\.ts$",
                       r"migrations/.*\.ts$", r"seed\.ts$"],
    ),
    DomainPattern(
        name="zod",
        priority=5,
        import_packages=["zod", "zod/lib", "@hookform/resolvers/zod"],
        keyword_exact=["z.object(", "z.string()", "z.number()", "z.boolean()",
                       "z.array(", "z.union(", "z.literal(", "z.enum(",
                       "z.infer<", ".parse(", ".safeParse(", ".parseAsync(",
                       "ZodSchema", "ZodError", "z.discriminatedUnion("],
        keyword_regex=[r"z\.\w+\(", r"ZodType\w*"],
        file_patterns=[r"schema\.ts$", r"validation\.ts$", r"validators/.*\.ts$",
                       r"schemas/.*\.ts$"],
    ),
    DomainPattern(
        name="general_ts",
        priority=0,
        import_packages=[],
        keyword_exact=["interface ", "namespace ", "declare module",
                       "declare global", "as const", "satisfies "],
        keyword_regex=[r"type\s+\w+\s*=", r"export\s+type\s+\w+"],
        file_patterns=[r"\.ts$", r"\.mts$", r"\.d\.ts$"],
        import_weight=0.1,
        keyword_weight=0.1,
        file_weight=0.1,
    ),
]
```

### Step 2: Heuristic Classifier

```python
# cola_coder/heuristic/classifier.py
import re
from typing import Optional
from .patterns import DOMAIN_PATTERNS, DomainPattern

@dataclass
class HeuristicResult:
    domain: str
    confidence: float
    scores: dict[str, float]
    evidence: list[str]  # What signals triggered the classification

    def __str__(self) -> str:
        return f"{self.domain} (conf={self.confidence:.3f})"


def extract_imports(code: str) -> list[str]:
    """Fast regex-based import extraction."""
    pattern = re.compile(
        r"""(?:import\s+(?:type\s+)?[\s\S]*?from\s+['"]([^'"]+)['"]"""
        r"""|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
        re.MULTILINE,
    )
    results = []
    for match in pattern.finditer(code):
        pkg = match.group(1) or match.group(2)
        if pkg:
            parts = pkg.split("/")
            if pkg.startswith("@"):
                results.append("/".join(parts[:2]))
            else:
                results.append(parts[0])
    return results


def classify(
    code: str,
    filepath: str = "",
    min_confidence: float = 0.1,
    return_all_scores: bool = False,
) -> Optional[HeuristicResult]:
    """
    Classify code snippet by domain using heuristic rules.
    Returns HeuristicResult or None if below min_confidence.
    """
    imports = set(extract_imports(code))
    raw_scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {p.name: [] for p in DOMAIN_PATTERNS}

    for pattern in DOMAIN_PATTERNS:
        score = 0.0

        # Import matching
        for pkg in imports:
            if pkg in pattern.import_packages:
                score += pattern.import_weight
                evidence[pattern.name].append(f"import:{pkg}")
            elif any(pkg.startswith(p) for p in pattern.import_packages):
                score += pattern.import_weight * 0.8
                evidence[pattern.name].append(f"import~:{pkg}")

        # Regex import patterns
        for pat_str in pattern.import_patterns:
            if re.search(pat_str, code):
                score += pattern.import_weight
                evidence[pattern.name].append(f"import_regex:{pat_str[:20]}")

        # Exact keyword matching
        for kw in pattern.keyword_exact:
            count = code.count(kw)
            if count > 0:
                score += pattern.keyword_weight * min(count, 5)
                evidence[pattern.name].append(f"kw:{kw[:20]}(x{count})")

        # Regex keyword patterns
        for pat_str in pattern.keyword_regex:
            matches = re.findall(pat_str, code, re.MULTILINE)
            if matches:
                score += pattern.keyword_weight * min(len(matches), 3)
                evidence[pattern.name].append(f"kw_re:{pat_str[:20]}")

        # File pattern matching
        if filepath:
            for fp_pat in pattern.file_patterns:
                if re.search(fp_pat, filepath):
                    score += pattern.file_weight
                    evidence[pattern.name].append(f"file:{fp_pat[:20]}")
                    break  # Only count file match once

        # Negative signals (reduce score)
        for neg in pattern.negative_signals:
            if neg in code:
                score *= 0.5

        raw_scores[pattern.name] = score

    # Apply priority tiebreaking
    best_domain, best_score = _resolve_priority(raw_scores)

    if best_score == 0:
        return HeuristicResult("general_ts", 0.5, raw_scores, []) if not min_confidence else None

    # Normalize confidence
    total = sum(raw_scores.values()) or 1.0
    confidence = best_score / total

    if confidence < min_confidence:
        return None

    return HeuristicResult(
        domain=best_domain,
        confidence=confidence,
        scores=raw_scores if return_all_scores else {},
        evidence=evidence[best_domain][:8],  # Top 8 signals
    )


def _resolve_priority(scores: dict[str, float]) -> tuple[str, float]:
    """
    Select winning domain, respecting priority ordering.
    If top scores are within 10% of each other, higher priority wins.
    """
    if not scores:
        return "general_ts", 0.0

    sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])
    top_domain, top_score = sorted_by_score[0]

    # Check if a higher-priority domain is close
    patterns_by_name = {p.name: p for p in DOMAIN_PATTERNS}
    top_priority = patterns_by_name[top_domain].priority

    for domain, score in sorted_by_score[1:4]:
        if score == 0:
            break
        domain_priority = patterns_by_name[domain].priority
        # If this domain has higher priority and score is within 20%
        if domain_priority > top_priority and score >= top_score * 0.8:
            top_domain, top_score = domain, score
            top_priority = domain_priority

    return top_domain, top_score
```

### Step 3: Batch Classification

```python
# cola_coder/heuristic/batch.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from .classifier import classify, HeuristicResult
from typing import Iterator

def classify_batch(
    snippets: list[tuple[str, str]],  # (code, filepath) pairs
    min_confidence: float = 0.1,
    workers: int = 4,
) -> list[Optional[HeuristicResult]]:
    """Classify multiple snippets in parallel."""
    results = [None] * len(snippets)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(classify, code, fp, min_confidence): i
            for i, (code, fp) in enumerate(snippets)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results
```

### Step 4: CLI Command

```python
@app.command()
def detect_domain(
    code_or_file: str = typer.Argument(..., help="Code snippet or file path"),
    show_evidence: bool = typer.Option(False, "--evidence", "-e"),
    show_all: bool = typer.Option(False, "--all-scores"),
    min_confidence: float = typer.Option(0.1, "--min-conf"),
):
    """Classify a code snippet or file by domain using heuristics."""
    from cola_coder.heuristic.classifier import classify
    import pathlib

    if pathlib.Path(code_or_file).exists():
        filepath = code_or_file
        code = pathlib.Path(code_or_file).read_text(encoding="utf-8", errors="ignore")
    else:
        filepath = ""
        code = code_or_file

    result = classify(code, filepath, min_confidence, return_all_scores=show_all)
    if result is None:
        console.print("[yellow]No confident domain detected[/yellow]")
        return

    console.print(f"[bold]Domain:[/bold] [cyan]{result.domain}[/cyan]  "
                  f"[bold]Confidence:[/bold] {result.confidence:.3f}")

    if show_evidence:
        console.print("[bold]Evidence:[/bold]")
        for ev in result.evidence:
            console.print(f"  [dim]{ev}[/dim]")

    if show_all and result.scores:
        from rich.table import Table
        table = Table(show_header=False, box=None)
        for domain, score in sorted(result.scores.items(), key=lambda x: -x[1]):
            bar = "█" * min(int(score * 2), 20)
            table.add_row(f"  {domain:14s}", f"[dim]{bar:20s}[/dim]", f"{score:.2f}")
        console.print(table)
```

---

## Key Files to Modify

- `cola_coder/heuristic/__init__.py` — new package
- `cola_coder/heuristic/patterns.py` — DomainPattern definitions
- `cola_coder/heuristic/classifier.py` — classify() function
- `cola_coder/heuristic/batch.py` — parallel batch classification
- `cola_coder/cli.py` — `detect-domain` command
- `tests/test_heuristic.py` — comprehensive rule tests

---

## Testing Strategy

```python
# tests/test_heuristic.py — comprehensive rule coverage

CASES = [
    ("import { useState } from 'react';\nconst C = () => <div/>;", "", "react"),
    ("import { getServerSideProps } from 'next';\n", "pages/index.tsx", "nextjs"),
    ("import { gql } from 'graphql-tag';\nconst Q = gql`query { id }`;", "", "graphql"),
    ("import { PrismaClient } from '@prisma/client';", "", "prisma"),
    ("import { z } from 'zod';\nconst s = z.object({ x: z.string() });", "", "zod"),
    ("describe('test', () => { it('passes', () => { expect(1).toBe(1); }); });", "", "testing"),
    ("const x: Record<string, number> = {};", "", "general_ts"),
]

@pytest.mark.parametrize("code,filepath,expected", CASES)
def test_heuristic_classification(code, filepath, expected):
    result = classify(code, filepath)
    assert result is not None
    assert result.domain == expected

def test_testing_overrides_react():
    code = ("import { render } from '@testing-library/react';\n"
            "import React from 'react';\n"
            "describe('suite', () => { it('works', () => {}); });")
    result = classify(code)
    assert result.domain == "testing"

def test_nextjs_overrides_react():
    code = "import React from 'react';\nexport const getServerSideProps = async () => ({props:{}});"
    result = classify(code)
    assert result.domain == "nextjs"

def test_empty_code():
    result = classify("", min_confidence=0.5)
    assert result is None

def test_evidence_populated():
    result = classify("import { z } from 'zod';\nz.string()", return_all_scores=True)
    assert len(result.evidence) > 0
    assert any("zod" in e for e in result.evidence)
```

---

## Performance Considerations

- **Speed:** Pure regex + string counting with no ML. Typical classification: <1ms for a 256-token snippet.
- **Batch throughput:** At 4 parallel workers, can classify ~10,000 snippets/second for Feature 17 dataset generation.
- **Memory:** No model weights — negligible memory usage.
- **Compilation:** Pre-compile all regex patterns at module load time (not inside the classify function).

```python
# Pre-compile at module level for performance
_COMPILED_PATTERNS = {
    p.name: {
        "keyword_regex": [re.compile(r) for r in p.keyword_regex],
        "import_patterns": [re.compile(r) for r in p.import_patterns],
        "file_patterns": [re.compile(r) for r in p.file_patterns],
    }
    for p in DOMAIN_PATTERNS
}
```

---

## Dependencies

- Python standard library only (`re`, `dataclasses`, `concurrent.futures`)
- No ML frameworks, no external packages
- Feature 17 (training data generator) — uses this module for labeling

---

## Estimated Complexity

| Task                          | Effort  |
|-------------------------------|---------|
| Pattern definitions           | 2h      |
| Classifier logic              | 2h      |
| Priority resolution           | 1h      |
| Batch classifier              | 0.5h    |
| CLI command                   | 0.5h    |
| Tests (comprehensive)         | 2h      |
| **Total**                     | **~8h** |

Overall complexity: **Low** (pure string processing, no ML, well-defined requirements)

---

## 2026 Best Practices

- **Extensible rule files:** Consider moving DOMAIN_PATTERNS to a YAML/JSON config file so domain rules can be updated without code changes. Load at startup.
- **Confidence calibration:** The raw score ratios are not calibrated probabilities. If using as a routing signal (not just labels), normalize outputs using isotonic regression on a validation set.
- **Negative example mining:** After training Feature 16's learned router, compare heuristic vs learned router outputs. Cases where they disagree are valuable hard examples — add them to the test suite (Feature 21).
- **Version the rule set:** Tag each release of DOMAIN_PATTERNS with a version string. Include version in generated JSONL records so training data can be traced back to rule version.
- **Framework evolution:** TypeScript ecosystem changes fast. Schedule a quarterly review of the import package lists to add newly popular frameworks (e.g., new routing libraries, new ORMs).
