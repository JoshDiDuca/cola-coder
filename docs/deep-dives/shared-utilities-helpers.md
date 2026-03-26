# Shared Utilities: DRY Infrastructure for the Scoring Pipeline

Every time two modules implement the same function, you get a maintenance debt
that compounds with every change. Module A hashes code with MD5. Module B hashes
code with MD5, but uses a different encoding. Module A gets updated to handle
Unicode correctly. Module B does not. Now your deduplication cache has a subtle
encoding mismatch that produces different hashes for the same code, and your
training data has duplicates that should have been caught.

This is not hypothetical. Before the shared utilities extraction, the scoring
pipeline had:

- Two implementations of code hashing (one in TscScorer, one in the RL reward)
- Three copies of "is this TypeScript?" logic (scorer, reward, data prep)
- Two different score-to-quality mappings that disagreed on thresholds
- Zero audit logging for most scoring operations

The shared utilities in `data/scorers/` exist to fix these violations. Each
module is small (30-70 lines), has one job, and is used by multiple consumers.
This document explains every utility, when to use each one, and the principles
behind the extraction.

---

## Table of Contents

1. [Why Shared Utilities Matter](#1-why-shared-utilities-matter)
2. [language_detect.py: Is This TypeScript?](#2-language_detectpy-is-this-typescript)
3. [utils.py: code_hash() and ScoreMapper](#3-utilspy-code_hash-and-scoremapper)
4. [audit.py: ScoringAuditLogger](#4-auditpy-scoringauditlogger)
5. [tsconfig_factory.py: Hardened tsconfig Generation](#5-tsconfig_factorypy-hardened-tsconfig-generation)
6. [Protocol-Based Design](#6-protocol-based-design)
7. [The DRY Rule](#7-the-dry-rule)
8. [Adding New Shared Utilities](#8-adding-new-shared-utilities)
9. [Testing Shared Utilities](#9-testing-shared-utilities)
10. [Reference: Module Dependency Map](#10-reference-module-dependency-map)

---

## 1. Why Shared Utilities Matter

The scoring pipeline has five scorers, a composite scorer, an RL reward function,
and a batch evaluation system. Each one needs to:

- Detect whether code is TypeScript before trying to type-check it
- Hash code for caching and deduplication
- Map raw metric counts to normalized 0.0-1.0 scores
- Log what it did for forensic analysis

Without shared utilities, each module implements these functions inline. With six
consumers and four shared operations, that is 24 potential implementations -- one
per consumer per operation. In practice, you end up with 8-10 slightly different
versions because some consumers share code by copy-paste and others diverge.

The cost is not the duplicated lines. It is the **inconsistency**. When TscScorer
and TypeCheckReward use different hashing, their caches are separate even for
identical code. When the data prep pipeline and the RL pipeline use different
TypeScript detection, some files get scored in one path but skipped in the other.
The training data and the reward signal disagree on what counts as TypeScript.

**TS analogy:** This is like having a `utils/` directory in a TypeScript project
with shared helpers like `isEmail()`, `slugify()`, `hashString()`. Without it,
every module has its own `isEmail()` regex, and some of them allow `user@localhost`
while others do not. Shared utilities are the `@/utils` that every module imports.

```
Before:                              After:

TscScorer    TypeCheckReward         TscScorer    TypeCheckReward
  |               |                    |               |
  +-- md5()       +-- md5()            +------+--------+
  +-- is_ts()     +-- is_ts()                 |
  +-- map_score() +-- map_score()      language_detect.py
                                       utils.py (code_hash, ScoreMapper)
                                       audit.py (ScoringAuditLogger)
                                       tsconfig_factory.py
```

---

## 2. language_detect.py: Is This TypeScript?

### The Problem

When scoring code from HuggingFace datasets, you get files with varying metadata.
Some have a `language` field (`"typescript"`). Some have a `file_path` field
(`"src/index.tsx"`). Some have neither. You need a single, canonical answer to
"is this TypeScript?" that works in all cases.

### The Module

`language_detect.py` exports five functions and a set of constants:

```python
# Constants
TYPESCRIPT_EXTENSIONS = frozenset({"ts", "tsx", "mts", "cts"})
JAVASCRIPT_EXTENSIONS = frozenset({"js", "jsx", "mjs", "cjs"})
JS_TS_EXTENSIONS = TYPESCRIPT_EXTENSIONS | JAVASCRIPT_EXTENSIONS

TYPESCRIPT_LANGUAGES = frozenset({"typescript", "ts", "tsx"})
JAVASCRIPT_LANGUAGES = frozenset({"javascript", "js", "jsx"})
JS_TS_LANGUAGES = TYPESCRIPT_LANGUAGES | JAVASCRIPT_LANGUAGES

# Functions
def get_language(metadata) -> str          # Extract normalized language string
def get_file_extension(metadata) -> str    # Extract extension without dot
def is_typescript(code, metadata) -> bool  # TypeScript only (not JavaScript)
def is_js_ts(code, metadata) -> bool       # JavaScript or TypeScript
def detect_extension(metadata) -> str      # Best-guess file extension with dot
```

### Detection Priority

`is_typescript()` checks three sources in priority order:

```
1. metadata["language"] in {"typescript", "ts", "tsx"}
   → Explicit language label. Most reliable.

2. file_path extension in {"ts", "tsx", "mts", "cts"}
   → File extension from metadata. Second most reliable.

3. Heuristic: 2+ TypeScript-specific patterns in the code
   → Fallback when metadata is missing.
```

The heuristic checks for TypeScript-specific syntax:

```python
ts_indicators = [
    ": string", ": number", ": boolean",  # Type annotations
    "interface ",                           # Interface declarations
    ": void",                              # Return type annotations
    "as const",                            # Const assertions
    "<T>",                                 # Generic type parameters
    "readonly ",                           # Readonly modifier
    "enum ",                               # Enum declarations
]
return sum(1 for ind in ts_indicators if ind in code) >= 2
```

Two or more indicators means TypeScript. One indicator is ambiguous -- `": string"`
could appear in a JSON file or a Python docstring. Two indicators are enough to be
confident without being overly strict.

**TS analogy:** This is like writing a `isReactComponent()` function that checks
for JSX syntax, `import React`, and hook calls. Each individual signal has false
positives, but two or more together are a strong signal.

### Why frozenset?

The constant sets use `frozenset` instead of `set`:

```python
TYPESCRIPT_EXTENSIONS = frozenset({"ts", "tsx", "mts", "cts"})
```

`frozenset` is immutable. No consumer can accidentally `add()` or `remove()`
elements from the shared constant. This is defensive programming -- the constants
are used across multiple modules, and a mutation in one module would silently affect
all others.

**TS analogy:** `frozenset` is `as const` for sets:

```typescript
const TYPESCRIPT_EXTENSIONS = ["ts", "tsx", "mts", "cts"] as const;
// Type: readonly ["ts", "tsx", "mts", "cts"]
// Cannot push, pop, or modify
```

### Consumers

| Consumer | Uses |
|----------|------|
| TscScorer | `is_typescript()` -- skip non-TS files |
| EslintScorer | `is_js_ts()` -- ESLint handles both JS and TS |
| HeuristicScorer | `is_js_ts()` -- heuristic signals apply to both |
| Data prep pipeline | `detect_extension()` -- choose file extension for temp files |
| TypeCheckReward | Does not use (only receives TS code from GRPO) |

---

## 3. utils.py: code_hash() and ScoreMapper

### code_hash(): Content-Addressable Deduplication

```python
def code_hash(code: str) -> str:
    """MD5 hash of code for dedup/caching. Used across all scorers."""
    return hashlib.md5(code.encode("utf-8")).hexdigest()
```

Eleven lines including the docstring. This is the smallest utility in the project,
but it fixes a real bug: inconsistent encoding.

Before extraction, one scorer used `code.encode()` (platform default, which is
`utf-8` on most systems but `cp1252` on some Windows installs) and another used
`code.encode("utf-8")`. On a Windows machine with a non-UTF-8 locale, the same
code would produce different hashes depending on which scorer processed it first.

The shared function enforces UTF-8 everywhere. One encoding, one hash, one truth.

### Why MD5?

MD5 is not cryptographically secure. An attacker could craft two different code
strings with the same MD5 hash. But we are not using it for security -- we are
using it for caching and deduplication. The attacker would need to craft
two valid TypeScript files that (a) have the same MD5 hash and (b) have different
tsc error outputs. That is not a realistic attack vector.

MD5 is fast. SHA-256 would be ~2x slower for no benefit. For hashing 50,000 code
files during batch scoring, the speed difference adds up.

**TS analogy:** This is like using a `Map` keyed by a fast hash instead of the
full string. You accept the theoretical collision risk for practical performance:

```typescript
function codeHash(code: string): string {
  return crypto.createHash('md5').update(code, 'utf-8').digest('hex');
}

const cache = new Map<string, TscError[]>();
const hash = codeHash(code);
if (cache.has(hash)) return cache.get(hash)!;
```

### ScoreMapper: Threshold-Based Score Conversion

```python
class ScoreMapper:
    """Map integer counts (errors, warnings) to 0.0-1.0 quality scores."""

    def __init__(self, thresholds: list[tuple[int, float]], fallback: float = 0.1):
        self._thresholds = thresholds
        self._fallback = fallback

    def map(self, count: int) -> float:
        for threshold, score in self._thresholds:
            if count <= threshold:
                return score
        return self._fallback
```

Every scorer needs to convert a raw count (errors, warnings, issues) into a
normalized 0.0-1.0 score. Without `ScoreMapper`, each scorer has a chain of
`if/elif/else` statements with hardcoded thresholds:

```python
# Without ScoreMapper (bad):
if num_errors == 0:
    score = 1.0
elif num_errors <= 2:
    score = 0.8
elif num_errors <= 5:
    score = 0.6
elif num_errors <= 10:
    score = 0.3
else:
    score = 0.1
```

With `ScoreMapper`, the thresholds are data, not control flow:

```python
# With ScoreMapper (good):
_TSC_SCORE_MAP = ScoreMapper([
    (0, 1.0),     # No errors = perfect
    (1, 0.8),     # 1 error = good
    (3, 0.6),     # 2-3 errors = decent
    (5, 0.4),     # 4-5 errors = average
    (10, 0.2),    # 6-10 errors = poor
])

score = _TSC_SCORE_MAP.map(num_errors)
```

**TS analogy:** This is a lookup table pattern, common in TypeScript:

```typescript
const TSC_SCORE_MAP: [number, number][] = [
  [0, 1.0],
  [1, 0.8],
  [3, 0.6],
  [5, 0.4],
  [10, 0.2],
];

function mapScore(count: number, fallback = 0.1): number {
  for (const [threshold, score] of TSC_SCORE_MAP) {
    if (count <= threshold) return score;
  }
  return fallback;
}
```

### Why a Class Instead of a Function?

`ScoreMapper` is a class instead of a bare function because it captures the
threshold configuration. Each scorer creates a `ScoreMapper` instance with its
own thresholds:

```python
# TscScorer: strict thresholds (type errors matter more)
_TSC_SCORE_MAP = ScoreMapper([(0, 1.0), (1, 0.8), (3, 0.6), (5, 0.4), (10, 0.2)])

# EslintScorer: lenient thresholds (lint warnings are less critical)
_ESLINT_SCORE_MAP = ScoreMapper([(0, 1.0), (3, 0.8), (8, 0.6), (15, 0.4), (30, 0.2)])
```

The same mapping logic, different thresholds. If `ScoreMapper` were a function,
you would pass thresholds on every call. As a class, you configure once and call
`map()` with just the count.

---

## 4. audit.py: ScoringAuditLogger

### Why Audit Logging Exists

When your model generates weird code, you need to trace backwards: what data did
it train on? What scores did those files get? Did the scorer even run, or did it
skip the file? Without an audit trail, debugging data quality issues is guesswork.

The `ScoringAuditLogger` records every scoring operation to a JSONL file:

```python
class ScoringAuditLogger:
    """Append-only JSONL audit log for all scoring operations."""

    def __init__(self, log_path: str | Path = "logs/scoring_audit.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_security_event(self, event: str, scorer: str = "", file_hash: str = ""):
        entry = AuditEntry(scorer=scorer, file_hash=file_hash, security_events=[event])
        self.log(entry)
```

### The JSONL Schema

Each line in the audit log is a JSON object:

```json
{
  "timestamp": "2026-03-25T14:32:07.123456+00:00",
  "scorer": "tsc",
  "file_hash": "a1b2c3d4e5f6...",
  "security_mode": "native",
  "command": ["tsc", "--project", ".", "--pretty", "false"],
  "exit_code": 1,
  "duration_ms": 47.3,
  "security_events": []
}
```

| Field | Type | Purpose |
|-------|------|---------|
| `timestamp` | ISO 8601 | When the operation occurred (UTC) |
| `scorer` | string | Which scorer ran ("tsc", "eslint", "heuristic") |
| `file_hash` | string | MD5 hash of the scored code (links to cache) |
| `security_mode` | string | "native" or "docker" |
| `command` | string[] | First 5 elements of the subprocess command |
| `exit_code` | int | tsc/eslint return code (0 = success, 1 = errors found) |
| `duration_ms` | float | Execution time in milliseconds |
| `security_events` | string[] | Any security-relevant events (credential found, timeout, etc.) |

### Why JSONL?

JSONL (JSON Lines) is one JSON object per line. Each line is independent and
self-contained. This format has three properties that matter for audit logging:

1. **Append-only writes are atomic.** On most filesystems, appending less than
   4KB to a file is atomic -- either the entire line is written or none of it.
   No locks needed. Multiple processes can append simultaneously without
   corruption.

2. **Easy to query.** `grep "tsc" scoring_audit.jsonl | python -m json.tool`
   gives you every tsc operation. `jq` works too. No database needed.

3. **Streaming-friendly.** You can tail the file in real time:
   `tail -f logs/scoring_audit.jsonl | jq .scorer` shows scorers as they run.

**TS analogy:** JSONL is like NDJSON (Newline Delimited JSON). If you have used
`ndjson-parse` or streamed JSON logs in Node.js, this is the same format.

### Security Events

The `log_security_event()` method records security-relevant incidents:

```python
# Called by credential scanner when it finds a secret
audit_logger.log_security_event(
    event="credential_detected:aws_key",
    scorer="credential_scanner",
    file_hash="a1b2c3d4...",
)
```

This creates a record that a specific file contained credentials. After a training
run, you can query the audit log for all security events to verify that no
secrets leaked into your training data.

### Thread Safety

The logger uses file `append` mode, which is thread-safe for short writes on most
operating systems. The `AuditEntry` dataclass is typically under 500 bytes when
serialized, well under the 4KB atomicity threshold.

For concurrent batch scoring with multiple worker processes, each process gets its
own file handle via `open()` in `log()`. Since each write is a single `f.write()`
call with one line of JSON, interleaving is impossible -- you get complete lines
or nothing.

---

## 5. tsconfig_factory.py: Hardened tsconfig Generation

The tsconfig factory is covered in detail in the
[TscRunner SOLID Architecture](tscrunner-solid-architecture.md) deep dive
(Section 4). Here we cover the API and usage patterns.

### Two Functions

```python
def create_hardened_tsconfig(
    strict: bool = True,
    include_files: list[str] | None = None,
) -> dict:
    """Returns a dict suitable for json.dumps() -> tsconfig.json."""

def write_hardened_tsconfig(
    directory: str | Path,
    strict: bool = True,
    include_files: list[str] | None = None,
) -> Path:
    """Writes hardened tsconfig.json to disk. Returns the path."""
```

`create_hardened_tsconfig()` is pure -- it returns a dict with no side effects.
`write_hardened_tsconfig()` is a convenience wrapper that creates the dict and
writes it to a file.

### Why a Factory Function Instead of a Constant?

The tsconfig varies in two ways:
- `strict` can be `True` or `False` (for scoring lenient code)
- `include_files` changes for each invocation (single file vs batch)

A constant dict would need to be deep-copied and mutated. A factory function
creates a fresh dict each time, with no mutation risk:

```python
# Good: factory function, no shared mutable state
config1 = create_hardened_tsconfig(include_files=["a.ts"])
config2 = create_hardened_tsconfig(include_files=["b.ts", "c.ts"])
# config1 and config2 are independent objects

# Bad: constant dict, shared mutable state
TSCONFIG = {"compilerOptions": {...}, "include": []}
config1 = copy.deepcopy(TSCONFIG)
config1["include"] = ["a.ts"]
# Hope nobody forgets deepcopy...
```

**TS analogy:** This is the factory pattern vs shared object pattern:

```typescript
// Good: factory
function createTsconfig(files: string[]): TsConfig {
  return { compilerOptions: { strict: true, ... }, include: files };
}

// Bad: shared mutable
const TSCONFIG = { compilerOptions: { strict: true, ... }, include: [] };
// Every consumer mutates the same object
```

### Security Invariants

The factory guarantees these fields are always present:

- `plugins: []` -- always empty, never omitted
- `types: []` -- always empty, never omitted
- `typeRoots: []` -- always empty, never omitted
- `noEmit: true` -- always set
- No `paths` key -- always absent
- No `baseUrl` key -- always absent

These invariants are tested. If someone modifies the factory to add a non-empty
`plugins` array, the test suite catches it immediately.

---

## 6. Protocol-Based Design

### ScorerProtocol

The scoring pipeline uses Python's `Protocol` for structural typing -- the same
concept as TypeScript interfaces:

```python
@runtime_checkable
class ScorerProtocol(Protocol):
    """Interface that all scorers must implement."""
    name: str
    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult: ...
    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]: ...
    @staticmethod
    def is_available() -> bool: ...
```

No class needs to explicitly inherit from `ScorerProtocol`. If a class has a
`name` attribute and the right method signatures, it satisfies the protocol:

```python
class MyScorer:  # No "implements ScorerProtocol" needed
    name = "my_scorer"

    def score(self, code, metadata=None):
        return ScorerResult(score=0.5, scorer_name=self.name)

    def score_batch(self, items):
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available():
        return True

# This works because MyScorer has the right shape:
assert isinstance(MyScorer(), ScorerProtocol)  # True at runtime
```

**TS analogy:** This is exactly how TypeScript interfaces work:

```typescript
interface ScorerProtocol {
  name: string;
  score(code: string, metadata?: Record<string, unknown>): ScorerResult;
  scoreBatch(items: [string, Record<string, unknown> | null][]): ScorerResult[];
  isAvailable(): boolean;
}

// No "implements ScorerProtocol" needed in TS either -- structural typing
const myScorer = {
  name: "my_scorer",
  score: (code) => ({ score: 0.5, scorerName: "my_scorer" }),
  scoreBatch: (items) => items.map(([code]) => myScorer.score(code)),
  isAvailable: () => true,
};
```

### MalwareScannerProtocol

The security module uses the same pattern for malware scanners:

```python
@runtime_checkable
class MalwareScannerProtocol(Protocol):
    name: str
    def scan_file(self, path: Path) -> MalwareScanResult: ...
    def scan_directory(self, path: Path) -> MalwareScanResult: ...
    def is_available(self) -> bool: ...
```

Three different scanners implement this protocol: YARA, Windows Defender, and
ClamAV. The `CompositeMalwareScanner` takes a list of scanners and runs them all:

```python
class CompositeMalwareScanner:
    def __init__(self, scanners: list[MalwareScannerProtocol]):
        self._scanners = [s for s in scanners if s.is_available()]

    def scan_file(self, path: Path) -> MalwareScanResult:
        result = MalwareScanResult(is_clean=True, threats=[], ...)
        for scanner in self._scanners:
            result = result.merge(scanner.scan_file(path))
        return result
```

### ScorerResult and CompositeResult

The data classes that flow through the protocol:

```python
@dataclass
class ScorerResult:
    score: float            # 0.0 - 1.0 normalized
    scorer_name: str        # "tsc", "eslint", "stars"
    details: dict[str, object] = field(default_factory=dict)

@dataclass
class CompositeResult:
    overall: float                       # Weighted average 0.0-1.0
    per_scorer: dict[str, ScorerResult]  # Individual results
    weight: float                        # Training weight (tier mapped)
```

`ScorerResult` is the return type of every scorer's `score()` method.
`CompositeResult` is the return type of `CompositeScorer.score()`, which
aggregates multiple `ScorerResult` objects into a weighted average.

The `weight` field in `CompositeResult` is the training weight -- how much
this sample should influence gradient updates:

```python
# CompositeScorer._score_to_weight()
if score >= 0.8: return 2.0    # excellent -> upweight
elif score >= 0.6: return 1.5  # good -> slight upweight
elif score >= 0.4: return 1.0  # average -> neutral
elif score >= 0.2: return 0.3  # poor -> downweight
else: return 0.0               # reject -> exclude
```

High-quality code gets 2x training weight. Low-quality code gets 0.3x. Rejected
code gets 0x (excluded entirely). This is the core of quality-weighted training:
the model sees more gradient signal from good code than from bad code.

---

## 7. The DRY Rule

### The Principle

DRY -- Don't Repeat Yourself -- is one of the oldest principles in software
engineering. Every piece of knowledge should have a single, unambiguous,
authoritative representation within a system.

In this codebase, DRY is enforced at the rule level. The project's coding
standards specify: if a function is used by two or more modules, extract it into
a shared utility.

### What Counts as Repetition

Not all similar code is repetition. The test for whether to extract:

1. **Same logic, same purpose** -- extract. Two modules both hash code for
   caching. Same logic, same purpose. Extract to `code_hash()`.

2. **Same logic, different purpose** -- do not extract. Two modules both iterate
   over a list and filter items. Similar logic, but the filter criteria are
   domain-specific. Keep them separate.

3. **Same constants** -- extract. Three modules all define
   `TYPESCRIPT_EXTENSIONS = {"ts", "tsx"}`. Extract to `language_detect.py`.

4. **Same pattern, different parameters** -- extract with parameters. Five scorers
   all have `if count <= N: return score` chains. Extract to `ScoreMapper(thresholds)`.

### The Extractions We Made

| Before (duplicated) | After (shared) | Modules affected |
|---------------------|---------------|-----------------|
| Inline `hashlib.md5(code.encode())` | `code_hash(code)` in `utils.py` | TscScorer, TscRunner, EslintScorer |
| Inline `if "typescript" in lang` checks | `is_typescript(code, metadata)` in `language_detect.py` | TscScorer, EslintScorer, HeuristicScorer, data prep |
| Inline `if count <= N: score = X` chains | `ScoreMapper(thresholds).map(count)` in `utils.py` | TscScorer, EslintScorer |
| Inline tsconfig dict literals | `create_hardened_tsconfig()` in `tsconfig_factory.py` | TscRunner, TscScorer (legacy) |
| No audit logging | `ScoringAuditLogger` in `audit.py` | SandboxedRunner, credential scanner |

### When NOT to Extract

Sometimes two modules have similar code that should stay separate:

- **TscScorer.score()** and **TypeCheckReward.score()** both compute scores from
  error lists, but with different scales (0.0-1.0 vs -0.5-1.0) and different
  thresholds. Extracting a shared `compute_score(errors)` function would require
  so many parameters that it would be less readable than the inline version.

- **Error parsing** in TscRunner and the diagnostic display in test output both
  parse tsc error strings, but the test display needs ANSI colors and pretty
  formatting. Sharing the parser would couple test display to production code.

The rule of thumb: extract when sharing reduces total complexity. Do not extract
when sharing increases coupling or reduces clarity.

---

## 8. Adding New Shared Utilities

### When to Create a New Utility

Create a new shared utility when:

1. **Two or more modules** need the same function, and
2. **The function is domain-agnostic** (not specific to one scorer), and
3. **The function is stable** (unlikely to change per-consumer)

### The Checklist

Before adding a shared utility:

- [ ] Is this function used by 2+ modules? (If only 1, keep it local)
- [ ] Is the interface stable? (If consumers need different signatures, do not force a shared one)
- [ ] Is there an existing utility that does 80% of what you need? (Extend, do not duplicate)
- [ ] Does the utility have zero side effects? (Pure functions are the best shared utilities)
- [ ] Can you write a unit test in under 10 lines? (If testing is complex, the utility is doing too much)

### Where to Put It

```
data/scorers/
  utils.py          -- Pure functions: hashing, mapping, conversion
  language_detect.py -- Language detection functions and constants
  audit.py           -- Audit logging (the one exception to "pure functions")
  tsconfig_factory.py -- tsconfig generation (pure function returning a dict)
  protocol.py        -- Protocol classes and shared data types
```

The naming convention: if the utility is about **what** (detection, mapping,
hashing), it goes in `utils.py` or its own module. If it is about **how** (the
interface contract), it goes in `protocol.py`.

### Example: Adding a Complexity Scorer Utility

Suppose you are adding a `CyclomaticComplexityScorer` and an
`ImportComplexityScorer`, and both need to count AST nodes. Extract:

```python
# ast_utils.py (new file in data/scorers/)
"""Shared AST analysis utilities for code scorers."""

import ast
from collections import Counter

def count_node_types(code: str) -> Counter[str]:
    """Count AST node types in Python code."""
    try:
        tree = ast.parse(code)
        return Counter(type(node).__name__ for node in ast.walk(tree))
    except SyntaxError:
        return Counter()

def count_branches(code: str) -> int:
    """Count branching statements (if, for, while, try)."""
    counts = count_node_types(code)
    return counts["If"] + counts["For"] + counts["While"] + counts["Try"]
```

Then both scorers import from `ast_utils`:

```python
from cola_coder.data.scorers.ast_utils import count_branches

class CyclomaticComplexityScorer:
    def score(self, code, metadata=None):
        branches = count_branches(code)
        score = self._mapper.map(branches)
        return ScorerResult(score=score, scorer_name="complexity")
```

---

## 9. Testing Shared Utilities

### Pure Functions Are Easy to Test

The best thing about shared utilities is testability. Pure functions with no side
effects are trivial to test:

```python
# test_language_detect.py
def test_is_typescript_by_language():
    assert is_typescript("", {"language": "typescript"}) is True
    assert is_typescript("", {"language": "javascript"}) is False

def test_is_typescript_by_extension():
    assert is_typescript("", {"file_path": "app.tsx"}) is True
    assert is_typescript("", {"file_path": "app.js"}) is False

def test_is_typescript_by_heuristic():
    code = "const x: number = 42;\ninterface Foo { bar: string; }"
    assert is_typescript(code) is True

def test_is_typescript_heuristic_needs_two_indicators():
    code = "const x: number = 42;"  # Only one indicator
    assert is_typescript(code) is False

# test_utils.py
def test_code_hash_consistent():
    assert code_hash("hello") == code_hash("hello")

def test_code_hash_different():
    assert code_hash("hello") != code_hash("world")

def test_score_mapper():
    mapper = ScoreMapper([(0, 1.0), (5, 0.5), (10, 0.2)])
    assert mapper.map(0) == 1.0
    assert mapper.map(3) == 0.5
    assert mapper.map(7) == 0.2
    assert mapper.map(100) == 0.1  # fallback
```

### Audit Logger Testing

The audit logger has side effects (file writes), so it needs a temp directory:

```python
def test_audit_logger_writes_jsonl(tmp_path):
    logger = ScoringAuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log(AuditEntry(scorer="tsc", exit_code=0, duration_ms=42.0))
    logger.log(AuditEntry(scorer="eslint", exit_code=1, duration_ms=87.0))

    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["scorer"] == "tsc"
    assert entry1["exit_code"] == 0

    entry2 = json.loads(lines[1])
    assert entry2["scorer"] == "eslint"

def test_audit_logger_security_event(tmp_path):
    logger = ScoringAuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_security_event("credential_detected:aws_key", scorer="scan")

    line = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert "credential_detected:aws_key" in line["security_events"]
```

### Protocol Conformance Testing

Verify that all scorers implement `ScorerProtocol`:

```python
def test_tsc_scorer_satisfies_protocol():
    assert isinstance(TscScorer(), ScorerProtocol)

def test_eslint_scorer_satisfies_protocol():
    assert isinstance(EslintScorer(), ScorerProtocol)

def test_heuristic_scorer_satisfies_protocol():
    assert isinstance(HeuristicScorer(), ScorerProtocol)
```

The `@runtime_checkable` decorator on `ScorerProtocol` makes `isinstance()`
checks work at runtime. Without it, you would need to inspect the class manually.

---

## 10. Reference: Module Dependency Map

Here is how the shared utilities connect to the rest of the scoring pipeline:

```
                    CompositeScorer
                    /      |      \
                   /       |       \
           TscScorer  EslintScorer  HeuristicScorer
               |          |             |
               v          v             v
           TscRunner   EslintRunner   (inline logic)
               |          |
               v          v
          SandboxedRunner
               |
               v
          subprocess.run()

  Shared utilities used by the above:
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  protocol.py ──── ScorerProtocol, ScorerResult,      │
  │                   CompositeResult, CompositeScorer    │
  │                                                      │
  │  language_detect.py ── is_typescript(), is_js_ts(),   │
  │                        detect_extension(), constants  │
  │                                                      │
  │  utils.py ──────── code_hash(), ScoreMapper           │
  │                                                      │
  │  tsconfig_factory.py ── create_hardened_tsconfig(),   │
  │                         write_hardened_tsconfig()     │
  │                                                      │
  │  audit.py ─────── ScoringAuditLogger, AuditEntry     │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### Import Graph

```
TscScorer
  <- language_detect.is_typescript
  <- protocol.ScorerResult
  <- utils.ScoreMapper
  <- TscRunner (from reasoning/rewards/)

TscRunner
  <- sandbox.SandboxedRunner
  <- tsconfig_factory.create_hardened_tsconfig

TypeCheckReward
  <- TscRunner (from reasoning/rewards/)

SandboxedRunner
  <- audit.ScoringAuditLogger (optional, via from_config())
  <- audit.AuditEntry

CompositeScorer (in protocol.py)
  <- protocol.ScorerProtocol
  <- protocol.ScorerResult
  <- protocol.CompositeResult
```

### File Sizes

| Module | Lines | Functions/Classes | Purpose |
|--------|-------|-------------------|---------|
| `protocol.py` | 120 | 4 (ScorerResult, ScorerProtocol, CompositeResult, CompositeScorer) | Interface contracts + composition |
| `language_detect.py` | 71 | 5 functions + 6 constants | Language detection |
| `tsconfig_factory.py` | 62 | 2 functions | Hardened tsconfig generation |
| `audit.py` | 60 | 2 (AuditEntry, ScoringAuditLogger) | JSONL audit trail |
| `utils.py` | 34 | 1 function + 1 class | Hashing + score mapping |
| **Total** | **347** | **15** | **5 modules** |

347 lines of shared infrastructure supporting a pipeline that processes millions
of files. Each module is small enough to read in one sitting, stable enough to
rarely change, and tested enough to trust unconditionally.

That is the point of shared utilities. You write them once, test them thoroughly,
and then forget they exist -- because they just work, everywhere, every time.
