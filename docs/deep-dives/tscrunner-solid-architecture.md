# TscRunner: SOLID Architecture for Safe TypeScript Compilation

You have two places in the codebase that need to type-check TypeScript: the data
quality scoring pipeline (TscScorer) and the RL reward function (TypeCheckReward).
Before TscRunner existed, each one rolled its own tsc subprocess logic. One used a
hardened tsconfig. The other did not. One ran through SandboxedRunner. The other
called subprocess directly. One cached results. The other re-compiled identical
code on every call.

Two paths to the compiler. One of them unsandboxed. That is the kind of bug that
does not show up in tests but shows up when someone feeds your pipeline a
TypeScript file with a malicious compiler plugin that runs `rm -rf /` at
type-check time.

TscRunner is the fix. A single class, in a single file, that handles every tsc
invocation in the entire codebase. It runs through SandboxedRunner, writes a
hardened tsconfig, manages temp files, caches results, and parses errors into
structured data. Nothing else in the project is allowed to call tsc directly.

This document explains the architecture, the SOLID principles behind it, and
every decision that went into making TypeScript compilation safe and fast.

---

## Table of Contents

1. [The Problem: Two Paths, One Unsandboxed](#1-the-problem-two-paths-one-unsandboxed)
2. [SOLID Principles Applied](#2-solid-principles-applied)
3. [TscRunner Internals](#3-tscrunner-internals)
4. [Hardened tsconfig Deep Dive](#4-hardened-tsconfig-deep-dive)
5. [How TypeCheckReward and TscScorer Share TscRunner](#5-how-typecheckreward-and-tscscorer-share-tscrunner)
6. [Batch Optimization](#6-batch-optimization)
7. [Error Classification](#7-error-classification)
8. [Testing Strategy](#8-testing-strategy)
9. [Extension: Adding Runners for Other Languages](#9-extension-adding-runners-for-other-languages)
10. [The Invariants](#10-the-invariants)

---

## 1. The Problem: Two Paths, One Unsandboxed

Before the refactor, the TypeScript compilation stack looked like this:

```
TscScorer (data scoring)           TypeCheckReward (RL training)
     |                                     |
     v                                     v
  SandboxedRunner                   subprocess.run() <-- DANGER
     |                                     |
     v                                     v
  hardened tsconfig                 default tsconfig <-- DANGER
     |                                     |
     v                                     v
    tsc                                   tsc
```

The scoring path was safe. The RL path was not. TypeCheckReward called
`subprocess.run(["tsc", ...])` directly, with a default tsconfig that allowed
compiler plugins, type root scanning, and path traversal.

**TS analogy:** Imagine you have a REST API with two endpoints that both query the
database. One uses parameterized queries. The other uses string concatenation. You
have SQL injection in exactly one endpoint, and you only find out when someone
exploits it.

The risks with the unsandboxed path:

1. **Compiler plugins** -- tsc supports plugins via `tsconfig.json`. A malicious
   file could include a tsconfig that loads a plugin, and that plugin runs
   arbitrary JavaScript at compile time. This is not theoretical; it is how
   TypeScript language server plugins work.

2. **Type root scanning** -- with `typeRoots` unset, tsc scans `node_modules/@types`
   automatically. If the temp directory inherits a parent's `node_modules`, tsc
   loads whatever types it finds, potentially triggering side effects.

3. **No timeout enforcement** -- a crafted file could cause tsc to hang forever
   (recursive type instantiation, `type A = A & { x: A }`). Without a timeout,
   your RL training loop blocks indefinitely.

4. **No process isolation** -- on Windows, a subprocess without `CREATE_NO_WINDOW`
   pops up a console window for every tsc call. During RL training with 1000+
   evaluations, that is 1000 flashing windows.

The solution: delete all direct tsc calls and funnel everything through one class.

---

## 2. SOLID Principles Applied

TscRunner is a textbook application of all five SOLID principles. This is not
accidental -- the design constraints (security, reuse, testability) naturally
lead to SOLID architecture.

### Single Responsibility Principle

TscRunner does ONE thing: execute tsc in a sandboxed environment and return
structured errors.

It does not:
- Decide whether code is TypeScript (that is `language_detect.py`)
- Map error counts to scores (that is `ScoreMapper` in `utils.py`)
- Decide what score thresholds mean (that is the caller's business)
- Log audit trails (that is `ScoringAuditLogger`)
- Manage Docker containers (that is `SandboxedRunner`)

```
                TscRunner
                    |
        +-----------+-----------+
        |           |           |
   Temp files   Hardened    Error parsing
   management   tsconfig
```

Each responsibility that is *not* tsc execution has been extracted into its own
module. The result: TscRunner is 194 lines. It could be 500 lines if it handled
scoring, language detection, and audit logging too. Instead, it handles subprocess
orchestration and nothing else.

### Open/Closed Principle

TscRunner is **open for extension** through new scorers and consumers. Adding a
new way to use tsc (say, a `TscLinter` that extracts warnings) requires zero
changes to TscRunner -- you just call `check()` or `check_batch()` and process
the `TscError` list differently.

TscRunner is **closed to tsc bypass**. There is no public method to run tsc
without the sandbox. There is no `unsafe_check()` escape hatch. The constructor
accepts a `SandboxedRunner` instance, but even if you pass a custom one, it still
goes through the same interface.

```python
# Open: new consumers can use TscRunner without changing it
class NewScorer:
    def __init__(self):
        self._tsc = TscRunner()

    def score(self, code: str) -> float:
        errors = self._tsc.check(code)
        # Your custom scoring logic here
        return 1.0 if not errors else 0.0
```

### Liskov Substitution Principle

TscScorer implements `ScorerProtocol`:

```python
@runtime_checkable
class ScorerProtocol(Protocol):
    name: str
    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult: ...
    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]: ...
    @staticmethod
    def is_available() -> bool: ...
```

Any code that accepts a `ScorerProtocol` can use TscScorer, EslintScorer,
HeuristicScorer, or any future scorer interchangeably. The `CompositeScorer`
does exactly this -- it takes a list of `(ScorerProtocol, float)` weight pairs
and calls `score()` on each one without knowing or caring which concrete class
it is.

**TS analogy:** This is like a TypeScript interface:

```typescript
interface Scorer {
  name: string;
  score(code: string, metadata?: Record<string, unknown>): ScorerResult;
  scoreBatch(items: [string, Record<string, unknown> | null][]): ScorerResult[];
  isAvailable(): boolean;
}
```

Any class that implements this interface works with CompositeScorer. No `extends`,
no base class -- just structural typing. Python's `Protocol` is structural typing
for Python. The `@runtime_checkable` decorator means you can do `isinstance(obj,
ScorerProtocol)` at runtime, just like TypeScript's type guards.

### Interface Segregation Principle

The `ScorerProtocol` has exactly three methods: `score()`, `score_batch()`, and
`is_available()`. Plus one attribute: `name`. That is the minimal interface.

There is no `configure()`, no `set_timeout()`, no `set_strict_mode()`. Configuration
is constructor-only. Once a scorer is created, it behaves the same way every time
you call `score()`. This immutability-after-construction pattern means callers
never need to worry about state -- they call `score()` and get a result.

If we had crammed audit logging into the protocol:

```python
# BAD: forced interface
class ScorerProtocol(Protocol):
    def score(self, code: str) -> ScorerResult: ...
    def score_batch(self, items: list) -> list[ScorerResult]: ...
    def is_available(self) -> bool: ...
    def configure_audit(self, logger: AuditLogger) -> None: ...  # NOT HERE
    def set_security_mode(self, mode: str) -> None: ...          # NOT HERE
```

Then every scorer would need to implement audit configuration, even the
HeuristicScorer that never runs a subprocess and has nothing to audit. ISP
says: do not force implementors to depend on methods they do not use.

### Dependency Inversion Principle

TscRunner depends on the `SandboxedRunner` *abstraction*, not on a specific
subprocess implementation:

```python
class TscRunner:
    def __init__(
        self,
        runner: SandboxedRunner | None = None,  # Inject the abstraction
    ) -> None:
        self._runner = runner or SandboxedRunner(timeout=timeout)
```

In production, `SandboxedRunner` might use native mode (temp dir + timeout) or
Docker mode (full container isolation). TscRunner does not know or care. In tests,
you inject a mock runner that returns canned output:

```python
class MockRunner:
    def run(self, cmd, cwd, **kwargs):
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

runner = MockRunner()
tsc = TscRunner(runner=runner)
errors = tsc.check("const x: number = 42;")
assert errors == []  # Mock returns no errors
```

The dependency arrow points from TscRunner to the SandboxedRunner interface, not
from TscRunner to subprocess or Docker. This inversion is what makes testing
possible without installing tsc or Docker.

---

## 3. TscRunner Internals

### The check() Method

The core method follows a strict sequence:

```
check(code: str) -> list[TscError]
    |
    +-- 1. Hash the code (MD5)
    +-- 2. Check LRU cache -> hit? return cached errors
    +-- 3. Create temp directory
    +-- 4. Write code to check.ts
    +-- 5. Write hardened tsconfig.json
    +-- 6. Run tsc through SandboxedRunner
    +-- 7. Parse stdout+stderr into TscError list
    +-- 8. Cache the result
    +-- 9. Return errors
```

Step-by-step:

```python
def check(self, code: str) -> list[TscError]:
    # Step 1: Hash for cache lookup and audit trail
    code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()

    # Step 2: LRU cache avoids re-compiling identical code
    if code_hash in self._cache:
        self._cache.move_to_end(code_hash)  # Touch for LRU
        return self._cache[code_hash]

    with tempfile.TemporaryDirectory(prefix="cola_tsc_") as tmpdir:
        # Step 4-5: Write files
        code_path = Path(tmpdir) / "check.ts"
        code_path.write_text(code, encoding="utf-8")
        tsconfig = create_hardened_tsconfig(strict=self._strict, include_files=["check.ts"])
        (Path(tmpdir) / "tsconfig.json").write_text(json.dumps(tsconfig))

        # Step 6: Sandboxed execution
        result = self._runner.run(
            [self._tsc_path, "--project", ".", "--pretty", "false"],
            cwd=tmpdir, label="tsc", file_hash=code_hash,
        )

        # Step 7-8: Parse and cache
        errors = self._parse_errors(result.stdout + "\n" + result.stderr)
        self._cache[code_hash] = errors
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # Evict oldest

        return errors
```

### The Cache

The cache is an `OrderedDict` used as an LRU (Least Recently Used) cache. When
an entry is accessed, `move_to_end()` bumps it to the back. When the cache
exceeds `cache_size` (default 256), `popitem(last=False)` evicts the oldest entry.

Why cache? During RL training, GRPO generates multiple completions per problem.
Many of these completions are identical or very similar. Without caching, you pay
the full tsc startup cost (~50ms) for code you already checked.

The cache key is an MD5 hash of the source code. Two identical code strings always
produce the same hash, so cache hits are exact matches. We do not cache near-misses
(code that differs by whitespace) because tsc's error output depends on exact
character positions.

**TS analogy:** This is a `Map<string, TscError[]>` with a max size:

```typescript
const cache = new Map<string, TscError[]>();

function check(code: string): TscError[] {
  const hash = md5(code);
  if (cache.has(hash)) return cache.get(hash)!;

  const errors = runTsc(code);
  cache.set(hash, errors);
  if (cache.size > 256) {
    // Delete oldest entry (first key in insertion order)
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
  return errors;
}
```

### Error Parsing

tsc outputs errors in a predictable format:

```
check.ts(3,7): error TS2322: Type 'string' is not assignable to type 'number'.
check.ts(10,1): error TS1005: ',' expected.
```

The regex:

```python
_ERROR_PATTERN = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$",
    re.MULTILINE,
)
```

Each match becomes a `TscError` dataclass:

```python
@dataclass
class TscError:
    file: str       # "check.ts"
    line: int       # 3
    col: int        # 7
    severity: str   # "error"
    code: str       # "TS2322"
    message: str    # "Type 'string' is not assignable to type 'number'."
```

The `--pretty false` flag is critical. Without it, tsc outputs colored text with
ANSI escape codes, and the error format changes to a multi-line format that breaks
the regex.

### The tsc Path Resolution

On Windows, `tsc` is actually a `.CMD` batch file (`tsc.cmd`). Python's
`subprocess.run(["tsc", ...])` cannot find `.CMD` files without `shell=True` or a
full path. We resolve this in the constructor:

```python
self._tsc_path = shutil.which("tsc") or "tsc"
```

`shutil.which()` searches `PATH` and returns the full path including the `.CMD`
extension. This avoids `shell=True`, which would re-introduce injection risks.

---

## 4. Hardened tsconfig Deep Dive

The tsconfig generated by `create_hardened_tsconfig()` is designed to make tsc
safe to run on code from the internet. Every field serves a security purpose:

```json
{
  "compilerOptions": {
    "strict": true,
    "noEmit": true,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "skipLibCheck": true,
    "plugins": [],
    "types": [],
    "typeRoots": []
  },
  "include": ["check.ts"],
  "exclude": ["node_modules", "**/*.js", "**/*.cjs", "**/*.mjs"]
}
```

### Why Each Field Matters

| Field | Value | Security Purpose |
|-------|-------|-----------------|
| `plugins: []` | Empty array | **Blocks compiler plugin execution.** tsc plugins are JavaScript modules that run at compile time. A malicious tsconfig could specify `"plugins": [{"name": "./evil.js"}]` and run arbitrary code when tsc loads. The empty array explicitly says "no plugins, period." |
| `types: []` | Empty array | **Blocks automatic @types resolution.** By default, tsc loads all packages under `node_modules/@types/`. If the temp directory has a `node_modules/@types/evil/index.d.ts`, tsc would load it. The empty array disables automatic type acquisition. |
| `typeRoots: []` | Empty array | **Blocks type root scanning.** Even with `types: []`, tsc could scan `typeRoots` directories for type definitions. The empty array prevents all scanning. Belt and suspenders. |
| `noEmit: true` | No output files | **No JavaScript output.** We only want type checking, not compilation. This prevents tsc from writing `.js` files that could contain transformed malicious code. |
| `strict: true` | Full strictness | **Catches more errors for training.** This is for quality, not security. Strict mode enables `noImplicitAny`, `strictNullChecks`, `strictFunctionTypes`, and more. The model learns to write properly typed code. |
| `skipLibCheck: true` | Skip .d.ts | **Faster execution.** We do not ship type definitions; skipping lib checks avoids errors from missing `@types` packages. |
| `include: ["check.ts"]` | Explicit file list | **Only checks named files.** No wildcards means tsc cannot pick up injected files in the temp directory. If an attacker somehow writes `evil.ts` next to `check.ts`, tsc ignores it. |
| No `paths`/`baseUrl` | Not set | **No path traversal.** With `baseUrl` set, tsc resolves imports relative to a base directory. A crafted import could escape the temp dir: `import "../../../etc/passwd"`. Without `baseUrl`, all imports are relative to the file, and since we are in a temp dir with nothing in it, there is nothing to import. |

### The Layered Defense

The hardened tsconfig is one layer in a defense stack:

```
Layer 1: SandboxedRunner (timeout, process isolation, optional Docker)
Layer 2: Temp directory (isolated filesystem, auto-cleaned)
Layer 3: Hardened tsconfig (no plugins, no type roots, explicit include)
Layer 4: No shell=True (prevents command injection)
Layer 5: Credential scanner (strips secrets before tsc sees them)
```

Any single layer failing is survivable. An attacker would need to bypass all five
to execute arbitrary code through tsc.

---

## 5. How TypeCheckReward and TscScorer Share TscRunner

The two consumers of TscRunner serve different purposes but use identical
tsc execution:

```
                        TscRunner
                       /         \
                      /           \
        TscScorer (data)    TypeCheckReward (RL)
        |                   |
        | ScorerProtocol    | score() -> float
        | score() -> Result | detailed_score() -> dict
        | score_batch()     |
        |                   |
        v                   v
    CompositeScorer      GRPO training loop
```

### TscScorer: Data Quality Scoring

TscScorer wraps TscRunner to implement `ScorerProtocol`. It converts TscRunner's
raw error list into a 0.0-1.0 quality score:

```python
class TscScorer:
    def score(self, code: str, metadata: dict | None = None) -> ScorerResult:
        if not is_typescript(code, metadata):
            return ScorerResult(score=0.5, scorer_name="tsc", details={"skipped": True})

        errors = self._tsc.check(code)
        num_errors = len(errors)
        has_syntax = any(e.code.startswith("TS1") for e in errors)

        score = _TSC_SCORE_MAP.map(num_errors)
        if has_syntax:
            score = min(score, 0.3)  # Syntax errors cap the score

        return ScorerResult(score=score, scorer_name="tsc", details={...})
```

The score mapping via `ScoreMapper`:

| Error Count | Score | Tier |
|------------|-------|------|
| 0 | 1.0 | Excellent |
| 1 | 0.8 | Excellent |
| 2-3 | 0.6 | Good |
| 4-5 | 0.4 | Average |
| 6-10 | 0.2 | Poor |
| 11+ | 0.1 | Reject |

Syntax errors (TS1xxx) cap the score at 0.3 regardless of count. Code that does
not parse is always at best "poor."

### TypeCheckReward: RL Training

TypeCheckReward uses TscRunner for reinforcement learning rewards. The score
range is different (-0.5 to 1.0) because GRPO needs variance between solutions:

```python
class TypeCheckReward:
    def score(self, code: str) -> float:
        errors = self._run_tsc(code)
        if not errors:
            return 1.0       # Perfect
        if has_syntax_error:
            return -0.5      # Negative reward for unparseable code
        if num_errors <= 2:
            return 0.7       # Minor issues
        elif num_errors <= 5:
            return 0.3       # Moderate issues
        else:
            return 0.0       # Major issues
```

Why negative rewards for syntax errors? GRPO computes advantages relative to the
group mean. If all solutions are bad (0.0 to 0.3), the gradient signal is weak.
By pushing syntax errors to -0.5, we create a clear cliff: "code that does not
parse is much worse than code with type errors." The model learns to produce
syntactically valid code first, then improve type correctness.

### The Key Insight

Both consumers share the exact same tsc execution engine. If we harden the
tsconfig, both benefit. If we add Docker mode, both get it. If we fix a Windows
edge case in path resolution, both are fixed. One change, two consumers, zero
divergence.

Before TscRunner, fixing a security issue in tsc execution meant remembering to
fix it in two places. After TscRunner, it means fixing it in one place.

---

## 6. Batch Optimization

### The Problem with One-at-a-Time

Each tsc invocation has overhead:

1. Start a subprocess (~10-30ms on Windows, ~5-10ms on Linux)
2. tsc reads `tsconfig.json` and initializes the compiler (~20-40ms)
3. tsc reads the source file, parses, type-checks (~10-50ms per file)
4. tsc exits, subprocess is reaped

For a single file, the total is ~50-100ms. For 100 files, that is 5-10 seconds.
Most of that time is startup overhead, not actual type checking.

### The Solution: check_batch()

`check_batch()` writes all files to a single temp directory and runs tsc once:

```python
def check_batch(self, codes: list[str]) -> dict[int, list[TscError]]:
    with tempfile.TemporaryDirectory(prefix="cola_tsc_batch_") as tmpdir:
        filenames = []
        for i, code in enumerate(codes):
            filename = f"check_{i}.ts"
            (Path(tmpdir) / filename).write_text(code)
            filenames.append(filename)

        tsconfig = create_hardened_tsconfig(
            strict=self._strict,
            include_files=filenames,  # All files in one tsconfig
        )
        (Path(tmpdir) / "tsconfig.json").write_text(json.dumps(tsconfig))

        # ONE tsc invocation for ALL files
        result = self._runner.run(
            [self._tsc_path, "--project", ".", "--pretty", "false"],
            cwd=tmpdir, label="tsc_batch",
        )

        # Parse errors and group by filename
        per_file = self._parse_per_file_errors(result.stdout + "\n" + result.stderr)
        return {i: per_file.get(f"check_{i}.ts", []) for i in range(len(codes))}
```

### The Speedup

```
100 files, one at a time:
  100 * (30ms startup + 30ms compile) = ~6,000ms = 6 seconds

100 files, batched:
  1 * (30ms startup + 30ms init + 100 * 5ms compile) = ~560ms = 0.56 seconds
```

That is a 10x speedup for batch scoring. During data preparation, where you might
score 50,000 files, the difference between 5 hours and 30 minutes is the
difference between "overnight job" and "go get coffee."

### The tsconfig Include List

The hardened tsconfig's `include` field lists every file explicitly:

```json
{
  "include": ["check_0.ts", "check_1.ts", "check_2.ts", "check_3.ts"]
}
```

This is critical for security. Without an explicit include list, tsc would use
`"include": ["*.ts"]`, which matches any `.ts` file in the directory. If an
attacker could write an extra file to the temp directory (race condition, symlink
attack), tsc would compile it. Explicit filenames prevent this.

### Mapping Errors Back to Indices

tsc outputs errors with filenames. The batch method parses filenames to map errors
back to the original index:

```
check_0.ts(5,3): error TS2322: ...  -> index 0
check_7.ts(1,1): error TS1005: ...  -> index 7
```

The `_parse_per_file_errors()` method groups errors by filename, then the caller
maps filenames to indices. Files with no errors simply have an empty list.

---

## 7. Error Classification

TypeScript diagnostic codes follow a numbering scheme. Understanding these ranges
is important for scoring because not all errors are created equal.

### The Ranges

```
TS1000-TS1999: Syntax errors
  Code does not parse. Missing semicolons, unclosed brackets, invalid tokens.
  These are the worst -- the code is not valid TypeScript at all.

TS2000-TS2999: Type errors
  The meat of type checking. Type mismatches, missing properties, incorrect
  argument types. These mean the code parses but has type system violations.

TS5000-TS5999: Configuration errors
  Bad tsconfig options, missing input files. Rarely seen in scored code because
  we control the tsconfig.

TS6000-TS6999: Informational messages
  Compiler output messages, not real errors.

TS7000-TS7999: Semantic/strictness errors
  Implicit any types, unused variables, missing return types. These are
  "style" violations under strict mode. Code would compile without --strict.
```

### How Each Consumer Uses Classification

**TscScorer** (data quality):
- TS1xxx (syntax): Caps score at 0.3. Unparseable code is always poor quality.
- TS2xxx (type): Graduated penalty via ScoreMapper. A few type errors is okay.
- TS7xxx (semantic): Counted the same as type errors. Strict mode violations
  reduce quality but are not catastrophic.

**TypeCheckReward** (RL):
- TS1xxx (syntax): Returns -0.5. Negative reward creates strong gradient.
- TS2xxx (type): Graduated by count (0.7 for 1-2, 0.3 for 3-5, 0.0 for 6+).
- TS7xxx (semantic): Same as type errors.

### Common Errors the Model Makes

From analyzing GRPO training runs, the most common generated errors:

| Code | Message | Frequency | What It Means |
|------|---------|-----------|---------------|
| TS2322 | Type 'X' is not assignable to type 'Y' | ~35% | The model knows it needs types but picks the wrong one |
| TS2339 | Property 'X' does not exist on type 'Y' | ~20% | The model invents properties that do not exist |
| TS2345 | Argument of type 'X' is not assignable to parameter of type 'Y' | ~15% | Function call with wrong argument types |
| TS7006 | Parameter implicitly has an 'any' type | ~10% | The model forgets to add parameter types |
| TS1005 | ',' expected / ';' expected | ~8% | Syntax errors -- usually truncated output |
| TS2304 | Cannot find name 'X' | ~5% | The model uses undefined variables |

The TS2322 dominance makes sense: the model learns to use the type annotation
syntax early (because syntax errors get -0.5), but it takes longer to learn which
types are compatible. The RL reward gradient pushes it from "has types" toward
"has correct types."

---

## 8. Testing Strategy

### The Testing Pyramid

```
                /\
               /  \
              / E2E \        test_execution_pipeline.py
             /  (real  \     (real tsc, real subprocess)
            /   tsc)    \
           /────────────\
          /  Integration  \   test_tsc_scorer.py
         /  (mock runner)  \  (mock SandboxedRunner, real parsing)
        /──────────────────\
       /      Unit tests     \  test_tsconfig_factory.py
      /    (pure functions)   \  test_event_transform.py
     /────────────────────────\
```

### Unit Tests: Pure Functions

The tsconfig factory and error parser are pure functions with no side effects:

```python
def test_hardened_tsconfig_blocks_plugins():
    config = create_hardened_tsconfig()
    assert config["compilerOptions"]["plugins"] == []

def test_hardened_tsconfig_blocks_types():
    config = create_hardened_tsconfig()
    assert config["compilerOptions"]["types"] == []
    assert config["compilerOptions"]["typeRoots"] == []

def test_hardened_tsconfig_explicit_include():
    config = create_hardened_tsconfig(include_files=["a.ts", "b.ts"])
    assert config["include"] == ["a.ts", "b.ts"]

def test_hardened_tsconfig_no_paths():
    config = create_hardened_tsconfig()
    assert "paths" not in config["compilerOptions"]
    assert "baseUrl" not in config["compilerOptions"]
```

These tests verify the security invariants of the tsconfig. If someone adds
`"plugins": [{"name": "some-tool"}]` to the hardened config, the test fails.

### Integration Tests: Mock Runner

TscRunner integration tests inject a mock `SandboxedRunner` that returns
predetermined output:

```python
class MockRunner:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls = []

    def run(self, cmd, cwd, **kwargs):
        self.calls.append({"cmd": cmd, "cwd": cwd})
        return CompletedProcess(
            args=cmd, returncode=self.returncode,
            stdout=self.stdout, stderr=self.stderr,
        )

def test_tsc_runner_parses_errors():
    runner = MockRunner(
        stdout='check.ts(3,7): error TS2322: Type "string" not assignable to "number".\n'
    )
    tsc = TscRunner(runner=runner)
    errors = tsc.check("const x: number = 'hello';")
    assert len(errors) == 1
    assert errors[0].code == "TS2322"
    assert errors[0].line == 3

def test_tsc_runner_caches_identical_code():
    runner = MockRunner(stdout="")
    tsc = TscRunner(runner=runner)
    tsc.check("const x = 1;")
    tsc.check("const x = 1;")  # Same code
    assert len(runner.calls) == 1  # Only one subprocess call

def test_tsc_runner_uses_hardened_tsconfig():
    runner = MockRunner(stdout="")
    tsc = TscRunner(runner=runner)
    tsc.check("const x = 1;")
    # Verify tsc was called with --project flag
    assert "--project" in runner.calls[0]["cmd"]
```

### Enforcement Tests

The most important tests verify that tsc execution *cannot* bypass the sandbox:

```python
def test_no_direct_subprocess_in_scorers():
    """No scorer module imports subprocess directly."""
    import ast
    scorer_dir = Path("src/cola_coder/data/scorers")
    for py_file in scorer_dir.glob("*.py"):
        if py_file.name in ("sandbox.py",):  # sandbox.py is allowed
            continue
        tree = ast.parse(py_file.read_text())
        imports = [
            node.names[0].name for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ]
        assert "subprocess" not in imports, (
            f"{py_file.name} imports subprocess directly -- use SandboxedRunner"
        )
```

This test ensures the architectural constraint is enforced: only `sandbox.py`
is allowed to call `subprocess`. Every other module must go through
`SandboxedRunner`.

---

## 9. Extension: Adding Runners for Other Languages

The TscRunner pattern is designed to be replicated for other languages. Here is
how you would add a Go type checker or a Rust compiler checker:

### Step 1: Create the Runner

```python
# go_runner.py
class GoVetRunner:
    """Sandboxed go vet execution for Go code quality scoring."""

    _ERROR_PATTERN = re.compile(r"^(.+?):(\d+):(\d+):\s+(.+)$", re.MULTILINE)

    def __init__(self, runner: SandboxedRunner | None = None, timeout: int = 10):
        self._runner = runner or SandboxedRunner(timeout=timeout)
        self._go_path = shutil.which("go") or "go"

    def check(self, code: str) -> list[GoError]:
        with tempfile.TemporaryDirectory(prefix="cola_govet_") as tmpdir:
            (Path(tmpdir) / "main.go").write_text(code)
            # Write go.mod to isolate the module
            (Path(tmpdir) / "go.mod").write_text("module check\n\ngo 1.21\n")

            result = self._runner.run(
                [self._go_path, "vet", "./..."],
                cwd=tmpdir, label="go_vet",
            )
            return self._parse_errors(result.stderr or "")
```

### Step 2: Create the Scorer

```python
# go_scorer.py
class GoVetScorer:
    """ScorerProtocol adapter for GoVetRunner."""
    name = "go_vet"

    def __init__(self, runner: SandboxedRunner | None = None):
        self._go = GoVetRunner(runner=runner)

    def score(self, code: str, metadata: dict | None = None) -> ScorerResult:
        if not is_go(code, metadata):
            return ScorerResult(score=0.5, scorer_name=self.name, details={"skipped": True})
        errors = self._go.check(code)
        score = _GO_SCORE_MAP.map(len(errors))
        return ScorerResult(score=score, scorer_name=self.name, details={...})
```

### Step 3: Register in CompositeScorer

```python
scorers = [
    (TscScorer(), 0.30),
    (EslintScorer(), 0.20),
    (GoVetScorer(), 0.20),  # New!
    (HeuristicScorer(), 0.15),
    (StarsScorer(), 0.15),
]
composite = CompositeScorer(scorers)
```

### The Pattern

Every language runner follows the same shape:

```
1. Runner class: manages temp files, runs tool through SandboxedRunner
2. Scorer class: wraps runner, implements ScorerProtocol
3. Hardened config: tool-specific config that blocks dangerous features
4. Language detect: is_go(), is_rust(), etc.
5. Score mapper: error count -> quality score
```

The existing utilities (SandboxedRunner, ScorerProtocol, ScoreMapper, language
detection) are all reusable. Adding a new language requires writing the runner
and scorer. The infrastructure is already there.

### What a Rust Version Would Look Like

```python
class RustcRunner:
    def check(self, code: str) -> list[RustError]:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.rs").write_text(code)
            result = self._runner.run(
                ["rustc", "--edition", "2021", "--crate-type", "lib",
                 "--error-format", "json", "main.rs"],
                cwd=tmpdir, label="rustc",
            )
            return self._parse_json_errors(result.stderr or "")
```

Rust's `--error-format json` makes parsing trivial compared to tsc's text output.
Each language has its own quirks, but the scaffolding (SandboxedRunner, temp dir,
hardened config) is identical.

---

## 10. The Invariants

These properties must always hold. If any of them break, the scoring and RL
pipelines are compromised:

1. **Single entry point.** All tsc execution in the codebase goes through
   TscRunner. No scorer, reward function, or script is allowed to call
   `subprocess.run(["tsc", ...])` directly.

2. **Always sandboxed.** TscRunner uses SandboxedRunner for every invocation.
   There is no `bypass_sandbox=True` flag. There is no `unsafe_check()` method.

3. **Always hardened.** Every tsc invocation uses a tsconfig with `plugins: []`,
   `types: []`, and `typeRoots: []`. The tsconfig is generated fresh for each
   invocation; it is never read from the input code.

4. **Explicit include.** The tsconfig's `include` field lists files by exact
   name. No wildcards, no glob patterns.

5. **No shell=True.** The subprocess command is passed as a list, never a string.
   No shell metacharacter expansion, no injection vector.

6. **Cache coherence.** The cache is keyed by MD5 of the source code. Identical
   code always produces the same cached result. Different code never collides
   (MD5 collision probability is 1 in 2^128 for random inputs).

7. **Protocol conformance.** TscScorer implements ScorerProtocol. It can be
   swapped with any other scorer in CompositeScorer without code changes.

8. **Batch consistency.** `check_batch()` produces the same errors as calling
   `check()` on each file individually. The batch optimization is purely for
   performance; it does not change results.

If you are modifying any of these files, run the tests:

```bash
.venv/Scripts/pytest tests/ -k "tsc" -v
```

If you are modifying TscRunner itself, also run the integration tests that use
a real tsc installation:

```bash
.venv/Scripts/pytest tests/test_execution_pipeline.py -v
```

The scoring pipeline processes millions of files. A single bug in TscRunner
propagates to every scored file in your training data. Get it right once. The
architecture makes sure it stays right.
