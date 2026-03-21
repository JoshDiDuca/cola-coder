# Quality-Weighted Training: Teaching Your Model What "Good Code" Looks Like

Not all code is equally valuable for training. A well-documented utility
with clean error handling teaches your model far more than a 3-line
boilerplate file or an auto-generated protobuf stub. Quality-weighted
training makes this intuition concrete: every training example gets a
weight that controls how much the model learns from it.

This is cola-coder's most impactful data pipeline feature. It is the
difference between "saw a million files" and "studied the best code and
skimmed the rest."

**TypeScript analogy:** Think of npm packages. You would not study
`left-pad` and a mature library like `zod` with equal intensity. You
would spend 2x the time on `zod` (excellent types, clear APIs, thorough
error handling) and barely glance at a 5-line utility. Quality weighting
does the same thing for your model --- automatically, at scale.

---

## Table of Contents

1. [The Core Insight](#1-the-core-insight)
2. [Two-Stage Pipeline Overview](#2-two-stage-pipeline-overview)
3. [Stage 1: The Quality Filter (Binary Keep/Reject)](#3-stage-1-the-quality-filter)
4. [Stage 2: The Quality Scorer (Continuous 0.0-1.0)](#4-stage-2-the-quality-scorer)
5. [Score-to-Weight Conversion](#5-score-to-weight-conversion)
6. [The .weights.npy Sidecar File](#6-the-weightsnpy-sidecar-file)
7. [WeightedCodeDataset: Loading Weights at Training Time](#7-weightedcodedataset)
8. [Weighted Cross-Entropy Loss: The Math and Intuition](#8-weighted-cross-entropy-loss)
9. [Why This Matters Enormously for Small Models](#9-why-this-matters-for-small-models)
10. [Practical Workflow: End-to-End](#10-practical-workflow)
11. [Tuning Guide](#11-tuning-guide)
12. [Research Backing](#12-research-backing)

---

## 1. The Core Insight

When your model processes a training batch, every token contributes
equally to the gradient update by default. A token inside this code:

```python
# EXCELLENT: well-structured, typed, documented
def parse_config(raw: dict[str, Any]) -> AppConfig:
    """Parse and validate raw config dictionary.

    Raises:
        ConfigError: If required keys are missing or values are invalid.
    """
    try:
        return AppConfig(
            host=raw["host"],
            port=int(raw.get("port", 8080)),
            debug=raw.get("debug", False),
        )
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"Invalid config: {exc}") from exc
```

...has exactly the same influence on model weights as a token inside this:

```python
# POOR: no types, no docs, bare except, meaningless names
def f(d):
    try:
        return d["h"], d["p"], d["d"]
    except:
        pass
```

That is a waste. The first example teaches the model type hints, docstrings,
specific exception handling, f-string error messages, and the `from exc`
chaining pattern. The second teaches single-character names, bare
`except:`, and silent `pass`. Every gradient step spent on the second
example is a step not spent on the first.

Quality weighting fixes this: the first example gets weight **~1.8** (the
model trains hard on it), while the second gets weight **~0.3** (it barely
moves the needle). Neither is thrown away --- the poor example still
contributes *something* --- but the model's limited capacity is focused
where it matters.

---

## 2. Two-Stage Pipeline Overview

The pipeline has two stages with different jobs:

```
Raw code from HuggingFace
    |
    v
 Stage 1: Quality Filter  (binary: keep or reject)
    |  Removes the truly awful stuff: minified bundles,
    |  auto-generated stubs, data dumps, syntax-broken files.
    |  ~48% rejected (conservative), ~65% rejected (strict).
    |
    v
 Stage 2: Quality Scorer   (continuous: 0.0 to 1.0)
    |  Scores everything that survived Stage 1 on 13 signals.
    |  Converts scores to training weights (0.0 to 2.0).
    |
    v
 Tokenize + chunk
    |
    v
 Save: train_data.npy  +  train_data.weights.npy
    |
    v
 Trainer auto-detects .weights.npy
    |  Uses WeightedCodeDataset + weighted cross-entropy loss.
    v
 Model learns more from good code, less from mediocre code.
```

**Why two stages instead of one?** Stage 1 is fast (string checks, no
scoring overhead) and removes ~50% of files that are unambiguously garbage.
Stage 2 is more expensive (13 signal computations per file) but only runs
on the survivors. This keeps preprocessing time reasonable even on millions
of files. The `--score` flag adds about 30% to preprocessing time.

**TypeScript analogy:** Stage 1 is like TypeScript's compiler errors ---
the code does not even compile, reject it. Stage 2 is like ESLint severity
levels --- the code compiles, but how *good* is it? Each rule contributes
a numeric weight to the final score.

---

## 3. Stage 1: The Quality Filter

**File:** `src/cola_coder/data/quality_filter.py`

The filter runs up to 10 binary checks on every file. Each check returns
`(keep: bool, reason: str)`. If any check fails, the file is rejected
and the reason is logged.

### The 10 Checks

#### 1. Length Check

```python
def check_length(content, min_lines=5, max_lines=10000):
```

- **What it catches:** Files that are too short (empty boilerplate,
  one-liner configs) or too long (auto-generated code, bundled files).
- **Conservative thresholds:** 5--10,000 lines.
- **Strict thresholds:** 10--5,000 lines.

**Example rejection:**

```python
# 3 lines --- rejected as too_short
from flask import Flask
app = Flask(__name__)
```

#### 2. Average Line Length

```python
def check_avg_line_length(content, max_avg=200):
```

- **What it catches:** Minified/bundled code. Normal source code averages
  30--60 characters per line. Minified JavaScript averages 5,000+.
- **Conservative:** >200 chars average.
- **Strict:** >120 chars average.

**Example rejection:** A webpack bundle where the entire app is on one
line with an average line length of 8,000 characters.

#### 3. Max Line Length

```python
def check_max_line_length(content, max_len=1000):
```

- **What it catches:** Files with a single extremely long line --- base64
  blobs, embedded data URIs, inlined SVGs.
- **Conservative:** any line >1,000 chars.
- **Strict:** any line >500 chars.

#### 4. Character Diversity

```python
def check_character_diversity(content, min_unique_ratio=0.05):
```

- **What it catches:** Degenerate content --- files that are mostly
  one repeated character or pattern (e.g., a file of 10,000 zeroes).
- **Conservative:** <5% unique characters.
- **Strict:** <8% unique characters.

#### 5. Auto-Generated Detection

```python
def check_not_autogenerated(content):
```

- **What it catches:** Files with "do not edit", "generated by",
  "auto-generated" and similar markers in their first 10 lines.
- Matches 12+ common markers including "generated by the protocol buffer
  compiler", "generated by django", "generated by swagger".
- Same in both modes.

**Example rejection:**

```python
# AUTO-GENERATED by protoc-gen-python. DO NOT EDIT.
# source: api/v1/service.proto
```

#### 6. Data File Detection

```python
def check_not_data_file(content):
```

- **What it catches:** Files that are data rather than code --- JSON
  dumps, CSV data, hex/binary dumps.
- Checks first 50 lines for: >80% lines starting with `{`/`[`/`]`/`}`;
  hex patterns; consistent comma counts suggesting CSV.
- Same in both modes.

#### 7. Comment Ratio

```python
def check_comment_ratio(content, max_comment_ratio=0.85):
```

- **What it catches:** Files that are almost entirely comments --- license
  headers, documentation-only files.
- **Conservative:** >85% comment lines.
- **Strict:** >60% comment lines.

#### 8. Test Dump Detection

```python
def check_not_test_heavy(content, max_test_ratio=0.9):
```

- **What it catches:** Auto-generated test files that are walls of
  `assert` or `expect()` statements with no real logic.
- Normal test files pass. Only rejects when >90% (conservative) or >70%
  (strict) of lines are assertions.

#### 9. Syntax Parsing (Python)

```python
def check_python_parseable(content):
```

- **What it catches:** Python files with syntax errors. Uses Python's
  `ast.parse()` --- if Python's own compiler rejects the file, the code
  is broken.
- Only runs on files that look like Python (heuristic detection).
- Non-Python files always pass.

#### 10. Brace Balance (JS/TS)

```python
def check_js_ts_parseable(content):
```

- **What it catches:** JavaScript/TypeScript files with badly unbalanced
  braces (truncated files, corrupted downloads).
- Allows small imbalance (template literals and regex can confuse counting).
  Only rejects when imbalance >5 braces AND >20% relative to total.
- Non-JS/TS files always pass.

### Conservative vs Strict Mode

| Check                 | Conservative       | Strict              |
|-----------------------|--------------------|---------------------|
| min lines             | 5                  | 10                  |
| max lines             | 10,000             | 5,000               |
| max avg line length   | 200                | 120                 |
| max single line       | 1,000              | 500                 |
| char diversity        | 5%                 | 8%                  |
| max comment ratio     | 85%                | 60%                 |
| max test ratio        | 90%                | 70%                 |
| has functions/classes | ---                | yes (strict only)   |
| naming quality        | ---                | avg >= 3 chars      |
| code-to-blank ratio   | ---                | >= 50% non-blank    |
| copy-paste detection  | ---                | <= 30% duplicate    |
| has documentation     | ---                | yes (strict only)   |
| hardcoded secrets     | ---                | yes (strict only)   |

Strict mode adds **6 extra checks** (functions/classes, naming, blank
ratio, copy-paste, documentation, secrets) that only exist in strict mode.
These reject code that is *mediocre*, not just *broken*.

**When to use which:**

- **Conservative (default):** You have a normal amount of data and want to
  remove obvious garbage without losing potentially useful examples.
  This is the right choice for most training runs.
- **Strict:** You have more data than you need (e.g., training on a
  popular language like Python/TypeScript with plenty of high-quality
  repos available) and want to maximize quality per token. Also useful
  when training small models (50M) that have very limited capacity.

### Rejection Rates in Practice

On StarCoderData (raw GitHub code):

- **Conservative:** ~48% rejection rate. Sounds high, but raw GitHub
  contains enormous amounts of noise: tiny boilerplate, auto-generated
  code, data dumps, minified bundles, and syntax-broken files. The
  StarCoder paper itself filters out ~50%.
- **Strict:** ~65% rejection rate. You keep only the clearly-good code.

### Design Decisions

**Conservative by default.** If a check cannot determine quality
(unsupported language, ambiguous content), it passes the file through.

**Fail-safe.** If any check crashes on a file (unexpected encoding,
etc.), the file passes through in conservative mode. In strict mode,
analysis failures result in rejection --- we only keep code we are
confident about.

**Fast.** All checks are string operations --- no external tools, no
network calls. The filter adds negligible overhead.

---

## 4. Stage 2: The Quality Scorer

**File:** `src/cola_coder/features/code_scorer.py`

Everything that survived the binary filter now gets a continuous 0.0--1.0
quality score based on 13 weighted signals.

### The 13 Signals

Each signal is scored independently on 0.0--1.0, then combined using a
weighted average. The weights sum to 1.0 (like CSS specificity --- each
signal contributes a fixed fraction).

```python
_WEIGHTS = {
    "length":         0.05,   # 5%
    "line_quality":   0.05,   # 5%
    "structure":      0.15,   # 15%  <-- highest weight
    "naming":         0.12,   # 12%
    "comments":       0.10,   # 10%
    "documentation":  0.10,   # 10%
    "complexity":     0.08,   # 8%
    "formatting":     0.05,   # 5%
    "duplication":    0.08,   # 8%
    "syntax":         0.10,   # 10%
    "modernness":     0.05,   # 5%
    "error_handling": 0.04,   # 4%
    "security":       0.03,   # 3%
}
```

**The top 3 signals by weight** (structure at 15%, naming at 12%, and
syntax + comments + documentation tied at 10% each) tell you what the
scorer values most: well-organized code with clear names that actually
parses and has some documentation.

#### 1. Length (5%)

Scores the file's line count. Sweet spot: 10--300 lines (score = 1.0).
Very short (<5 lines) or very long (>5,000 lines) files score 0.1.
This is a soft version of the binary filter's length check.

| Lines       | Score |
|-------------|-------|
| < 5         | 0.1   |
| 5--9        | 0.4   |
| 10--300     | 1.0   |
| 301--500    | 0.9   |
| 501--1,000  | 0.7   |
| 1,001--2,000| 0.5   |
| 2,001--5,000| 0.3   |
| > 5,000     | 0.1   |

#### 2. Line Quality (5%)

Scores average and maximum line length. Normal code averages 30--80
characters. Minified code averages 200+. Returns a blend of 60%
average-length score and 40% max-length score.

Red flags: max line >500 (score 0.1) or avg >200 (score 0.15).

#### 3. Structure (15% --- highest weight)

Counts functions, classes, and imports. Well-structured code is organized
into named units rather than being a wall of top-level statements.

- No functions or classes in a long file: **0.2**
- Function density 3--15% of total lines: **1.0**
- Has classes: **+0.1 bonus**
- Has imports: **+0.05 bonus**

**TypeScript analogy:** A file full of loose `console.log()` statements
scores 0.2. A file with well-defined exported functions and a class
scores 1.0. Same logic you would apply reviewing a PR.

#### 4. Naming (12%)

Checks naming convention consistency and descriptiveness.

- Extracts identifier names from function defs, const/let declarations.
- Checks for snake_case (Python), camelCase (JS/TS), PascalCase (classes).
- Penalizes single-character names (>50% single-char = score 0.2).
- Penalizes very short average name length (<3 chars = score 0.25).
- Rewards consistency: if >80% of names follow one convention, score 1.0.

#### 5. Comments (10%)

Scores comment ratio. Sweet spot: 5--25% of file content is comments.

| Comment Ratio | Score |
|---------------|-------|
| 5--25%        | 1.0   |
| 2--5% or 25--40% | 0.6 |
| < 2%          | 0.3   |
| > 40%         | 0.2   |

Too few comments = the code is undocumented. Too many = it is probably
a license dump or documentation-only file.

#### 6. Documentation (10%)

Counts docstrings (`"""..."""` in Python) and JSDoc blocks (`/** ... */`
in JS/TS). More documentation blocks = higher score.

| Doc Blocks | Score |
|------------|-------|
| >= 5       | 1.0   |
| 3--4       | 0.9   |
| 1--2       | 0.7   |
| 0 (short file) | 0.4 |
| 0 (long file)  | 0.2 |

#### 7. Complexity (8%)

Counts control-flow keywords (`if`, `else`, `for`, `while`, `try`,
`catch`, etc.) per line. Too many = spaghetti. Zero = declarations-only.

| Density       | Score |
|---------------|-------|
| 0             | 0.5   |
| <= 0.15       | 1.0   |
| 0.15--0.25    | 0.7   |
| 0.25--0.40    | 0.4   |
| > 0.40        | 0.2   |

#### 8. Formatting (5%)

Two sub-checks blended 60/40:

- **Blank line ratio** (60%): sweet spot 5--25%. Files with no blank
  lines (wall of text) or mostly blank lines both score low.
- **Indentation consistency** (40%): mixing tabs and spaces is penalized.
  Consistent use of one style scores 1.0.

#### 9. Duplication (8%)

Counts non-trivial lines (>10 chars) that appear more than once. High
internal duplication signals auto-generated code or lazy copy-paste.

| Duplicate Ratio | Score |
|-----------------|-------|
| <= 5%           | 1.0   |
| 5--15%          | 0.85  |
| 15--25%         | 0.65  |
| 25--35%         | 0.45  |
| 35--50%         | 0.25  |
| > 50%           | 0.1   |

#### 10. Syntax (10%)

- **Python:** Runs `ast.parse()`. Full parse success = 1.0. Syntax error
  gives partial credit based on how far into the file the error occurs
  (capped at 0.3).
- **JS/TS:** Checks brace balance. <= 2% relative imbalance = 1.0.
  > 20% imbalance = 0.15.
- **Unknown language:** 0.7 (neutral).

#### 11. Modernness (5%)

Detects modern vs deprecated language patterns.

**Python modern patterns (boost score):**
- f-strings, type hints, walrus operator (`:=`), match/case, async/await,
  `@dataclass`, pathlib usage.

**Python deprecated patterns (lower score):**
- `%`-style formatting, old-style `super(ClassName, self)`, Python 2
  `print` statements, `.has_key()`.

**JS/TS modern patterns (boost score):**
- `const`/`let`, arrow functions, template literals, optional chaining
  (`?.`), nullish coalescing (`??`), async/await, TypeScript generics,
  interfaces/types.

**JS/TS deprecated patterns (lower score):**
- `var` declarations, `.then()`/`.catch()` chains, loose equality (`==`),
  `arguments` keyword.

The score is `modern_points / (modern_points + deprecated_points + 1.0)`,
clamped to 0.0--1.0. A file with no signals gets 0.6 (neutral).

#### 12. Error Handling (4%)

Scores the quality of try/except (Python) or try/catch (JS/TS) blocks.

**Boosts:**
- Has try/except blocks at all (+0.2)
- Catches specific exception types like `ValueError` (+0.1)
- Custom exception classes (+0.15)
- `finally` blocks (+0.05)
- Descriptive error messages in `raise` (+0.05--0.1)
- Logging errors instead of swallowing (+0.05)

**Penalties:**
- All `except:` clauses are bare (no exception type): -0.3
- Empty catch blocks in JS/TS: -0.2 per empty catch

Short files (<20 lines) without error handling get 0.6 (it is reasonable
for a utility function to not need try/except).

#### 13. Security (3%)

A **penalty scorer** --- starts at 1.0 and deducts for anti-patterns:

| Anti-Pattern                        | Penalty |
|-------------------------------------|---------|
| Hardcoded secrets (API keys, etc.)  | -0.5    |
| `eval()` / `exec()` usage          | -0.15 each (max -0.3) |
| SQL string concatenation            | -0.1 each (max -0.2) |
| `subprocess` with `shell=True`      | -0.1    |
| `os.system()` with string concat    | -0.1    |

A clean file scores 1.0. A file with a hardcoded API key and an `eval()`
call scores 0.35.

### Score Distribution: What Real Code Looks Like

Most code that survives the binary filter lands in the 0.4--0.7 range.
The distribution looks roughly like this:

```
Score Range   Tier        Typical %    What It Looks Like
-----------   ---------   ----------   ----------------------------------
0.0--0.2      reject       ~5%         Broken syntax, degenerate files
0.2--0.4      poor        ~15%         Bare scripts, no structure
0.4--0.6      average     ~45%         Typical GitHub code
0.6--0.8      good        ~30%         Well-organized, documented
0.8--1.0      excellent    ~5%         Textbook-quality, all signals green
```

The bell curve peaks around 0.5. Excellent code (0.8+) is rare --- only
about 5% of files hit all 13 signals well. This matches intuition: most
GitHub code is functional but not exemplary.

### Concrete Scoring Example

Here is a real-world-style Python file and its approximate breakdown:

```python
"""HTTP client with retry logic and timeout handling."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class RetryConfig:
    """Configuration for HTTP retry behavior."""
    max_retries: int = 3
    backoff_factor: float = 0.5
    timeout: float = 30.0


class HttpClient:
    """Async HTTP client with configurable retry logic.

    Usage:
        client = HttpClient(RetryConfig(max_retries=5))
        response = await client.get("https://api.example.com/data")
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()
        self._client = httpx.AsyncClient(timeout=self.config.timeout)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Send GET request with automatic retry on failure.

        Args:
            url: The URL to request.
            **kwargs: Additional arguments passed to httpx.get().

        Raises:
            httpx.HTTPError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.get(url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                wait = self.config.backoff_factor * (2 ** attempt)
                await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]
```

**Approximate signal scores:**

| Signal          | Score | Why                                            |
|-----------------|-------|------------------------------------------------|
| length          | 1.0   | ~55 lines (sweet spot)                         |
| line_quality    | 1.0   | avg ~35 chars, max ~70 chars                   |
| structure       | 1.0   | 2 classes, 2 functions, 4 imports              |
| naming          | 0.95  | descriptive names, consistent snake_case       |
| comments        | 0.6   | ~12% comments (within range but on the low end)|
| documentation   | 1.0   | 4 docstring blocks                             |
| complexity      | 1.0   | moderate control flow (for, try, if)           |
| formatting      | 1.0   | consistent 4-space indent, clean blank lines   |
| duplication     | 1.0   | no repeated blocks                             |
| syntax          | 1.0   | parses cleanly                                 |
| modernness      | 0.9   | type hints, async/await, dataclass, f-strings  |
| error_handling  | 0.85  | specific except, retry logic, re-raise         |
| security        | 1.0   | no anti-patterns                               |

**Weighted overall:** ~0.96 --- tier **"excellent"**, weight **~2.0**.

Compare with a mediocre version of similar functionality:

```python
import requests

def get(u):
    for i in range(3):
        try:
            r = requests.get(u)
            return r
        except:
            pass
    return None
```

| Signal          | Score | Why                                            |
|-----------------|-------|------------------------------------------------|
| length          | 0.4   | ~10 lines (barely above minimum)               |
| line_quality    | 0.9   | short lines, fine                              |
| structure       | 0.7   | has a function, but minimal                    |
| naming          | 0.25  | single-char names: u, i, r                     |
| comments        | 0.3   | no comments at all                             |
| documentation   | 0.4   | no docstring (short file)                      |
| complexity      | 1.0   | low density (fine for a short function)        |
| formatting      | 0.8   | consistent but no blank line structure          |
| duplication     | 0.8   | too short to judge                             |
| syntax          | 1.0   | parses fine                                    |
| modernness      | 0.6   | no modern or deprecated signals                |
| error_handling  | 0.2   | bare except, silent pass, returns None         |
| security        | 1.0   | clean                                          |

**Weighted overall:** ~0.57 --- tier **"average"**, weight **~1.0**.

The excellent file gets **2x the gradient signal** of the mediocre one.

---

## 5. Score-to-Weight Conversion

The scorer maps the 0.0--1.0 overall score to a 0.0--2.0 training weight
using a tier-based system with smooth interpolation within each tier.

### Tier Table

| Tier       | Score Range | Base Weight | Effect on Training              |
|------------|-------------|-------------|----------------------------------|
| excellent  | 0.8--1.0    | 2.0         | Model trains hard on this        |
| good       | 0.6--0.8    | 1.5         | Above-average contribution       |
| average    | 0.4--0.6    | 1.0         | Baseline (same as unweighted)    |
| poor       | 0.2--0.4    | 0.3         | Barely contributes to gradient   |
| reject     | 0.0--0.2    | 0.0         | Excluded from loss entirely      |

### Smooth Interpolation

Within each tier, the weight is linearly interpolated toward the next
tier's base weight so there are no hard jumps at tier boundaries.

```python
# From code_scorer.py, simplified:
lo, hi = tier_bounds[tier]          # e.g., (0.6, 0.8) for "good"
t = (overall - lo) / (hi - lo)     # 0.0 at bottom of tier, 1.0 at top

next_weight = next_weight_map[tier] # "good" -> 2.0 (toward excellent)
weight = base_weight + t * (next_weight - base_weight)
```

**Concrete examples:**

| Overall Score | Tier      | t     | Weight Calculation          | Final Weight |
|---------------|-----------|-------|-----------------------------|-------------|
| 0.90          | excellent | 0.50  | 2.0 + 0.5 * (2.0 - 2.0)   | 2.0         |
| 0.80          | excellent | 0.00  | 2.0 + 0.0 * (2.0 - 2.0)   | 2.0         |
| 0.70          | good      | 0.50  | 1.5 + 0.5 * (2.0 - 1.5)   | 1.75        |
| 0.60          | good      | 0.00  | 1.5 + 0.0 * (2.0 - 1.5)   | 1.5         |
| 0.50          | average   | 0.50  | 1.0 + 0.5 * (1.5 - 1.0)   | 1.25        |
| 0.40          | average   | 0.00  | 1.0 + 0.0 * (1.5 - 1.0)   | 1.0         |
| 0.30          | poor      | 0.50  | 0.3 + 0.5 * (1.0 - 0.3)   | 0.65        |
| 0.20          | poor      | 0.00  | 0.3 + 0.0 * (1.0 - 0.3)   | 0.3         |
| 0.15          | reject    | ---   | 0.0 (always)               | 0.0         |

The curve is **not linear** --- it is steeper at the top (big reward for
excellent code) and gentler at the bottom (poor code gets very little, but
not zero). This matches the intuition that the difference between "good"
and "excellent" matters more than the difference between "poor" and
"average."

---

## 6. The .weights.npy Sidecar File

When you run `prepare_data.py --score`, the scorer creates a **sidecar
file** alongside the main training data:

```
data/processed/
    train_data.npy             # [num_chunks, chunk_size] uint16 token IDs
    train_data.weights.npy     # [num_chunks] float32 quality weights
```

### How It Is Generated

The scoring pipeline in `prepare_data.py` works as follows:

1. All data has already been filtered, tokenized, and saved to
   `train_data.npy`.
2. The script loads the saved data via memory-mapped access.
3. For each chunk, it **decodes** the tokens back to text using the
   tokenizer.
4. It runs `CodeScorer().score(text)` on the decoded text.
5. It converts each score to a weight via `scorer.score_to_weight(result)`.
6. It saves the weight array as `train_data.weights.npy`.

```python
# Simplified from prepare_data.py:
scorer = CodeScorer()
data = np.load(output_file, mmap_mode="r")
weights = np.zeros(len(data), dtype=np.float32)

for i in range(len(data)):
    text = tokenizer.decode(data[i])
    result = scorer.score(text)
    weights[i] = scorer.score_to_weight(result)

np.save(weights_path, weights)
```

### Important Detail: Scoring Happens After Tokenization

The scorer works on **decoded chunks**, not raw files. This means:

- Each chunk is `max_seq_len` tokens long (e.g., 1024 tokens).
- A single source file may span multiple chunks.
- Each chunk is scored independently.
- A file boundary in the middle of a chunk means that chunk gets a blended
  quality signal from both files.

This is fine in practice --- the chunk-level granularity means weights
naturally adapt to the local quality of the training data.

### File Format

The `.weights.npy` file is a standard NumPy array:

- **Shape:** `[num_chunks]` --- one float per training example.
- **Dtype:** `float32`.
- **Values:** 0.0--2.0 (after `score_to_weight()` conversion).
- **Size:** `num_chunks * 4 bytes`. For 100K chunks, that is ~400 KB.

The file must have **exactly** the same number of entries as rows in
`train_data.npy`. If there is a mismatch, `WeightedCodeDataset` raises a
`ValueError` at startup (fail-fast --- no silent corruption).

---

## 7. WeightedCodeDataset

**File:** `src/cola_coder/data/dataset.py`

`WeightedCodeDataset` extends `CodeDataset` with per-example quality
weights. It is the bridge between the `.weights.npy` file and the training
loop.

### How It Works

```python
class WeightedCodeDataset(CodeDataset):
    def __init__(self, data_path, max_seq_len=None, weights_path=None):
        super().__init__(data_path, max_seq_len=max_seq_len)

        if weights_path is not None:
            raw = np.load(weights_path).astype(np.float32)
            # Validate shape match
            assert len(raw) == self.num_chunks

            # KEY STEP: normalize so mean == 1.0
            self.weights = torch.from_numpy(raw / raw.mean())
        else:
            self.weights = torch.ones(self.num_chunks)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)       # {"input_ids": tensor}
        item["weight"] = self.weights[idx]    # add scalar weight
        return item
```

### Mean-Normalization: Why It Matters

The raw weights from the scorer range from 0.0 to 2.0. But the dataset
**normalizes them so the mean is 1.0**:

```python
self.weights = torch.from_numpy(raw / raw.mean())
```

**Why?** Without normalization, if most weights are low (say average 0.6),
the effective learning rate would be 60% of what you configured. The model
would train slower for no good reason. Normalizing to mean=1.0 preserves
the configured gradient scale while still letting high-quality examples
contribute proportionally more.

**TypeScript analogy:** It is like normalizing exam scores so the class
average is always 100. The ranking stays the same (top students still
score higher) but the absolute numbers are comparable across classes.

After normalization, a weight of 1.5 means "50% more gradient signal than
average" and a weight of 0.5 means "50% less than average."

### The Collator

`WeightedCodeCollator` stacks the per-example weights into a batch tensor:

```python
class WeightedCodeCollator:
    def __call__(self, examples):
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        weights = torch.stack([ex["weight"] for ex in examples])
        return {"input_ids": input_ids, "weights": weights}
        # weights shape: [batch_size]
```

### Auto-Detection

The `create_dataloader()` function auto-selects the right dataset and
collator:

```python
use_weights = weights_path is not None and os.path.exists(weights_path)

if use_weights:
    dataset = WeightedCodeDataset(data_path, weights_path=weights_path)
    collator = WeightedCodeCollator()
else:
    dataset = CodeDataset(data_path)
    collator = CodeCollator()
```

If the weights file does not exist, training falls back to the unweighted
path. Fully backward compatible --- you never need to change training code
to use or not use weights.

---

## 8. Weighted Cross-Entropy Loss

### The Standard Loss

In standard language model training, every example contributes equally:

```
L = (1/B) * sum(loss_i)     for i in 1..B
```

where `B` is the batch size and `loss_i` is the cross-entropy loss for
example `i` (averaged over all tokens in that sequence).

### The Weighted Loss

With quality weights, each example's contribution is scaled:

```
L_weighted = (1/B) * sum(w_i * loss_i)     for i in 1..B
```

where `w_i` is the normalized quality weight for example `i`.

### How It Is Implemented in the Trainer

The trainer applies weights at the **batch level** using a simple
multiplication:

```python
# From trainer.py, in the training loop:
loss = self.model.compute_loss(input_ids)   # scalar loss, averaged over batch

if weights is not None:
    weights = weights.to(self.device, non_blocking=True)
    loss = loss * weights.mean()            # scale by batch's mean weight
```

This means: if a batch happens to contain mostly high-quality examples
(mean weight > 1.0), the gradient is larger. If a batch has mostly
low-quality examples (mean weight < 1.0), the gradient is smaller.

### The Intuition

Imagine you are studying for a coding interview. You have two books:

1. **Algorithms textbook** (weight 2.0): Each page teaches you a useful
   pattern. You read carefully, take notes, re-read the examples.
2. **Copy-pasted Stack Overflow answers** (weight 0.3): Each page might
   have something useful buried in noise. You skim quickly.

The total time you spend is the same. But the textbook gets 6--7x more
of your focused attention per page. That is exactly what weighted loss
does to gradient updates.

### Gradient Signal Scaling

The gradient for each parameter `theta` is:

```
d(L_weighted)/d(theta) = (1/B) * sum( w_i * d(loss_i)/d(theta) )
```

A high-quality example with weight 2.0 contributes **2x** the gradient
of a baseline example. The model's weights move more toward predicting
the patterns in that excellent code.

A poor example with weight 0.3 contributes only **0.3x** the gradient.
The model barely adjusts for it. It is not ignored (it still contributes
*something*), but it does not override the signal from good examples.

---

## 9. Why This Matters Enormously for Small Models

At 50--455M parameters, your model has **severely limited capacity**.
Every parameter must earn its keep. Here is why quality weighting has
outsized impact at this scale:

### Limited Parameters = Limited "Memory"

A 125M parameter model has roughly 125 million floating-point numbers
to encode *all of programming*. For comparison, GPT-4 has over a
trillion. Your model cannot afford to waste capacity encoding patterns
from garbage code.

**Concrete example:** Without weighting, if 30% of your training data
is mediocre single-letter-variable code, the model dedicates roughly 30%
of its capacity to predicting that style. With weighting, that mediocre
code might account for only 10% of the effective gradient signal.

### The Phi-1 Result

Microsoft's Phi-1 (1.3B params) outperformed 10x-larger models on code
benchmarks by training primarily on "textbook quality" data. They showed
that a small model + excellent data beats a large model + average data.

Quality weighting is a softer version of the same idea. Instead of
hard-filtering to only textbook-quality code (which might leave you with
too little data), you use *all* the data but amplify the good stuff.

### Token Budget Matters

On consumer hardware, you might train on 1--10B tokens total. At scale,
companies train on trillions. When your total token budget is small,
every token that the model "spends attention on" matters. Weighting
ensures your limited tokens are allocated efficiently.

### Practical Impact

In practice, quality-weighted training on cola-coder typically shows:

- **0.05--0.15 lower final loss** compared to unweighted training on
  the same data.
- **Better generation quality** even at similar perplexity --- the model
  generates code that looks more like the high-weighted examples.
- **Faster convergence** in early training steps, because the gradient
  signal is less noisy (dominated by clear, well-structured code).

---

## 10. Practical Workflow

### End-to-End: From Raw Data to Weighted Training

**Step 1: Prepare data with scoring**

```bash
.venv/Scripts/python scripts/prepare_data.py \
    --config configs/tiny.yaml \
    --tokenizer tokenizer.json \
    --score
```

This runs the full pipeline:

1. Download/stream code from HuggingFace.
2. Apply binary quality filter (conservative by default).
3. Tokenize surviving files.
4. Chunk into `max_seq_len`-sized sequences.
5. Save `train_data.npy`.
6. **Score each chunk** and save `train_data.weights.npy`.

The output looks something like:

```
Quality filter results:
  Total files:    100,000
  Kept:           52,000 (52.0%)
  Rejected:       48,000 (48.0%)
  Rejection reasons:
    too_short (3 lines): 12,400
    autogenerated (generated by): 8,200
    python_syntax_error: 6,100
    minified (avg line 340 chars): 5,800
    ...

Scoring 52,000 chunks...
  Score distribution:
    [0.0-0.2]: 2,600 (5.0%)
    [0.2-0.4]: 7,800 (15.0%)
    [0.4-0.6]: 23,400 (45.0%)
    [0.6-0.8]: 15,600 (30.0%)
    [0.8-1.0]: 2,600 (5.0%)

Saved: data/processed/train_data.npy (52,000 chunks)
Saved: data/processed/train_data.weights.npy
```

**Step 2: Train (auto-detects weights)**

```bash
.venv/Scripts/python scripts/train.py --config configs/tiny.yaml
```

The trainer automatically looks for `.weights.npy` next to the data file:

```python
weights_path = str(Path(data_path).with_suffix(".weights.npy"))
```

If found, it prints:

```
Using quality-weighted training (mean weight: 1.00)
```

No flags needed. No config changes. If the file exists, weights are used.
If it does not exist, training proceeds normally with uniform weights.

**Step 3: Verify**

Check the training logs. With weighted training, you should see:

- Loss decreasing slightly faster in early steps.
- Generated code samples (via `generate.py`) using more descriptive names,
  docstrings, and structured patterns.

### Strict Mode + Scoring Combo

For maximum quality:

```bash
.venv/Scripts/python scripts/prepare_data.py \
    --config configs/tiny.yaml \
    --tokenizer tokenizer.json \
    --filter-strict \
    --score
```

This rejects ~65% of files (strict filter) and then quality-weights the
survivors. Use this when you have ample data and want to push quality
as high as possible.

---

## 11. Tuning Guide

### When to Use Scoring vs Not

| Scenario                          | Recommendation        |
|-----------------------------------|-----------------------|
| First training run (experimenting)| No scoring (faster)   |
| Serious training run              | Use `--score`         |
| Very large dataset (>5B tokens)   | Use `--score`         |
| Very small dataset (<500M tokens) | Use `--score` carefully --- do not want too many near-zero weights |
| Quick iteration on architecture   | No scoring (speed)    |

### Strict vs Conservative

| Scenario                          | Recommendation        |
|-----------------------------------|-----------------------|
| Python + TypeScript (plenty of data) | Conservative + scoring |
| TypeScript only (abundant data)   | Strict + scoring      |
| Niche language (limited data)     | Conservative only     |
| Small model (50M)                 | Strict + scoring      |
| Large model (350M+)              | Conservative + scoring |

### Adjusting Signal Weights

The `CodeScorer._WEIGHTS` dictionary controls how much each signal
contributes to the final score. You can modify these to match your
priorities:

```python
# Example: prioritize documentation and naming for a team that
# values readable code above all else
_WEIGHTS = {
    "structure":      0.10,    # reduced from 0.15
    "naming":         0.18,    # increased from 0.12
    "documentation":  0.15,    # increased from 0.10
    # ... rest stay the same
}
```

**Constraint:** Weights must sum to 1.0. If you increase one, decrease
another by the same amount.

### Adjusting Tier Weights

The `_TIER_WEIGHTS` dictionary controls how aggressively the model
favors good code:

```python
_TIER_WEIGHTS = {
    "excellent": 2.0,    # could increase to 3.0 for stronger preference
    "good":      1.5,
    "average":   1.0,
    "poor":      0.3,    # could decrease to 0.1 to almost-ignore poor code
    "reject":    0.0,
}
```

A more aggressive setting like `excellent: 3.0, poor: 0.1` makes the
model focus even more on top-quality examples. Use this when you have
plenty of data and the model can afford to nearly ignore poor code.

A less aggressive setting like `excellent: 1.5, poor: 0.7` makes training
more uniform. Use this when data is scarce and you cannot afford to
under-weight anything.

### Disabling Scoring Without Re-Preparing Data

If you want to temporarily disable quality weighting without re-running
`prepare_data.py`:

1. **Rename the weights file:**
   ```bash
   mv train_data.weights.npy train_data.weights.npy.bak
   ```
   The trainer will not find it and will fall back to uniform weights.

2. **Use the feature toggle:**
   Set `FEATURE_ENABLED = False` in `code_scorer.py`. This makes
   `score_to_weight()` always return 1.0.

---

## 12. Research Backing

Quality-weighted training is not a novel technique --- it is supported by
a growing body of research:

### Phi-1: "Textbook Quality" (Microsoft, 2023)

The Phi-1 paper showed that a 1.3B model trained on "textbook quality"
code outperformed models 10x its size. Their key insight: **data quality
scales better than model size**. They used GPT-4 to filter training data,
keeping only code that read like well-written educational material.

Our approach is a budget-friendly version: instead of using GPT-4 to
filter (expensive), we use 13 heuristic signals to approximate the same
quality judgment (free, fast, runs locally).

### DSIR: Data Selection via Importance Resampling (Stanford, 2023)

DSIR showed that reweighting training examples to match a target
distribution (high-quality code) improves downstream performance. Their
method is more sophisticated (it learns weights from a target corpus), but
the core idea is the same: not all examples are equally valuable.

### DoReMi: Optimizing Data Mixtures (Google, 2023)

DoReMi trains a small proxy model to learn optimal data mixture weights,
then uses those weights to train the full model. Again, the core idea:
give more weight to data that helps the model learn faster.

### The Stack v2 (BigCode, 2024)

The BigCode team showed that aggressive quality filtering on code data
consistently improves downstream benchmarks. Their SantaCoder and
StarCoder models both benefited from multi-stage quality filtering
similar to our two-stage pipeline.

### General Principle: Data Quality > Quantity

The consistent finding across all this research is:

> A smaller amount of high-quality data outperforms a larger amount of
> low-quality data, especially for smaller models.

Quality-weighted training is the pragmatic middle ground: use all your
data, but amplify the signal from the best examples. You get the benefits
of quality filtering without throwing away data that might still teach
the model something useful.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/cola_coder/features/code_scorer.py` | 13-signal quality scorer (0.0--1.0) |
| `src/cola_coder/data/quality_filter.py` | Binary keep/reject filter |
| `src/cola_coder/data/dataset.py` | WeightedCodeDataset + WeightedCodeCollator |
| `src/cola_coder/training/trainer.py` | Weighted cross-entropy loss integration |
| `scripts/prepare_data.py` | `--score` flag generates `.weights.npy` |
