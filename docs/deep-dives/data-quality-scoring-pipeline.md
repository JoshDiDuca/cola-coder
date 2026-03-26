# Data Quality Scoring Pipeline: Five Scorers, One Score, Better Models

Not all code teaches your model the same lessons. A file with clean TypeScript
interfaces, comprehensive JSDoc, and zero type errors is worth far more to a
code generation model than a file with 47 eslint warnings and a `// TODO: fix
this` comment where the type annotations should be. The scoring pipeline makes
this value judgment automatically, at scale, across millions of files.

The Seed-Coder paper demonstrated this dramatically: training on just **21% of
the original data** (the highest-quality slice) produced a model that
outperformed the one trained on 100% of the data by **3.7x** on coding
benchmarks. Quality beats quantity. Every time.

This document covers the complete scoring pipeline: the protocol that all
scorers share, each of the five scorers in detail, the composite weighting
system, curriculum ordering, and how to add your own scorer.

**TypeScript analogy:** Think of the scoring pipeline as a CI check with five
reporters. Each reporter (tsc, eslint, a heuristic linter, GitHub stars, and a
trained classifier) writes a report card. A composite function reads all five
report cards, weights them by importance, and produces a single GPA. That GPA
determines how much the model studies each file.

---

## Table of Contents

1. [Why Scoring Matters](#1-why-scoring-matters)
2. [The ScorerProtocol --- Unified Interface](#2-the-scorerprotocol--unified-interface)
3. [CompositeScorer --- Weighted Combination](#3-compositescorer--weighted-combination)
4. [The Five Scorers in Detail](#4-the-five-scorers-in-detail)
5. [LLM-as-Judge Workflow](#5-llm-as-judge-workflow)
6. [Curriculum Ordering](#6-curriculum-ordering)
7. [Score Storage --- .weights.npy and .scores.jsonl](#7-score-storage--weightsnpy-and-scoresjsonl)
8. [Configuration in scoring.yaml](#8-configuration-in-scoringyaml)
9. [CLI --- score_data.py and train_judge_classifier.py](#9-cli--score_datapy-and-train_judge_classifierpy)
10. [Pipeline Integration --- Where Scoring Fits](#10-pipeline-integration--where-scoring-fits)
11. [Adding a Custom Scorer](#11-adding-a-custom-scorer)
12. [Common Mistakes and Debugging](#12-common-mistakes-and-debugging)

---

## 1. Why Scoring Matters

When you train a language model, every token in every training example pushes
the model weights by the same amount. A beautifully typed TypeScript utility
with proper error handling and comprehensive JSDoc gets exactly the same
gradient influence as a file that starts with `// @ts-nocheck` and ends with
five bare `catch {}` blocks.

That is wasteful. Your model has limited capacity --- especially at the 50M to
150M parameter scale where cola-coder operates. Every gradient step spent
learning from sloppy code is a step not spent learning from exemplary code.

Quality scoring fixes this by assigning each training example a weight between
0.0 and 2.0:

```
Excellent code (score >= 0.8) --> weight 2.0  (model studies it hard)
Good code      (score >= 0.6) --> weight 1.5  (model pays attention)
Average code   (score >= 0.4) --> weight 1.0  (normal training)
Poor code      (score >= 0.2) --> weight 0.3  (model barely glances)
Reject         (score <  0.2) --> weight 0.0  (skipped entirely)
```

The weighted cross-entropy loss scales each example's contribution to the
gradient. The math is simple: multiply each example's loss by its weight before
averaging. The effect is profound: the model converges faster, produces
higher-quality completions, and uses its limited parameters more efficiently.

**Real numbers from cola-coder experiments:** On the StarCoderData TypeScript
subset (~2.3M files), scoring with the default 4-scorer configuration
(tsc + eslint + stars + heuristic) produces this distribution:

```
Tier         % of data    Training weight
---------    ---------    ---------------
excellent       ~8%       2.0x (16% of gradient)
good           ~25%       1.5x (37.5% of gradient)
average        ~40%       1.0x (40% of gradient)
poor           ~20%       0.3x (6% of gradient)
reject          ~7%       0.0x (0% of gradient)
```

The top 33% of data (excellent + good) contributes over 53% of the gradient
signal. The bottom 27% (poor + reject) contributes just 6%. Your model spends
its capacity where it matters.

---

## 2. The ScorerProtocol --- Unified Interface

**File:** `src/cola_coder/data/scorers/protocol.py`

Every scorer in the pipeline implements the same three-method interface. This
is Python's structural typing --- the same concept as TypeScript interfaces,
enforced at runtime with `@runtime_checkable`.

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

**TypeScript equivalent:**

```typescript
interface Scorer {
  name: string;
  score(code: string, metadata?: Record<string, unknown>): ScorerResult;
  scoreBatch(items: [string, Record<string, unknown> | null][]): ScorerResult[];
  isAvailable(): boolean;  // static in Python, but you get the idea
}
```

### The Three Methods

**`score(code, metadata)`** --- Score a single code sample. Returns a
`ScorerResult` with a normalized 0.0--1.0 score, the scorer's name, and an
optional details dict for debugging.

**`score_batch(items)`** --- Score multiple samples. The default naive
implementation just loops over `score()`, but scorers like `EslintScorer` and
`TscScorer` override this to run the tool once on all files simultaneously.
This is a huge performance win --- tsc startup time is ~500ms, so scoring 100
files individually takes 50 seconds, while batching takes ~2 seconds.

**`is_available()`** --- Check if the scorer can run. TscScorer checks if `tsc`
is on the PATH. EslintScorer checks for `eslint` or `npx`. StarsScorer always
returns True (it is pure Python). The registry uses this to silently skip
unavailable scorers instead of crashing.

### ScorerResult

Every scorer returns the same result type:

```python
@dataclass
class ScorerResult:
    score: float            # 0.0 - 1.0 normalized
    scorer_name: str        # e.g. "tsc", "eslint", "stars"
    details: dict[str, object] = field(default_factory=dict)
```

The `details` dict is scorer-specific. TscScorer puts error codes and messages
in there. EslintScorer puts warning and error counts. StarsScorer puts the raw
star count. This is purely for debugging and the `.scores.jsonl` detail log ---
only the `score` float matters for the composite.

---

## 3. CompositeScorer --- Weighted Combination

**File:** `src/cola_coder/data/scorers/protocol.py`

The `CompositeScorer` takes multiple scorers, each with a weight, and produces
a single 0.0--1.0 overall score. It then maps that score to a training weight
via the tier system.

```
 TscScorer (weight=0.3)     ----\
 EslintScorer (weight=0.2)  -----+--> CompositeScorer --> overall: 0.72
 StarsScorer (weight=0.15)  -----+                        tier: "good"
 HeuristicScorer (weight=0.2)---/                         weight: 1.5
```

### How Weights Work

Scorer weights are normalized to sum to 1.0. If you configure:

```yaml
scorers:
  tsc:      { weight: 0.3 }
  eslint:   { weight: 0.2 }
  stars:    { weight: 0.15 }
  heuristic: { weight: 0.2 }
```

The total is 0.85. After normalization: tsc=0.353, eslint=0.235, stars=0.176,
heuristic=0.235. This means you can think of weights as relative importance ---
doubling tsc's weight makes it twice as influential regardless of what the
other weights are.

```python
# From CompositeScorer.__init__:
total = sum(w for _, w in self._scorers)
self._scorers = [(s, w / total) if total > 0 else (s, 0.0) for s, w in self._scorers]
```

### The Tier-to-Weight Mapping

The 0.0--1.0 composite score maps to a training weight via five tiers:

```python
DEFAULT_TIER_WEIGHTS = {
    "excellent": 2.0,   # score >= 0.8
    "good": 1.5,        # score >= 0.6
    "average": 1.0,     # score >= 0.4
    "poor": 0.3,        # score >= 0.2
    "reject": 0.0,      # score < 0.2
}
```

**Why tiers instead of a continuous function?** Two reasons:

1. **Interpretability.** "This file is in the 'good' tier with weight 1.5" is
   immediately understandable. "This file has weight 1.347" is not.

2. **Stability.** Small score fluctuations near tier boundaries can change the
   weight significantly, but within a tier the weight is constant. This makes
   the system less sensitive to noise in individual scorers.

The mapping is configurable in `scoring.yaml`:

```yaml
tier_weights:
  excellent: 2.0
  good: 1.5
  average: 1.0
  poor: 0.3
  reject: 0.0
```

### Batch Scoring

`CompositeScorer.score_batch()` calls each scorer's `score_batch()` method,
then combines results per-item. This lets tool-based scorers (tsc, eslint)
batch their subprocess calls while pure-Python scorers (stars, heuristic)
iterate normally.

```python
def score_batch(self, items):
    # Each scorer processes the full batch its own way
    all_results = {}
    for scorer, _ in self._scorers:
        all_results[scorer.name] = scorer.score_batch(items)

    # Combine per-item
    results = []
    for i in range(len(items)):
        overall = sum(
            all_results[scorer.name][i].score * weight
            for scorer, weight in self._scorers
        )
        overall = max(0.0, min(1.0, overall))
        results.append(CompositeResult(
            overall=overall,
            per_scorer={...},
            weight=self._score_to_weight(overall),
        ))
    return results
```

---

## 4. The Five Scorers in Detail

### 4.1 TscScorer --- TypeScript Compiler Checking

**File:** `src/cola_coder/data/scorers/tsc_scorer.py`

The TypeScript compiler is the gold standard for TypeScript code quality. If
tsc reports zero errors with `--strict` enabled, the code has correct types,
no implicit `any`, no unused locals, and no unreachable code. This is the
single most informative signal for TypeScript training data.

**How it works:**

1. Check if the code is TypeScript (via `language_detect.is_typescript()`).
   Non-TypeScript files get a neutral 0.5 score and are skipped.
2. Delegate to `TscRunner.check(code)`, which:
   - Creates a temp directory
   - Writes the code as `check.ts`
   - Writes a **hardened tsconfig.json** (plugins=[], types=[], typeRoots=[])
   - Runs `tsc --project . --pretty false` through `SandboxedRunner`
   - Parses the error output into structured `TscError` objects
3. Map the error count to a score via `ScoreMapper`.

**Score mapping:**

```python
_TSC_SCORE_MAP = ScoreMapper([
    (0, 1.0),     # No errors = perfect
    (1, 0.8),     # 1 error = good
    (3, 0.6),     # 2-3 errors = decent
    (5, 0.4),     # 4-5 errors = average
    (10, 0.2),    # 6-10 errors = poor
])
# 10+ errors = 0.1 (fallback)
```

**Syntax error penalty:** If any error has a TS1xxx code (syntax errors), the
score is capped at 0.3 regardless of the total count. A single syntax error
is worse than five type errors because it means the code does not parse at all.

```python
has_syntax = any(e.code.startswith("TS1") for e in errors)
if has_syntax:
    score = min(score, 0.3)
```

**Batch optimization:** `score_batch()` separates TypeScript and non-TypeScript
files, sends only the TypeScript files to `TscRunner.check_batch()`, and
reassembles results. The batch check writes all files to a single temp
directory and runs tsc once, which is dramatically faster than individual
invocations.

**What the error codes mean:**

| Code Range | Category | Example |
|------------|----------|---------|
| TS1xxx | Syntax errors | TS1005: `;` expected |
| TS2xxx | Semantic errors | TS2322: Type 'string' not assignable to 'number' |
| TS6xxx | Compiler config | TS6133: Declared but never read |
| TS7xxx | Strict mode | TS7006: Parameter implicitly has 'any' type |

**TypeScript analogy:** This scorer is literally running tsc on your training
data. If you would reject a PR with 10 type errors, your model should not be
studying that code intensely either.

### 4.2 EslintScorer --- Linting Quality

**File:** `src/cola_coder/data/scorers/eslint_scorer.py`

ESLint catches a broader category of issues than tsc: unused variables, missing
return types, inconsistent formatting, unreachable code, complexity warnings,
and style violations. Where tsc checks "does this compile?", ESLint checks
"is this well-written?"

**How it works:**

1. Check if the code is JavaScript or TypeScript.
2. Write the file to a temp directory with the correct extension (.ts, .tsx,
   .js, .jsx).
3. Run `eslint --format json --no-eslintrc <files>` through `SandboxedRunner`.
   The `--no-eslintrc` flag uses ESLint's default rules, ensuring consistent
   scoring regardless of per-project configs.
4. Parse the JSON output to extract error and warning counts.
5. Map total issue count to a score.

**Score mapping:**

```python
_ESLINT_SCORE_MAP = ScoreMapper([
    (0, 1.0),     # 0 issues = perfect
    (2, 0.9),     # 1-2 issues = great
    (5, 0.7),     # 3-5 issues = good
    (10, 0.5),    # 6-10 issues = average
    (20, 0.3),    # 11-20 issues = poor
])
# 20+ issues = 0.1 (fallback)
```

**Batch optimization --- the key performance trick:** Unlike `score()` which
creates one file per invocation, `score_batch()` writes ALL files to a single
temp directory and runs ESLint once with all file paths. ESLint's JSON output
includes per-file results, so we parse them back to individual scores.

```python
def score_batch(self, items):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_map = {}
        for i, (code, metadata) in enumerate(items):
            filename = f"file_{i}{ext}"
            Path(tmpdir, filename).write_text(code)
            file_map[str(filepath)] = i

        # One ESLint invocation for the entire batch
        eslint_result = self._run_eslint(tmpdir, list(file_map.keys()))
```

This takes ESLint from ~1 second per file to ~2 seconds per 100 files.

**Fallback strategy:** The scorer tries `eslint` directly first, then falls
back to `npx eslint`. This handles both global and local ESLint installations.
If both fail, the file gets a neutral 0.5 score.

### 4.3 StarsScorer --- Repository Popularity

**File:** `src/cola_coder/data/scorers/stars_scorer.py`

This is the simplest scorer and the only one that does not look at the code
itself. Instead, it uses the repository's GitHub star count as a quality proxy.

The intuition: code from popular, well-maintained repositories (React, Zod,
tRPC) tends to be higher quality than code from abandoned hobby projects with
zero stars. Stars are an imperfect proxy --- there are excellent zero-star
libraries and terrible popular ones --- but as a signal averaged across millions
of files, it is surprisingly useful.

**Log-scale normalization:**

Stars follow a heavy power law. Most repos have 0--5 stars. A few have
100,000+. Linear normalization would make everything look the same except the
top 0.01%. Log scale compresses this:

```python
@staticmethod
def _normalize_stars(stars: int) -> float:
    """
    0      -> 0.10
    1      -> 0.10
    10     -> 0.30
    100    -> 0.50
    1000   -> 0.80
    10000  -> 1.00
    100000 -> 1.00
    """
    if stars <= 0:
        return 0.1
    log_stars = math.log10(max(stars, 1))
    if log_stars <= 1:       # 1-10 stars
        return 0.1 + (log_stars / 1.0) * 0.2
    elif log_stars <= 2:     # 10-100 stars
        return 0.3 + ((log_stars - 1) / 1.0) * 0.2
    elif log_stars <= 3:     # 100-1000 stars
        return 0.5 + ((log_stars - 2) / 1.0) * 0.3
    elif log_stars <= 4:     # 1000-10000 stars
        return 0.8 + ((log_stars - 3) / 1.0) * 0.2
    else:
        return 1.0
```

**Fallback for missing data:** Many HuggingFace datasets (like StarCoderData
parquet files) do not include star counts. When `repo_stars` is missing from
metadata, the scorer returns a configurable default score (0.3 by default).
This puts starless files in the "poor-to-average" range --- they are not
penalized heavily, but they do not get a free pass either.

```yaml
scorers:
  stars:
    enabled: true
    weight: 0.15
    default_score: 0.3   # Fallback when no star data available
```

**`is_available()` always returns True.** No external tools needed --- this is
pure Python math on metadata fields.

### 4.4 HeuristicScorer --- 13-Signal Code Analysis

**File:** `src/cola_coder/data/scorers/heuristic_scorer.py`

This scorer wraps the original `CodeScorer` (from `features/code_scorer.py`)
--- the same 13-signal analysis described in the
[Quality-Weighted Training deep dive](quality-weighted-training.md). It
analyzes:

1. **Length** (5%) --- sweet spot 10--300 lines
2. **Line Quality** (5%) --- average and max line length
3. **Structure** (15%) --- functions, classes, imports
4. **Naming** (12%) --- convention consistency, descriptiveness
5. **Comments** (10%) --- comment ratio sweet spot 5--25%
6. **Documentation** (10%) --- docstrings, JSDoc blocks
7. **Complexity** (8%) --- control flow density
8. **Formatting** (5%) --- blank lines, indentation consistency
9. **Duplication** (8%) --- internal copy-paste detection
10. **Syntax** (10%) --- ast.parse / brace balance
11. **Modernness** (5%) --- modern vs deprecated patterns
12. **Error Handling** (4%) --- try/catch quality
13. **Security** (3%) --- hardcoded secrets, eval, SQL injection

The HeuristicScorer is a thin adapter --- it delegates everything to
`CodeScorer` and wraps the result in a `ScorerResult`:

```python
class HeuristicScorer:
    name: str = "heuristic"

    def score(self, code, metadata=None):
        language = str(metadata.get("language", "")) if metadata else ""
        result = self._get_scorer().score(code, language)
        return ScorerResult(
            score=result.overall,
            scorer_name=self.name,
            details={"tier": result.tier, "breakdown": result.breakdown},
        )
```

The `breakdown` dict in the details contains all 13 individual signal scores,
which is invaluable for debugging why a file scored low.

**`is_available()` checks for the import.** If `CodeScorer` is not importable
(unlikely in a normal installation), the scorer is skipped.

### 4.5 ClassifierScorer --- Distilled LLM Judgment

**File:** `src/cola_coder/data/scorers/classifier.py`

This is the most interesting scorer architecturally. It uses a TF-IDF +
logistic regression model that was trained on LLM-generated annotations.
Think of it as distilling Claude's (or CodeLlama's) code quality judgment
into a tiny, fast classifier that runs in microseconds.

**How the classifier works:**

1. **TF-IDF vectorization** converts code to a sparse feature vector. The
   vectorizer uses unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear
   term frequency, capturing up to 10,000 features.

2. **Logistic regression** predicts a quality class (0--5) from the TF-IDF
   features. The `class_weight="balanced"` parameter handles the imbalanced
   distribution (most code is average).

3. **Score normalization** divides the 0--5 prediction by 5.0 to get a
   0.0--1.0 score.

```python
def predict(self, code: str) -> float:
    X = self._vectorizer.transform([code])
    pred = self._model.predict(X)[0]
    return float(pred) / 5.0

def predict_batch(self, codes: list[str]) -> list[float]:
    X = self._vectorizer.transform(codes)
    preds = self._model.predict(X)
    return [float(p) / 5.0 for p in preds]
```

**Why this works:** The LLM sees patterns that heuristic rules miss ---
idiomatic usage, API design quality, code organization that is not captured by
simple metrics. By distilling thousands of LLM judgments into a classifier,
you get most of the LLM's quality signal at a fraction of the cost.

**Disabled by default.** The classifier requires a trained model. You must run
the annotation and training pipeline first (see Section 5). Once trained,
enable it in config:

```yaml
scorers:
  classifier:
    enabled: true     # Enable after training
    weight: 0.15
    model_dir: "models/quality_classifier"
```

---

## 5. LLM-as-Judge Workflow

**Files:** `src/cola_coder/data/scorers/llm_judge.py`,
`scripts/train_judge_classifier.py`

The LLM-as-Judge workflow is a three-stage process that converts expensive
LLM quality assessments into a fast classifier:

```
Stage 1: Annotate      (slow, expensive --- LLM scores 10K samples)
    |
    v
Stage 2: Train         (fast --- TF-IDF + LR on annotations)
    |
    v
Stage 3: Predict       (very fast --- classifier scores millions of files)
```

### Stage 1: Annotation

The `LlmJudge` class sends code samples to an LLM (Claude API or local
Ollama) with a structured prompt that asks for a 0--5 quality rating:

```
Rate this TypeScript code on a scale of 0-5 for training data quality.

0 = Garbage: auto-generated, minified, data dump, or broken
1 = Very poor: no structure, terrible naming, no documentation
2 = Poor: some structure but significant quality issues
3 = Average: functional, reasonable structure, some documentation
4 = Good: well-structured, good naming, clear documentation
5 = Excellent: production-quality, idiomatic, educational value

Reply in exactly this format:
Score: <0-5>
Reason: <one sentence>
```

The annotation batch is **resume-capable** --- it tracks which code hashes have
already been annotated, so you can interrupt and resume without re-processing.

**Security:** Before sending code to external APIs, the `CredentialScanner`
processes the code according to the configured mode (strip secrets, reject
files with secrets, or warn). This prevents leaking API keys and passwords from
training data to LLM providers.

### Stage 2: Training

The `QualityClassifierTrainer` trains a TF-IDF + logistic regression pipeline
on the annotations:

```python
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
model = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    class_weight="balanced",
)
```

It saves `vectorizer.pkl`, `model.pkl`, and `meta.json` to the output
directory. The meta file includes accuracy and MAE metrics for verification.

### Stage 3: Inference

`ClassifierScorer` loads the trained model and runs inference as part of the
composite scorer. Because it is just TF-IDF + matrix multiplication, it scores
thousands of files per second --- orders of magnitude faster than calling an LLM.

---

## 6. Curriculum Ordering

**File:** `src/cola_coder/data/scorers/curriculum.py`

Curriculum learning is the idea that the order of training data matters.
Research from Arctic-SnowCoder shows that training on high-quality data first,
then progressively introducing lower-quality data, improves model performance
compared to random ordering.

### Four Strategies

```python
class CurriculumStrategy(str, Enum):
    EASY_TO_HARD = "easy_to_hard"     # High quality first (recommended)
    HARD_TO_EASY = "hard_to_easy"     # Low quality first
    STAGED = "staged"                  # Split into N quality phases
    RANDOM = "random"                  # Shuffle (baseline)
```

**easy_to_hard (recommended):** Sort by weight descending. The model sees
excellent code first, learns the patterns, then encounters progressively
messier code. The intuition: learn the rules before the exceptions.

**hard_to_easy:** Sort by weight ascending. Occasionally useful for curriculum
research, but not recommended for production training.

**staged:** Split data into N phases by quality. Phase 1 is the top third,
phase 2 the middle, phase 3 the bottom. Each phase gets its own `.npy` file.
This lets you control exactly how many epochs of each quality tier the model
sees.

**random:** Baseline shuffle. Useful for A/B comparisons.

### What Happens When You Enable Curriculum

```bash
python scripts/score_data.py \
  --data train_data.npy \
  --tokenizer tokenizer.json \
  --curriculum easy_to_hard
```

The orderer:

1. Loads `train_data.npy` and `train_data.weights.npy`.
2. Sorts both arrays by weight (descending for easy_to_hard).
3. Overwrites both files with the reordered data.
4. Saves a `train_data.curriculum.json` schedule file with phase boundaries.

The schedule file looks like:

```json
{
  "strategy": "easy_to_hard",
  "total_samples": 500000,
  "phases": [
    {
      "phase": 1,
      "start_idx": 0,
      "end_idx": 166666,
      "num_samples": 166666,
      "mean_score": 1.78,
      "min_score": 1.5,
      "max_score": 2.0
    },
    {
      "phase": 2,
      "start_idx": 166666,
      "end_idx": 333333,
      "num_samples": 166667,
      "mean_score": 0.95,
      "min_score": 0.3,
      "max_score": 1.5
    },
    {
      "phase": 3,
      "start_idx": 333333,
      "end_idx": 500000,
      "num_samples": 166667,
      "mean_score": 0.08,
      "min_score": 0.0,
      "max_score": 0.3
    }
  ]
}
```

**TypeScript analogy:** Curriculum ordering is like structuring a coding
bootcamp. Week 1: clean, well-documented examples. Week 2: typical production
code with some warts. Week 3: legacy code with technical debt. Students learn
the ideal patterns first, then learn to recognize and navigate imperfect code.

---

## 7. Score Storage --- .weights.npy and .scores.jsonl

The scoring pipeline produces two output files:

### .weights.npy --- The Sidecar File

A NumPy array of float32 values, one per training chunk. This is the file that
the training loop actually uses.

```
data/processed/
    train_data.npy             # [num_chunks, chunk_size] uint16 token IDs
    train_data.weights.npy     # [num_chunks] float32 quality weights
```

- **Shape:** `[num_chunks]`
- **Dtype:** float32
- **Values:** 0.0--2.0 (from tier mapping)
- **Size:** ~400 KB for 100K chunks

The weights file must have exactly the same number of entries as rows in the
data file. A mismatch causes `WeightedCodeDataset` to raise a `ValueError`
at training startup --- fail-fast, never silent corruption.

### .scores.jsonl --- The Detail Log

A JSONL file with per-chunk scoring breakdowns. Each line is a JSON object:

```json
{"index": 0, "overall": 0.7234, "weight": 1.5, "tsc": 1.0, "eslint": 0.7, "stars": 0.5, "heuristic": 0.65}
{"index": 1, "overall": 0.4112, "weight": 1.0, "tsc": 0.4, "eslint": 0.3, "stars": 0.3, "heuristic": 0.55}
```

This file is for debugging and analysis. You can use it to:

- Find which scorer is dragging scores down
- Verify that the tier distribution looks reasonable
- Spot anomalies (e.g., all tsc scores are 0.5, meaning tsc is not available)
- Compare scorer agreement

**The scores.jsonl file is not used by training.** Only `.weights.npy` matters
at training time. You can safely delete `.scores.jsonl` after analysis.

---

## 8. Configuration in scoring.yaml

**File:** `configs/scoring.yaml`

The complete scoring configuration lives in a single YAML file:

```yaml
scoring:
  # Security settings (see security-architecture.md)
  security:
    mode: "native"              # off | native | docker
    require_docker: false
    timeout: 10
    memory_mb: 512
    docker_image: "node:20-alpine"
    audit_log: "logs/scoring_audit.jsonl"
    credential_scan:
      mode: "strip"             # off | warn | strip | reject

  # Which scorers to enable and their composite weights
  scorers:
    tsc:
      enabled: true
      weight: 0.3          # tsc is the most informative signal
      strict: true          # Enable TypeScript strict mode
      timeout: 10           # Seconds per file

    eslint:
      enabled: true
      weight: 0.2
      timeout: 15

    stars:
      enabled: true
      weight: 0.15
      default_score: 0.3   # When star data is unavailable

    heuristic:
      enabled: true
      weight: 0.2

    classifier:
      enabled: false        # Enable after training with train_judge_classifier.py
      weight: 0.15
      model_dir: "models/quality_classifier"

  # Score-to-training-weight tier mapping
  tier_weights:
    excellent: 2.0    # score >= 0.8
    good: 1.5         # score >= 0.6
    average: 1.0      # score >= 0.4
    poor: 0.3         # score >= 0.2
    reject: 0.0       # score < 0.2

# LLM-as-Judge settings
llm_judge:
  provider: "ollama"
  model: "codellama"
  base_url: "http://localhost:11434"
  num_samples: 10000
  output_path: "data/annotations.jsonl"

# Curriculum ordering
curriculum:
  enabled: false
  strategy: "easy_to_hard"
  num_phases: 3
```

### Key Configuration Decisions

**tsc weight=0.3 is the highest.** Type checking is the strongest signal for
TypeScript code quality. A file that passes strict tsc with zero errors is
almost certainly well-written.

**classifier is disabled by default.** It requires a trained model that does
not exist until you run the LLM-as-Judge pipeline. Enabling it without a
trained model causes graceful degradation (score=0.5 for all files), but that
defeats the purpose.

**stars default_score=0.3** puts starless files in the poor-to-average range.
If your dataset reliably includes star counts, you can lower this. If your
dataset never has star counts, consider disabling the stars scorer entirely
(set `weight: 0` or `enabled: false`).

---

## 9. CLI --- score_data.py and train_judge_classifier.py

### score_data.py

The primary scoring script. Supports both JSONL (raw GitHub data) and NPY
(tokenized training data).

**Score a JSONL file:**

```bash
python scripts/score_data.py --jsonl github_scraped.jsonl
```

This produces `github_scraped.weights.npy` and `github_scraped.scores.jsonl`.

**Score tokenized data:**

```bash
python scripts/score_data.py \
  --data train_data.npy \
  --tokenizer tokenizer.json
```

This decodes each chunk back to text (using the tokenizer), scores the text,
and produces the weights sidecar.

**Score with specific scorers only:**

```bash
python scripts/score_data.py \
  --data train_data.npy \
  --tokenizer tokenizer.json \
  --scorers tsc,eslint
```

**Score with curriculum ordering:**

```bash
python scripts/score_data.py \
  --data train_data.npy \
  --tokenizer tokenizer.json \
  --curriculum easy_to_hard
```

**Limit sample count for testing:**

```bash
python scripts/score_data.py \
  --jsonl data.jsonl \
  --max-samples 1000
```

### train_judge_classifier.py

Three-stage pipeline for the ClassifierScorer.

**Stage 1: Annotate with LLM:**

```bash
# Using local Ollama
python scripts/train_judge_classifier.py annotate \
  --provider ollama --model codellama \
  --data data.jsonl --num-samples 10000

# Using Claude API
python scripts/train_judge_classifier.py annotate \
  --provider claude --model claude-sonnet-4-6 \
  --data data.jsonl --num-samples 10000
```

**Stage 2: Train classifier:**

```bash
python scripts/train_judge_classifier.py train \
  --annotations data/annotations.jsonl \
  --output-dir models/quality_classifier
```

**Stage 3: Evaluate:**

```bash
python scripts/train_judge_classifier.py evaluate \
  --model-dir models/quality_classifier \
  --annotations data/annotations.jsonl
```

---

## 10. Pipeline Integration --- Where Scoring Fits

The scoring pipeline is stage 6 of the 10-stage data processing pipeline:

```
 1. Download raw data (HuggingFace / GitHub scraping)
 2. Deduplication (exact + near-duplicate removal)
 3. Quality filter (binary keep/reject -- Stage 1)
 4. Tokenization (BPE encoding)
 5. Chunking (split into max_seq_len sequences)
 6. >>> SCORING (this pipeline) <<<
 7. Weight generation (.weights.npy sidecar)
 8. Curriculum ordering (optional reorder)
 9. Training data packaging (final .npy + .weights.npy)
10. Training (WeightedCodeDataset + weighted cross-entropy)
```

Scoring happens AFTER tokenization and chunking. This means:

- **The scorer receives decoded chunks, not raw files.** Each chunk is
  `max_seq_len` tokens (~1024 tokens, roughly 3--5 KB of code).
- **A single source file may span multiple chunks.** Each chunk is scored
  independently.
- **File boundaries within chunks** mean some chunks contain code from two
  different files. The scorer handles this gracefully --- the composite score
  reflects the blended quality.

The reason scoring happens after chunking (not before) is practical: the
training loop needs one weight per chunk, not one weight per file. Scoring
before chunking would require a mapping from files to chunks, with decisions
about how to handle chunks that span file boundaries. Scoring after chunking
avoids all of this complexity.

---

## 11. Adding a Custom Scorer

Implementing a custom scorer requires three steps:

### Step 1: Implement ScorerProtocol

Create a new file in `src/cola_coder/data/scorers/`:

```python
# src/cola_coder/data/scorers/my_scorer.py
from cola_coder.data.scorers.protocol import ScorerResult


class MyScorer:
    """Custom scorer that checks for something interesting."""

    name: str = "my_scorer"

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        # Your scoring logic here
        # Must return a ScorerResult with score in 0.0-1.0
        my_score = self._analyze(code)
        return ScorerResult(
            score=my_score,
            scorer_name=self.name,
            details={"threshold": self._threshold},
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        # Default: loop over score(). Override for batch optimization.
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True  # Or check for external dependencies

    def _analyze(self, code: str) -> float:
        # Your analysis logic
        return 0.7
```

### Step 2: Register in the Registry

Add your scorer to `_instantiate_scorer()` in `registry.py`:

```python
elif name == "my_scorer":
    from cola_coder.data.scorers.my_scorer import MyScorer
    return MyScorer(
        threshold=cfg.get("threshold", 0.5),
    )
```

### Step 3: Add to scoring.yaml

```yaml
scorers:
  my_scorer:
    enabled: true
    weight: 0.15
    threshold: 0.5
```

The `build_composite_scorer()` function will pick it up automatically. Scorer
weights are normalized, so adding a new scorer does not require adjusting
existing weights --- the relative proportions stay the same.

**Testing tip:** Use `list_available_scorers()` to verify your scorer is
detected:

```python
from cola_coder.data.scorers.registry import list_available_scorers
for s in list_available_scorers():
    print(f"{s['name']}: available={s['available']}, enabled={s['enabled']}")
```

---

## 12. Common Mistakes and Debugging

### "All tsc scores are 0.5"

This means tsc is not available on your PATH. The scorer skips non-TypeScript
files (returns 0.5), but if `is_available()` returns False, the registry
skips the scorer entirely. Check:

```bash
which tsc      # Unix
where tsc      # Windows
```

If tsc is installed but not found, it might be a local `node_modules/.bin/tsc`
that is not on the global PATH. TscRunner resolves the path with
`shutil.which("tsc")`.

### "ESLint scores are all neutral (0.5)"

Same issue --- eslint is not installed. The scorer also tries `npx eslint` as a
fallback, so make sure either `eslint` or `npx` is available.

### "My classifier scorer returns 0.5 for everything"

The model has not been trained yet. Run the annotation and training pipeline
first:

```bash
python scripts/train_judge_classifier.py annotate --data data.jsonl
python scripts/train_judge_classifier.py train --annotations data/annotations.jsonl
```

Then enable it in `scoring.yaml`.

### "Scoring is very slow (>1 file/second)"

Likely causes:

1. **Not using batch mode.** The CLI scripts use batch scoring automatically,
   but if you are calling `composite.score()` in a loop, switch to
   `composite.score_batch()`.
2. **Docker mode is enabled.** Docker adds ~2 seconds overhead per invocation.
   For development, use `security.mode: native`.
3. **tsc or eslint timeout.** Check if specific files cause tsc to hang (some
   pathological TypeScript files trigger exponential type inference). The
   timeout defaults to 10 seconds --- if many files hit this, lower it to 5.

### "Data and weights length mismatch"

`WeightedCodeDataset` raises this when `train_data.npy` has a different number
of rows than `train_data.weights.npy`. This happens when you re-run
preprocessing (creating new chunks) without re-running scoring. Always re-score
after re-chunking.

### "Scorer weights do not add up to 1.0"

They do not need to. CompositeScorer normalizes them automatically. If you
configure weights of 0.3, 0.2, 0.15, 0.2 (total 0.85), they become 0.353,
0.235, 0.176, 0.235 (total 1.0). Think of config weights as relative
importance, not absolute fractions.

### What Happens When a Scorer Crashes

If a scorer raises an exception during `score()`, it does NOT crash the
pipeline. The `_instantiate_scorer()` function in the registry catches import
errors and returns None. Individual scorer `score()` methods handle their own
exceptions and return neutral 0.5 scores. The CompositeScorer weights
automatically adjust to exclude missing scorers.

This fail-safe design means a broken ESLint installation does not block you
from scoring with the remaining scorers. You will see a warning in the output,
but processing continues.
