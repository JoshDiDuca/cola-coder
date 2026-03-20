# Feature 47: Thinking Quality Scorer

**Status:** Proposed
**CLI Flag:** `--thinking-quality-reward`
**Complexity:** Medium

---

## Overview

A GRPO reward component that evaluates whether a model's `<think>` trace is semantically and logically related to the generated code. Rather than rewarding the model just for passing tests, this scorer rewards *how well the model reasoned* — penalizing copy-paste thinking, irrelevant rambling, and rewarding coherent, code-aligned reasoning traces.

---

## Motivation

Without a quality signal on the thinking trace, GRPO training can produce two degenerate behaviors:

1. **Copy-paste thinking:** The model simply repeats the prompt verbatim inside `<think>`, then generates code. This wastes tokens and provides no generalization benefit.
2. **Irrelevant rambling:** The model produces plausible-sounding but unrelated reasoning ("Let me think about sorting algorithms...") regardless of the actual problem.

A quality scorer creates a direct training signal toward *grounded* reasoning — traces that mention the functions, types, and approaches actually used in the final code. This complements test-execution rewards by rewarding the reasoning process, not just the output.

Evidence from chain-of-thought research (Wei et al., 2022; Lightman et al., 2023) shows that process rewards significantly improve solution quality on hard problems, even when outcome accuracy is held constant.

---

## Architecture / Design

```
<think> trace ──┐
                ├── ThinkingQualityScorer ──► quality_score: float [0, 1]
<code> output ──┘

ThinkingQualityScorer
├── SemanticSimilarityScore   (embedding cosine)
├── KeywordOverlapScore        (function names, types)
├── LogicalCoherenceScore      (approach mention)
└── PenaltyDetector            (copy-paste, rambling)

quality_score = w1*semantic + w2*keyword + w3*coherence - penalties
```

### Score Components

| Component | Weight | Description |
|---|---|---|
| Semantic similarity | 0.35 | Cosine similarity between think and code embeddings |
| Keyword overlap | 0.30 | % of code identifiers mentioned in think |
| Logical coherence | 0.20 | Does think mention the approach used in code? |
| Copy-paste penalty | -0.40 | Think overlaps > 80% with prompt |
| Rambling penalty | -0.20 | Think has < 10% keyword overlap with code |

---

## Implementation Steps

### Step 1: Extract Think and Code Sections

```python
# src/rewards/think_extractor.py
import re
from dataclasses import dataclass

@dataclass
class ParsedGeneration:
    think: str
    code: str
    prompt: str

def parse_generation(text: str, prompt: str = "") -> ParsedGeneration:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    code_match  = re.search(r"```(?:typescript|javascript|ts|js)?\n(.*?)```", text, re.DOTALL)

    think = think_match.group(1).strip() if think_match else ""
    code  = code_match.group(1).strip()  if code_match  else text

    return ParsedGeneration(think=think, code=code, prompt=prompt)
```

### Step 2: Semantic Similarity Score

Uses a small embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`, 22M params) to compute cosine similarity between think and code.

```python
# src/rewards/semantic_similarity.py
from sentence_transformers import SentenceTransformer
import numpy as np

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def semantic_similarity_score(think: str, code: str) -> float:
    if not think or not code:
        return 0.0
    model = _get_model()
    embeddings = model.encode([think, code], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    # Map from [-1,1] to [0,1]
    return (score + 1.0) / 2.0
```

**Note:** During GRPO training, run this on CPU in a reward worker process to avoid competing with the main GPU training loop.

### Step 3: Keyword Overlap Score

Extract identifiers from the generated code and check what fraction appear in the thinking trace.

```python
# src/rewards/keyword_overlap.py
import re

def extract_identifiers(code: str) -> set[str]:
    # Match camelCase, PascalCase, snake_case identifiers
    # Exclude keywords and very short names
    KEYWORDS = {
        "const", "let", "var", "function", "return", "if", "else",
        "for", "while", "class", "interface", "type", "import", "export",
        "async", "await", "new", "this", "true", "false", "null", "undefined",
        "string", "number", "boolean", "void", "any",
    }
    raw = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b', code))
    return raw - KEYWORDS

def keyword_overlap_score(think: str, code: str) -> float:
    code_ids = extract_identifiers(code)
    if not code_ids:
        return 0.5  # neutral if no identifiers found

    think_lower = think.lower()
    hits = sum(1 for ident in code_ids if ident.lower() in think_lower)
    return hits / len(code_ids)
```

### Step 4: Logical Coherence Score

Check if the thinking trace mentions the approach actually used in the code. Uses a pattern-matching approach over common algorithmic strategies.

```python
# src/rewards/logical_coherence.py
import re

APPROACH_PATTERNS = [
    # (code_pattern, think_keywords)
    (r'\bfor\s*\(', ["loop", "iterate", "traverse", "for loop"]),
    (r'\.sort\(', ["sort", "order", "ascending", "descending"]),
    (r'\brecursi', ["recursion", "recursive", "base case", "call itself"]),
    (r'\bMap\b|\bSet\b|\bObject\b', ["hash", "map", "lookup", "dictionary"]),
    (r'\bwhile\s*\(', ["while", "loop until", "condition"]),
    (r'\bPromise\b|\basync\b|\bawait\b', ["async", "promise", "await", "asynchronous"]),
    (r'class\s+\w+', ["class", "object", "instance", "constructor"]),
    (r'\.reduce\(', ["reduce", "accumulate", "fold"]),
    (r'\?\?|\?\.',  ["null", "undefined", "optional", "nullish"]),
]

def logical_coherence_score(think: str, code: str) -> float:
    if not think or not code:
        return 0.0

    think_lower = think.lower()
    matched = 0
    checked = 0

    for code_pattern, think_keywords in APPROACH_PATTERNS:
        if re.search(code_pattern, code, re.IGNORECASE):
            checked += 1
            if any(kw in think_lower for kw in think_keywords):
                matched += 1

    if checked == 0:
        return 0.5  # no detectable patterns, neutral score
    return matched / checked
```

### Step 5: Penalty Detector

```python
# src/rewards/penalty_detector.py
from src.rewards.keyword_overlap import extract_identifiers

def copy_paste_penalty(think: str, prompt: str, threshold: float = 0.8) -> float:
    """Returns penalty (positive float to subtract) if think copies the prompt."""
    if not think or not prompt:
        return 0.0

    prompt_words = set(prompt.lower().split())
    think_words  = set(think.lower().split())

    if not prompt_words:
        return 0.0

    overlap = len(prompt_words & think_words) / len(prompt_words)
    if overlap >= threshold:
        return 0.4 * ((overlap - threshold) / (1.0 - threshold))
    return 0.0

def rambling_penalty(think: str, code: str, threshold: float = 0.10) -> float:
    """Returns penalty if think mentions very few code identifiers."""
    code_ids = extract_identifiers(code)
    if not code_ids:
        return 0.0

    think_lower = think.lower()
    overlap = sum(1 for ident in code_ids if ident.lower() in think_lower) / len(code_ids)

    if overlap < threshold:
        return 0.2 * (1.0 - overlap / threshold)
    return 0.0
```

### Step 6: Composite Quality Scorer

```python
# src/rewards/thinking_quality_scorer.py
from dataclasses import dataclass
from src.rewards.think_extractor import parse_generation
from src.rewards.semantic_similarity import semantic_similarity_score
from src.rewards.keyword_overlap import keyword_overlap_score
from src.rewards.logical_coherence import logical_coherence_score
from src.rewards.penalty_detector import copy_paste_penalty, rambling_penalty

@dataclass
class QualityScoreBreakdown:
    semantic: float
    keyword: float
    coherence: float
    copy_paste_penalty: float
    rambling_penalty: float
    total: float

WEIGHTS = {
    "semantic":  0.35,
    "keyword":   0.30,
    "coherence": 0.20,
}

def score_thinking_quality(
    generation: str,
    prompt: str,
    use_semantic: bool = True,
) -> QualityScoreBreakdown:
    parsed = parse_generation(generation, prompt)

    semantic  = semantic_similarity_score(parsed.think, parsed.code) if use_semantic else 0.5
    keyword   = keyword_overlap_score(parsed.think, parsed.code)
    coherence = logical_coherence_score(parsed.think, parsed.code)

    cpp = copy_paste_penalty(parsed.think, prompt)
    rp  = rambling_penalty(parsed.think, parsed.code)

    total = (
        WEIGHTS["semantic"]  * semantic +
        WEIGHTS["keyword"]   * keyword  +
        WEIGHTS["coherence"] * coherence
        - cpp
        - rp
    )
    total = max(0.0, min(1.0, total))

    return QualityScoreBreakdown(
        semantic=semantic,
        keyword=keyword,
        coherence=coherence,
        copy_paste_penalty=cpp,
        rambling_penalty=rp,
        total=total,
    )
```

### Step 7: GRPO Integration

In `src/training/grpo_trainer.py`, add the quality score as an additional reward component:

```python
# Inside reward computation loop
from src.rewards.thinking_quality_scorer import score_thinking_quality

def compute_reward(
    generation: str,
    prompt: str,
    test_result: float,
    type_check_result: float,
    thinking_quality_weight: float = 0.15,
) -> float:
    base_reward = 0.6 * test_result + 0.25 * type_check_result

    if thinking_quality_weight > 0:
        tq = score_thinking_quality(generation, prompt, use_semantic=True)
        quality_reward = thinking_quality_weight * tq.total
    else:
        quality_reward = 0.0

    return base_reward + quality_reward
```

**Reward weights (suggested starting point):**
```
test_execution:    0.60
type_check:        0.25
thinking_quality:  0.15
```

Tune via grid search — see Performance Considerations.

### Step 8: CLI Flags

```python
# cli/train.py
parser.add_argument("--thinking-quality-reward", action="store_true",
    help="Enable thinking quality as a GRPO reward component.")
parser.add_argument("--thinking-quality-weight", type=float, default=0.15,
    help="Weight of thinking quality in composite reward (default: 0.15).")
parser.add_argument("--disable-semantic-similarity", action="store_true",
    help="Use only keyword/coherence scores (no embedding model, faster).")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/rewards/thinking_quality_scorer.py` | New composite scorer |
| `src/rewards/semantic_similarity.py` | New embedding-based scorer |
| `src/rewards/keyword_overlap.py` | New identifier overlap scorer |
| `src/rewards/logical_coherence.py` | New pattern-based coherence scorer |
| `src/rewards/penalty_detector.py` | New copy-paste/rambling detector |
| `src/rewards/think_extractor.py` | New parser for think/code sections |
| `src/training/grpo_trainer.py` | Integrate quality reward |
| `cli/train.py` | Add CLI flags |

---

## Testing Strategy

```python
# tests/test_thinking_quality.py

COPY_PASTE_EXAMPLE = {
    "prompt": "Write a function that returns the sum of two numbers.",
    "think":  "Write a function that returns the sum of two numbers.",
    "code":   "function add(a: number, b: number): number { return a + b; }",
}

GOOD_THINK_EXAMPLE = {
    "prompt": "Write a function that returns the sum of two numbers.",
    "think":  "I need to add two numbers. I'll use a simple function with numeric parameters and return their sum.",
    "code":   "function add(a: number, b: number): number { return a + b; }",
}

def test_copy_paste_penalized():
    score = score_thinking_quality(
        f"<think>{COPY_PASTE_EXAMPLE['think']}</think>\n```ts\n{COPY_PASTE_EXAMPLE['code']}\n```",
        COPY_PASTE_EXAMPLE['prompt'],
    )
    assert score.copy_paste_penalty > 0.2

def test_good_thinking_scores_higher():
    good = score_thinking_quality(
        f"<think>{GOOD_THINK_EXAMPLE['think']}</think>\n```ts\n{GOOD_THINK_EXAMPLE['code']}\n```",
        GOOD_THINK_EXAMPLE['prompt'],
    )
    bad = score_thinking_quality(
        f"<think>{COPY_PASTE_EXAMPLE['think']}</think>\n```ts\n{COPY_PASTE_EXAMPLE['code']}\n```",
        COPY_PASTE_EXAMPLE['prompt'],
    )
    assert good.total > bad.total

def test_empty_think_gets_zero():
    score = score_thinking_quality("```ts\nconst x = 1;\n```", "Do something.")
    assert score.total < 0.3

def test_keyword_overlap():
    from src.rewards.keyword_overlap import keyword_overlap_score
    code = "function binarySearch(arr: number[], target: number): number { ... }"
    think = "I'll implement binarySearch using a loop with arr and target as parameters."
    assert keyword_overlap_score(think, code) > 0.5
```

**Ablation study:** Train two GRPO models — one with and one without thinking quality reward — for 1000 steps each. Compare HumanEval pass@1 and average thinking trace keyword overlap.

---

## Performance Considerations

- The `all-MiniLM-L6-v2` model is 22M parameters. On CPU, encoding a pair takes ~5ms. For a batch of 64 reward computations, this is ~320ms — acceptable as a reward worker.
- If semantic similarity is too slow during large-scale training, disable it with `--disable-semantic-similarity` and rely on keyword/coherence only. These are pure Python and take < 1ms each.
- Cache embeddings for repeated prompts using an LRU cache keyed on the text hash.
- Run reward computation in a separate process (or thread) from gradient updates to avoid blocking the GPU.

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1024)
def _embed_cached(text_hash: str, text: str):
    return _get_model().encode([text], normalize_embeddings=True)[0]

def semantic_similarity_score_cached(think: str, code: str) -> float:
    h_think = hashlib.md5(think.encode()).hexdigest()
    h_code  = hashlib.md5(code.encode()).hexdigest()
    e_think = _embed_cached(h_think, think)
    e_code  = _embed_cached(h_code, code)
    score = float(np.dot(e_think, e_code))
    return (score + 1.0) / 2.0
```

---

## Dependencies

```
sentence-transformers>=2.7.0   # for semantic similarity
numpy>=1.26.0                  # already present
```

---

## Estimated Complexity

**Development time:** 3-4 days
**Risk:** Low-Medium. The scorer is purely additive to reward computation.
**Main uncertainty:** Calibrating weights so quality reward doesn't overwhelm outcome reward.
**Lines of new code:** ~400

---

## 2026 Best Practices

- **Process rewards over outcome rewards:** Research in 2024-2025 (DeepMind's process reward models, OpenAI's ORM vs PRM studies) consistently shows process rewards improve generalization. This feature directly implements that principle.
- **Lightweight embeddings for reward:** Using a 22M param sentence transformer avoids the cost of calling the full model for reward computation, matching the efficiency patterns of RLHF at scale.
- **Composable reward functions:** Define rewards as independent, testable functions and compose them with explicit weights rather than a single monolithic reward. This makes ablation and debugging straightforward.
- **Reward hacking monitoring:** Log the distribution of each reward component separately (semantic, keyword, coherence, penalties) to detect reward hacking early. If the model learns to maximize thinking quality without improving code quality, the penalty terms need tuning.
