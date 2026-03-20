# Feature 50: Self-Verification Loop

**Status:** Proposed
**CLI Flag:** `--self-verify`
**Complexity:** High

---

## Overview

After generating an initial code solution, the model re-reads its own output in a `<think>` block and can optionally revise the code. The loop runs up to 2-3 iterations, stopping early if the model declares the code correct or makes no substantive changes. A safeguard keeps the best version by perplexity to prevent degradation loops.

---

## Motivation

Current single-pass generation has no mechanism for self-correction. Human programmers routinely re-read and revise their code before submitting. Self-consistency and self-refinement techniques (Madaan et al., "Self-Refine", 2023; Shinn et al., "Reflexion", 2023) show that iterative revision improves accuracy on code tasks by 5-15% with no additional training.

Key benefits:
- Catches obvious bugs (off-by-one, typos, missing returns) visible on re-reading.
- Improves type safety when the model notices type mismatches in its own code.
- Provides a training signal: pairs of (original, revision) for supervised fine-tuning.

Key risks:
- **Degradation loops:** Each revision may introduce new bugs. The model can "fix" working code into broken code.
- **Hallucinated corrections:** The model may claim to fix something without actually changing the code meaningfully.
- **Latency:** Each revision pass adds a full generation sequence.

---

## Architecture / Design

```
Initial generation
       │
       ▼
[Code v0]
       │
       ▼
Self-Verify Loop (max_revisions=3)
  ┌────────────────────────────────┐
  │  Append: "[Review your code]"  │
  │  Generate: <think>review</think>│
  │  Parse: is_correct / revision  │
  │  If revision: generate Code v1 │
  │  Compare perplexity v0 vs v1   │
  │  Keep lower perplexity version │
  └────────────────────────────────┘
       │  stop when: is_correct=True
       │              OR no_change=True
       │              OR max_revisions reached
       ▼
Best version (by perplexity)
```

### Perplexity Safeguard

Perplexity of the generated code under the model itself is used as a proxy for quality. Lower perplexity = more "model-natural" = less likely to contain degenerate outputs. This is not a perfect quality metric, but it reliably prevents the worst degradation cases.

```
perplexity(code) = exp(-1/N * Σ log P(token_i | context))
```

If `perplexity(new) > perplexity(old) * alpha` (e.g., alpha=1.3), revert to old.

---

## Implementation Steps

### Step 1: Perplexity Scorer

```python
# src/generation/perplexity_scorer.py
import torch
import math
from transformers import PreTrainedModel, PreTrainedTokenizerFast

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    device: str = "cuda",
) -> float:
    """Compute per-token perplexity of `text` under the model."""
    encoding = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encoding.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())
```

### Step 2: Revision Detector

```python
# src/generation/revision_detector.py
import re

CORRECT_PHRASES = [
    "looks correct", "seems correct", "this is correct", "no issues",
    "no changes needed", "the code is fine", "this should work",
    "no bugs", "no errors", "implementation is correct",
]

def parse_review(think_text: str) -> dict:
    lower = think_text.lower()

    is_correct = any(p in lower for p in CORRECT_PHRASES)

    # Detect if a specific fix is mentioned
    fix_indicators = ["should be", "mistake", "bug", "error", "fix", "change", "incorrect", "wrong"]
    has_fix = any(ind in lower for ind in fix_indicators)

    return {
        "is_correct": is_correct,
        "has_fix": has_fix,
        "raw_think": think_text,
    }

def extract_revision_prompt(original_code: str) -> str:
    return (
        f"\n\n[Review your code below and fix any bugs. "
        f"If the code is correct, say 'looks correct' in your thinking.]\n\n"
        f"```typescript\n{original_code}\n```\n\n"
        f"<think>"
    )
```

### Step 3: Self-Verification Loop

```python
# src/generation/self_verify.py
import re
from dataclasses import dataclass, field
from src.generation.perplexity_scorer import compute_perplexity
from src.generation.revision_detector import parse_review, extract_revision_prompt

@dataclass
class VerificationState:
    versions: list[str] = field(default_factory=list)
    perplexities: list[float] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    best_idx: int = 0
    stopped_reason: str = ""

def self_verify(
    model,
    tokenizer,
    initial_code: str,
    prompt_context: str,
    max_revisions: int = 3,
    perplexity_tolerance: float = 1.30,
    device: str = "cuda",
    verbose: bool = False,
) -> VerificationState:
    state = VerificationState()

    # Evaluate initial version
    ppl0 = compute_perplexity(model, tokenizer, initial_code, device)
    state.versions.append(initial_code)
    state.perplexities.append(ppl0)
    state.best_idx = 0

    for revision_num in range(max_revisions):
        current_code = state.versions[-1]
        review_prompt = prompt_context + extract_revision_prompt(current_code)

        # Generate review
        review_text = _generate_think_block(model, tokenizer, review_prompt, max_tokens=256, device=device)
        review = parse_review(review_text)
        state.reviews.append(review)

        if verbose:
            print(f"[Revision {revision_num+1}] correct={review['is_correct']} has_fix={review['has_fix']}")

        if review["is_correct"] or not review["has_fix"]:
            state.stopped_reason = "model_satisfied"
            break

        # Generate revised code
        revision_full_prompt = review_prompt + review_text + "</think>\n```typescript\n"
        new_code = _generate_code_block(model, tokenizer, revision_full_prompt, max_tokens=1024, device=device)

        if _is_trivial_change(current_code, new_code):
            state.stopped_reason = "no_change"
            break

        new_ppl = compute_perplexity(model, tokenizer, new_code, device)
        state.versions.append(new_code)
        state.perplexities.append(new_ppl)

        # Perplexity safeguard
        best_ppl = state.perplexities[state.best_idx]
        if new_ppl <= best_ppl:
            state.best_idx = len(state.versions) - 1
        elif new_ppl > best_ppl * perplexity_tolerance:
            if verbose:
                print(f"[Revision {revision_num+1}] Perplexity degraded ({new_ppl:.2f} > {best_ppl:.2f}*{perplexity_tolerance}). Reverting.")
            state.stopped_reason = "perplexity_degradation"
            # Do not update best_idx; optionally break
            break
    else:
        state.stopped_reason = "max_revisions"

    return state

def best_version(state: VerificationState) -> str:
    return state.versions[state.best_idx]

def _is_trivial_change(old: str, new: str) -> bool:
    """Returns True if new code is essentially the same as old code."""
    old_stripped = re.sub(r'\s+', ' ', old).strip()
    new_stripped = re.sub(r'\s+', ' ', new).strip()
    return old_stripped == new_stripped

def _generate_think_block(model, tokenizer, prompt, max_tokens, device) -> str:
    import torch
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
    generated = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    # Extract up to </think>
    m = re.search(r"(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text

def _generate_code_block(model, tokenizer, prompt, max_tokens, device) -> str:
    import torch
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
    generated = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    # Extract up to closing ```
    m = re.search(r"(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text
```

### Step 4: Improvement Scorer

```python
# src/generation/improvement_scorer.py
from src.generation.self_verify import VerificationState

def score_improvement(state: VerificationState, run_tests_fn=None) -> dict:
    """
    Measure improvement across revisions.
    If run_tests_fn is provided, runs tests on each version.
    Otherwise uses perplexity delta as proxy.
    """
    n = len(state.versions)
    ppl_delta = state.perplexities[0] - state.perplexities[state.best_idx]

    result = {
        "n_revisions": n - 1,
        "ppl_initial": state.perplexities[0],
        "ppl_best": state.perplexities[state.best_idx],
        "ppl_delta": ppl_delta,
        "best_version_idx": state.best_idx,
        "stopped_reason": state.stopped_reason,
    }

    if run_tests_fn is not None:
        test_scores = [run_tests_fn(v) for v in state.versions]
        result["test_scores"] = test_scores
        result["test_improvement"] = test_scores[state.best_idx] - test_scores[0]

    return result
```

### Step 5: Training Data Generation

The self-verification loop can generate (original, revision) pairs for supervised fine-tuning:

```python
# src/data/revision_pair_generator.py
from src.generation.self_verify import self_verify, best_version, VerificationState

def generate_revision_pairs(
    model,
    tokenizer,
    prompts: list[str],
    max_revisions: int = 2,
) -> list[dict]:
    pairs = []
    for prompt in prompts:
        # Generate initial solution
        initial = _generate_initial(model, tokenizer, prompt)
        state = self_verify(model, tokenizer, initial, prompt, max_revisions=max_revisions)

        if state.best_idx > 0 and state.stopped_reason != "perplexity_degradation":
            pairs.append({
                "prompt": prompt,
                "original_code": state.versions[0],
                "revised_code": state.versions[state.best_idx],
                "n_revisions": state.best_idx,
                "reviews": state.reviews[:state.best_idx],
            })
    return pairs
```

### Step 6: CLI Integration

```python
# In cli/generate.py
parser.add_argument("--self-verify", action="store_true",
    help="Enable self-verification loop after initial generation.")
parser.add_argument("--max-revisions", type=int, default=2,
    help="Maximum number of self-revision passes (default: 2).")
parser.add_argument("--perplexity-tolerance", type=float, default=1.30,
    help="Allow up to this factor increase in perplexity before reverting (default: 1.30).")
parser.add_argument("--show-revisions", action="store_true",
    help="Show all revision versions and their perplexity scores.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/generation/self_verify.py` | New — main loop |
| `src/generation/perplexity_scorer.py` | New — perplexity computation |
| `src/generation/revision_detector.py` | New — parse review text |
| `src/generation/improvement_scorer.py` | New — measure improvement |
| `src/data/revision_pair_generator.py` | New — SFT pair generation |
| `cli/generate.py` | Add CLI flags |
| `src/eval/evaluator.py` | Record revision stats |

---

## Testing Strategy

```python
# tests/test_self_verify.py

def test_trivial_change_detection():
    old = "function add(a, b) { return a + b; }"
    new = "function add( a, b ) { return a + b; }"  # whitespace only
    assert _is_trivial_change(old, new)

def test_correct_detection():
    review = parse_review("The code looks correct to me, no issues found.")
    assert review["is_correct"] is True
    assert review["has_fix"] is False

def test_fix_detection():
    review = parse_review("There's a bug here — the loop should use <= not <, fix that.")
    assert review["has_fix"] is True
    assert review["is_correct"] is False

def test_best_version_tracks_minimum_perplexity():
    state = VerificationState()
    state.versions = ["v0", "v1", "v2"]
    state.perplexities = [10.0, 8.5, 12.0]
    state.best_idx = 0
    # Simulate logic: v1 is best
    for i, ppl in enumerate(state.perplexities):
        if ppl < state.perplexities[state.best_idx]:
            state.best_idx = i
    assert state.best_idx == 1
    assert best_version(state) == "v1"
```

**Regression test:** On a 50-problem benchmark, verify that enabling `--self-verify` never produces a *lower* test pass rate than single-pass generation (the safeguard should guarantee this in the common case).

---

## Performance Considerations

- Each revision pass costs one full forward pass for perplexity + one generation sequence.
- With `max_revisions=2`, worst-case latency is 3x single-pass generation.
- For training-time use (SFT pair generation), run in batches and cache initial generations.
- Use `do_sample=False` (greedy) for revision generation to get deterministic behavior; sampling introduces variance that makes the improvement signal noisy.
- Early stopping (model says "looks correct") reduces average revisions to ~1.2 per sample in practice.
- Consider running perplexity computation with `torch.no_grad()` and `model.half()` for 2x speed.

---

## Dependencies

No new dependencies — uses existing model and tokenizer.

---

## Estimated Complexity

**Development time:** 5-7 days (including safeguards and testing)
**Risk:** High. Degradation loops are a real risk; the perplexity safeguard mitigates but does not eliminate this. Thorough benchmarking is required before using in production.
**Lines of new code:** ~400

---

## 2026 Best Practices

- **Perplexity as a safeguard, not a quality metric:** Perplexity correlates poorly with code correctness in general, but it is a strong signal for degenerate outputs (very high perplexity = incoherent text). Using it only as a safeguard (revert if perplexity spikes) rather than a quality ranking is the right design.
- **Stop conditions over max iterations:** Always implement explicit stop conditions (is_correct, no_change) rather than relying solely on the max revisions limit. This keeps average latency low.
- **Revision pairs for SFT:** The (original, revision) pairs generated by this loop are valuable supervised fine-tuning data. Many 2024-2025 coding models (WizardCoder, Magicoder) use iterative refinement data. Capturing these pairs during evaluation runs is free.
- **Limit to inference only:** Self-verification during GRPO training would make sequences extremely long. Restrict this feature to inference/evaluation only, and use the generated revision pairs for a separate SFT phase.
