# Feature 46: Thinking Budget Controller

**Status:** Proposed
**CLI Flag:** `--thinking-budget`
**Complexity:** Medium-High

---

## Overview

The Thinking Budget Controller dynamically allocates a token budget for the model's `<think>...</think>` section based on estimated problem difficulty. Easy problems get a small budget (e.g., 128 tokens); hard problems get a larger one (e.g., 1024 tokens). During generation the controller tracks live token consumption, warns when approaching the limit, and forces a transition to code generation if the model appears stuck in a repetition loop.

---

## Motivation

Without a budget, the model can waste inference compute by producing arbitrarily long thinking traces on trivial problems, or conversely cut short its reasoning on complex ones. In production settings (latency-sensitive APIs) and during GRPO training (where long traces inflate the sequence length and thus memory), unbounded thinking is costly. A budget controller:

- Reduces average sequence length by ~30-40% on simple tasks.
- Prevents degenerate "thinking loops" where the model repeats the same reasoning.
- Makes training more stable by bounding the variance in sequence lengths within a batch.
- Provides interpretable CLI feedback so developers can tune budgets per task type.

---

## Architecture / Design

```
Prompt
  │
  ▼
DifficultyClassifier
  │  → difficulty_score: float [0, 1]
  │  → budget: int (tokens)
  ▼
GenerationLoop
  │  tracks: think_token_count
  │  on budget_warning threshold → inject soft stop hint
  │  on budget_exceeded → force </think> injection
  │  on repetition_detected → force </think> injection
  ▼
Output: <think>...</think><code>...</code>
```

### DifficultyClassifier

A lightweight heuristic classifier (no neural network required at first; can be upgraded later).

**Signals:**
| Signal | Weight |
|---|---|
| Prompt length (tokens) | 0.25 |
| Presence of algorithmic keywords | 0.30 |
| Cyclomatic keyword density (`for`, `while`, `if`, `else`) | 0.20 |
| Presence of data structure keywords (`tree`, `graph`, `heap`) | 0.15 |
| Explicit complexity hints (`O(n log n)`, `optimize`) | 0.10 |

**Budget mapping:**
```python
BUDGET_TABLE = {
    (0.0, 0.2): 64,    # trivial
    (0.2, 0.4): 128,   # easy
    (0.4, 0.6): 256,   # medium
    (0.6, 0.8): 512,   # hard
    (0.8, 1.0): 1024,  # very hard
}
```

### Repetition Detector

Sliding window n-gram overlap check over the last N generated tokens in the thinking section. If a 4-gram appears more than 3 times in a 64-token window, declare the model stuck.

```python
def is_repetition(token_ids: list[int], window: int = 64, ngram: int = 4, threshold: int = 3) -> bool:
    if len(token_ids) < window:
        return False
    recent = token_ids[-window:]
    counts: dict[tuple, int] = {}
    for i in range(len(recent) - ngram + 1):
        gram = tuple(recent[i:i+ngram])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= threshold:
            return True
    return False
```

---

## Implementation Steps

### Step 1: Difficulty Classifier Module

Create `src/generation/difficulty_classifier.py`:

```python
import re
from dataclasses import dataclass

ALGO_KEYWORDS = {
    "sort", "search", "binary", "graph", "tree", "heap", "hash",
    "dynamic", "recursion", "backtrack", "greedy", "optimal",
    "complexity", "O(n", "O(log", "matrix", "dp", "memoize",
}

DS_KEYWORDS = {
    "linkedlist", "stack", "queue", "trie", "segment", "fenwick",
    "disjoint", "union-find", "adjacency", "topological",
}

@dataclass
class DifficultyResult:
    score: float        # 0.0 - 1.0
    budget: int         # token budget for <think>
    signals: dict       # breakdown for debugging

def classify_difficulty(prompt: str, tokenizer=None) -> DifficultyResult:
    tokens = tokenizer.encode(prompt) if tokenizer else prompt.split()
    length_score = min(len(tokens) / 512, 1.0)

    lower = prompt.lower()
    algo_hits = sum(1 for kw in ALGO_KEYWORDS if kw in lower)
    ds_hits   = sum(1 for kw in DS_KEYWORDS if kw in lower)
    control_density = len(re.findall(r'\b(for|while|if|else|switch)\b', lower)) / max(len(tokens), 1)
    explicit_hint   = 1.0 if re.search(r'O\(n', prompt) or "optimize" in lower else 0.0

    algo_score    = min(algo_hits / 5, 1.0)
    ds_score      = min(ds_hits  / 3, 1.0)
    control_score = min(control_density * 20, 1.0)

    score = (
        0.25 * length_score +
        0.30 * algo_score   +
        0.20 * control_score +
        0.15 * ds_score      +
        0.10 * explicit_hint
    )

    budget = _score_to_budget(score)
    return DifficultyResult(
        score=score,
        budget=budget,
        signals={
            "length_score": length_score,
            "algo_score": algo_score,
            "ds_score": ds_score,
            "control_score": control_score,
            "explicit_hint": explicit_hint,
        }
    )

def _score_to_budget(score: float) -> int:
    thresholds = [(0.2, 64), (0.4, 128), (0.6, 256), (0.8, 512)]
    for threshold, budget in thresholds:
        if score < threshold:
            return budget
    return 1024
```

### Step 2: Budget-Aware Generation Hook

Modify `src/generation/generate.py` (or equivalent):

```python
from src.generation.difficulty_classifier import classify_difficulty, DifficultyResult
from src.generation.budget_controller import ThinkingBudgetController

def generate_with_budget(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    thinking_budget: int | None = None,   # None = auto-classify
    verbose: bool = False,
) -> str:
    difficulty = classify_difficulty(prompt, tokenizer)
    budget = thinking_budget if thinking_budget is not None else difficulty.budget

    if verbose:
        print(f"[Budget] difficulty={difficulty.score:.2f}  budget={budget} tokens")

    controller = ThinkingBudgetController(
        tokenizer=tokenizer,
        budget=budget,
        warn_at=int(budget * 0.8),
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = []
    in_think = False

    for step in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]

        # Force </think> if budget exceeded or repetition detected
        if controller.should_force_close(generated, in_think):
            force_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
            next_token = force_token
            if verbose:
                print(f"[Budget] Forcing </think> at step {step}")
        else:
            next_token = logits.argmax(-1).item()

        generated.append(next_token)
        decoded = tokenizer.decode([next_token])

        if "<think>" in decoded:
            in_think = True
        elif "</think>" in decoded:
            in_think = False

        controller.update(next_token, in_think)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=False)
```

### Step 3: Budget Controller Class

Create `src/generation/budget_controller.py`:

```python
from src.generation.difficulty_classifier import is_repetition

class ThinkingBudgetController:
    def __init__(self, tokenizer, budget: int, warn_at: int):
        self.tokenizer  = tokenizer
        self.budget     = budget
        self.warn_at    = warn_at
        self.think_count = 0
        self.warned     = False

    def update(self, token_id: int, in_think: bool):
        if in_think:
            self.think_count += 1
            if self.think_count >= self.warn_at and not self.warned:
                self.warned = True
                # Could inject soft hint token here in future

    def should_force_close(self, generated_ids: list[int], in_think: bool) -> bool:
        if not in_think:
            return False
        if self.think_count >= self.budget:
            return True
        if is_repetition(generated_ids):
            return True
        return False

    @property
    def usage(self) -> dict:
        return {
            "think_tokens_used": self.think_count,
            "budget": self.budget,
            "pct_used": self.think_count / self.budget * 100 if self.budget else 0,
        }
```

### Step 4: Training Data Variation

In `src/data/format_examples.py`, vary `<think>` section lengths proportionally:

```python
def format_with_think_budget(
    prompt: str,
    think_text: str,
    code: str,
    difficulty: float,
    max_think_tokens: int = 1024,
    tokenizer=None,
) -> str:
    budget = _score_to_budget(difficulty)
    if tokenizer:
        think_tokens = tokenizer.encode(think_text)
        if len(think_tokens) > budget:
            # Truncate think section to budget
            think_tokens = think_tokens[:budget]
            think_text = tokenizer.decode(think_tokens)
    return f"<think>\n{think_text}\n</think>\n{code}"
```

### Step 5: CLI Integration

In `cli/generate.py` or main entry point:

```python
import argparse
from rich.console import Console
from rich.table import Table

console = Console()

def add_budget_args(parser: argparse.ArgumentParser):
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Max tokens for <think> section. Auto-classified if not set.")
    parser.add_argument("--show-budget", action="store_true",
                        help="Display thinking budget usage after generation.")
    parser.add_argument("--disable-thinking-budget", action="store_true",
                        help="Disable budget controller entirely.")

def display_budget_usage(controller, console: Console):
    usage = controller.usage
    table = Table(title="Thinking Budget Usage")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Tokens Used", str(usage["think_tokens_used"]))
    table.add_row("Budget", str(usage["budget"]))
    table.add_row("Usage %", f"{usage['pct_used']:.1f}%")
    console.print(table)
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/generation/generate.py` | Add budget-aware generation loop |
| `src/generation/difficulty_classifier.py` | New file |
| `src/generation/budget_controller.py` | New file |
| `src/data/format_examples.py` | Truncate think sections to budget |
| `cli/generate.py` | Add `--thinking-budget`, `--show-budget` flags |
| `src/training/grpo_trainer.py` | Pass difficulty scores to formatter |

---

## Testing Strategy

```python
# tests/test_budget_controller.py

def test_trivial_problem_gets_small_budget():
    result = classify_difficulty("Write a function that adds two numbers.")
    assert result.budget <= 128

def test_hard_problem_gets_large_budget():
    result = classify_difficulty(
        "Implement a segment tree with lazy propagation for range sum queries O(log n)."
    )
    assert result.budget >= 512

def test_repetition_detection():
    # Repeat same 4-gram many times
    ids = [1, 2, 3, 4] * 20
    assert is_repetition(ids, window=64, ngram=4, threshold=3)

def test_budget_forces_close():
    controller = ThinkingBudgetController(tokenizer=None, budget=10, warn_at=8)
    # Simulate 10 think tokens
    for i in range(10):
        controller.update(i, in_think=True)
    assert controller.should_force_close([], in_think=True)

def test_no_force_outside_think():
    controller = ThinkingBudgetController(tokenizer=None, budget=5, warn_at=4)
    for i in range(10):
        controller.update(i, in_think=True)
    # Outside think block, should not force
    assert not controller.should_force_close([], in_think=False)
```

**Integration test:** Generate 100 samples with `--thinking-budget 64` on easy prompts, verify mean think length < 70 tokens.

---

## Performance Considerations

- Difficulty classification is O(|prompt_tokens|) and adds < 1ms per call — negligible.
- Repetition detection over a 64-token window is O(64) per generation step — negligible.
- Budget enforcement reduces mean sequence length, which reduces memory and speeds up batch generation by up to 30% on easy problem sets.
- During GRPO training: shorter sequences = larger effective batch size given fixed GPU memory.
- Use `torch.compile` on the generation loop; the budget controller sits outside the compiled region.

---

## Dependencies

- No new pip dependencies for the heuristic classifier.
- `rich` (already likely present) for CLI display.
- Optional upgrade path: replace heuristic classifier with a small fine-tuned DeBERTa for better accuracy.

---

## Estimated Complexity

**Development time:** 3-5 days
**Risk:** Low — purely additive; existing generation is unaffected when flag is off.
**Lines of new code:** ~300

---

## 2026 Best Practices

- **Budget as a first-class prompt attribute:** Modern reasoning models (o3, Gemini 2.0 Flash Thinking) expose explicit "thinking budget" parameters. Aligning Cola-Coder with this paradigm makes the interface familiar.
- **Structured generation constraints:** Use logit processors (HuggingFace `LogitsProcessor`) rather than post-hoc filtering for cleaner integration with beam search and sampling.
- **Token budget logging:** Emit structured JSON logs (e.g., `{"event": "think_budget", "used": 128, "budget": 256}`) for observability tooling rather than raw print statements.
- **Dynamic batching alignment:** Ensure the budget controller integrates with vLLM's continuous batching if the project moves to serving infrastructure.
