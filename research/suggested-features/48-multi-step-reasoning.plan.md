# Feature 48: Multi-Step Reasoning (Think → Plan → Code)

**Status:** Proposed
**CLI Flag:** `--multi-step-reasoning`
**Complexity:** Medium-High

---

## Overview

Introduces a three-phase generation pipeline with distinct structured sections:
- `<think>...</think>` — problem understanding and analysis
- `<plan>...</plan>` — structured solution outline (pseudocode-level)
- `<code>...</code>` (or fenced block) — full implementation

Two new special tokens `<plan>` and `</plan>` are added to the vocabulary. Training data is reformatted to include all three sections. Evaluation includes an ablation comparing 2-phase (think+code) against 3-phase (think+plan+code) solution quality.

---

## Motivation

Current 2-phase generation collapses understanding and solution design into a single unstructured `<think>` block. This conflates two cognitively distinct activities:

1. **Problem understanding:** What is being asked? What are the constraints? What are the edge cases?
2. **Solution design:** Which algorithm? What data structures? What is the structure of the code?

Separating these into dedicated sections encourages the model to:
- Commit to a plan before writing code (reducing backtracking and incoherence)
- Produce plans that can be independently evaluated and rewarded
- Generate code that actually follows the stated plan (measurable alignment)

Human expert programmers follow this pattern naturally. Structured prompting research (e.g., "Chain of Code", Li et al., 2023) shows explicit intermediate representations improve code generation accuracy by 8-15% on hard problems.

---

## Architecture / Design

```
Prompt
  │
  ▼
<think>
  Understand the problem.
  Identify constraints and edge cases.
  Note relevant concepts.
</think>
  │
  ▼
<plan>
  1. Parse input parameters.
  2. Initialize result array.
  3. Iterate with two-pointer approach.
  4. Return sorted result.
</plan>
  │
  ▼
```typescript
function twoSum(nums: number[], target: number): number[] {
  // implementation following the plan
}
```

### Token Vocabulary Extension

```
New special tokens:
  <plan>    (token ID: assigned during tokenizer extension)
  </plan>   (token ID: assigned during tokenizer extension)

Existing tokens (already present):
  <think>   </think>
```

### Format Contract

The model is trained to always generate in order: `<think>` → `</think>` → `<plan>` → `</plan>` → code. The generation loop enforces this order via a state machine.

---

## Implementation Steps

### Step 1: Extend Tokenizer

```python
# src/tokenizer/extend_tokenizer.py
from transformers import PreTrainedTokenizerFast

MULTI_STEP_TOKENS = ["<plan>", "</plan>"]

def add_multi_step_tokens(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    existing = set(tokenizer.additional_special_tokens)
    new_tokens = [t for t in MULTI_STEP_TOKENS if t not in existing]

    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added tokens: {new_tokens}")
    else:
        print("Multi-step tokens already present.")

    return tokenizer

def resize_model_embeddings(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    # Initialize new token embeddings near the mean of existing embeddings
    import torch
    with torch.no_grad():
        embed = model.get_input_embeddings()
        mean_embed = embed.weight[:-len(MULTI_STEP_TOKENS)].mean(0)
        for i in range(1, len(MULTI_STEP_TOKENS) + 1):
            embed.weight[-i] = mean_embed + torch.randn_like(mean_embed) * 0.01
```

### Step 2: Training Data Formatter

```python
# src/data/multi_step_formatter.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class MultiStepExample:
    prompt: str
    think: str
    plan: str
    code: str

def format_multi_step(ex: MultiStepExample) -> str:
    return (
        f"{ex.prompt}\n\n"
        f"<think>\n{ex.think.strip()}\n</think>\n"
        f"<plan>\n{ex.plan.strip()}\n</plan>\n"
        f"```typescript\n{ex.code.strip()}\n```"
    )

def parse_multi_step(text: str) -> Optional[MultiStepExample]:
    import re
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    plan_m  = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    code_m  = re.search(r"```(?:typescript|ts)?\n(.*?)```", text, re.DOTALL)
    if not (think_m and plan_m and code_m):
        return None
    return MultiStepExample(
        prompt="",
        think=think_m.group(1).strip(),
        plan=plan_m.group(1).strip(),
        code=code_m.group(1).strip(),
    )
```

### Step 3: Synthetic Plan Generation for Existing Data

For existing training examples that have `<think>` and code but no `<plan>`, use a rule-based plan extractor to bootstrap training data:

```python
# src/data/plan_bootstrapper.py
import re
import ast

def extract_plan_from_code(code: str) -> str:
    """Generate a simple numbered plan from code structure."""
    lines = []
    step = 1

    # Extract function signature
    fn_match = re.search(r'function\s+(\w+)\s*\(([^)]*)\)', code)
    if fn_match:
        lines.append(f"{step}. Define function `{fn_match.group(1)}` with parameters: {fn_match.group(2)}")
        step += 1

    # Detect major constructs
    if re.search(r'\bfor\s*\(', code):
        lines.append(f"{step}. Iterate using a for loop")
        step += 1
    if re.search(r'\.sort\(', code):
        lines.append(f"{step}. Sort the collection")
        step += 1
    if re.search(r'\bMap\b|\bnew Map\b', code):
        lines.append(f"{step}. Use a Map for O(1) lookups")
        step += 1
    if re.search(r'\breturn\b', code):
        lines.append(f"{step}. Return the result")
        step += 1

    if not lines:
        lines.append("1. Implement the required logic")
        lines.append("2. Return the result")

    return "\n".join(lines)

def upgrade_example_to_multi_step(think: str, code: str) -> tuple[str, str]:
    """
    Given an existing think+code pair, produce (think, plan) where
    plan is extracted from code structure.
    """
    plan = extract_plan_from_code(code)
    return think, plan
```

### Step 4: Generation State Machine

```python
# src/generation/multi_step_generator.py
from enum import Enum, auto
from typing import Optional

class GenPhase(Enum):
    PROMPT    = auto()
    THINKING  = auto()
    PLANNING  = auto()
    CODING    = auto()
    DONE      = auto()

class MultiStepGenerator:
    def __init__(self, tokenizer, model, max_think=256, max_plan=128, max_code=1024):
        self.tokenizer  = tokenizer
        self.model      = model
        self.max_think  = max_think
        self.max_plan   = max_plan
        self.max_code   = max_code
        self.phase      = GenPhase.PROMPT
        self.phase_counts = {p: 0 for p in GenPhase}

    def _get_phase_token_id(self, phase: GenPhase) -> Optional[int]:
        mapping = {
            GenPhase.THINKING: self.tokenizer.convert_tokens_to_ids("<think>"),
            GenPhase.PLANNING: self.tokenizer.convert_tokens_to_ids("<plan>"),
            GenPhase.CODING:   self.tokenizer.convert_tokens_to_ids("```"),
        }
        return mapping.get(phase)

    def should_transition(self, token_id: int, decoded: str) -> Optional[GenPhase]:
        if self.phase == GenPhase.THINKING and "</think>" in decoded:
            return GenPhase.PLANNING
        if self.phase == GenPhase.PLANNING and "</plan>" in decoded:
            return GenPhase.CODING
        if self.phase == GenPhase.CODING and self.tokenizer.eos_token_id == token_id:
            return GenPhase.DONE
        # Enforce phase token limits
        if self.phase == GenPhase.THINKING and self.phase_counts[GenPhase.THINKING] >= self.max_think:
            return GenPhase.PLANNING
        if self.phase == GenPhase.PLANNING and self.phase_counts[GenPhase.PLANNING] >= self.max_plan:
            return GenPhase.CODING
        return None

    def generate(self, prompt: str) -> str:
        import torch
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = []

        # Inject opening <think> token
        think_token = self.tokenizer.convert_tokens_to_ids("<think>")
        input_ids = torch.cat([input_ids, torch.tensor([[think_token]])], dim=-1)
        self.phase = GenPhase.THINKING

        for _ in range(self.max_think + self.max_plan + self.max_code + 64):
            logits = self.model(input_ids).logits[:, -1, :]
            next_id = logits.argmax(-1).item()
            decoded = self.tokenizer.decode([next_id])

            next_phase = self.should_transition(next_id, decoded)
            if next_phase:
                # Force close current phase, open next
                next_id = self._force_transition(next_phase, generated, input_ids)
                self.phase = next_phase

            generated.append(next_id)
            self.phase_counts[self.phase] = self.phase_counts.get(self.phase, 0) + 1
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=-1)

            if self.phase == GenPhase.DONE:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def _force_transition(self, next_phase: GenPhase, generated, input_ids) -> int:
        # Returns the ID of the closing token for current phase
        close_tokens = {
            GenPhase.PLANNING: "</think>",
            GenPhase.CODING:   "</plan>",
            GenPhase.DONE:     "```",
        }
        if next_phase in close_tokens:
            return self.tokenizer.convert_tokens_to_ids(close_tokens[next_phase])
        return self.tokenizer.eos_token_id
```

### Step 5: Plan-Code Alignment Evaluator

```python
# src/eval/plan_code_alignment.py
import re
from src.data.multi_step_formatter import parse_multi_step

def plan_code_alignment_score(generation: str) -> float:
    """
    Check whether the plan's stated steps are reflected in the code.
    Returns a float [0, 1].
    """
    parsed = parse_multi_step(generation)
    if not parsed:
        return 0.0

    plan_steps = [line.strip() for line in parsed.plan.split("\n") if line.strip()]
    code_lower = parsed.code.lower()
    hits = 0

    for step in plan_steps:
        # Extract key nouns/verbs from the step
        words = set(re.findall(r'\b([a-z][a-z0-9_]{2,})\b', step.lower()))
        words -= {"the", "and", "with", "for", "use", "return", "define", "create"}
        if words and any(w in code_lower for w in words):
            hits += 1

    return hits / len(plan_steps) if plan_steps else 0.0
```

### Step 6: Ablation Setup

```python
# src/eval/ablation_runner.py
"""
Compare 2-phase vs 3-phase on a held-out benchmark.
"""
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AblationConfig:
    mode: Literal["2phase", "3phase"] = "3phase"
    n_samples: int = 200
    benchmark: str = "humaneval_ts"

def run_ablation(config: AblationConfig, model, tokenizer):
    results = []
    for problem in load_benchmark(config.benchmark, n=config.n_samples):
        if config.mode == "2phase":
            gen = generate_2phase(model, tokenizer, problem.prompt)
        else:
            gen = generate_3phase(model, tokenizer, problem.prompt)
        passed = run_tests(gen, problem.tests)
        results.append({"mode": config.mode, "passed": passed, "problem": problem.id})
    return summarize_ablation(results)
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/tokenizer/extend_tokenizer.py` | Add `<plan>`, `</plan>` tokens |
| `src/data/multi_step_formatter.py` | New 3-phase formatter and parser |
| `src/data/plan_bootstrapper.py` | Generate plans for existing 2-phase data |
| `src/generation/multi_step_generator.py` | State machine generator |
| `src/eval/plan_code_alignment.py` | Plan-code alignment metric |
| `src/eval/ablation_runner.py` | 2-phase vs 3-phase ablation |
| `src/training/grpo_trainer.py` | Use plan-code alignment as reward signal |
| `cli/generate.py` | `--multi-step-reasoning` flag |

---

## Testing Strategy

```python
# tests/test_multi_step.py

def test_formatter_round_trip():
    ex = MultiStepExample(
        prompt="Add two numbers",
        think="I need a simple addition function.",
        plan="1. Accept two number parameters.\n2. Return their sum.",
        code="function add(a: number, b: number): number { return a + b; }",
    )
    formatted = format_multi_step(ex)
    parsed = parse_multi_step(formatted)
    assert parsed.think == ex.think
    assert parsed.plan  == ex.plan
    assert parsed.code  == ex.code

def test_plan_bootstrapper_produces_non_empty():
    code = """
    function binarySearch(arr: number[], target: number): number {
        let lo = 0, hi = arr.length - 1;
        while (lo <= hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (arr[mid] === target) return mid;
            else if (arr[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }
    """
    think, plan = upgrade_example_to_multi_step("thinking text", code)
    assert len(plan.strip()) > 0
    assert "loop" in plan.lower() or "while" in plan.lower()

def test_plan_code_alignment_perfect():
    generation = (
        "<think>Solve it</think>\n"
        "<plan>\n1. Use a Map for lookups.\n2. Return the result.\n</plan>\n"
        "```ts\nconst m = new Map();\nreturn m.get(key);\n```"
    )
    score = plan_code_alignment_score(generation)
    assert score > 0.5
```

---

## Performance Considerations

- Adding 2 tokens to the vocabulary is negligible — the embedding matrix grows by 2 rows.
- The state machine generator adds ~0 overhead; it operates on already-generated token IDs.
- Plan bootstrapping for existing data is a one-time offline preprocessing step. At 10k examples/second, a 1M example dataset takes ~100 seconds.
- Ablation studies: run on a subset (200 problems) to get statistically significant results without full retraining.

---

## Dependencies

No new pip dependencies required. Uses existing tokenizer infrastructure.

---

## Estimated Complexity

**Development time:** 5-7 days (includes data reformatting and ablation setup)
**Risk:** Medium. Tokenizer extension requires model retraining or careful embedding initialization. Existing checkpoints are incompatible without embedding resize.
**Lines of new code:** ~500

---

## 2026 Best Practices

- **Structured intermediate representations:** Industry models (GPT-4o, Gemini 2.5) use implicit multi-step reasoning. Making it explicit with dedicated tokens aligns with interpretability goals and makes the pipeline inspectable.
- **Ablation-first development:** Never ship a structural change without an ablation. The 2-phase vs 3-phase comparison provides the evidence base needed to justify the training cost.
- **Token initialization:** Initialize new special token embeddings near the centroid of existing token embeddings (with small noise) rather than random initialization. This accelerates convergence.
- **Minimal vocabulary expansion:** Adding only 2 tokens (not full structured XML schema) keeps the vocabulary size impact negligible while providing the structural anchors the model needs.
