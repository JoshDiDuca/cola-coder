# Feature 53: Constitutional Coding

**Status:** Proposed
**CLI Flag:** `--constitutional`
**Complexity:** High

---

## Overview

Implements a Constitutional AI-inspired pipeline for code generation: (1) generate code, (2) model critiques it against a set of coding principles, (3) model revises based on the critique. Training uses (code, critique, revision) triples. This is an RLAIF (Reinforcement Learning from AI Feedback) approach where the model provides its own feedback.

---

## Motivation

Constitutional AI (Bai et al., Anthropic 2022) demonstrated that models trained with self-critique outperform those trained only on human feedback, particularly for nuanced quality dimensions that are hard to specify in rewards (readability, error handling, code style). For code:

- Test rewards capture functional correctness but miss readability and maintainability.
- Type rewards capture type safety but miss runtime safety (null checks, bounds checks).
- A "constitutional" critique captures principles that are hard to mechanically test.

**Coding Constitution (the principles used for critique):**
1. **Correctness:** Does the code handle all edge cases mentioned in the prompt?
2. **Type Safety:** Are all types explicitly annotated? No implicit `any`?
3. **Error Handling:** Does the code handle null/undefined inputs gracefully?
4. **Readability:** Are variable names descriptive? Is the logic easy to follow?
5. **Performance:** Are there obvious O(n²) algorithms that could be O(n log n)?

---

## Architecture / Design

```
Prompt
  │
  ▼
[Initial Generation] → Code v0
  │
  ▼
[Critique Phase]
  Model reads: Code v0 + Constitution
  Generates: <critique>...</critique>
  e.g., "The code doesn't handle null input. Variable 'x' is not descriptive."
  │
  ▼
[Revision Phase]
  Model reads: Code v0 + Critique
  Generates: Code v1 (revised)
  │
  ▼
Training data: (prompt, code_v0, critique, code_v1) triple
SFT on code_v1 only (the revised version)
```

---

## Implementation Steps

### Step 1: Constitutional Principles Definition

```python
# src/constitutional/principles.py
from dataclasses import dataclass

@dataclass
class Principle:
    id: str
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str

CODING_CONSTITUTION = [
    Principle(
        id="correctness",
        name="Correctness",
        description="The code correctly implements the specified behavior for all cases.",
        critique_prompt=(
            "Does this code correctly handle all edge cases from the problem statement? "
            "Consider: empty inputs, null values, boundary values, negative numbers. "
            "List any cases that are not handled."
        ),
        revision_prompt="Fix the edge cases you identified in your critique.",
    ),
    Principle(
        id="type_safety",
        name="Type Safety",
        description="All values have explicit TypeScript types. No implicit any.",
        critique_prompt=(
            "Are all function parameters and return types explicitly typed? "
            "Is there any implicit `any` or missing type annotation? "
            "List specific lines with type issues."
        ),
        revision_prompt="Add or fix the type annotations you identified.",
    ),
    Principle(
        id="error_handling",
        name="Error Handling",
        description="The code handles null/undefined and invalid inputs gracefully.",
        critique_prompt=(
            "Does the code validate inputs? Does it handle null or undefined? "
            "What happens if an invalid value is passed? "
            "List missing null checks or input validation."
        ),
        revision_prompt="Add the null checks and input validation you identified.",
    ),
    Principle(
        id="readability",
        name="Readability",
        description="Variable and function names are descriptive. Logic is clear.",
        critique_prompt=(
            "Are variable names descriptive (not single letters like 'x', 'i' unless conventional)? "
            "Is the logic easy to follow without comments? "
            "List any naming or clarity issues."
        ),
        revision_prompt="Improve the variable names and clarity issues you identified.",
    ),
    Principle(
        id="performance",
        name="Performance",
        description="No obvious algorithmic inefficiencies.",
        critique_prompt=(
            "Is there an obvious O(n²) algorithm that could be more efficient? "
            "Are there redundant computations inside loops? "
            "List any performance issues."
        ),
        revision_prompt="Optimize the performance issues you identified if possible.",
    ),
]
```

### Step 2: Critique Generator

```python
# src/constitutional/critique_generator.py
from src.constitutional.principles import CODING_CONSTITUTION, Principle
import re

CRITIQUE_PROMPT_TEMPLATE = """You are reviewing TypeScript code for quality.

Problem statement:
{prompt}

Generated code:
```typescript
{code}
```

Principle to check: {principle_name}
{critique_question}

Write a brief, specific critique (2-4 sentences). If the code satisfies this principle, say "No issues with {principle_name}."
"""

REVISION_PROMPT_TEMPLATE = """Revise the following TypeScript code based on the critique.

Problem statement:
{prompt}

Original code:
```typescript
{code}
```

Critique:
{critique}

{revision_instruction}

Write only the revised TypeScript code. Keep all logic that was correct.
"""

def generate_critique(
    model,
    tokenizer,
    prompt: str,
    code: str,
    principle: Principle,
    max_tokens: int = 200,
    device: str = "cuda",
) -> str:
    critique_input = CRITIQUE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        code=code,
        principle_name=principle.name,
        critique_question=principle.critique_prompt,
    )
    return _generate(model, tokenizer, critique_input, max_tokens, device)

def generate_revision(
    model,
    tokenizer,
    prompt: str,
    code: str,
    critique: str,
    principle: Principle,
    max_tokens: int = 512,
    device: str = "cuda",
) -> str:
    revision_input = REVISION_PROMPT_TEMPLATE.format(
        prompt=prompt,
        code=code,
        critique=critique,
        revision_instruction=principle.revision_prompt,
    )
    full = _generate(model, tokenizer, revision_input, max_tokens, device)
    # Extract code block
    m = re.search(r"```(?:typescript|ts)?\n(.*?)```", full, re.DOTALL)
    return m.group(1).strip() if m else full

def _generate(model, tokenizer, text: str, max_tokens: int, device: str) -> str:
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### Step 3: Constitutional Pipeline

```python
# src/constitutional/pipeline.py
from dataclasses import dataclass, field
from src.constitutional.principles import CODING_CONSTITUTION, Principle
from src.constitutional.critique_generator import generate_critique, generate_revision

NO_ISSUE_PHRASES = [
    "no issues", "satisfies this principle", "no problems", "looks good",
    "no concerns", "correctly handles", "well done",
]

@dataclass
class ConstitutionalTriple:
    prompt: str
    code_v0: str
    principle: str
    critique: str
    code_v1: str
    improved: bool     # True if code_v1 differs meaningfully from code_v0

@dataclass
class ConstitutionalResult:
    prompt: str
    original_code: str
    final_code: str
    triples: list[ConstitutionalTriple] = field(default_factory=list)
    n_principles_applied: int = 0

def run_constitutional_pipeline(
    model,
    tokenizer,
    prompt: str,
    initial_code: str,
    principles: list[Principle] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> ConstitutionalResult:
    if principles is None:
        principles = CODING_CONSTITUTION

    current_code = initial_code
    triples = []

    for principle in principles:
        critique = generate_critique(model, tokenizer, prompt, current_code, principle, device=device)

        if verbose:
            print(f"[Constitution] {principle.name}: {critique[:80]}...")

        # Check if critique found issues
        critique_lower = critique.lower()
        has_issues = not any(p in critique_lower for p in NO_ISSUE_PHRASES)

        if has_issues:
            revised = generate_revision(model, tokenizer, prompt, current_code, critique, principle, device=device)
            improved = _meaningful_change(current_code, revised)
            triples.append(ConstitutionalTriple(
                prompt=prompt,
                code_v0=current_code,
                principle=principle.id,
                critique=critique,
                code_v1=revised,
                improved=improved,
            ))
            if improved:
                current_code = revised
        else:
            triples.append(ConstitutionalTriple(
                prompt=prompt,
                code_v0=current_code,
                principle=principle.id,
                critique=critique,
                code_v1=current_code,
                improved=False,
            ))

    return ConstitutionalResult(
        prompt=prompt,
        original_code=initial_code,
        final_code=current_code,
        triples=triples,
        n_principles_applied=sum(1 for t in triples if t.improved),
    )

def _meaningful_change(old: str, new: str) -> bool:
    import re
    o = re.sub(r'\s+', ' ', old).strip()
    n = re.sub(r'\s+', ' ', new).strip()
    return o != n and len(new) > 10
```

### Step 4: Training Data Generation

```python
# src/constitutional/data_generator.py
import json
from pathlib import Path
from src.constitutional.pipeline import run_constitutional_pipeline, ConstitutionalResult

def generate_constitutional_dataset(
    model,
    tokenizer,
    prompts: list[str],
    output_path: str,
    device: str = "cuda",
) -> int:
    """
    Generate (code, critique, revision) triples from a list of prompts.
    Saves as JSONL. Returns count of improved examples.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    improved_count = 0

    with open(path, "w") as f:
        for i, prompt in enumerate(prompts):
            # Generate initial code
            initial_code = _generate_initial_code(model, tokenizer, prompt, device)
            result = run_constitutional_pipeline(model, tokenizer, prompt, initial_code, device=device)

            for triple in result.triples:
                if triple.improved:
                    record = {
                        "prompt": triple.prompt,
                        "code_v0": triple.code_v0,
                        "principle": triple.principle,
                        "critique": triple.critique,
                        "code_v1": triple.code_v1,
                    }
                    f.write(json.dumps(record) + "\n")
                    improved_count += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(prompts)} prompts ({improved_count} improvements)")

    return improved_count

def format_for_sft(triple: dict) -> dict:
    """Format a constitutional triple for supervised fine-tuning."""
    return {
        "instruction": triple["prompt"],
        "input": f"Critique: {triple['critique']}",
        "output": f"```typescript\n{triple['code_v1']}\n```",
    }
```

### Step 5: CLI Integration

```python
# cli/generate.py
parser.add_argument("--constitutional", action="store_true",
    help="Apply constitutional critique-revision pipeline after generation.")
parser.add_argument("--constitutional-principles", nargs="+",
    choices=["correctness", "type_safety", "error_handling", "readability", "performance"],
    default=None, help="Principles to apply (default: all).")
parser.add_argument("--generate-constitutional-dataset", metavar="OUTPUT.jsonl",
    help="Generate (critique, revision) dataset from prompts file.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/constitutional/principles.py` | New — constitution definition |
| `src/constitutional/critique_generator.py` | New — generate/revise helpers |
| `src/constitutional/pipeline.py` | New — full pipeline |
| `src/constitutional/data_generator.py` | New — SFT dataset generation |
| `cli/generate.py` | Add `--constitutional` flag |
| `src/training/sft_trainer.py` | Train on constitutional revision triples |

---

## Testing Strategy

```python
# tests/test_constitutional.py

def test_no_issue_detection():
    from src.constitutional.pipeline import NO_ISSUE_PHRASES
    critique = "No issues with type safety. All types are correctly annotated."
    assert any(p in critique.lower() for p in NO_ISSUE_PHRASES)

def test_issue_detection():
    critique = "The function doesn't handle null input. Line 3 should check for undefined."
    assert not any(p in critique.lower() for p in NO_ISSUE_PHRASES)

def test_meaningful_change():
    old = "function f(x) { return x; }"
    new = "function f(x: number): number { return x; }"
    assert _meaningful_change(old, new)

def test_trivial_change():
    old = "function f(x: number): number { return x; }"
    new = "function f(x: number): number { return x; }"
    assert not _meaningful_change(old, new)

def test_triple_format_for_sft():
    triple = {
        "prompt": "Return x",
        "code_v0": "function f(x) { return x; }",
        "principle": "type_safety",
        "critique": "Missing type annotation on x.",
        "code_v1": "function f(x: number): number { return x; }",
    }
    sft = format_for_sft(triple)
    assert "typescript" in sft["output"]
    assert "critique" in sft["input"].lower()
```

---

## Performance Considerations

- The pipeline calls the model 2× per principle per example (critique + revision). With 5 principles, that is 10 model calls per example vs 1 for standard generation.
- **Use for offline SFT data generation only**, not for inference-time use on every generation (too slow).
- Parallelize across multiple GPUs using a process pool if generating large datasets.
- Skip revision generation if critique says "no issues" — this halves the calls in the best case.
- Cache critiques by (code_hash, principle_id) to avoid re-critiquing identical code.

---

## Dependencies

No new pip dependencies. Uses existing model and tokenizer.

---

## Estimated Complexity

**Development time:** 6-8 days (including SFT dataset generation pipeline)
**Risk:** Medium-High. Quality of critiques depends entirely on model capability. A weak model may produce unhelpful or incorrect critiques that make code worse. Requires validation.
**Lines of new code:** ~500

---

## 2026 Best Practices

- **RLAIF at scale:** Constitutional AI and RLAIF are now standard practice at leading labs (Anthropic, Google DeepMind). Implementing this for Cola-Coder is forward-looking.
- **Principle selection matters:** Start with 2-3 principles (correctness + type_safety) rather than all 5. Each additional principle increases generation cost 2× and may produce noise.
- **Validate critique quality separately:** Before using constitutional triples for training, measure: "what fraction of critiques correctly identify a real issue?" Use a held-out set with known bugs.
- **Filter by improvement:** Only include triples where the revision actually improved test scores or type-check results. Unvalidated revisions can introduce bugs.
