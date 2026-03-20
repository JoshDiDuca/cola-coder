# Feature 52: Reward Shaping (Partial Credit)

**Status:** Proposed
**CLI Flag:** `--reward-shaping`
**Complexity:** Medium

---

## Overview

Replaces binary pass/fail rewards with a composable reward function that gives partial credit across four components: syntax validity, type correctness, structural match, and test passage. Weights are tunable via CLI or grid search. Enables finer-grained GRPO training signals, especially on hard problems where the model rarely passes all tests.

---

## Motivation

Binary rewards create a sparse signal: if the model never passes the tests, it receives reward=0 for every sample and cannot learn from the relative quality of its outputs. Partial credit rewards:

- Provide dense feedback even when tests fail (e.g., "your code at least parsed and typechecked")
- Enable meaningful GRPO policy gradients on hard problems
- Reflect real-world code quality more accurately than binary pass/fail

The total reward is:
```
R = w1*syntax + w2*types + w3*structure + w4*tests
  = 0.1*syntax + 0.2*types + 0.3*structure + 0.4*tests
```

This sums to 1.0 when all components score 1.0.

---

## Architecture / Design

```
Generated code
     │
     ├── SyntaxReward       → [0, 1]   (parse attempt)
     ├── TypeReward         → [0, 1]   (tsc --noEmit)
     ├── StructureReward    → [0, 1]   (signature/return type match)
     └── TestReward         → [0, 1]   (test execution pass rate)
          │
          ▼
     CompositeReward = w1*R1 + w2*R2 + w3*R3 + w4*R4
```

---

## Implementation Steps

### Step 1: Syntax Reward

```python
# src/rewards/syntax_reward.py
import subprocess
import tempfile
import os

def syntax_reward(code: str) -> float:
    """
    Attempt to parse the TypeScript code.
    Returns 1.0 if it parses without errors, 0.0 otherwise.
    Uses `npx tsc --noEmit --allowJs --checkJs false` for basic syntax check.
    """
    if not code.strip():
        return 0.0

    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["npx", "ts-node", "--transpileOnly", "--eval", code],
            capture_output=True, text=True, timeout=5
        )
        # If transpileOnly succeeds, syntax is valid
        return 1.0 if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: naive bracket balance check
        return _naive_syntax_check(code)
    finally:
        os.unlink(tmp_path)

def _naive_syntax_check(code: str) -> float:
    """Quick bracket-balance check as fallback."""
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    in_string = False
    string_char = None

    for ch in code:
        if in_string:
            if ch == string_char:
                in_string = False
        elif ch in ('"', "'", '`'):
            in_string = True
            string_char = ch
        elif ch in '({[':
            stack.append(ch)
        elif ch in ')}]':
            if not stack or stack[-1] != pairs[ch]:
                return 0.0
            stack.pop()

    return 0.8 if not stack else 0.0  # 0.8 (not 1.0) because this is a weaker check
```

### Step 2: Type Reward

```python
# src/rewards/type_reward.py
import subprocess
import tempfile
import os
import json

def type_reward(code: str, timeout: int = 10) -> float:
    """
    Run TypeScript type checker. Returns a score based on error count.
    0 errors → 1.0
    1-2 errors → 0.5
    3+ errors → 0.0
    """
    if not code.strip():
        return 0.0

    tsconfig = {
        "compilerOptions": {
            "strict": True,
            "noEmit": True,
            "target": "ES2020",
            "module": "commonjs",
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "solution.ts")
        cfg_path = os.path.join(tmpdir, "tsconfig.json")

        with open(src_path, "w") as f:
            f.write(code)
        with open(cfg_path, "w") as f:
            json.dump(tsconfig, f)

        try:
            result = subprocess.run(
                ["npx", "tsc", "--project", cfg_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmpdir,
            )
            error_count = result.stdout.count("error TS") + result.stderr.count("error TS")
            if error_count == 0:
                return 1.0
            elif error_count <= 2:
                return 0.5
            else:
                return 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except FileNotFoundError:
            # tsc not available; skip gracefully
            return 0.5  # neutral
```

### Step 3: Structure Reward

```python
# src/rewards/structure_reward.py
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class FunctionSignature:
    name: str
    params: list[str]
    return_type: Optional[str]

def parse_signature(code: str) -> Optional[FunctionSignature]:
    pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{]+))?'
    m = re.search(pattern, code)
    if not m:
        return None
    name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return_type = m.group(3).strip() if m.group(3) else None
    return FunctionSignature(name=name, params=params, return_type=return_type)

def structure_reward(
    generated_code: str,
    expected_signature: Optional[FunctionSignature] = None,
    expected_from_prompt: Optional[str] = None,
) -> float:
    """
    Score how well the code structure matches the expected signature.
    If expected_signature is None, try to infer from prompt.
    """
    gen_sig = parse_signature(generated_code)
    if gen_sig is None:
        return 0.0

    if expected_signature is None and expected_from_prompt:
        expected_signature = parse_signature(expected_from_prompt)

    if expected_signature is None:
        # No reference: check that function exists and has some params
        score = 0.5
        if gen_sig.params:
            score += 0.2
        if gen_sig.return_type:
            score += 0.2
        if "return" in generated_code:
            score += 0.1
        return min(score, 1.0)

    score = 0.0
    # Name match
    if gen_sig.name.lower() == expected_signature.name.lower():
        score += 0.4
    elif expected_signature.name.lower() in gen_sig.name.lower():
        score += 0.2

    # Param count match
    if len(gen_sig.params) == len(expected_signature.params):
        score += 0.3
    elif abs(len(gen_sig.params) - len(expected_signature.params)) == 1:
        score += 0.1

    # Return type match
    if expected_signature.return_type and gen_sig.return_type:
        exp_rt = expected_signature.return_type.strip().lower()
        gen_rt = gen_sig.return_type.strip().lower()
        if exp_rt == gen_rt:
            score += 0.3
        elif exp_rt in gen_rt or gen_rt in exp_rt:
            score += 0.1

    return min(score, 1.0)
```

### Step 4: Composite Reward Function

```python
# src/rewards/composite_reward.py
from dataclasses import dataclass
from typing import Callable, Optional
from src.rewards.syntax_reward import syntax_reward
from src.rewards.type_reward import type_reward
from src.rewards.structure_reward import structure_reward, FunctionSignature

@dataclass
class RewardWeights:
    syntax:    float = 0.10
    types:     float = 0.20
    structure: float = 0.30
    tests:     float = 0.40

    def validate(self):
        total = self.syntax + self.types + self.structure + self.tests
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.3f}")

@dataclass
class RewardBreakdown:
    syntax:    float
    types:     float
    structure: float
    tests:     float
    total:     float
    weights:   RewardWeights

def compute_composite_reward(
    code: str,
    prompt: str,
    test_result: float,                           # from existing test runner
    weights: Optional[RewardWeights] = None,
    expected_sig: Optional[FunctionSignature] = None,
    run_tsc: bool = True,
) -> RewardBreakdown:
    if weights is None:
        weights = RewardWeights()
    weights.validate()

    r_syntax    = syntax_reward(code)
    r_types     = type_reward(code) if run_tsc else 0.5
    r_structure = structure_reward(code, expected_sig, prompt)
    r_tests     = float(test_result)

    total = (
        weights.syntax    * r_syntax    +
        weights.types     * r_types     +
        weights.structure * r_structure +
        weights.tests     * r_tests
    )

    return RewardBreakdown(
        syntax=r_syntax,
        types=r_types,
        structure=r_structure,
        tests=r_tests,
        total=total,
        weights=weights,
    )
```

### Step 5: Weight Grid Search

```python
# src/rewards/weight_tuner.py
import itertools
from src.rewards.composite_reward import RewardWeights, compute_composite_reward

def grid_search_weights(
    eval_examples: list[dict],
    reward_grid: list[tuple[float, float, float, float]] = None,
    metric: str = "correlation_with_tests",
) -> RewardWeights:
    """
    Search over weight combinations to find the one that best correlates
    with final test pass rate (or other metric).
    """
    if reward_grid is None:
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        reward_grid = [
            combo for combo in itertools.product(values, repeat=4)
            if abs(sum(combo) - 1.0) < 0.01
        ]

    best_weights = None
    best_score = -float("inf")

    for w1, w2, w3, w4 in reward_grid:
        weights = RewardWeights(syntax=w1, types=w2, structure=w3, tests=w4)
        rewards = []
        test_scores = []

        for ex in eval_examples:
            bd = compute_composite_reward(
                ex["code"], ex["prompt"], ex["test_score"], weights
            )
            rewards.append(bd.total)
            test_scores.append(ex["test_score"])

        # Correlation of total reward with test score
        import numpy as np
        if len(set(rewards)) > 1:
            corr = float(np.corrcoef(rewards, test_scores)[0, 1])
        else:
            corr = 0.0

        if corr > best_score:
            best_score = corr
            best_weights = weights

    print(f"Best weights: {best_weights} (correlation={best_score:.3f})")
    return best_weights
```

### Step 6: CLI Integration

```python
# cli/train.py
parser.add_argument("--reward-shaping", action="store_true",
    help="Use composable partial-credit reward instead of binary pass/fail.")
parser.add_argument("--reward-w-syntax",    type=float, default=0.10)
parser.add_argument("--reward-w-types",     type=float, default=0.20)
parser.add_argument("--reward-w-structure", type=float, default=0.30)
parser.add_argument("--reward-w-tests",     type=float, default=0.40)
parser.add_argument("--tune-reward-weights", action="store_true",
    help="Run grid search on a validation set to find optimal reward weights before training.")
parser.add_argument("--disable-tsc", action="store_true",
    help="Disable TypeScript type checking in reward (faster but less accurate).")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/rewards/syntax_reward.py` | New |
| `src/rewards/type_reward.py` | New (extends existing TSC reward) |
| `src/rewards/structure_reward.py` | New |
| `src/rewards/composite_reward.py` | New |
| `src/rewards/weight_tuner.py` | New |
| `src/training/grpo_trainer.py` | Replace binary reward with composite |
| `cli/train.py` | Add CLI flags |

---

## Testing Strategy

```python
# tests/test_reward_shaping.py

def test_perfect_code_scores_near_one():
    code = "function add(a: number, b: number): number { return a + b; }"
    bd = compute_composite_reward(code, "Add two numbers", test_result=1.0)
    assert bd.total > 0.85

def test_empty_code_scores_zero():
    bd = compute_composite_reward("", "Add two numbers", test_result=0.0)
    assert bd.total < 0.1

def test_weights_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        RewardWeights(syntax=0.5, types=0.5, structure=0.5, tests=0.5).validate()

def test_syntax_check_bracket_mismatch():
    code = "function broken() { return {"   # unclosed brace
    score = _naive_syntax_check(code)
    assert score == 0.0

def test_structure_reward_name_match():
    code = "function twoSum(nums: number[], target: number): number[] { return []; }"
    from src.rewards.structure_reward import FunctionSignature
    expected = FunctionSignature(name="twoSum", params=["nums", "target"], return_type="number[]")
    score = structure_reward(code, expected)
    assert score > 0.7
```

---

## Performance Considerations

- `syntax_reward` using `npx ts-node --transpileOnly` takes ~300-800ms per call. For GRPO training batches of 64, this adds ~20-50s per training step — prohibitive.
- **Solution:** Run reward computations in a pool of worker processes (e.g., 8 workers), parallelizing across the batch. Net overhead reduces to ~8-10s per step.
- Alternatively, use the `_naive_syntax_check` fallback for syntax during training and only use full TSC for evaluation.
- Cache reward results by code hash to avoid recomputing identical generations.

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=4096)
def _cached_syntax_reward(code_hash: str, code: str) -> float:
    return syntax_reward(code)
```

---

## Dependencies

- `numpy>=1.26.0` for grid search correlation
- Node.js + TypeScript (`npm install -g typescript`) — already required for type checking rewards

---

## Estimated Complexity

**Development time:** 3-4 days
**Risk:** Low-Medium. The composable design is straightforward; main risk is TSC subprocess speed.
**Lines of new code:** ~450

---

## 2026 Best Practices

- **Partial credit is standard:** All major coding LLM benchmarks (SWE-Bench, LiveCodeBench) have shifted toward partial credit metrics. This feature aligns training with evaluation.
- **Log reward components separately:** Always log r_syntax, r_types, r_structure, r_tests individually in addition to the total. This makes reward hacking visible and enables per-component analysis.
- **Weight calibration:** Do not use fixed weights without validation. The grid search step is essential — optimal weights vary significantly by problem distribution and model size.
- **Subprocess isolation:** Run TypeScript tools in isolated subprocesses with strict timeouts. Untrusted generated code can hang or produce side effects; never run it in the main process.
