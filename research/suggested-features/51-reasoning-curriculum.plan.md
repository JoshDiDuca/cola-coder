# Feature 51: Reasoning Curriculum

**Status:** Proposed
**CLI Flag:** `--curriculum`
**Complexity:** High

---

## Overview

Trains the model on progressively harder problems across five difficulty levels: L1 (trivial arithmetic/strings) through L5 (system design). A curriculum scheduler advances the model to the next level when it exceeds a pass-rate threshold. Synthetic problems are generated at each level to supplement real data. Level progression is tracked and logged.

---

## Motivation

Training on a random mix of easy and hard problems is inefficient. The model wastes capacity trying to solve L5 problems before it has mastered L1 fundamentals. Curriculum learning (Bengio et al., 2009; Soviany et al., 2022) consistently improves:
- Sample efficiency (fewer training steps to reach a target accuracy)
- Generalization to harder problems
- Training stability (lower gradient variance early in training)

Code generation-specific curriculum research (CodeRL, Li et al., 2022) shows that difficulty-ordered training improves HumanEval pass@1 by up to 12% vs random ordering.

---

## Architecture / Design

```
Curriculum Scheduler
├── Level definitions (L1-L5)
├── Problem pool per level (real + synthetic)
├── Pass-rate tracker per level
└── Advancement policy

Training loop:
  current_level = L1
  while not converged:
      batch = sample_from_level(current_level)
      train_step(batch)
      if steps % eval_interval == 0:
          pass_rate = evaluate_level(current_level)
          if pass_rate >= threshold:
              current_level = advance(current_level)
```

### Level Definitions

| Level | Name | Problem Types | Target Pass Rate |
|---|---|---|---|
| L1 | Foundations | arithmetic, string ops, basic I/O | 0.90 |
| L2 | Basics | array manipulation, sorting, searching | 0.80 |
| L3 | Intermediate | data structures, recursion, trees | 0.70 |
| L4 | Advanced | algorithms, DP, graphs | 0.60 |
| L5 | Expert | system design, concurrency, architecture | 0.50 |

---

## Implementation Steps

### Step 1: Level Definitions and Problem Classifier

```python
# src/curriculum/levels.py
from dataclasses import dataclass
from enum import IntEnum

class Level(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5

@dataclass
class LevelConfig:
    level: Level
    name: str
    description: str
    advancement_threshold: float
    max_steps: int              # max steps before forced advancement
    synthetic_ratio: float      # fraction of synthetic problems in batches

LEVEL_CONFIGS = {
    Level.L1: LevelConfig(Level.L1, "Foundations",
        "Arithmetic, string manipulation, basic conditionals",
        advancement_threshold=0.90, max_steps=500, synthetic_ratio=0.5),
    Level.L2: LevelConfig(Level.L2, "Basics",
        "Array operations, sorting, binary search",
        advancement_threshold=0.80, max_steps=1000, synthetic_ratio=0.4),
    Level.L3: LevelConfig(Level.L3, "Intermediate",
        "Trees, linked lists, recursion, hash maps",
        advancement_threshold=0.70, max_steps=2000, synthetic_ratio=0.3),
    Level.L4: LevelConfig(Level.L4, "Advanced",
        "Dynamic programming, graph algorithms, backtracking",
        advancement_threshold=0.60, max_steps=4000, synthetic_ratio=0.2),
    Level.L5: LevelConfig(Level.L5, "Expert",
        "System design, concurrency, architectural patterns",
        advancement_threshold=0.50, max_steps=8000, synthetic_ratio=0.1),
}

# Keywords per level for problem classification
LEVEL_KEYWORDS = {
    Level.L1: {"add", "sum", "multiply", "reverse", "upper", "lower", "concat", "length"},
    Level.L2: {"sort", "search", "filter", "map", "reduce", "find", "count", "max", "min"},
    Level.L3: {"tree", "node", "linked", "recursive", "hash", "stack", "queue", "depth"},
    Level.L4: {"dynamic", "dp", "graph", "dijkstra", "backtrack", "greedy", "optimal"},
    Level.L5: {"architect", "system", "concurrent", "thread", "cache", "distributed", "scale"},
}

def classify_level(prompt: str) -> Level:
    lower = prompt.lower()
    scores = {}
    for level, keywords in LEVEL_KEYWORDS.items():
        scores[level] = sum(1 for kw in keywords if kw in lower)
    if not any(scores.values()):
        return Level.L2  # default to basics
    return max(scores, key=lambda l: scores[l])
```

### Step 2: Synthetic Problem Generator

```python
# src/curriculum/synthetic_generator.py
import random
from src.curriculum.levels import Level

L1_TEMPLATES = [
    ("Add {a} and {b}", "function add(a: number, b: number): number {{ return a + b; }}"),
    ("Multiply {a} by {b}", "function multiply(a: number, b: number): number {{ return a * b; }}"),
    ("Reverse string {s}", "function reverseString(s: string): string {{ return s.split('').reverse().join(''); }}"),
    ("Check if {n} is even", "function isEven(n: number): boolean {{ return n % 2 === 0; }}"),
    ("Count vowels in {s}", "function countVowels(s: string): number {{ return s.match(/[aeiou]/gi)?.length ?? 0; }}"),
]

L2_TEMPLATES = [
    ("Find max in array {arr}", "function findMax(arr: number[]): number {{ return Math.max(...arr); }}"),
    ("Sort array {arr}", "function sortArray(arr: number[]): number[] {{ return [...arr].sort((a, b) => a - b); }}"),
    ("Binary search {target} in {arr}", """function binarySearch(arr: number[], target: number): number {{
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (arr[mid] === target) return mid;
    arr[mid] < target ? lo = mid + 1 : hi = mid - 1;
  }}
  return -1;
}}"""),
]

L3_TEMPLATES = [
    ("Implement a Stack with push/pop/peek", """class Stack<T> {{
  private items: T[] = [];
  push(item: T): void {{ this.items.push(item); }}
  pop(): T | undefined {{ return this.items.pop(); }}
  peek(): T | undefined {{ return this.items[this.items.length - 1]; }}
  isEmpty(): boolean {{ return this.items.length === 0; }}
}}"""),
]

TEMPLATES_BY_LEVEL = {
    Level.L1: L1_TEMPLATES,
    Level.L2: L2_TEMPLATES,
    Level.L3: L3_TEMPLATES,
}

def generate_synthetic(level: Level, n: int = 100) -> list[dict]:
    templates = TEMPLATES_BY_LEVEL.get(level, [])
    if not templates:
        return []
    examples = []
    for _ in range(n):
        template_prompt, template_code = random.choice(templates)
        examples.append({
            "prompt": template_prompt,
            "code": template_code,
            "level": level,
            "synthetic": True,
        })
    return examples
```

### Step 3: Problem Pool per Level

```python
# src/curriculum/problem_pool.py
import random
from pathlib import Path
import numpy as np
from src.curriculum.levels import Level, classify_level, LEVEL_CONFIGS
from src.curriculum.synthetic_generator import generate_synthetic

class LevelProblemPool:
    def __init__(self, real_data: list[dict], level: Level):
        self.level = level
        config = LEVEL_CONFIGS[level]

        self.real = [d for d in real_data if classify_level(d["prompt"]) == level]
        self.synthetic = generate_synthetic(level, n=max(200, len(self.real)))
        self.synthetic_ratio = config.synthetic_ratio

        print(f"[Curriculum] L{level}: {len(self.real)} real + {len(self.synthetic)} synthetic")

    def sample_batch(self, batch_size: int) -> list[dict]:
        n_synthetic = int(batch_size * self.synthetic_ratio)
        n_real = batch_size - n_synthetic

        real_sample = random.choices(self.real, k=n_real) if self.real else []
        syn_sample  = random.choices(self.synthetic, k=n_synthetic)
        batch = real_sample + syn_sample
        random.shuffle(batch)
        return batch
```

### Step 4: Curriculum Scheduler

```python
# src/curriculum/scheduler.py
from dataclasses import dataclass, field
from src.curriculum.levels import Level, LEVEL_CONFIGS

@dataclass
class CurriculumState:
    current_level: Level = Level.L1
    steps_at_level: int = 0
    total_steps: int = 0
    history: list[dict] = field(default_factory=list)
    level_pass_rates: dict[Level, list[float]] = field(default_factory=dict)

class CurriculumScheduler:
    def __init__(self, eval_interval: int = 100):
        self.state = CurriculumState()
        self.eval_interval = eval_interval

    def should_evaluate(self) -> bool:
        return self.state.steps_at_level > 0 and self.state.steps_at_level % self.eval_interval == 0

    def record_eval(self, pass_rate: float):
        level = self.state.current_level
        if level not in self.state.level_pass_rates:
            self.state.level_pass_rates[level] = []
        self.state.level_pass_rates[level].append(pass_rate)

        config = LEVEL_CONFIGS[level]
        recent_avg = sum(self.state.level_pass_rates[level][-3:]) / min(3, len(self.state.level_pass_rates[level]))

        should_advance = (
            recent_avg >= config.advancement_threshold
            or self.state.steps_at_level >= config.max_steps
        )

        if should_advance and level < Level.L5:
            self._advance_level(pass_rate)

    def _advance_level(self, pass_rate: float):
        old_level = self.state.current_level
        new_level = Level(int(old_level) + 1)
        self.state.history.append({
            "from": old_level,
            "to": new_level,
            "at_step": self.state.total_steps,
            "pass_rate": pass_rate,
        })
        self.state.current_level = new_level
        self.state.steps_at_level = 0
        print(f"[Curriculum] Advanced L{old_level} → L{new_level} (pass_rate={pass_rate:.2f})")

    def step(self):
        self.state.steps_at_level += 1
        self.state.total_steps += 1

    @property
    def current_level(self) -> Level:
        return self.state.current_level
```

### Step 5: Integration with Training Loop

```python
# In src/training/grpo_trainer.py

from src.curriculum.scheduler import CurriculumScheduler
from src.curriculum.problem_pool import LevelProblemPool
from src.curriculum.levels import Level, LEVEL_CONFIGS

def train_with_curriculum(
    model,
    tokenizer,
    real_data: list[dict],
    max_steps: int,
    eval_fn,            # callable(model, problems) -> pass_rate
    eval_problems: dict[Level, list[dict]],
    curriculum_eval_interval: int = 100,
):
    scheduler = CurriculumScheduler(eval_interval=curriculum_eval_interval)
    pools = {level: LevelProblemPool(real_data, level) for level in Level}

    for step in range(max_steps):
        level = scheduler.current_level
        batch = pools[level].sample_batch(batch_size=8)
        train_step(model, tokenizer, batch)
        scheduler.step()

        if scheduler.should_evaluate():
            pass_rate = eval_fn(model, eval_problems[level])
            scheduler.record_eval(pass_rate)
            _log_curriculum_progress(scheduler.state, step)

def _log_curriculum_progress(state, step: int):
    import json
    print(json.dumps({
        "step": step,
        "level": int(state.current_level),
        "steps_at_level": state.steps_at_level,
        "level_pass_rates": {
            int(k): round(sum(v) / len(v), 3)
            for k, v in state.level_pass_rates.items() if v
        },
    }))
```

### Step 6: CLI Integration

```python
# cli/train.py
parser.add_argument("--curriculum", action="store_true",
    help="Enable curriculum learning (L1→L5 difficulty progression).")
parser.add_argument("--curriculum-start-level", type=int, default=1, choices=[1,2,3,4,5],
    help="Starting curriculum level (default: 1).")
parser.add_argument("--curriculum-eval-interval", type=int, default=100,
    help="Steps between curriculum level evaluations (default: 100).")
parser.add_argument("--curriculum-threshold-override", type=float, default=None,
    help="Override advancement threshold for all levels.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/curriculum/levels.py` | New — level definitions and classifier |
| `src/curriculum/synthetic_generator.py` | New — synthetic problem templates |
| `src/curriculum/problem_pool.py` | New — per-level data pools |
| `src/curriculum/scheduler.py` | New — advancement logic |
| `src/training/grpo_trainer.py` | Integrate curriculum scheduler |
| `src/data/preprocess.py` | Tag training examples with difficulty levels |
| `cli/train.py` | Add CLI flags |

---

## Testing Strategy

```python
# tests/test_curriculum.py

def test_level_classification():
    assert classify_level("Write a function to add two numbers") == Level.L1
    assert classify_level("Implement Dijkstra's algorithm on a weighted graph") == Level.L4

def test_scheduler_advances():
    scheduler = CurriculumScheduler(eval_interval=10)
    for _ in range(10):
        scheduler.step()
    assert scheduler.should_evaluate()
    scheduler.record_eval(pass_rate=0.95)  # exceeds L1 threshold of 0.90
    assert scheduler.current_level == Level.L2

def test_scheduler_does_not_advance_early():
    scheduler = CurriculumScheduler(eval_interval=10)
    for _ in range(10):
        scheduler.step()
    scheduler.record_eval(pass_rate=0.50)  # below L1 threshold
    assert scheduler.current_level == Level.L1

def test_pool_batch_ratio():
    pool = LevelProblemPool([{"prompt": "add", "code": "return 0"}] * 100, Level.L1)
    batch = pool.sample_batch(10)
    assert len(batch) == 10
```

---

## Performance Considerations

- Level classification is O(|prompt|) per example — negligible.
- Problem pools are in-memory. At 10k examples per level and 5 levels = 50k examples total — fits easily in RAM.
- Synthetic generation is done once at startup; no per-step overhead.
- Curriculum evaluation (every 100 steps) adds ~2-5% overhead depending on eval set size. Use a small, fixed 50-problem eval set per level for speed.

---

## Dependencies

No new pip dependencies.

---

## Estimated Complexity

**Development time:** 5-7 days
**Risk:** Medium. The main risk is the synthetic problem templates not covering the level's skills well enough. Mitigation: weight real data more heavily (low synthetic ratio) and validate templates against actual HumanEval performance.
**Lines of new code:** ~600

---

## 2026 Best Practices

- **Adaptive thresholds:** Rather than fixed advancement thresholds, consider computing them from the distribution of pass rates across a validation set. This makes the curriculum self-calibrating.
- **Mixed-level batches at boundaries:** When advancing from L2 to L3, include a fraction of L2 problems in L3 batches for the first N steps to prevent catastrophic forgetting.
- **Log level progression as a metric:** Track level advancement steps in your experiment tracker (wandb/mlflow). Models that advance quickly are better initializations for harder levels.
- **Synthetic data annotation:** Always mark synthetic examples with `"synthetic": true` in the training manifest. This allows post-hoc analysis of whether synthetic data helped or hurt.
