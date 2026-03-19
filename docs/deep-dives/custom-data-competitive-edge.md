# Custom Data: How Small Models Beat Big Ones

The most counterintuitive result in modern ML: a 1.3B parameter model
(Phi-1) outperformed GPT-3.5 (175B — 130x larger) on code benchmarks.
The secret wasn't architecture, wasn't training tricks, wasn't hardware.
It was the data.

This guide covers how to create custom training data that gives your model
an unfair advantage over models trained on raw GitHub scrapes — and whether
it's realistic to compete with commercial models at our scale.

---

## Table of Contents

1. [The Phi-1 Result: Why This Matters](#1-the-phi-1-result-why-this-matters)
2. [Why Custom Data Beats More Data](#2-why-custom-data-beats-more-data)
3. [The Three Types of Custom Data](#3-the-three-types-of-custom-data)
4. [Textbook-Quality Code Generation](#4-textbook-quality-code-generation)
5. [Exercise-Solution Pairs](#5-exercise-solution-pairs)
6. [Distillation: Learning from Bigger Models](#6-distillation-learning-from-bigger-models)
7. [Building a Custom Data Pipeline](#7-building-a-custom-data-pipeline)
8. [Mixing Custom and Real Data](#8-mixing-custom-and-real-data)
9. [Where Small Models Can Win](#9-where-small-models-can-win)
10. [Where Small Models Can't Win (Yet)](#10-where-small-models-cant-win-yet)
11. [Measuring Whether You're Winning](#11-measuring-whether-youre-winning)
12. [The Cost of Custom Data](#12-the-cost-of-custom-data)
13. [Advanced: Self-Improving Data Loops](#13-advanced-self-improving-data-loops)
14. [Practical Playbook](#14-practical-playbook)

---

## 1. The Phi-1 Result: Why This Matters

In June 2023, Microsoft Research released Phi-1. The headline numbers:

| Model | Parameters | HumanEval pass@1 | Training Data |
|-------|-----------|-------------------|---------------|
| GPT-3.5 | 175,000M | 48.1% | Enormous web crawl |
| StarCoder | 15,500M | 33.6% | The Stack (GitHub) |
| **Phi-1** | **1,300M** | **50.6%** | **Custom "textbook" data** |
| CodeGen-16B | 16,000M | 29.3% | GitHub + web |

Phi-1 — 130x smaller than GPT-3.5 — beat it on HumanEval. Not by a
fluke. Consistently. The paper's title was literally "Textbooks Are All
You Need."

Their approach:

1. Used GPT-3.5 to generate ~1B tokens of "textbook quality" Python code.
   Clean, well-documented, educational code with explanations.
2. Generated ~180M tokens of exercises and solutions.
3. Trained the 1.3B model on this synthetic data + a filtered subset of
   The Stack.

That's it. No architectural innovations. No training tricks. Just better
data.

**What this means for us:** At our scale (125M-1B), the quality of training
data is the single most important variable. A 350M model trained on
carefully crafted data can genuinely compete with 3-7B general models on
the specific tasks you optimize for.

---

## 2. Why Custom Data Beats More Data

### The information density argument

GitHub code has extremely uneven quality. Rough estimates from research:

```
Top 5% of GitHub code:   Excellent — clean, documented, correct
Next 15%:                Good — works, reasonable quality
Middle 30%:              Mediocre — works but messy, no docs
Next 30%:                Poor — hacky, copied from StackOverflow, bugs
Bottom 20%:              Garbage — broken, abandoned, auto-generated
```

When you train on all of it, every gradient update is a weighted average
across this quality spectrum. A lot of your training compute is spent
learning bad patterns that you then have to "unlearn."

Custom data is 100% top-quality. Every training step pushes the model in
the right direction. No wasted compute.

### The concentration argument

A general model at 125M parameters spreads its capacity across:

- 80+ programming languages
- Every coding pattern from "hello world" to kernel drivers
- Good code AND bad code
- Multiple coding styles and conventions

A specialized model with custom data concentrates on:

- 1-3 languages you care about
- Patterns relevant to YOUR domain
- Only correct, clean code
- A consistent style

It's the difference between studying every textbook in the library vs
studying the three books on your exam topic.

### The signal-to-noise argument

Training is gradient descent. Each batch gives the model a gradient signal:
"adjust weights this direction." Noisy data gives noisy gradients — the
model oscillates instead of converging cleanly.

Custom data gives clean, consistent gradients. The model converges faster
and to a better final quality. This effect is largest for small models,
where every parameter matters.

---

## 3. The Three Types of Custom Data

### Type 1: Curated real code

Take real code from GitHub but filter it heavily. Instead of training on
all TypeScript code, find the best TypeScript code. Selection criteria:

- Popular repos (>500 stars)
- Active maintenance (commits in last 6 months)
- Has tests that pass
- Uses modern patterns (no var, uses async/await, has types)
- Well-documented (JSDoc, README)

This gives you real-world code at high quality. The patterns are authentic
— they come from production codebases, not contrived examples.

**Effort:** Low-medium. Write filters, curate a list of quality repos.
**Quality:** High.
**Volume:** Medium (limited by how much great code exists).

### Type 2: Synthetic "textbook" code

Use an LLM to generate clean, educational code examples. Each example is:

- Self-contained (doesn't depend on unknown context)
- Well-documented (explains what and why)
- Correct (ideally verified by execution)
- Covers a specific concept clearly

This is the Phi-1 approach. The "textbook" framing is key — the code
teaches a concept, not just implements a feature.

**Effort:** Medium. Requires LLM API access and prompt engineering.
**Quality:** Very high (each example is purpose-built).
**Volume:** Unlimited (you can generate as much as you can afford).

### Type 3: Exercise-solution pairs

Structured training data where a problem statement is paired with its
solution. Closest to how the model will actually be used (given a prompt,
generate code).

```
Problem: Write a function that takes an array of numbers and returns
         the second largest unique value.

Solution:
function secondLargest(nums: number[]): number | undefined {
  const unique = [...new Set(nums)].sort((a, b) => b - a);
  return unique.length >= 2 ? unique[1] : undefined;
}
```

**Effort:** Medium-high. Need diverse, well-specified problems.
**Quality:** Highest (directly trains the skill you're testing).
**Volume:** Medium (limited by problem diversity).

---

## 4. Textbook-Quality Code Generation

This is the most impactful technique. Here's how to actually do it.

### The prompt template

```
You are writing a chapter of a programming textbook for intermediate
TypeScript developers. Write a self-contained code example that teaches
the following concept:

{concept}

Requirements:
- The code must be complete and runnable
- Include JSDoc documentation for all public functions
- Include 2-3 usage examples as comments at the bottom
- Explain WHY the code is structured this way, not just WHAT it does
- Use modern TypeScript features and idioms
- Handle edge cases properly
- Keep it under 80 lines
```

### Concept list (generate hundreds of these)

```
Core Language:
- "Generic type constraints with extends and keyof"
- "Discriminated unions for type-safe state machines"
- "Recursive types for nested data structures"
- "Template literal types for string pattern matching"
- "Using satisfies for type-safe configuration objects"
- "Conditional types with infer keyword"
- "Mapped types with key remapping"

Patterns:
- "Builder pattern with method chaining and type inference"
- "Result type pattern (no exceptions) for error handling"
- "Repository pattern with generic CRUD operations"
- "Middleware pipeline pattern for request processing"
- "Event emitter with type-safe event maps"
- "Dependency injection without decorators"
- "Retry with exponential backoff and jitter"

Algorithms:
- "LRU cache with O(1) get and put using Map"
- "Debounce and throttle with proper TypeScript types"
- "Deep merge of nested objects with type safety"
- "Topological sort for dependency resolution"
- "Trie implementation for autocomplete"
- "Rate limiter using token bucket algorithm"

Real-world:
- "Type-safe API client with path parameters and response types"
- "Form validation using Zod schema inference"
- "Pagination cursor with type-safe next/previous"
- "WebSocket reconnection manager with typed messages"
- "File upload with progress tracking and cancellation"
```

### What the output looks like

A good textbook generation:

```typescript
/**
 * Result type pattern — handle errors without exceptions.
 *
 * Instead of throwing errors and hoping callers catch them, encode
 * success/failure in the return type. The compiler forces you to handle
 * both cases. This is similar to Rust's Result<T, E>.
 *
 * Why this pattern?
 * - Exceptions break type safety (any function might throw)
 * - Try/catch blocks are easy to forget
 * - Result types make error handling explicit and type-checked
 */

/** Represents either a successful value or an error. */
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

/** Create a success result. */
function Ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

/** Create an error result. */
function Err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

/**
 * Parse a JSON string safely, returning a Result instead of throwing.
 *
 * @param json - The JSON string to parse
 * @returns Ok with parsed value, or Err with the parse error
 */
function safeJsonParse<T>(json: string): Result<T> {
  try {
    return Ok(JSON.parse(json) as T);
  } catch (e) {
    return Err(e instanceof Error ? e : new Error(String(e)));
  }
}

/**
 * Chain multiple Result-returning operations.
 * If any step fails, short-circuit and return the error.
 */
function andThen<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>,
): Result<U, E> {
  return result.ok ? fn(result.value) : result;
}

// --- Usage ---
// The compiler FORCES you to check .ok before accessing .value:

const parsed = safeJsonParse<{ name: string }>('{"name": "Alice"}');
if (parsed.ok) {
  console.log(parsed.value.name); // TS knows .value exists here
} else {
  console.error(parsed.error);    // TS knows .error exists here
}
```

Every line teaches something. The model learns the pattern, the types, the
documentation style, and the reasoning behind design decisions.

### Volume targets

| Model Size | Synthetic Textbook Data | Exercise Data | Real Code |
|------------|------------------------|---------------|-----------|
| 50M (tiny) | 50M tokens (~5K examples) | 10M tokens | 500M tokens |
| 125M (small) | 200M tokens (~20K examples) | 50M tokens | 5B tokens |
| 350M (medium) | 500M tokens (~50K examples) | 100M tokens | 12B tokens |
| 1B+ (large) | 1B tokens (~100K examples) | 200M tokens | 50B tokens |

The sweet spot is roughly 5-15% synthetic textbook data mixed into the
total training set.

---

## 5. Exercise-Solution Pairs

Exercises directly train the skill the model is evaluated on: given a
specification, produce correct code.

### Generating diverse exercises

**Difficulty tiers:**

```
Tier 1 (Easy):
  "Write a function that reverses a string"
  "Write a function that checks if a number is prime"
  "Write a function that flattens a nested array"

Tier 2 (Medium):
  "Write a function that finds the longest common subsequence"
  "Write a type-safe event emitter class"
  "Write a function that deep-clones an object including Date and RegExp"

Tier 3 (Hard):
  "Write a SQL query builder with type-safe column references"
  "Write a reactive state management system with computed values"
  "Write a parser for a simplified Markdown syntax"
```

**Include test cases with every exercise:**

```typescript
// Exercise: Write a function that groups array elements by a key function.

// Signature:
function groupBy<T, K extends string>(
  items: T[],
  keyFn: (item: T) => K,
): Record<K, T[]>;

// Tests:
assert.deepEqual(
  groupBy([1, 2, 3, 4, 5], n => n % 2 === 0 ? "even" : "odd"),
  { odd: [1, 3, 5], even: [2, 4] },
);
assert.deepEqual(
  groupBy(["apple", "banana", "avocado"], w => w[0]),
  { a: ["apple", "avocado"], b: ["banana"] },
);

// Solution:
function groupBy<T, K extends string>(
  items: T[],
  keyFn: (item: T) => K,
): Record<K, T[]> {
  const result = {} as Record<K, T[]>;
  for (const item of items) {
    const key = keyFn(item);
    (result[key] ??= []).push(item);
  }
  return result;
}
```

### Multiple solutions per problem

For each exercise, generate 3-5 different valid solutions. This teaches the
model that there are multiple correct approaches:

```typescript
// Solution 1: Imperative
function flatten<T>(arr: (T | T[])[]): T[] {
  const result: T[] = [];
  for (const item of arr) {
    if (Array.isArray(item)) {
      result.push(...flatten(item));
    } else {
      result.push(item);
    }
  }
  return result;
}

// Solution 2: Recursive with reduce
function flatten<T>(arr: (T | T[])[]): T[] {
  return arr.reduce<T[]>((acc, item) =>
    acc.concat(Array.isArray(item) ? flatten(item) : item), []);
}

// Solution 3: Using built-in (modern)
function flatten<T>(arr: (T | T[])[]): T[] {
  return arr.flat(Infinity) as T[];
}
```

The model learns that all three are valid, and picks up different approaches
for different contexts.

---

## 6. Distillation: Learning from Bigger Models

**What it is:** Use a large model (Claude, GPT-4, etc.) to generate
training data for your small model. Your small model "distills" the large
model's knowledge into its limited parameters.

**Why it works:** The large model has seen trillions of tokens and learned
deep patterns. By asking it to generate clean, correct, well-explained code,
you get training data that encodes that knowledge. Your small model can't
learn everything the large model knows, but it can learn the specific
patterns present in the generated data.

### Distillation strategies

**Strategy 1: Direct generation**

Ask the large model to generate code examples directly.

```
Prompt to Claude/GPT-4:
"Generate a TypeScript implementation of [concept].
 Make it production-quality with full type safety, error handling,
 JSDoc documentation, and edge case handling."
```

**Strategy 2: Code improvement**

Take real (messy) code, ask the large model to improve it.

```
Prompt:
"Here is a TypeScript function from a real codebase. Rewrite it to be:
 - Fully type-safe (no 'any')
 - Well-documented (JSDoc)
 - Handle edge cases
 - Use modern TypeScript features
 Original: [paste messy code]"
```

This produces pairs of (bad code, good code) that teach the model quality
improvements.

**Strategy 3: Explanation generation**

Take code and ask the large model to explain it. Train your model on
(explanation + code) pairs.

```
Prompt:
"Explain this TypeScript code step by step, then show the complete
 implementation with comments explaining each section:
 [paste function signature and docstring]"
```

**Strategy 4: Test generation**

Ask the large model to generate comprehensive test suites for functions.

```
Prompt:
"Write comprehensive tests for this TypeScript function using Vitest.
 Cover: normal cases, edge cases, error cases, type edge cases.
 [paste function]"
```

### Important: verify synthetic data

Never trust synthetic data blindly. Generated code can have bugs, even from
GPT-4 or Claude. For every batch of synthetic data:

1. **Run the code** if possible (does it parse? does it compile with `tsc`?)
2. **Run the tests** if tests were generated alongside the code.
3. **Sample check** — manually review 5% of generated examples.

Bad synthetic data is worse than no synthetic data. A model trained on buggy
"textbook" code learns bugs with high confidence.

---

## 7. Building a Custom Data Pipeline

Here's the concrete pipeline for generating and integrating custom data.

### Step 1: Generate with an LLM API

```python
import json
from pathlib import Path

# Your concept list (hundreds of items)
concepts = [
    "Generic type constraints with extends",
    "Discriminated unions for state machines",
    "Builder pattern with method chaining",
    # ... hundreds more
]

PROMPT_TEMPLATE = """Write a self-contained TypeScript code example that
teaches the following concept: {concept}

Requirements:
- Complete and runnable code
- JSDoc on public functions
- 2-3 usage examples at the bottom
- Handle edge cases
- Under 80 lines
- Modern TypeScript idioms"""

def generate_examples(concepts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, concept in enumerate(concepts):
        prompt = PROMPT_TEMPLATE.format(concept=concept)

        # Call your LLM API here (Claude, GPT-4, etc.)
        response = call_llm_api(prompt)

        # Save the generated code
        (output_dir / f"textbook_{i:05d}.ts").write_text(response)

        print(f"Generated {i+1}/{len(concepts)}: {concept}")
```

### Step 2: Validate the generated code

```bash
# Type-check all generated files
npx tsc --noEmit --strict generated/*.ts

# Or for individual files:
for f in generated/*.ts; do
    npx tsc --noEmit --strict "$f" 2>/dev/null && echo "OK: $f" || echo "FAIL: $f"
done
```

Remove or regenerate any files that fail type-checking.

### Step 3: Mix into training data

```python
# In your data preparation script:
from pathlib import Path

def mixed_data_stream(real_data_stream, custom_data_dir, custom_ratio=0.15):
    """Interleave custom data into the real data stream.

    Args:
        real_data_stream: Iterator of real code files.
        custom_data_dir: Path to directory of custom .ts files.
        custom_ratio: Fraction of training examples that are custom (0.0-1.0).
    """
    custom_files = list(Path(custom_data_dir).glob("*.ts"))
    custom_index = 0

    for real_file in real_data_stream:
        yield real_file

        # Interleave custom data at the target ratio
        # For ratio=0.15: yield ~1 custom file per ~6 real files
        if random.random() < custom_ratio / (1 - custom_ratio):
            yield custom_files[custom_index % len(custom_files)].read_text()
            custom_index += 1
```

---

## 8. Mixing Custom and Real Data

The ratio matters enormously. Too much synthetic data and the model sounds
like a textbook — technically correct but sterile. Too little and you don't
get the quality boost.

### Research-backed ratios

| Study | Synthetic % | Real % | Finding |
|-------|------------|--------|---------|
| Phi-1 | ~20% | ~80% | Best HumanEval scores at this ratio |
| Phi-1.5 | ~30% | ~70% | Slightly higher synthetic worked for broader tasks |
| WizardCoder | ~10% | ~90% | Evol-Instruct synthetic data on top of StarCoder |
| CodeAlpaca | ~100% | 0% | Pure synthetic — worked but plateau'd fast |

**The consensus:** 10-20% synthetic is the sweet spot. Going higher helps on
benchmarks but hurts on real-world code generation (the model becomes too
"clean" and struggles with messy real codebases).

### Recommended mix for Cola-Coder

```
85% Real TypeScript code (filtered, from StarCoderData)
10% Textbook-quality synthetic code (generated by LLM)
 5% Exercise-solution pairs (generated by LLM, verified by execution)
```

### When to show each type

Curriculum matters. One approach:

```
First 30% of training:  90% real + 10% textbook (learn real patterns first)
Middle 40% of training: 80% real + 15% textbook + 5% exercises
Last 30% of training:   85% real + 10% textbook + 5% exercises (fine-tune)
```

The exercises are most valuable later in training, when the model can
actually benefit from structured problem-solving practice.

---

## 9. Where Small Models Can Win

With custom data, a small specialized model can genuinely beat larger general
models on specific tasks. Here are the realistic wins:

### Single-language code completion

A 350M TypeScript-only model with custom data can match a 3-7B general model
on TypeScript autocomplete. The general model spreads its capacity across
dozens of languages. Yours doesn't.

### Framework-specific generation

If you fine-tune on Next.js patterns, your model knows that
`export default function Page()` in `app/page.tsx` is a route handler. A
general model might generate a generic React component instead.

### Type-level programming

TypeScript's type system is complex and most training data doesn't cover
advanced types well. If your custom data emphasizes generics, conditional
types, and mapped types, your model will be better at these than models
10x its size.

### Domain-specific code

If you add training data from your specific domain (fintech, game dev,
data pipelines, etc.), your model understands domain-specific patterns that
general models haven't seen enough of.

### Structured output

Custom training data in consistent formats (JSON schemas, API responses,
config files) makes small models excellent at generating structured output
that general models fumble.

---

## 10. Where Small Models Can't Win (Yet)

Being honest about limitations:

### Long-range reasoning

A 350M model can't plan a complex algorithm the way a 70B model can. It can
learn patterns (if-then-else, loop-accumulate, divide-and-conquer) but not
truly novel algorithmic reasoning. For LeetCode hard problems, you need
scale.

### Multi-file refactoring

Understanding how a change in one file affects 10 other files requires
holding a lot of context. Even with multi-file training, small models
struggle with cross-file reasoning at scale.

### Natural language understanding

Interpreting complex, ambiguous instructions ("make this more robust")
requires world knowledge that small models don't have. They can follow
specific patterns but can't reason about intent.

### Rare languages and frameworks

If you've specialized in TypeScript, your model is useless for Python. You
traded breadth for depth. This is a feature, not a bug, but it's a real
limitation.

### Keeping up with ecosystem changes

Your model's knowledge is frozen at training time. When Next.js ships a new
API pattern, your model doesn't know about it. Retraining or fine-tuning is
needed. Larger models from API providers get updated regularly.

---

## 11. Measuring Whether You're Winning

### HumanEval (our built-in benchmark)

Our evaluation harness has 20 Python problems. For TypeScript specialization,
you'd want to:

1. Port the problems to TypeScript
2. Add TypeScript-specific problems (type inference, generic functions)
3. Compare pass@1 against published results from other models

### Published benchmarks to compare against

| Benchmark | What It Tests | Where to Find |
|-----------|--------------|---------------|
| HumanEval | Basic code generation | Built into our project |
| MBPP | More basic problems (974) | Google research |
| MultiPL-E | HumanEval in 18 languages | bigcode/MultiPL-E on HF |
| DS-1000 | Data science problems | xlang-ai/DS-1000 on HF |
| TypeScript problems on LeetCode | Real-world TS challenges | Manual collection |

### Apples-to-apples comparison

When comparing your model to published results, check:

- **Same benchmark version** — HumanEval has multiple versions.
- **Same pass@k** — pass@1 vs pass@10 are very different numbers.
- **Same temperature** — temperature=0.2 for pass@1, temperature=0.8 for
  pass@10 is standard.
- **Same number of samples** — for pass@k computation.

### Track your own progress

```bash
# After each training run, evaluate and log results
python scripts/evaluate.py \
  --checkpoint ./checkpoints/small/latest \
  --num-samples 10 \
  --temperature 0.2

# Compare: base model vs custom-data model vs reasoning model
```

---

## 12. The Cost of Custom Data

### LLM API costs for data generation

| What | Volume | Estimated Cost |
|------|--------|---------------|
| 5K textbook examples (~50M tokens) | Input: ~5M tokens, Output: ~50M tokens | ~$15-50 |
| 20K textbook examples (~200M tokens) | Input: ~20M tokens, Output: ~200M tokens | ~$60-200 |
| 2K exercise-solution pairs | Input: ~2M tokens, Output: ~20M tokens | ~$10-30 |
| Code improvement (10K files) | Input: ~20M tokens, Output: ~20M tokens | ~$40-80 |

Using Claude Haiku or GPT-4o-mini for generation keeps costs low. Use the
larger models (Opus, GPT-4) only for the hardest/most important examples.

### Time costs

- Prompt engineering: 2-4 hours to get good templates.
- Generation: runs in parallel, ~1-4 hours for 20K examples.
- Validation: `tsc --noEmit` takes seconds per file, ~1 hour total.
- Manual review of 5% sample: 2-3 hours.
- Integration into pipeline: 1-2 hours.

**Total:** A weekend project for a meaningful dataset. Not free, but far
cheaper than renting A100s for an extra week of training.

### The ROI calculation

```
Option A: Train 125M model for 1 extra week on raw data
  Cost: ~$200 (cloud GPU) or 1 week of your electricity
  Improvement: ~2-5% on benchmarks (diminishing returns)

Option B: Generate 20K custom examples and train for the same time
  Cost: ~$100 (API) + same GPU cost
  Improvement: ~10-20% on benchmarks (concentrated quality)
```

Custom data is almost always the better investment at our scale.

---

## 13. Advanced: Self-Improving Data Loops

The most powerful technique: use your own model to improve its own
training data. This is what DeepSeek and others do at scale.

### The loop

```
1. Train model V1 on real data + initial custom data
2. Use V1 to generate solutions to coding problems
3. Run solutions against tests, keep only correct ones
4. Use correct solutions as new training data
5. Train model V2 on real data + improved custom data
6. Use V2 to generate better solutions
7. Repeat
```

Each iteration, the model generates slightly better training data, which
trains a slightly better model, which generates even better data.

### Why this works

- **Selection pressure:** Only correct solutions survive. Each iteration
  filters for quality.
- **Diversity:** The model generates different solutions each time
  (temperature > 0), so the training data gets more diverse.
- **Specialization:** The model's own correct solutions are perfectly
  calibrated to its vocabulary and style. Training on them is maximally
  efficient.

### When it stops working

The loop hits diminishing returns when:

- The model solves all problems it's given (no new signal).
- The generated solutions become homogeneous (loss of diversity).
- The model starts overfitting to its own output patterns.

Mitigation: add new, harder problems each iteration. The problems should
always be just beyond the model's current capability — the "zone of
proximal development."

### Connecting to GRPO

This loop is essentially what GRPO does online: generate, evaluate, reinforce.
The difference is that here we're collecting the data for supervised
re-training rather than using RL gradients directly. Both work.

---

## 14. Practical Playbook

Here's what to do, step by step.

### Phase 1: Baseline (what you already have)

```bash
# Train the small model on StarCoderData
make train-small

# Evaluate baseline
python scripts/evaluate.py --checkpoint ./checkpoints/small/latest
# Record: pass@1 = X%
```

### Phase 2: Generate custom data ($50-100, one weekend)

```bash
# 1. Write your concept list (see Section 4 for templates)
#    Save as concepts.json with 200-500 concepts

# 2. Generate textbook examples using Claude/GPT API
#    Write a script that iterates concepts and calls the API
#    Save output to data/custom/textbook/

# 3. Generate exercise-solution pairs
#    Save to data/custom/exercises/

# 4. Validate: type-check everything
npx tsc --noEmit --strict data/custom/textbook/*.ts
npx tsc --noEmit --strict data/custom/exercises/*.ts
# Remove or regenerate failures

# 5. Manual spot-check: read 20-30 examples, fix any issues
```

### Phase 3: Retrain with mixed data

Modify `prepare_data.py` to accept a custom data directory, or manually
concatenate custom data files into the raw data stream before tokenization.

```bash
# Prepare mixed dataset
python scripts/prepare_data.py \
  --config configs/small.yaml \
  --tokenizer tokenizer.json \
  --custom-data ./data/custom/   # <-- you'd add this flag

# Train on mixed data
python scripts/train.py --config configs/small.yaml
```

### Phase 4: Evaluate and compare

```bash
python scripts/evaluate.py --checkpoint ./checkpoints/small/latest
# Record: pass@1 = Y%
# Compare Y to baseline X
# Expected: Y > X by 5-15%
```

### Phase 5: Iterate

- If pass@1 improved but not enough: generate more custom data in weak areas.
- If pass@1 improved a lot: try the self-improving loop (Phase 3 of Section 13).
- Add GRPO reasoning training on top for another boost.

### The target

| Model | Expected pass@1 (Python HumanEval) |
|-------|-----------------------------------|
| Your 125M, raw data only | ~15% |
| Your 125M, + custom data | ~20-28% |
| Your 125M, + custom + GRPO | ~25-33% |
| Published StarCoder 1B | ~22% |
| Published Phi-1 1.3B | ~50% (extreme custom data) |

With custom data and reasoning training, a 125M model can match the raw
performance of general models 5-10x its size on the specific tasks you
optimize for. That's the power of data.
