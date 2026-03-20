# Research: Self-Code-Align — Generating Instruction Data from Raw Code

## Status: Ready to implement — template mode works today, LLM mode when budget allows

## The Core Insight

You have raw code (from GitHub scrapes, HuggingFace datasets). But for instruction
tuning, you need (instruction, response) pairs — "Write a function that..." followed
by the actual code. SelfCodeAlign and related techniques generate these pairs
automatically from raw code, eliminating the need for expensive human annotation.

For a TS dev: imagine you have a codebase of 10,000 functions. SelfCodeAlign reads
each function and generates a coding interview question that would produce it, plus
a model answer. Now you have 10,000 instruction-tuning examples for free.

## Techniques Overview

### 1. SelfCodeAlign (BigCode, 2024)

The full pipeline:

```
Raw Code → Extract Seeds → Generate Instructions → Generate Solutions → Filter
```

Steps:
1. **Seed extraction**: Pull standalone functions, classes, and patterns from raw code
2. **Instruction generation**: Ask an LLM (or use templates) to write a natural
   language instruction that would produce this code
3. **Solution generation**: Given just the instruction, generate a fresh solution
   (not copying the seed — this tests if the instruction is clear enough)
4. **Filtering**: Keep only pairs where the generated solution is high quality
   (passes syntax checks, type checks, or even execution tests)

Key insight: The seed code is NOT the output. The seed inspires the instruction,
but the model generates a fresh solution. This avoids data leakage and produces
instruction-following examples rather than code-copying examples.

### 2. Self-Instruct (Wang et al., 2023)

The original self-instruction technique:

```
Seed Tasks (175) → Generate New Tasks → Generate Inputs → Generate Outputs → Filter
```

Steps:
1. Start with a small set of seed tasks (human-written examples)
2. Prompt the model to generate new, diverse tasks similar to the seeds
3. For each task, generate input examples
4. For each (task, input) pair, generate the output
5. Filter out low-quality or duplicate examples

For cola-coder: Start with ~50 hand-written TypeScript coding tasks, then let the
model generate thousands more. Each round's outputs can seed the next round.

### 3. OSS-Instruct (Magicoder, Wei et al., 2024)

Uses open-source code as inspiration seeds:

```
GitHub Code Snippet → "Inspired by this, create a problem..." → Solution
```

Unlike SelfCodeAlign, the instruction doesn't need to match the seed exactly.
The seed is just inspiration — "Here's a snippet using async/await with retry
logic. Create a coding problem that involves similar concepts."

Advantages:
- More diverse instructions (not limited to what the seed does)
- Works with partial/messy code (doesn't need clean functions)
- Generates problems at varying difficulty levels

### 4. Evol-Instruct (WizardCoder, Luo et al., 2024)

Evolve simple instructions into complex ones:

```
Simple Instruction → Evolve (add constraints, edge cases, complexity) → Complex Instruction
```

Evolution operators:
- **Add constraints**: "...and handle the case where the input is empty"
- **Increase reasoning**: "...optimize for O(n log n) time complexity"
- **Concretize**: "...for a React component that manages form state"
- **Deepen**: "...also implement error handling and retry logic"
- **Broaden**: "...extend this to work with any iterable, not just arrays"

For cola-coder: Start with "Write a function that sorts an array" and evolve it to
"Write a generic TypeScript function that sorts any array of objects by a key,
handles undefined values, supports ascending/descending order, and is stable."

## Bootstrapping Without an External LLM

The expensive approach uses Claude/GPT API to generate instructions. But you can
bootstrap with cheaper alternatives:

### Template-Based Generation (Free)

Use code analysis + templates to generate instructions without any LLM:

```python
# If we detect a function with a docstring:
template = "Write a TypeScript function called {name} that {docstring}"

# If we detect a class:
template = "Implement a TypeScript class called {name} with methods: {methods}"

# If we detect error handling:
template = "Write a function that {does_thing} with proper error handling"
```

Quality is lower but cost is zero. Good for initial bootstrapping.

### Self-Instruct Bootstrapping

1. Train a small base model on raw code (we already do this)
2. Use that base model to generate instructions from seeds
3. Filter aggressively (keep only the best 20-30%)
4. Fine-tune on the filtered instruction data
5. Repeat — each round the model gets better at generating instructions

This is a virtuous cycle: better model → better instructions → better fine-tuning → better model.

### Hybrid Approach (Recommended)

1. **Phase 1**: Template-based generation (free) — 5,000 examples
2. **Phase 2**: Use Claude Haiku ($5) to generate 2,000 high-quality examples
3. **Phase 3**: Fine-tune base model on Phase 1+2 data
4. **Phase 4**: Use fine-tuned model for self-instruct (free) — 10,000+ examples
5. **Phase 5**: Filter everything, combine with real code, train final model

Total cost: ~$5 for a complete instruction-tuning dataset.

## Practical Implementation for TypeScript

### Seed Categories

TypeScript code has rich structure we can exploit:

| Category | How to Extract | Example Instruction |
|----------|---------------|-------------------|
| Functions with types | Regex/AST: `function name(args): RetType` | "Write a function that takes X and returns Y" |
| Interfaces/Types | Regex: `interface Name { ... }` | "Define a TypeScript interface for..." |
| Generic functions | Regex: `function name<T>` | "Write a generic function that..." |
| Async functions | Regex: `async function` | "Write an async function that..." |
| React components | Regex: `function Component(props:` | "Create a React component that..." |
| Express handlers | Regex: `app.get\|post\|put` | "Write an Express endpoint that..." |
| Class with methods | Regex: `class Name` | "Implement a class that..." |
| Zod schemas | Regex: `z.object\|z.string` | "Define a Zod validation schema for..." |
| Test files | Regex: `describe\|it\|test` | "Write tests for a function that..." |

### TypeScript-Specific Instruction Templates

```
# Type-focused
"Add proper TypeScript type annotations to the following JavaScript code: {seed}"
"Write a type-safe version of {description} using generics"
"Define the TypeScript types needed for {scenario}"

# Refactoring
"Refactor this code to use async/await instead of callbacks: {seed}"
"Convert this class-based React component to a functional component with hooks: {seed}"
"Extract the repeated logic in this code into a reusable utility: {seed}"

# Testing
"Write unit tests for this function using Jest: {seed}"
"Add edge case tests for: {seed}"

# Documentation
"Write JSDoc comments for: {seed}"
```

### Quality Filtering for TypeScript

After generating instruction-solution pairs, filter by:

1. **Syntax check**: Does the solution parse? (Use a TS parser or regex heuristics)
2. **Type annotation presence**: Does it use TypeScript types? (not just plain JS)
3. **Length check**: Is the solution substantive? (>5 lines, <500 lines)
4. **Instruction clarity**: Does the instruction clearly describe a task? (not vague)
5. **Diversity**: MinHash deduplication across all examples
6. **Brace/bracket balance**: Quick structural check

## Connection to Existing Pipeline

This integrates with the existing data pipeline (`src/cola_coder/data/pipeline.py`):

```python
# SelfAlignSource is a DataSource that generates instruction examples
pipeline = DataPipeline(
    sources=[SelfAlignSource(mode="template", max_examples=5000)],
    filters=[QualityFilterPlugin()],  # reuse existing quality filters
)

for record in pipeline.stream():
    # record.content = formatted instruction-solution pair
    # record.metadata = {"source": "self_align", "seed": "...", "quality_score": 0.8}
    pass
```

The output format for instruction tuning:
```json
{
    "instruction": "Write a TypeScript function that...",
    "input": "",
    "output": "function doThing(x: number): string { ... }",
    "source": "self_align_template",
    "quality_score": 0.85
}
```

## Connection to Synthetic Curriculum (Research 05)

SelfCodeAlign complements the synthetic curriculum:

- **Synthetic curriculum** (05): Generates code examples organized by difficulty
  - Focus: pretraining data, concept progression, raw code
- **SelfCodeAlign** (this): Generates instruction-response pairs from existing code
  - Focus: instruction tuning data, task following, Q&A format

Training pipeline:
1. Pretrain on raw code + synthetic curriculum (Research 05)
2. Instruction-tune on SelfCodeAlign data (this research)
3. RLHF/GRPO on reasoning tasks (existing reasoning module)

## Cost Estimates

| Method | Examples | Cost | Quality |
|--------|----------|------|---------|
| Template only | 5,000 | $0 | Medium |
| Template + Haiku | 7,000 | ~$5 | Medium-High |
| Full LLM (Sonnet) | 5,000 | ~$30 | High |
| Self-instruct (after base) | 10,000+ | $0 | Varies |
| Hybrid (recommended) | 10,000+ | ~$5 | High |

## Prior Art

- SelfCodeAlign (BigCode, 2024): Full pipeline for code instruction generation
- Self-Instruct (Wang et al., 2023): Original self-instruction technique
- Magicoder / OSS-Instruct (Wei et al., 2024): Open-source code as seed
- WizardCoder / Evol-Instruct (Luo et al., 2024): Instruction evolution
- phi-1 / phi-2 (Microsoft, 2023): Synthetic data for code models
- Code Alpaca (Chaudhary, 2023): Self-Instruct applied to code
- Octopack (Muennighoff et al., 2023): Instruction data from Git commits

## Implementation

See `src/cola_coder/data/sources/self_align.py` for the implementation:
- `SeedExtractor`: Extracts code seeds from raw files
- `InstructionGenerator`: Generates instructions from seeds (template/LLM/self modes)
- `SelfAlignPipeline`: Orchestrates the full pipeline
- `SelfAlignSource`: DataSource adapter for the data pipeline

CLI: `scripts/generate_instructions.py` — interactive menu for generating data.
