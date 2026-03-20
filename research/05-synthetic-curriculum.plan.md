# Research: Synthetic Curriculum Learning

## Status: Emerging — phi-1/phi-2 showed synthetic data works, but curriculum is unexplored

## The Core Insight

Microsoft's phi-1 (2023) proved that a 1.3B model trained on "textbook quality"
synthetic data can beat 10x larger models trained on raw data. But phi just
filtered for quality — it didn't TEACH progressively.

What if we created a CURRICULUM of synthetic training data, organized from
simple → complex, designed to teach concepts in the optimal learning order?

## How Humans Learn Programming

A good programming course teaches:
1. Variables and types
2. Functions
3. Control flow (if/else, loops)
4. Data structures (arrays, objects)
5. Classes and OOP
6. Async/await
7. Error handling
8. Design patterns
9. Full applications

But current LLMs learn from RANDOM GitHub code — they see complex async GraphQL
resolvers before they've seen a for-loop. This is like teaching calculus before
arithmetic.

## The Approach: Staged Synthetic Curriculum

### Stage 1: Foundation (Steps 0-2000)

Generate and train on simple, focused examples:

```typescript
// Variables and types
const name: string = "Alice";
const age: number = 30;
const isActive: boolean = true;

// Simple functions
function add(a: number, b: number): number {
    return a + b;
}

// Basic control flow
function isEven(n: number): boolean {
    return n % 2 === 0;
}
```

**Generation prompt for Claude/GPT:**
"Generate 10 simple TypeScript functions that demonstrate basic variable types
and arithmetic. Each function should be 3-8 lines, have clear type annotations,
and include a one-line comment explaining what it does. Vary the function names
and logic."

### Stage 2: Intermediate (Steps 2000-6000)

```typescript
// Array operations
function filterActive(users: User[]): User[] {
    return users.filter(u => u.isActive);
}

// Object manipulation
function mergeConfigs(base: Config, override: Partial<Config>): Config {
    return { ...base, ...override };
}

// Error handling
function parseJSON<T>(input: string): T | null {
    try {
        return JSON.parse(input);
    } catch {
        return null;
    }
}
```

### Stage 3: Advanced (Steps 6000-12000)

```typescript
// Async operations
async function fetchWithRetry(url: string, retries: number = 3): Promise<Response> {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url);
            if (res.ok) return res;
        } catch (err) {
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
        }
    }
    throw new Error("Unreachable");
}

// Generics
function groupBy<T, K extends string>(items: T[], key: (item: T) => K): Record<K, T[]> {
    return items.reduce((acc, item) => {
        const group = key(item);
        (acc[group] ??= []).push(item);
        return acc;
    }, {} as Record<K, T[]>);
}
```

### Stage 4: Real-World Patterns (Steps 12000-20000)

Mix synthetic curriculum data with real GitHub data:
- 30% synthetic (complex patterns, design patterns, full modules)
- 70% real code (from curated repos)

This gives the model structured knowledge foundation PLUS real-world pattern exposure.

## Synthetic Data Generation Pipeline

```python
class CurriculumGenerator:
    """Generate synthetic training data organized by difficulty.

    Uses an LLM (Claude API) to generate high-quality code examples
    that teach specific concepts.
    """

    STAGES = [
        {
            "name": "foundation",
            "concepts": ["variables", "types", "functions", "arithmetic", "strings"],
            "complexity": "simple",
            "lines": (3, 15),
            "examples_per_concept": 1000,
        },
        {
            "name": "intermediate",
            "concepts": ["arrays", "objects", "control_flow", "error_handling", "closures"],
            "complexity": "moderate",
            "lines": (10, 40),
            "examples_per_concept": 1000,
        },
        {
            "name": "advanced",
            "concepts": ["generics", "async", "decorators", "streams", "iterators"],
            "complexity": "complex",
            "lines": (20, 80),
            "examples_per_concept": 500,
        },
        {
            "name": "patterns",
            "concepts": ["factory", "observer", "middleware", "repository", "dependency_injection"],
            "complexity": "architectural",
            "lines": (40, 200),
            "examples_per_concept": 200,
        },
    ]

    def generate_stage(self, stage: dict) -> list[str]:
        """Generate all examples for a curriculum stage.

        Uses Claude API (or any LLM) with carefully crafted prompts
        to generate diverse, high-quality examples.
        """
        examples = []
        for concept in stage["concepts"]:
            prompt = self._build_prompt(concept, stage["complexity"], stage["lines"])
            for batch in range(stage["examples_per_concept"] // 10):
                response = self._call_llm(prompt)
                parsed = self._parse_examples(response)
                # Validate: syntax check, type check, dedup
                validated = [ex for ex in parsed if self._validate(ex)]
                examples.extend(validated)
        return examples
```

### Diversity Enforcement

A critical problem with synthetic data: repetition. LLMs tend to generate similar
patterns. Mitigation strategies:

1. **Seed variation**: Each prompt includes 3-5 random seed examples from real code
2. **Name randomization**: "Generate functions with random but realistic names"
3. **Domain variation**: "This batch should be about e-commerce / social media / CLI tools"
4. **Style variation**: "Write in functional style / OOP style / imperative style"
5. **Deduplication**: MinHash across all generated examples, reject >60% similarity

### Cost Estimation

Using Claude Haiku ($0.25/1M input, $1.25/1M output):

| Stage | Examples | Avg tokens/example | Generation cost |
|-------|----------|-------------------|----------------|
| Foundation | 5,000 | 200 | ~$5 |
| Intermediate | 5,000 | 500 | ~$10 |
| Advanced | 2,500 | 1,000 | ~$10 |
| Patterns | 1,000 | 2,000 | ~$10 |

**Total: ~$35 for a complete curriculum.** This is incredibly cheap.

## Training Schedule

```python
class CurriculumScheduler:
    """Controls which data the model sees at each training step.

    Instead of random sampling, we follow the curriculum:
    - Steps 0-2000: 100% Stage 1 (foundation)
    - Steps 2000-4000: 50% Stage 1 + 50% Stage 2
    - Steps 4000-8000: 100% Stage 2 + Stage 3
    - Steps 8000-12000: 50% synthetic (Stage 3+4) + 50% real code
    - Steps 12000+: 30% synthetic + 70% real code
    """

    def get_data_source(self, step: int) -> DataLoader:
        """Return the appropriate data loader for this training step."""
```

## Why This Could Win

1. **phi showed it works**: Small model + synthetic data > big model + raw data
2. **Curriculum is proven in ML**: But nobody's applied it to code LLMs
3. **Cheap**: ~$35 to generate a complete curriculum
4. **Composable**: Combine with real data for best of both worlds
5. **Controllable**: You decide exactly what the model learns
6. **Reproducible**: Same prompts → same curriculum → reproducible training
7. **TypeScript advantage**: You can generate TypeScript-specific curriculum that no
   generic code model has seen (Zod schemas, Next.js patterns, Prisma queries)

## Connection to Multi-Agent Vision

Each specialist model gets its OWN curriculum:
- React specialist: curriculum focused on JSX, hooks, component patterns
- GraphQL specialist: curriculum focused on resolvers, schemas, mutations
- Prisma specialist: curriculum focused on queries, migrations, relations

A $35 curriculum per specialist × 6 specialists = $210 for a complete multi-agent
training curriculum. Absurdly cheap for the potential quality gain.

## Prior Art

- phi-1 / phi-2 (Microsoft, 2023): Proved synthetic "textbook" data works
- Orca (Microsoft, 2023): Used GPT-4 to generate reasoning traces
- TinyStories (2023): Curriculum of simple stories for tiny language models
- Self-Instruct (2023): LLM generates its own instruction data
- **Nobody has combined curriculum learning + synthetic code + real code mixing**
