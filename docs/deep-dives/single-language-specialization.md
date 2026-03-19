# Single-Language Specialization: Building a TypeScript-Only Model

What happens if you take all your compute budget and pour it into one
language instead of spreading it across six? Short answer: yes, it gets
significantly better at that language. But there are tradeoffs, tricks,
and a whole set of techniques that only make sense when you go deep on one
language.

---

## Table of Contents

1. [Does Single-Language Training Actually Work?](#1-does-single-language-training-actually-work)
2. [The Transfer Learning Question](#2-the-transfer-learning-question)
3. [How to Structure a TypeScript-Only Dataset](#3-how-to-structure-a-typescript-only-dataset)
4. [Technique: Type-Aware Training](#4-technique-type-aware-training)
5. [Technique: AST-Level Data Augmentation](#5-technique-ast-level-data-augmentation)
6. [Technique: Multi-File Context](#6-technique-multi-file-context)
7. [Technique: Dependency-Aware Training](#7-technique-dependency-aware-training)
8. [Technique: Test-Driven Training Data](#8-technique-test-driven-training-data)
9. [Technique: Compiler Error Training](#9-technique-compiler-error-training)
10. [Technique: Documentation-Code Alignment](#10-technique-documentation-code-alignment)
11. [Technique: Framework-Specific Fine-Tuning](#11-technique-framework-specific-fine-tuning)
12. [Technique: Repo-Level Training](#12-technique-repo-level-training)
13. [Technique: Custom Tokenizer for TypeScript](#13-technique-custom-tokenizer-for-typescript)
14. [The Hybrid Approach: Pre-train Multi, Specialize Single](#14-the-hybrid-approach-pre-train-multi-specialize-single)
15. [Scaling: What Changes at 1B+ Parameters](#15-scaling-what-changes-at-1b-parameters)
16. [Practical Plan: TypeScript-Specialized Cola-Coder](#16-practical-plan-typescript-specialized-cola-coder)

---

## 1. Does Single-Language Training Actually Work?

Yes. The evidence is clear.

**CodeGen-Mono** (Salesforce, 2022): They trained identical-size models on
multi-language data vs Python-only data. The Python-only model scored **30%
higher** on Python benchmarks than the multi-language model at the same
parameter count. The multi-language model knew six languages but was mediocre
at all of them.

**StarCoder** vs **StarCoderBase**: StarCoderBase was trained on 80+
languages. StarCoder fine-tuned it on Python-heavy data and immediately
improved Python performance.

**The intuition:** A 125M parameter model has a fixed amount of capacity
(think of it as a fixed-size brain). If you teach it six languages, each
language gets roughly 1/6 of that capacity. If you teach it one language,
that language gets all of it. The model learns deeper patterns: idiomatic
usage, framework conventions, type system nuances, common error patterns.

**The tradeoff:** Your model becomes useless at everything else. A
TypeScript-only model won't help you write Python. For many use cases, that's
fine — if you only write TypeScript, you don't need a Python model.

### How much better, quantitatively?

Rough estimates based on published results and scaling laws:

| Setup | Equivalent Quality |
|-------|-------------------|
| 125M model, 6 languages | ~125M / 6 = ~20M effective capacity per language |
| 125M model, TS only | Full 125M capacity on TypeScript |
| 350M model, 6 languages | ~60M effective per language |
| 350M model, TS only | Full 350M — approaches multi-language 1B model on TS |

A 350M TypeScript-only model could match a general 1B model on TypeScript
tasks. That's trainable on your local GPU.

---

## 2. The Transfer Learning Question

**"Should I train from scratch on TypeScript only, or pre-train on
everything then fine-tune on TypeScript?"**

Both work. The hybrid approach (Section 14) is usually better, but
pure single-language training is simpler and surprisingly competitive.

**Why multi-language pre-training helps TypeScript:**

- Programming languages share deep structure: variables, functions,
  control flow, data structures. A model pre-trained on Python already
  "knows" what a for loop is before it sees TypeScript.
- Type system concepts from Java/C# transfer to TypeScript's type system.
- String manipulation, array methods, and async patterns are similar
  across JS/TS/Python.

**Why pure TypeScript training can win:**

- No capacity wasted on irrelevant syntax (Python's `self.`, Go's `:=`,
  Rust's `&mut`).
- The tokenizer is optimized for TypeScript tokens (see Section 13).
- The model sees far more TypeScript per training step — it's getting
  concentrated experience.

**Recommendation:** If you have the compute for two training runs, do the
hybrid (pre-train multi, fine-tune TS). If you only have one shot, go
pure TypeScript.

---

## 3. How to Structure a TypeScript-Only Dataset

Not all TypeScript code is equally useful. Here's how to build a
high-quality TS-specific dataset.

### Data sources (in order of quality)

1. **DefinitelyTyped** — The largest collection of TypeScript type
   definitions. Teaches the model the type system deeply. ~60K packages.

2. **Popular TS repos** (>1000 stars) — Framework source code, well-maintained
   libraries. The best code in the ecosystem.

3. **TypeScript compiler source** — The TS compiler is itself written in
   TypeScript. Training on it teaches deep language understanding.

4. **Framework examples** — Official examples from Next.js, Remix, Deno,
   NestJS, tRPC, Prisma, etc. Clean, idiomatic, representative of real usage.

5. **StarCoderData TypeScript split** — Bulk data. Lower average quality but
   massive volume.

### Data proportions

```
Tier 1 (upsample 3x):  DefinitelyTyped, compiler, top-1000-star repos
Tier 2 (upsample 2x):  Framework official examples, well-linted repos
Tier 3 (1x):           General StarCoderData TypeScript split
Tier 4 (downsample):   JS files with JSDoc types (some signal, lots of noise)
```

### What to include alongside TypeScript

Even for a "TypeScript-only" model, include some related content:

- **`.d.ts` type definition files** — Critical for understanding the type
  system and API surfaces.
- **`tsconfig.json` files** — The model should know config options.
- **`package.json` files** — Dependency awareness.
- **JSX/TSX files** — React components are a massive part of the TS ecosystem.
- **`.test.ts` and `.spec.ts` files** — Test patterns.
- **Markdown from TS repos** — API docs, READMEs. Teaches the model what
  functions are supposed to do, not just their implementation.

### What to exclude

- `.js` output files (compiled from `.ts` — the model should learn the
  source, not the transpiled output)
- `node_modules/` (obviously)
- `.min.js` / bundle output
- Migration files
- Auto-generated API clients

---

## 4. Technique: Type-Aware Training

**What it is:** Structure training data to emphasize TypeScript's type
system — the thing that makes TS different from JavaScript.

**Why it matters:** General code models treat types as an afterthought.
A TypeScript-specialized model should understand types deeply: generics,
conditional types, mapped types, template literal types, discriminated
unions.

### Training data format: type-first examples

Instead of just showing the model complete files, create training examples
that emphasize type definitions:

```typescript
// Example 1: Type + implementation pair
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function tryParse<T>(json: string, schema: ZodSchema<T>): Result<T> {
  try {
    return { ok: true, value: schema.parse(JSON.parse(json)) };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e : new Error(String(e)) };
  }
}
```

```typescript
// Example 2: Complex type gymnastics
type DeepPartial<T> = T extends object
  ? { [K in keyof T]?: DeepPartial<T[K]> }
  : T;

type PathKeys<T, Prefix extends string = ""> = T extends object
  ? { [K in keyof T & string]:
      | `${Prefix}${K}`
      | PathKeys<T[K], `${Prefix}${K}.`>
    }[keyof T & string]
  : never;
```

### Synthetic type exercises

Generate training data that specifically exercises the type system:

```
Prompt: "Define a type-safe event emitter where event names and
         their payload types are enforced at compile time."

Solution:
type EventMap = Record<string, any>;

class TypedEmitter<Events extends EventMap> {
  private handlers = new Map<keyof Events, Set<Function>>();

  on<K extends keyof Events>(
    event: K,
    handler: (payload: Events[K]) => void
  ): void {
    // ...
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    // ...
  }
}
```

**Impact:** High. This is the single biggest differentiator for a TS model.
General models are mediocre at complex types because they've seen 100x more
Python (which has no real type system) than TypeScript type gymnastics.

---

## 5. Technique: AST-Level Data Augmentation

**What it is:** Parse TypeScript code into an Abstract Syntax Tree (AST),
transform it, and generate new training examples from the transformed AST.

**Why it works:** The model sees the same logic expressed in different ways,
which teaches it that these are equivalent. This improves generalization.

### Augmentation transforms

**Variable renaming:** Parse the AST, identify all variable declarations,
randomly rename them to different valid names. The model learns that `const
userCount = users.length` and `const numUsers = users.length` mean the
same thing.

**Function reordering:** If a file has multiple independent functions,
shuffle their order. The model learns that function order doesn't affect
correctness (within a scope).

**Style variation:** Convert between equivalent syntax forms:

```typescript
// Original
const add = (a: number, b: number): number => a + b;

// Augmented variant 1: function declaration
function add(a: number, b: number): number {
  return a + b;
}

// Augmented variant 2: explicit return
const add = (a: number, b: number): number => {
  return a + b;
};
```

**Type annotation toggling:** For expressions where TypeScript can infer
the type, randomly add or remove explicit annotations:

```typescript
// With annotation
const names: string[] = users.map((u: User) => u.name);

// Without (inferred)
const names = users.map(u => u.name);
```

**Implementation:** Use the TypeScript compiler API itself (`ts.createSourceFile`,
`ts.transform`) to parse and transform. Since you're building a TS model,
you already have Node.js available.

**Impact:** Medium-high. Data augmentation is well-proven in vision ML and
increasingly used for code. The key is making sure transforms preserve
correctness — which is guaranteed when you work at the AST level.

---

## 6. Technique: Multi-File Context

**What it is:** Instead of training on individual files in isolation, show
the model multiple related files together — like how a developer actually
works.

**The problem with single-file training:** In practice, TypeScript code
spans many files. A component imports types from one file, utilities from
another, hooks from a third. When the model only ever sees isolated files,
it can't learn cross-file patterns.

### Implementation: repo-context windows

```
Training example format:

// File: src/types/user.ts
export interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "user";
}

// File: src/services/userService.ts
import { User } from "../types/user";

export async function getUser(id: string): Promise<User | null> {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) return null;
  return res.json();
}

// File: src/components/UserProfile.tsx
import { User } from "../types/user";
import { getUser } from "../services/userService";

export function UserProfile({ userId }: { userId: string }) {
  // ... model generates the component using the imported types
```

The model sees the type definition, the service that uses it, and the
component that consumes both. It learns that `User.role` is
`"admin" | "user"`, not just `string`.

### How to build this

1. Clone popular TS repos (or process them from StarCoderData's repo metadata).
2. Group files by import graph — files that import each other go together.
3. Concatenate related files into single training examples (with file path
   headers).
4. Respect the context window limit (2048 or 4096 tokens). If the import
   cluster is too large, use BFS from a random starting file and include
   as many neighbors as fit.

**Impact:** High. This is one of the most underexplored techniques for code
models. Current models are surprisingly bad at cross-file reasoning because
they're trained on isolated files. A TS model with multi-file context would
have a real advantage.

---

## 7. Technique: Dependency-Aware Training

**What it is:** Include type definitions for commonly used npm packages in
the training context, so the model learns API surfaces.

**The problem:** When the model sees `import express from "express"`, it
needs to know what `express.Router()` returns, what methods are available,
what the type signatures are. Without this, it's guessing.

### Include `.d.ts` stubs

For the top 100 npm packages, include their type definitions as training
data:

```typescript
// Context: node_modules/@types/express/index.d.ts (abbreviated)
declare namespace Express {
  interface Request {
    body: any;
    params: Record<string, string>;
    query: Record<string, string>;
    headers: Record<string, string>;
  }
  interface Response {
    status(code: number): Response;
    json(body: any): Response;
    send(body: string): Response;
  }
}

// Training example that uses it:
import express, { Request, Response } from "express";

const app = express();

app.get("/users/:id", async (req: Request, res: Response) => {
  const user = await db.users.findById(req.params.id);
  if (!user) return res.status(404).json({ error: "Not found" });
  res.json(user);
});
```

When the type stub and usage appear together in training, the model learns
the correct API surface. It won't hallucinate methods that don't exist.

**Top packages to include:**

```
react, react-dom, next, express, zod, prisma, trpc,
node (built-in types), typescript (built-in lib types),
vitest/jest, tailwindcss, drizzle-orm, hono, bun
```

**Impact:** High for practical code generation. The model's usefulness
depends heavily on whether it knows the APIs developers actually use. This
is a straightforward way to inject that knowledge.

---

## 8. Technique: Test-Driven Training Data

**What it is:** Structure training examples as test → implementation pairs.
The model sees what the code should do (the test) before seeing how to do
it (the implementation).

**Format:**

```typescript
// Test (context)
describe("parseQueryString", () => {
  it("parses simple key-value pairs", () => {
    expect(parseQueryString("a=1&b=2")).toEqual({ a: "1", b: "2" });
  });

  it("handles empty string", () => {
    expect(parseQueryString("")).toEqual({});
  });

  it("decodes URI components", () => {
    expect(parseQueryString("name=hello%20world")).toEqual({ name: "hello world" });
  });

  it("handles duplicate keys as arrays", () => {
    expect(parseQueryString("a=1&a=2")).toEqual({ a: ["1", "2"] });
  });
});

// Implementation (what the model generates)
function parseQueryString(query: string): Record<string, string | string[]> {
  if (!query) return {};
  const result: Record<string, string | string[]> = {};
  for (const pair of query.split("&")) {
    const [key, value] = pair.split("=").map(decodeURIComponent);
    if (key in result) {
      const existing = result[key];
      result[key] = Array.isArray(existing)
        ? [...existing, value]
        : [existing, value];
    } else {
      result[key] = value;
    }
  }
  return result;
}
```

**Why this is powerful:** The test acts as a specification. The model learns
to read tests as requirements and generate code that satisfies them. This
directly improves the model's performance on benchmarks (which are
structured as "given this spec, write the code").

**How to generate this data:**

1. From real repos: find `.test.ts` files, match them with their source
   files by naming convention (`foo.ts` ↔ `foo.test.ts`).
2. Synthetic: generate test-implementation pairs using an LLM.
3. From our HumanEval problems: these are already in this format.

**Impact:** High. This teaches the model the skill that matters most for
practical coding assistance: understanding a specification and producing
correct code.

---

## 9. Technique: Compiler Error Training

**What it is:** Show the model code with type errors, paired with the
compiler's error message and the fix. The model learns to diagnose and fix
TypeScript errors.

**Format:**

```
// Broken code:
function greet(name: string): string {
  return "Hello, " + name;
}

greet(42);

// Error: TS2345: Argument of type 'number' is not assignable to
//        parameter of type 'string'.

// Fixed code:
function greet(name: string): string {
  return "Hello, " + name;
}

greet("Alice");
```

**More complex example:**

```
// Broken code:
interface Config {
  port: number;
  host: string;
}

function startServer(config: Config) {
  console.log(`Starting on ${config.url}`);
}

// Error: TS2339: Property 'url' does not exist on type 'Config'.
//        Did you mean 'host'?

// Fixed code:
function startServer(config: Config) {
  console.log(`Starting on ${config.host}:${config.port}`);
}
```

**How to generate this data at scale:**

1. Take working TypeScript code from repos.
2. Introduce common errors programmatically:
   - Wrong type in argument
   - Missing property access
   - Incorrect generic parameter
   - Missing await on Promise
   - Assigning to readonly property
3. Run `tsc --noEmit` to get the error message.
4. Create the triplet: (broken code, error, fixed code).

This can be fully automated — no LLM needed.

**Impact:** Medium-high. Models trained with error-correction data become
significantly better at debugging tasks. Since TypeScript's compiler errors
are highly structured and deterministic, this is one of the cleanest
training signals you can generate.

---

## 10. Technique: Documentation-Code Alignment

**What it is:** Pair JSDoc/TSDoc comments with the code they describe.
The model learns what code is supposed to do (from the docs) and how it
does it (from the implementation).

**Format:**

```typescript
/**
 * Retries an async operation with exponential backoff.
 *
 * @param fn - The async function to retry
 * @param maxRetries - Maximum number of retry attempts (default: 3)
 * @param baseDelay - Initial delay in ms before first retry (default: 1000)
 * @returns The result of the first successful call
 * @throws The error from the last failed attempt if all retries exhausted
 *
 * @example
 * const data = await retry(() => fetch("/api/data"), 5, 500);
 */
async function retry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000,
): Promise<T> {
  let lastError: Error;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, baseDelay * 2 ** attempt));
      }
    }
  }
  throw lastError!;
}
```

**Why it works for TypeScript specifically:** TypeScript's type annotations
+ JSDoc create a very rich specification layer. The model can learn to
generate code from type signatures and doc comments alone — which is exactly
what developers want during autocomplete.

**How to source this data:**

- Filter for files with high JSDoc coverage (>60% of exported functions
  have JSDoc).
- DefinitelyTyped has extensive documentation in `.d.ts` files.
- Official framework docs often ship as markdown + code blocks.

**Impact:** Medium. Improves the model's ability to generate code from
natural language descriptions and to produce well-documented code.

---

## 11. Technique: Framework-Specific Fine-Tuning

**What it is:** After training the base model on general TypeScript, do a
short fine-tuning phase focused on a specific framework (Next.js, Remix,
NestJS, etc.).

**Why:** Frameworks have conventions that go beyond the language. Next.js has
`page.tsx`, `layout.tsx`, `loading.tsx` file conventions. NestJS has
decorators and dependency injection patterns. A model that knows these
conventions can generate framework-correct code, not just syntactically
valid code.

### Framework training data sources

| Framework | Data Sources |
|-----------|-------------|
| **Next.js** | `examples/` dir in vercel/next.js, App Router docs, popular templates |
| **React** | Official tutorial, react.dev examples, top component libraries |
| **NestJS** | nestjs/nest source, official samples, CLI-generated projects |
| **tRPC** | trpc/trpc source, examples, integration tests |
| **Prisma** | prisma/prisma source, schema files, migration files |
| **Deno** | deno_std library, Fresh framework, official examples |

### Multi-phase training schedule

```
Phase 1 (pre-training):     General TypeScript code      (100K steps)
Phase 2 (specialization):   High-quality TS repos        (20K steps, lower LR)
Phase 3 (framework focus):  Next.js/React-specific code  (5K steps, very low LR)
```

Each phase uses a lower learning rate than the previous one. This is
standard transfer learning: learn broad patterns first, then specialize.

**Impact:** High for framework-specific use cases. A Next.js-tuned model
would know that `export default function Page()` in `app/page.tsx` is a
route handler, while a general model just sees a function export.

---

## 12. Technique: Repo-Level Training

**What it is:** Instead of training on shuffled individual files, train on
complete repositories — preserving the file structure, import relationships,
and project context.

**Standard approach (what we do now):**

```
File 1 from repo A → tokenize → chunk
File 7 from repo B → tokenize → chunk
File 3 from repo A → tokenize → chunk
... (files are shuffled randomly across repos)
```

**Repo-level approach:**

```
Repo A:
  tsconfig.json → package.json → src/index.ts → src/types.ts → src/utils.ts
  → src/routes/users.ts → src/routes/posts.ts → tests/users.test.ts
  [end of repo]

Repo B:
  tsconfig.json → package.json → ...
```

**Why this matters:** The model learns project structure. It learns that
`tsconfig.json` comes with `package.json`, that types are defined in one
file and imported in another, that tests mirror the source structure. This
is implicit knowledge that developers have but models don't.

**Implementation challenge:** Repos are large. A single repo might be 100K+
tokens, but our context window is 2048-4096. Solutions:

1. **Sliding window:** Take a 4096-token window that slides across the
   concatenated repo files. Adjacent windows overlap.
2. **Sampled subgraph:** Pick a random file, include its imports and
   importers, fit as much as possible in the context.
3. **File ordering as context:** Even without seeing all files at once,
   processing a repo's files in import-graph order (dependencies before
   dependents) creates implicit context through the training sequence.

**Impact:** Medium-high. This is an active research area. Google's "repo-level
code completion" work showed significant improvements from repo-level context.
Hard to implement well, but a meaningful differentiator.

---

## 13. Technique: Custom Tokenizer for TypeScript

**What it is:** Train a BPE tokenizer specifically on TypeScript code instead
of using a general-purpose tokenizer trained on all languages.

**Why it matters:** A general tokenizer trained on Python + JS + Java + etc.
makes compromises. It might tokenize `interface` as two tokens (`inter` +
`face`) because `interface` is rare in Python. A TS-specific tokenizer would
make `interface` a single token.

**Benefits of a TS-specific tokenizer:**

- **Shorter sequences:** Common TS tokens (`interface`, `readonly`, `extends`,
  `keyof`, `typeof`, `Promise`, `async`, `await`) become single tokens
  instead of 2-3 pieces. Your 2048-token context window fits more code.
- **Better predictions:** The model predicts one token instead of two-three
  for common keywords. Less room for error.
- **Smaller vocab waste:** No tokens wasted on Python's `self.`, Go's `:=`,
  Rust's `fn`, Java's `public static void`.

**How to do it:**

```bash
# Train tokenizer only on TypeScript data
python scripts/train_tokenizer.py \
  --languages typescript \
  --num-samples 50000 \
  --vocab-size 32768
```

More samples = better tokenizer. 50K TypeScript files gives a strong
vocabulary.

**Measuring the improvement:**

```python
# Compare token counts between general and TS-specific tokenizers
general_tokens = general_tokenizer.encode(ts_code)
ts_tokens = ts_tokenizer.encode(ts_code)
print(f"General: {len(general_tokens)} tokens")
print(f"TS-specific: {len(ts_tokens)} tokens")
# Expect 10-20% fewer tokens with the TS-specific tokenizer
```

10-20% fewer tokens means:
- 10-20% more code fits in each context window
- 10-20% faster training (fewer tokens to process)
- Slightly better prediction accuracy (coarser but more meaningful units)

**Impact:** Medium. A straightforward improvement that compounds with
everything else. Already supported by our existing tokenizer training script.

---

## 14. The Hybrid Approach: Pre-train Multi, Specialize Single

**The best of both worlds.** This is what most production code models do.

### Phase 1: Multi-language pre-training

Train on Python + TypeScript + JavaScript + 3-5 other languages. This builds
foundational knowledge: what a function is, how control flow works, what
variables do, what types mean.

```yaml
# configs/small.yaml (original multi-language config)
data:
  languages: ["python", "typescript", "javascript", "java", "go", "rust"]
training:
  max_steps: 100000
  learning_rate: 6.0e-4
```

### Phase 2: TypeScript specialization

Take the multi-language checkpoint. Fine-tune on TypeScript-only data with
a lower learning rate.

```yaml
# configs/ts-specialize.yaml
data:
  languages: ["typescript"]
  # Also include: .d.ts files, tsconfig.json, package.json
training:
  max_steps: 30000       # Fewer steps than pre-training
  learning_rate: 2.0e-4  # 3x lower than pre-training
  warmup_steps: 500
```

### Phase 3 (optional): Framework fine-tuning

Narrow further to a specific framework.

```yaml
# Even lower LR, even fewer steps
training:
  max_steps: 5000
  learning_rate: 5.0e-5
```

### Why this works better than TS-only from scratch

The multi-language phase teaches general programming concepts efficiently.
The model learns what a `for` loop is from seeing it in 6 languages — each
one reinforces the concept. By the time it specializes in TypeScript, it
already understands programming. The TS specialization phase then teaches
TS-specific patterns (types, generics, decorators, JSX) on top of that
foundation.

A from-scratch TS model has to learn both "what is programming" and "what is
TypeScript" from TypeScript alone. It works, but it's less sample-efficient.

**Impact:** This is the recommended approach. Pre-train multi (which our
default configs already do), then specialize.

---

## 15. Scaling: What Changes at 1B+ Parameters

If you access cloud GPUs and train a large (1B+) TypeScript-specialized
model, some dynamics change.

### More capacity = deeper patterns

At 125M, the model learns syntax and common patterns.
At 350M, it learns idioms and API usage.
At 1B+, it starts learning:

- **Algorithmic patterns:** Sorting, searching, tree traversal in TS.
- **Design patterns:** Observer, factory, strategy — in TypeScript idioms.
- **Type system depth:** Conditional types, mapped types, template literal
  types, infer keyword, recursive types.
- **Framework conventions:** File-based routing, middleware chains, hook
  composition.

### Data becomes the bottleneck

At 1B params, you need roughly 20-50B tokens for proper training. If you're
TS-only, that means you need a LOT of TypeScript code — or you need to
augment heavily with synthetic data and augmentation techniques.

**Rough calculation:**

```
StarCoderData TypeScript split: ~10-15B tokens
DefinitelyTyped:                ~2B tokens
Add .d.ts, tests, configs:     ~3B tokens
Total real TS data:             ~15-20B tokens

To reach 50B, you need:
- 2-3 epochs over the real data (diminishing returns after 2)
- Plus synthetic data to fill the gap
- Plus augmented variants from AST transforms
```

### Multi-language pre-training becomes more important

At small scale (125M), going TS-only is competitive with hybrid because
capacity is the bottleneck. At 1B+, capacity is less constrained and data
diversity matters more. The hybrid approach pulls ahead clearly at this
scale.

---

## 16. Practical Plan: TypeScript-Specialized Cola-Coder

If you want to build this, here's the concrete path.

### Step 1: Retrain the tokenizer on TypeScript

```bash
python scripts/train_tokenizer.py \
  --languages typescript \
  --num-samples 50000 \
  --output tokenizer-ts.json
```

### Step 2: Prepare TypeScript-heavy data

Create a new config:

```yaml
# configs/ts-specialist.yaml
model:
  vocab_size: 32768
  dim: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 4
  max_seq_len: 2048

training:
  batch_size: 8
  gradient_accumulation: 4
  learning_rate: 6.0e-4
  max_steps: 100000
  precision: "bf16"

data:
  dataset: "bigcode/starcoderdata"
  languages: ["typescript"]
  max_tokens_per_file: 4096
```

```bash
python scripts/prepare_data.py \
  --config configs/ts-specialist.yaml \
  --tokenizer tokenizer-ts.json
```

### Step 3: Train

```bash
python scripts/train.py --config configs/ts-specialist.yaml
```

### Step 4: Evaluate on TypeScript problems

Extend the HumanEval problems in `evaluation/humaneval.py` with TypeScript
versions. You'd define the prompts in TypeScript syntax and write JS-based
test execution.

### Step 5: Reasoning fine-tuning with TS examples

Add TypeScript-specific chain-of-thought examples to `reasoning/cot_data.py`
and run GRPO.

### Alternative: Hybrid approach

```bash
# Phase 1: Multi-language pre-training (use existing small config)
python scripts/train.py --config configs/small.yaml

# Phase 2: TypeScript specialization (use ts-specialist config with lower LR)
# Create a fine-tuning config that loads from the multi-language checkpoint
python scripts/train.py \
  --config configs/ts-specialist.yaml \
  --resume ./checkpoints/small/latest
```

For the hybrid approach, modify `ts-specialist.yaml` to use a lower
learning rate (2e-4 instead of 6e-4) and fewer steps (30K instead of 100K).

---

## Summary

| Technique | Effort | Impact | Best For |
|-----------|--------|--------|----------|
| TS-only tokenizer | Low | Medium | All setups |
| TS-only data filtering | Low | Medium | All setups |
| Type-aware training data | Medium | High | Type system quality |
| Multi-file context | Medium | High | Import-aware completions |
| Dependency-aware (.d.ts) | Medium | High | Framework/API usage |
| Test-driven data | Medium | High | Correctness |
| Compiler error training | Medium | Medium-high | Debugging assistance |
| AST augmentation | High | Medium-high | Data efficiency |
| Repo-level training | High | Medium-high | Project-level understanding |
| Framework fine-tuning | Low | High (per framework) | Framework-specific use |
| Synthetic data | Medium | High | Small model performance |
| Hybrid pre-train/specialize | Medium | Highest | Overall best approach |

The core answer to your question: **yes, a model trained exclusively on
high-quality TypeScript data would be significantly better at TypeScript than
a general model of the same size.** The gap is roughly 30-50% on benchmarks
at the 125M scale. With the techniques above, you can push it even further.
