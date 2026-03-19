# Multi-Agent Specialization: One Brain, Many Experts

What if instead of building one model that tries to know everything, you
build a system where a large general model delegates to small specialist
models? A "lead developer" that calls on a React expert, a GraphQL expert,
a TypeORM expert — each one deeply trained on their domain.

This is not hypothetical. This architecture works, it's trainable on
consumer hardware, and it's arguably better than a single monolithic model
at every scale.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Why This Beats a Single Model](#2-why-this-beats-a-single-model)
3. [Architecture: Router + Specialists](#3-architecture-router--specialists)
4. [What the Router Model Does](#4-what-the-router-model-does)
5. [What Specialist Models Do](#5-what-specialist-models-do)
6. [How to Train the Specialists](#6-how-to-train-the-specialists)
7. [How to Train the Router](#7-how-to-train-the-router)
8. [The Routing Problem: How to Pick the Right Expert](#8-the-routing-problem-how-to-pick-the-right-expert)
9. [Concrete Example: A Full-Stack TypeScript System](#9-concrete-example-a-full-stack-typescript-system)
10. [Inference: How the System Actually Runs](#10-inference-how-the-system-actually-runs)
11. [VRAM Budget: Can This Run Locally?](#11-vram-budget-can-this-run-locally)
12. [Comparison to Mixture of Experts (MoE)](#12-comparison-to-mixture-of-experts-moe)
13. [Comparison to Tool Use / Function Calling](#13-comparison-to-tool-use--function-calling)
14. [Hybrid: API Router + Local Specialists](#14-hybrid-api-router--local-specialists)
15. [Training Pipeline for Cola-Coder Multi-Agent](#15-training-pipeline-for-cola-coder-multi-agent)
16. [What Could Go Wrong](#16-what-could-go-wrong)
17. [The Honest Assessment: When This Wins and When It Doesn't](#17-the-honest-assessment-when-this-wins-and-when-it-doesnt)

---

## 1. The Core Idea

Instead of one model doing everything:

```
User prompt → [Single 350M General Model] → Generated code
                     ↑
              Knows a little about everything.
              Expert at nothing.
```

Build a system:

```
User prompt → [Router Model]
                    |
          ┌────────┼────────┐
          ↓        ↓        ↓
     [React      [GraphQL   [TypeORM
     Specialist]  Specialist] Specialist]
          ↓        ↓        ↓
          └────────┼────────┘
                   ↓
        [Router assembles final output]
                   ↓
            Generated code
```

The router is the senior dev. It reads the prompt, decides which
specialist(s) to call, passes them the relevant context, and combines
their output into a coherent response.

Each specialist is a small model (50M-125M) trained deeply on one
framework or domain. It doesn't need to know about anything else.

---

## 2. Why This Beats a Single Model

### The parameter budget argument

A 350M model spread across 6 frameworks:

```
350M / 6 frameworks = ~58M effective parameters per framework
```

Six 50M specialists + one 125M router:

```
Router:     125M parameters (understands task structure, delegates)
React:       50M parameters (ALL dedicated to React patterns)
GraphQL:     50M parameters (ALL dedicated to GraphQL)
TypeORM:     50M parameters (ALL dedicated to TypeORM)
Next.js:     50M parameters (ALL dedicated to Next.js)
Prisma:      50M parameters (ALL dedicated to Prisma)
Zod:         50M parameters (ALL dedicated to Zod)

Total:      475M parameters
Active:     175M per request (router + 1 specialist)
```

Each specialist has 50M params fully dedicated to its domain — comparable
to the general model's effective budget per domain, but without any
capacity wasted on unrelated frameworks. And the router has 125M params
dedicated solely to understanding what the user wants and coordinating
the response.

### The training data argument

A general model trains on ALL code. The React specialist trains ONLY on:
- React source code
- React component libraries
- React tutorials and docs
- React test files
- JSX/TSX patterns

Per training step, the specialist sees more relevant examples than the
general model ever would. It doesn't waste any steps learning Express
middleware patterns or Prisma schema syntax.

### The interference argument

When a single model learns React patterns AND GraphQL patterns, they can
interfere. The model's weights are shared — learning new GraphQL patterns
can slightly degrade React performance (this is called "catastrophic
forgetting"). Specialists don't have this problem because they never learn
conflicting domains.

### The update argument

When Next.js 15 ships with new patterns, you retrain one 50M specialist.
Training time: hours. With a single 350M model, you'd retrain the whole
thing — days or weeks — and risk degrading performance on everything else.

---

## 3. Architecture: Router + Specialists

### The three components

```
┌─────────────────────────────────────────────────────────┐
│                     ROUTER MODEL                         │
│                                                         │
│  Input:   User prompt + conversation context            │
│  Job:     1. Understand what the user wants              │
│           2. Decide which specialist(s) to call          │
│           3. Extract the relevant context for each       │
│           4. Combine specialist outputs                  │
│  Size:    125M-350M (needs general understanding)        │
│  Trained: On routing examples + general code             │
└─────────────────────────────────────────────────────────┘
              |              |              |
              v              v              v
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ SPECIALIST: React│ │ SPECIALIST: GQL │ │ SPECIALIST: ORM │
│                 │ │                 │ │                 │
│ Input: Focused  │ │ Input: Focused  │ │ Input: Focused  │
│        sub-task │ │        sub-task │ │        sub-task │
│ Output: Code in │ │ Output: Code in │ │ Output: Code in │
│         domain  │ │         domain  │ │         domain  │
│ Size: 50M-125M  │ │ Size: 50M-125M  │ │ Size: 50M-125M  │
│ Trained: React  │ │ Trained: GQL    │ │ Trained: ORM    │
│         only    │ │         only    │ │         only    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
              |              |              |
              v              v              v
┌─────────────────────────────────────────────────────────┐
│                    ASSEMBLER                             │
│                                                         │
│  Can be the router model again, or a simple combiner.   │
│  Merges specialist outputs into a coherent response.    │
│  Handles imports, naming consistency, integration.       │
└─────────────────────────────────────────────────────────┘
```

### What makes this different from just "calling multiple models"

The key innovation is **structured delegation**. The router doesn't just
blindly forward the prompt. It:

1. **Decomposes** the task into framework-specific sub-tasks.
2. **Provides context** each specialist needs (types, interfaces, existing
   code) without flooding it with irrelevant information.
3. **Validates** specialist output before including it in the final response.
4. **Resolves conflicts** between specialists (e.g., naming inconsistencies).

---

## 4. What the Router Model Does

The router is a general-purpose model that understands code at a structural
level. It doesn't need to know React's `useEffect` API — it needs to know
that the user's prompt involves React and should be routed to the React
specialist.

### Router input/output format

```
<|system|>You are a code task router. Analyze the user's request and
determine which specialist(s) should handle it. Extract the relevant
context for each specialist.

Available specialists: react, graphql, typeorm, nextjs, prisma, zod

<|user|>Create an API endpoint that fetches users from the database
using TypeORM, validates the response with Zod, and returns it as a
GraphQL resolver.

<|router|>
<route to="typeorm">
  Task: Write a TypeORM query to fetch all users with their roles.
  Entity context: User entity has id, name, email, role fields.
  Return: The repository query function.
</route>
<route to="zod">
  Task: Write a Zod schema to validate a User object.
  Fields: id (string uuid), name (string), email (string email),
          role (enum: admin, user, viewer).
  Return: The schema definition and inferred type.
</route>
<route to="graphql">
  Task: Write a GraphQL resolver that calls a data-fetching function
        and validates the result with a schema.
  Context: The function is called `getUsers()` and returns `Promise<User[]>`.
           The validation schema is called `UserSchema`.
  Return: The resolver with type definitions.
</route>
```

The router's output is structured — it tells the system which specialists
to call and what to pass them. This is a skill that can be trained with
relatively little data because it's about task decomposition, not code
generation.

### What the router needs to be good at

1. **Framework detection:** Recognizing which technologies are involved
   from natural language descriptions.
2. **Task decomposition:** Breaking a complex request into independent
   sub-tasks that can run in parallel.
3. **Context extraction:** Pulling out the relevant types, interfaces,
   and constraints each specialist needs.
4. **Output assembly:** Combining specialist outputs with correct imports,
   consistent naming, and proper integration code.

### What the router does NOT need

- Deep knowledge of any framework's API.
- The ability to write complex React components or GraphQL schemas.
- Knowledge of framework-specific best practices.

That's what the specialists are for.

---

## 5. What Specialist Models Do

Each specialist is a small model that is extremely good at one thing.

### React Specialist (50M-125M)

**Training data:**
- React source code (facebook/react)
- Top 500 React component libraries by stars
- React official docs and tutorials
- JSX/TSX files from popular projects
- React testing patterns (React Testing Library, Vitest)
- Hooks patterns, context patterns, performance patterns

**What it generates:**
- Functional components with proper hook usage
- Custom hooks
- Context providers and consumers
- Component composition patterns
- Proper memo/callback/ref usage
- Accessible JSX with ARIA attributes

**What it doesn't know:**
- How to write a GraphQL schema
- How to define a TypeORM entity
- How to configure Next.js routing
- Any backend pattern

### GraphQL Specialist (50M-125M)

**Training data:**
- graphql-js source code
- Apollo Server/Client source and examples
- GraphQL schema definitions from popular APIs
- Resolver implementations
- Type definitions and code generation patterns
- Subscription and real-time patterns

**What it generates:**
- Type definitions (SDL and code-first)
- Resolvers with proper typing
- DataLoader patterns for N+1 prevention
- Input validation
- Error handling in GraphQL style
- Subscription resolvers

### TypeORM Specialist (50M-125M)

**Training data:**
- TypeORM source code and docs
- Entity definitions with decorators
- Repository patterns and query builders
- Migration files
- Relationship patterns (OneToMany, ManyToMany, etc.)
- Transaction handling

**What it generates:**
- Entity classes with proper decorators
- Repository queries
- Complex joins and subqueries
- Migration code
- Relationship configuration

### You get the idea.

Each specialist is narrow but deep. It's seen thousands of examples of
its specific framework and nothing else. Within its domain, it's more
capable than a general model 5-10x its size.

---

## 6. How to Train the Specialists

Each specialist follows the single-language specialization playbook from
the previous deep-dive, but narrowed to a single framework.

### Step 1: Collect framework-specific data

```python
# For a React specialist, collect:
sources = {
    "core": ["facebook/react source code"],
    "libraries": [
        # Top component libraries
        "shadcn/ui", "radix-ui", "chakra-ui", "mantine",
        "react-hook-form", "tanstack/query", "tanstack/router",
        "zustand", "jotai", "react-spring",
    ],
    "examples": [
        # Official examples and tutorials
        "react.dev code samples",
        "Next.js app directory examples",
        "Remix examples",
    ],
    "types": [
        "@types/react",
        "@types/react-dom",
    ],
}
```

### Step 2: Custom tokenizer per specialist (optional but recommended)

A React tokenizer would make `useState`, `useEffect`, `className`, `onClick`
single tokens. A TypeORM tokenizer would make `@Entity`, `@Column`,
`@ManyToOne` single tokens.

```bash
# Train tokenizer on React code only
python scripts/train_tokenizer.py \
  --data ./data/react/ \
  --vocab-size 16384 \
  --output tokenizer-react.json
```

Smaller vocab (16K vs 32K) is fine for specialists because their domain
vocabulary is smaller.

### Step 3: Train from a base checkpoint (recommended)

Start from your general TypeScript model, then specialize:

```bash
# Fine-tune the small model on React-only data
python scripts/train.py \
  --config configs/specialist-react.yaml \
  --resume ./checkpoints/small/latest
```

With a specialist config using lower learning rate and fewer steps:

```yaml
# configs/specialist-react.yaml
model:
  vocab_size: 32768
  dim: 512
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  max_seq_len: 2048

training:
  batch_size: 16
  gradient_accumulation: 2
  learning_rate: 1.0e-4   # Low — fine-tuning, not from scratch
  max_steps: 20000        # React corpus is smaller
  precision: "bf16"

data:
  dataset: "local"        # Pre-collected React data
  data_dir: "./data/react"
```

### Step 4: Generate synthetic training data for the specialist

```
Prompt to Claude/GPT-4:

"Write a React component that implements {concept}.
 Use functional components, TypeScript, and modern React patterns.
 Include proper type definitions and JSDoc.
 Handle loading, error, and empty states."

Concepts:
- "infinite scroll with virtualization"
- "optimistic UI updates with rollback"
- "compound component pattern for a form builder"
- "real-time search with debounced API calls"
- etc.
```

This is cheaper per specialist because you need fewer examples (5K-10K)
to cover a single framework well.

---

## 7. How to Train the Router

The router is different from the specialists. It's not generating framework
code — it's generating routing decisions and assembly instructions.

### Training data format

```
Input: "Build a user registration form with email validation,
        store in the database, and show a success toast"

Output:
<route to="zod">
  Define a registration schema: email (email format), password (min 8 chars),
  name (string, min 2 chars). Export the schema and inferred type.
</route>
<route to="react">
  Build a registration form component using react-hook-form.
  Props: onSubmit(data: RegistrationData) => Promise<void>.
  Show validation errors inline. Show a loading spinner during submission.
  Types needed: RegistrationData = { email: string, password: string, name: string }
</route>
<route to="typeorm">
  Define a User entity with id (uuid), email (unique), passwordHash, name,
  createdAt. Write a createUser function that hashes the password and saves.
</route>
<assemble>
  1. Import Zod schema and inferred type
  2. Import React form component
  3. Import createUser from the ORM layer
  4. Wire: form.onSubmit → validate with schema → call createUser → show toast
  5. Handle errors: validation → show inline, server → show toast
</assemble>
```

### How to generate router training data

You don't need a huge amount — routing is a much simpler task than code
generation. A few thousand examples cover most patterns.

**Method 1: Template-based generation**

Write templates for common multi-framework tasks:

```python
tasks = [
    {
        "prompt": "CRUD API for {entity}",
        "routes": ["typeorm", "zod", "graphql"],
    },
    {
        "prompt": "Form that submits to {endpoint}",
        "routes": ["react", "zod"],
    },
    {
        "prompt": "Real-time {feature} with WebSockets",
        "routes": ["react", "graphql"],
    },
]

# Fill in entities, endpoints, features from a list
# Generate 500-1000 examples from 50 templates
```

**Method 2: LLM-generated routing examples**

Ask Claude/GPT-4 to decompose coding tasks:

```
"Given this coding request, decompose it into sub-tasks for these
 specialists: react, graphql, typeorm, prisma, zod, nextjs.
 Show which specialists should be called and what context each needs.

 Request: {complex_coding_task}"
```

**Training the router:**

The router can be trained with standard supervised fine-tuning. The task
is sequence-to-sequence: given a user prompt, generate the routing
instructions. This is much easier than code generation and can be trained
on a smaller model (125M is plenty).

---

## 8. The Routing Problem: How to Pick the Right Expert

This is the hardest part of the system. Bad routing = the whole system fails.

### Simple approach: keyword matching

```python
FRAMEWORK_KEYWORDS = {
    "react": ["component", "hook", "usestate", "useeffect", "jsx", "tsx",
              "props", "context", "render", "react", "dom"],
    "graphql": ["query", "mutation", "resolver", "schema", "graphql",
                "subscription", "typedefs", "apollo"],
    "typeorm": ["entity", "repository", "column", "migration", "typeorm",
                "relation", "onetomany", "manytomany", "querybuilder"],
    "nextjs": ["page", "layout", "middleware", "api route", "server component",
               "client component", "next", "app router", "getserversideprops"],
    "prisma": ["prisma", "model", "schema.prisma", "findmany", "findunique",
               "prisma client"],
    "zod": ["schema", "validate", "parse", "zod", "z.object", "z.string",
            "infer", "safeParse"],
}

def route_by_keywords(prompt: str) -> list[str]:
    prompt_lower = prompt.lower()
    scores = {}
    for framework, keywords in FRAMEWORK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[framework] = score
    return sorted(scores, key=scores.get, reverse=True)
```

This works surprisingly well for explicit requests ("write a React
component") but fails on implicit ones ("build a signup page" — doesn't
mention React).

### Better approach: trained classifier

A tiny classification head on top of embeddings:

```python
class RouterClassifier:
    """Classify which specialists a prompt needs.

    This is a multi-label classifier — a prompt can need multiple
    specialists (e.g., React + GraphQL + Zod).
    """

    def __init__(self, embedding_dim: int, num_specialists: int):
        # Small MLP on top of the router model's embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_specialists),
            nn.Sigmoid(),  # Multi-label: each specialist is independent
        )

    def predict(self, prompt_embedding: torch.Tensor) -> dict[str, float]:
        scores = self.classifier(prompt_embedding)
        return {
            name: score.item()
            for name, score in zip(SPECIALIST_NAMES, scores[0])
        }
```

Train on a few thousand labeled (prompt → specialist list) examples. This
is a simple classification task — not a generative one — so it trains fast
and needs little data.

### Best approach: let the router model decide

If the router model is trained on routing examples (Section 7), it generates
the routing decision as text. No separate classifier needed — routing is
part of the model's output.

This is the most flexible because the router can explain WHY it chose
each specialist and provide custom context. But it's slower (full model
inference for routing).

### Fallback strategy

When the router isn't confident, fall back to the general model:

```python
def route_request(prompt: str) -> list[str]:
    specialists = router.predict(prompt)
    confident = [s for s, score in specialists.items() if score > 0.7]

    if not confident:
        # No clear specialist match — use general model
        return ["general"]

    return confident
```

This is critical for the "just works or falls back" principle. Unknown
domains, ambiguous requests, or purely algorithmic tasks that don't map to
any framework → use the general model.

---

## 9. Concrete Example: A Full-Stack TypeScript System

Here's what a complete multi-agent code generation system looks like for
a TypeScript full-stack developer.

### The specialists

```
┌──────────────────────────────────────────────────────────────┐
│ Router (125M)     — Task decomposition, context extraction   │
│ General TS (125M) — Fallback for non-framework code         │
│ React (50M)       — Components, hooks, JSX, state mgmt     │
│ Next.js (50M)     — App router, server components, API      │
│ GraphQL (50M)     — Schemas, resolvers, Apollo patterns     │
│ Prisma/ORM (50M)  — Database models, queries, migrations    │
│ Zod (50M)         — Validation schemas, type inference      │
│ Testing (50M)     — Vitest, React Testing Library, mocks    │
│                                                              │
│ Total: 550M params. Active per request: ~175M               │
└──────────────────────────────────────────────────────────────┘
```

### Example flow

**User prompt:** "Add a settings page where users can update their email
and notification preferences. Validate the input, save to the database,
and show a success message."

**Router output:**

```
Analysis: Full-stack feature touching UI, validation, API, and database.

1. [prisma] Define NotificationPreferences model with fields:
   userId, emailNotifications (bool), pushNotifications (bool),
   marketingEmails (bool). Add relation to User model.

2. [zod] Create UpdateSettingsSchema:
   email: z.string().email(),
   emailNotifications: z.boolean(),
   pushNotifications: z.boolean(),
   marketingEmails: z.boolean()

3. [nextjs] Create server action in app/settings/actions.ts:
   - Validate input with UpdateSettingsSchema
   - Update user email in database
   - Upsert notification preferences
   - Revalidate the settings page

4. [react] Create SettingsForm component:
   - Props: initialData (current settings)
   - Use react-hook-form with zodResolver
   - Toggle switches for notification preferences
   - Email input with validation
   - Submit button with loading state
   - Success toast using sonner

5. [nextjs] Create app/settings/page.tsx:
   - Server component that fetches current settings
   - Passes to SettingsForm as initialData
```

**Each specialist generates its piece.** The router then assembles with
correct imports and integration.

### The output

A complete, working feature across 4-5 files, each generated by the
specialist that knows that domain best. The React specialist writes better
components than a general model. The Prisma specialist writes better
schemas. The Next.js specialist knows server actions and revalidation.

---

## 10. Inference: How the System Actually Runs

### Sequential routing (simple, slower)

```python
def generate_multi_agent(prompt: str) -> str:
    # Step 1: Router analyzes and decomposes
    routing = router_model.generate(format_routing_prompt(prompt))
    sub_tasks = parse_routing_output(routing)

    # Step 2: Call each specialist
    specialist_outputs = {}
    for task in sub_tasks:
        specialist = load_specialist(task.specialist_name)
        output = specialist.generate(task.context)
        specialist_outputs[task.specialist_name] = output
        unload_specialist(specialist)  # Free VRAM

    # Step 3: Assemble
    assembly_prompt = format_assembly_prompt(prompt, specialist_outputs)
    final_output = router_model.generate(assembly_prompt)

    return final_output
```

Key detail: **load and unload specialists one at a time.** They share GPU
memory. Only one specialist is in VRAM at any moment. This means you only
need VRAM for the router + one specialist (175M-250M), not all of them.

### Parallel routing (faster, more VRAM)

If you have enough VRAM (24GB+), run independent specialists in parallel:

```python
import asyncio

async def generate_parallel(prompt: str) -> str:
    routing = router_model.generate(format_routing_prompt(prompt))
    sub_tasks = parse_routing_output(routing)

    # Run independent specialists concurrently
    results = await asyncio.gather(*[
        run_specialist(task) for task in sub_tasks
    ])

    # Assemble
    return router_model.generate(
        format_assembly_prompt(prompt, dict(zip(sub_tasks, results)))
    )
```

### Inference latency

```
Sequential:
  Router analysis:        ~200ms
  Specialist 1 (load):    ~500ms
  Specialist 1 (generate): ~1-3s
  Specialist 2 (load):    ~500ms
  Specialist 2 (generate): ~1-3s
  Specialist 3 (load):    ~500ms
  Specialist 3 (generate): ~1-3s
  Assembly:               ~500ms
  Total:                  ~5-12s

Parallel (enough VRAM):
  Router analysis:        ~200ms
  All specialists:        ~1-3s (parallel)
  Assembly:               ~500ms
  Total:                  ~2-4s
```

Slower than a single model call (~1-3s), but the output quality is
higher because each piece is generated by a domain expert.

---

## 11. VRAM Budget: Can This Run Locally?

### Sequential loading (minimum VRAM)

Only the router + one specialist in memory at a time:

```
Router (125M, bf16):       ~250 MB
Specialist (50M, bf16):    ~100 MB
KV-cache + overhead:       ~500 MB
                           --------
Total:                     ~850 MB
```

This fits on literally any GPU. Even a 4GB GPU can run this. The tradeoff
is latency — loading/unloading specialists takes ~500ms each.

### Keep everything loaded (fast, more VRAM)

All models resident in memory:

```
Router (125M):               ~250 MB
6 specialists (50M each):    ~600 MB
General fallback (125M):     ~250 MB
KV-caches + overhead:        ~1 GB
                              --------
Total:                        ~2.1 GB
```

Fits easily on any 4GB+ GPU. All models are tiny. Inference is fast
because there's no loading/unloading.

### With larger models

If you use a 350M router and 125M specialists:

```
Router (350M):               ~700 MB
6 specialists (125M each):   ~1.5 GB
General fallback (350M):     ~700 MB
KV-caches + overhead:        ~2 GB
                              --------
Total:                        ~4.9 GB
```

Still fits on an 8GB GPU. This is the nice thing about this architecture —
the total parameter count is large (1.1B) but only a fraction is active per
request, and all models are small enough to have tiny VRAM footprints.

### Training VRAM

You train each model independently. The router and each specialist are
trained separately. Maximum training VRAM is determined by the SINGLE
largest model (the router), which is much smaller than a monolithic model
of the same total parameter count.

```
Train a 125M router:      ~6.5 GB VRAM  (your RTX 3080/4080 handles this)
Train a 50M specialist:   ~3.6 GB VRAM  (fits on anything)
```

Compare: training a single 550M monolithic model would need ~12 GB with
gradient checkpointing.

---

## 12. Comparison to Mixture of Experts (MoE)

MoE (Mixture of Experts) is an architecture used by models like Mixtral,
GPT-4 (rumored), and Switch Transformer. It's similar in spirit but
different in implementation.

### How MoE works

```
Standard transformer:
  Input → [FFN layer] → Output
  (one FFN, always active)

MoE transformer:
  Input → [Router] → selects 2 of 8 expert FFNs → Output
  (8 expert FFNs, only 2 active per token)
```

MoE operates at the **layer level** — inside a single model, each token
is routed to different expert sub-networks within each transformer block.
The routing happens per-token, thousands of times per forward pass.

### How our multi-agent system works

Our system operates at the **task level** — the entire prompt is routed to
a specialist model that handles it completely. The routing happens once per
request, not per token.

### Comparison

| Aspect | MoE (Mixtral-style) | Multi-Agent (ours) |
|--------|--------------------|--------------------|
| Routing granularity | Per token | Per task/request |
| Expert type | FFN sub-layers | Complete models |
| Routing decision | Learned gate network | Trained router model |
| Training | All experts train together | Each expert trains independently |
| Expert specialization | Emergent (model decides) | Explicit (we decide) |
| Total params | 8x47B = 376B (Mixtral) | Router + 6 specialists = ~500M |
| Active params | 2x47B = 94B per token | 175M per request |
| Can add new expert | No (requires full retrain) | Yes (train new specialist) |
| Can update one expert | No (entangled weights) | Yes (retrain just that specialist) |

### Key advantage of our approach

**Modularity.** You can add a new specialist (say, a Drizzle ORM specialist)
without touching anything else. Train it independently, plug it into the
routing table, done. In MoE, adding a new expert means retraining the
entire model.

**Interpretability.** You can see exactly which specialist was called and
what it generated. In MoE, the routing is opaque — you can't easily tell
which expert handled which token.

**Trainability on consumer hardware.** Each component is small enough to
train on a single consumer GPU. MoE models require multi-GPU setups just
for inference.

---

## 13. Comparison to Tool Use / Function Calling

Modern LLMs can call tools — search the web, run code, query APIs. How is
this different from our specialist models?

### Tool use

```
LLM generates: "I need to search for React form validation patterns"
                → calls search_web("React form validation")
                → gets results
                → incorporates into response
```

The LLM is still doing ALL the code generation. Tools provide information,
not code. The LLM reads the search results and writes the code itself.

### Our approach

```
Router generates: "This needs a React form with Zod validation"
                  → calls react_specialist("form component for settings")
                  → react_specialist GENERATES the component code
                  → calls zod_specialist("settings validation schema")
                  → zod_specialist GENERATES the schema code
                  → router assembles the pieces
```

The specialists ARE the code generators. They don't provide information
for the router to use — they write the actual code. The router just
coordinates.

### Can you combine both?

Yes. And you should. The router model can use both tools AND specialists:

```
Router decides:
  1. Call search_web() to find the latest Next.js 15 API conventions
  2. Route to next_specialist with the search results as context
  3. Route to react_specialist for the frontend component
  4. Assemble

The router uses tools for information gathering
and specialists for code generation.
```

---

## 14. Hybrid: API Router + Local Specialists

Here's a pragmatic architecture that gets you started quickly:

**Router:** Use Claude or GPT-4 via API. They're excellent at task
decomposition, context extraction, and assembly — the exact skills the
router needs. Cost: $0.002-0.01 per routing call.

**Specialists:** Your locally trained Cola-Coder models. Small, fast,
free to run, specialized in your exact tech stack.

```
User prompt
     ↓
[Claude API: routing & decomposition]     ← $0.005 per request
     ↓              ↓              ↓
[Local React    [Local GraphQL  [Local Prisma
 50M model]     50M model]      50M model]    ← Free, runs on your GPU
     ↓              ↓              ↓
[Claude API: assembly & polish]               ← $0.005 per request
     ↓
Final output
```

**Total cost per request:** ~$0.01 (the API calls).
**Quality:** API-grade routing + specialist-grade code generation.
**Latency:** ~3-5 seconds (API + local inference).

### Why this is clever

- The expensive API model handles the EASY part (routing is a simple task
  for a 200B model — it doesn't use much of the model's capacity).
- The cheap local models handle the HARD part (generating framework-specific
  code — this is where domain expertise matters most).
- You could gradually replace the API router with a local 125M router as
  you collect enough routing examples from the API model's outputs.

### Gradually replacing the API

```
Phase 1: Claude routes, local specialists generate     (now)
Phase 2: Collect Claude's routing decisions as training data
Phase 3: Train local 125M router on collected data     (weeks later)
Phase 4: Local router + local specialists              (fully local)
```

Each Claude routing call becomes a training example for your local router.
After a few thousand examples, your local router can handle most cases. Use
Claude as a fallback for edge cases.

---

## 15. Training Pipeline for Cola-Coder Multi-Agent

Here's how to build this with the existing Cola-Coder codebase.

### Step 1: Train the base TypeScript model

```bash
# This is your foundation — general TypeScript capability
make train-small
```

### Step 2: Create specialist datasets

For each framework, collect data into separate directories:

```
data/specialists/
├── react/          # .tsx files, hooks, components
├── graphql/        # schemas, resolvers, type defs
├── prisma/         # schema.prisma, queries, migrations
├── nextjs/         # page.tsx, layout.tsx, server actions
├── zod/            # schemas, validators, type inference
└── testing/        # .test.ts, .spec.ts files
```

### Step 3: Train each specialist

```bash
# Each one starts from the base model and specializes
for specialist in react graphql prisma nextjs zod testing; do
    python scripts/train.py \
        --config configs/specialist-${specialist}.yaml \
        --resume ./checkpoints/small/latest
done
```

Each takes ~1-2 days on the tiny config, ~3-5 days on small.

### Step 4: Train the router

```bash
# Generate routing examples (or use Claude API to bootstrap)
python scripts/generate_routing_data.py

# Train the router model
python scripts/train.py \
    --config configs/router.yaml \
    --data ./data/routing/train_data.npy
```

### Step 5: Build the orchestration layer

Create a new module `src/cola_coder/agents/` that ties it all together:

```python
# src/cola_coder/agents/orchestrator.py (sketch)

class MultiAgentOrchestrator:
    def __init__(self, router_checkpoint, specialist_dir):
        self.router = load_model(router_checkpoint)
        self.specialist_checkpoints = {
            name: path
            for name, path in discover_specialists(specialist_dir)
        }
        self.active_specialist = None

    def generate(self, prompt: str) -> str:
        # Route
        routing = self.router.generate(format_routing(prompt))
        tasks = parse_routes(routing)

        # Generate with specialists
        results = {}
        for task in tasks:
            specialist = self.load_specialist(task.name)
            results[task.name] = specialist.generate(task.context)

        # Assemble
        return self.router.generate(
            format_assembly(prompt, results)
        )
```

---

## 16. What Could Go Wrong

### Bad routing kills everything

If the router sends a GraphQL task to the React specialist, you get
garbage. The quality floor is determined by routing accuracy.

**Mitigation:** Conservative routing with a general-model fallback. If
the router's confidence is below a threshold, use the general model.

### Context loss between router and specialist

The router extracts context for each specialist, but it might miss
something important. The specialist generates code that's correct in
isolation but doesn't integrate properly.

**Mitigation:** Include more context than you think is necessary. Pass
relevant type definitions, import paths, and naming conventions to every
specialist. Over-context is better than under-context.

### Inconsistent output between specialists

Specialist A uses `userId`, specialist B uses `user_id`, specialist C
uses `UserID`. The assembled output has inconsistent naming.

**Mitigation:** The router's assembly step should enforce consistency.
Alternatively, establish naming conventions in each specialist's system
prompt.

### Increased latency

Multiple model loads and inferences add up. For real-time autocomplete,
this might be too slow. For batch code generation or chat-based workflows,
it's fine.

**Mitigation:** Keep all specialists loaded in memory (only ~2 GB total
for 6x50M models). Use the sequential-load approach only if VRAM is tight.

### Training overhead

Instead of training one model, you're training 7-8. More configs, more
data pipelines, more evaluation.

**Mitigation:** Automate aggressively. A Makefile target for each specialist.
Shared base training reduces per-specialist training time.

---

## 17. The Honest Assessment: When This Wins and When It Doesn't

### This approach WINS when:

- **You have a defined tech stack.** If you know you use React + Next.js +
  Prisma + Zod, you can build specialists for exactly those tools. The
  system becomes very good at your exact workflow.

- **You want updateability.** New framework version? Retrain one 50M
  specialist in a day. Don't touch anything else.

- **You're on consumer hardware.** Each component is small enough to train
  and run on a single GPU. The total system is more capable than a single
  model you could afford to train.

- **You need depth over breadth.** The React specialist will know hooks,
  patterns, and conventions that no general model under 7B learns well.

- **You're building a product.** If you're building a coding assistant for
  a specific audience (full-stack TS developers), this architecture lets
  you optimize for exactly their needs.

### This approach LOSES when:

- **You need general-purpose coding across many languages.** If users ask
  for Python one minute and Rust the next, you need specialists for
  everything, and the routing complexity explodes.

- **Tasks are highly interleaved.** If a single function mixes React hooks,
  GraphQL queries, and database access in 10 lines, no specialist can
  handle it alone. The router has to break it into pieces that might not
  decompose cleanly.

- **Latency is critical.** Real-time autocomplete (50-100ms) can't afford
  router → specialist → assembly overhead. Use a single model for that.

- **You're competing with frontier models.** GPT-4 and Claude already do
  multi-framework code well because they have billions of parameters and
  trained on the whole internet. This architecture helps you compete at
  small scale, but at large scale a monolith with enough parameters wins.

### The sweet spot

This architecture is ideal for:

```
Scale:     125M router + 50-125M specialists (500M-1B total)
Use case:  Coding assistant for a specific tech stack
Hardware:  Consumer GPU (8-16 GB VRAM)
Audience:  You or your team (known tech stack, known patterns)
```

At this scale, multi-agent specialization genuinely outperforms a single
monolithic model. The specialists know their domain better, the router
coordinates effectively, and the whole system trains on hardware you
already own.

Would it beat Claude or GPT-4? No. Would it beat any open-source model
under 3B on your specific tech stack? Very possibly yes. And that's the
point — build something that's excellent at what YOU need, not something
that's mediocre at everything.
