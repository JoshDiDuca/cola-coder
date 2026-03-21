# Mixture of Experts: A Committee of Specialists Inside One Model

What if your neural network could have 8x the parameters but only pay for 2x
the compute? That is the core promise of Mixture of Experts (MoE). Instead of
one monolithic feed-forward network that processes every token the same way,
MoE replaces it with multiple "expert" sub-networks and a tiny router that
decides which experts handle each token.

This is not the same as our multi-agent specialization system (discrete
specialist models). MoE operates *inside* a single transformer, at the
layer level, on a per-token basis. Both are "experts" architectures, but
they solve different problems at different granularities.

---

## Table of Contents

1. [The Core Idea: Microservices Inside Your Model](#1-the-core-idea-microservices-inside-your-model)
2. [Dense vs Sparse: Why MoE Cheats the Parameter Budget](#2-dense-vs-sparse-why-moe-cheats-the-parameter-budget)
3. [The Routing Mechanism: How Tokens Find Their Experts](#3-the-routing-mechanism-how-tokens-find-their-experts)
4. [Load Balancing: The Silent Killer of MoE Training](#4-load-balancing-the-silent-killer-of-moe-training)
5. [The Auxiliary Loss: Penalizing Lazy Routers](#5-the-auxiliary-loss-penalizing-lazy-routers)
6. [Our Implementation: `MoEFFN` and `ExpertRouter`](#6-our-implementation-moeffn-and-expertrouter)
7. [MoE vs Multi-Agent Specialization: When to Use Which](#7-moe-vs-multi-agent-specialization-when-to-use-which)
8. [The Domain Router Model: MLP vs Transformer](#8-the-domain-router-model-mlp-vs-transformer)
9. [Training the Router: Data Generation and Pipeline](#9-training-the-router-data-generation-and-pipeline)
10. [Practical Considerations: VRAM, Throughput, and Small Models](#10-practical-considerations-vram-throughput-and-small-models)
11. [Configuration: Enabling MoE in Cola-Coder](#11-configuration-enabling-moe-in-cola-coder)

---

## 1. The Core Idea: Microservices Inside Your Model

### The TypeScript analogy

Think of a standard transformer FFN layer as a single Express.js handler that
processes every request identically:

```typescript
// Dense FFN: one handler for everything
app.use("/api/*", (req, res) => {
  // This single function handles React questions,
  // GraphQL queries, Prisma schemas, testing, everything.
  // It has to be good at ALL of them.
  const result = doEverything(req.body);
  res.json(result);
});
```

Now imagine replacing that with a router and multiple specialized handlers:

```typescript
// MoE: a router picks the best handler(s) per request
const router = new ExpertRouter();

const experts = {
  0: handlePatternMatching,     // Good at syntax patterns
  1: handleControlFlow,         // Good at if/else/loops
  2: handleTypeSignatures,      // Good at type-level reasoning
  3: handleStringTemplates,     // Good at template literals
  4: handleImportResolution,    // Good at module relationships
  5: handleNumericComputation,  // Good at math operations
  6: handleErrorHandling,       // Good at try/catch patterns
  7: handleDataTransformation,  // Good at map/filter/reduce
};

app.use("/api/*", (req, res) => {
  // Router picks the top-2 most relevant experts
  const [expert1, expert2] = router.selectTopK(req.body, k=2);

  // Blend their outputs with learned weights
  const result =
    expert1.weight * experts[expert1.id](req.body) +
    expert2.weight * experts[expert2.id](req.body);

  res.json(result);
});
```

Each "expert" is a full feed-forward network (in our case, a SwiGLU FFN). The
router is a tiny linear layer that looks at the token embedding and outputs a
probability distribution over experts. Only the top-k experts actually run,
so you get the capacity of 8 experts but the compute cost of 2.

### What actually happens per token

```
Input token embedding (dim=512)
            |
            v
    ┌───────────────┐
    │  ExpertRouter  │   Just a Linear(512, 8) + softmax
    │  (tiny: 4096   │   Produces: [0.05, 0.02, 0.41, 0.03, 0.38, 0.01, 0.07, 0.03]
    │   parameters)  │
    └───────────────┘
            |
            v
     Top-k selection (k=2)
     Expert 2: weight 0.52 (renormalized)
     Expert 4: weight 0.48 (renormalized)
            |
     ┌──────┴──────┐
     v             v
 ┌────────┐   ┌────────┐
 │Expert 2│   │Expert 4│    Each is a full SwiGLU FFN
 │ SwiGLU │   │ SwiGLU │    gate_proj + up_proj + down_proj
 │  FFN   │   │  FFN   │    ~3x dim^2 parameters each
 └────────┘   └────────┘
     |             |
     v             v
  output_2      output_4
     |             |
     v             v
  0.52 * output_2 + 0.48 * output_4
            |
            v
   Final FFN output (dim=512)
```

The other 6 experts? They never run for this token. Their parameters exist in
memory but consume zero FLOPs. That is the "sparse" in sparse MoE.

---

## 2. Dense vs Sparse: Why MoE Cheats the Parameter Budget

### Dense models: every parameter fires every time

In a standard (dense) transformer, every token passes through every parameter:

```
Standard FFN parameters = 3 * dim * hidden_dim
For dim=512, hidden_dim=1376: 3 * 512 * 1376 = 2,113,536 params

Every token uses ALL 2.1M parameters.
Compute per token: 2 * 2.1M = 4.2M FLOPs (multiply-add)
```

### Sparse MoE: more params, same compute

With 8 experts and top-2 routing:

```
Total FFN parameters = 8 * 3 * dim * hidden_dim = 8 * 2.1M = 16.9M params
Active per token    = 2 * 3 * dim * hidden_dim  = 2 * 2.1M = 4.2M params

8x the parameters. Same FLOPs per token.
```

This is why MoE models punch above their weight class. A 350M dense model
and a 350M-active-parameter MoE model have the same inference speed, but
the MoE model has ~1.4B total parameters (since the FFN is typically ~2/3 of
total params, and FFN params are multiplied by num_experts/top_k = 4x).

### The catch: memory

Those extra parameters still live in VRAM. You get compute efficiency but not
memory efficiency. For a model with 8 experts and top-2:

```
Dense 350M model:   ~700MB in fp16
MoE equivalent:     ~700MB (attention, etc.) + 4x the FFN portion
                    ≈ ~700MB + ~930MB extra = ~1.6GB in fp16
```

On your RTX 4080 with 16GB, this is fine. On a 10GB card, it starts to matter.
More on this in section 10.

### Why this works at all

A key insight: not every token needs the same processing. A `{` brace token,
an `import` keyword, and a complex generic type like `Record<string, T[]>` all
require different kinds of reasoning. In a dense model, the same FFN neurons
fire for all of them, wasting capacity. With MoE, the router can learn to
send punctuation tokens to a "simple syntax" expert and complex type tokens to
a "type reasoning" expert.

Research consistently shows that experts do organically specialize. In code
models, experts tend to cluster around:
- Syntax/punctuation (braces, semicolons, operators)
- Identifiers and naming
- String literals and templates
- Type annotations
- Control flow keywords
- Import/module patterns

---

## 3. The Routing Mechanism: How Tokens Find Their Experts

### The router itself

Our `ExpertRouter` is almost absurdly simple:

```python
class ExpertRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)           # (tokens, num_experts)
        probs = F.softmax(logits, dim=-1)  # normalize to probabilities
        return logits, probs, probs
```

That is it. A single matrix multiply: `(tokens, dim) @ (dim, num_experts)`.
For 8 experts and dim=512, that is a 512x8 weight matrix with 4,096
parameters. Compared to the millions of parameters in each expert, the
router is essentially free.

### Top-k selection

After the router produces probabilities, we take the top-k:

```python
# For each token, pick the k experts with highest probability
top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

# Renormalize so the weights sum to 1
top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
```

With `top_k=2` and 8 experts, a token might get:

```
Raw probs:  [0.05, 0.02, 0.41, 0.03, 0.38, 0.01, 0.07, 0.03]
Top-2:      expert_2 (0.41), expert_4 (0.38)
Renormalized: expert_2 (0.52), expert_4 (0.48)
```

The renormalization is important: it ensures the output has the same scale
regardless of how confident the router is. Without it, a token the router is
uncertain about (low top-k probabilities) would produce a weak output.

### Why top-k instead of just top-1?

Top-1 routing is simpler but has two problems:

1. **Gradient sparsity**: Only one expert gets gradient signal per token.
   With top-2, two experts learn from each token, making training more stable.

2. **Robustness**: If the router makes a bad choice, top-1 gives you one
   expert's (possibly wrong) answer. Top-2 gives you a blend, which is more
   forgiving of routing errors.

The standard in modern MoE models (Mixtral, DeepSeek-V2, Switch) is top-2.
Our default matches this.

### Token-level routing, not sequence-level

A critical detail: routing happens per token, not per sequence. Within the
same prompt, different tokens go to different experts. The `import` token
might go to expert 3, while the `useState` token goes to expert 7. This
is much more fine-grained than our multi-agent system (which routes entire
prompts to a specialist model).

In TypeScript terms:

```typescript
// Multi-agent: route the WHOLE request to one service
const specialist = router.pickService(entirePrompt); // "react"
const result = specialist.generate(entirePrompt);

// MoE: route EACH token to different handlers within the same model
for (const token of tokens) {
  const experts = router.pickTopK(token.embedding, k=2);
  token.output = weightedSum(experts.map(e => e.process(token)));
}
```

---

## 4. Load Balancing: The Silent Killer of MoE Training

### Expert collapse: the default failure mode

Without intervention, MoE training almost always collapses. Here is why:

1. Early in training, the router is random. By chance, it sends a few more
   tokens to expert 3.
2. Expert 3 gets more gradient updates, so it gets slightly better.
3. Because expert 3 is slightly better, the router sends even more tokens
   to it (the router is optimizing the main loss).
4. Expert 3 gets even more updates, gets even better, attracts even more
   tokens.
5. After a few hundred steps, expert 3 handles 80% of all tokens. The
   other 7 experts barely train and stay near random initialization.

This is a classic "rich get richer" feedback loop. In TypeScript terms, imagine
a load balancer that routes based on response quality:

```typescript
// This is what UNBALANCED MoE does
class NaiveLoadBalancer {
  route(request: Request): Server {
    // Always pick the server with highest success rate
    return this.servers.sort((a, b) => b.successRate - a.successRate)[0];
    // Result: one server handles everything, others idle
  }
}
```

The result is effectively a dense model (one expert doing all the work) that
wasted VRAM on 7 unused expert copies. You get all the memory cost of MoE
with none of the capacity benefit.

### What balanced routing looks like

With 8 experts and even distribution, each expert handles `1/8 = 12.5%` of
tokens. Perfect balance is unrealistic (and undesirable — some tokens
genuinely need specific experts), but we want something like:

```
Healthy:     Expert 0: 11%  Expert 1: 14%  Expert 2: 13%  Expert 3: 10%
             Expert 4: 15%  Expert 5: 12%  Expert 6: 11%  Expert 7: 14%

Collapsed:   Expert 0: 1%   Expert 1: 2%   Expert 2: 1%   Expert 3: 82%
             Expert 4: 3%   Expert 5: 1%   Expert 6: 8%   Expert 7: 2%
```

---

## 5. The Auxiliary Loss: Penalizing Lazy Routers

### The load balancing loss formula

We add an auxiliary loss to the main training loss that penalizes uneven
routing. The formula from our implementation:

```
aux_loss = num_experts * sum_i(f_i * P_i)

where:
  f_i = fraction of tokens actually routed to expert i
  P_i = mean routing probability for expert i (across all tokens)
```

The key insight is that this loss is minimized when routing is perfectly
uniform. If every expert gets `1/N` of the tokens and `1/N` of the
probability mass:

```
aux_loss = N * N * (1/N * 1/N) = N * N * (1/N^2) = 1.0
```

If routing collapses to one expert:

```
f_collapsed = 1.0, P_collapsed ≈ 1.0, all others ≈ 0
aux_loss = N * (1.0 * 1.0 + 0 + 0 + ...) = N = 8.0
```

So the loss goes from 1.0 (uniform) to 8.0 (fully collapsed). The training
loop sees this spike and pushes the router back toward balance.

### Why f_i * P_i and not just f_i?

Using only the fraction `f_i` would be a hard constraint that ignores the
router's *intent*. The product `f_i * P_i` captures both what the router
*did* (f_i) and what it *wanted to do* (P_i). This lets gradients flow back
through the softmax to adjust the router weights, not just the top-k
selection.

### The weight coefficient

Our implementation uses `aux_loss_weight = 0.01`:

```python
self.aux_loss_weight = 0.01
# ...
return self.aux_loss_weight * aux_loss
```

This means the balancing loss is 1% the scale of the main language modeling
loss. Too high and the model prioritizes balance over quality (every expert
handles equal tokens but none are good). Too low and collapse still happens.
Values between 0.001 and 0.05 are typical; 0.01 is a safe default.

The total training loss becomes:

```
total_loss = language_modeling_loss + sum(layer.moe.aux_loss for each MoE layer)
```

### In TypeScript terms

```typescript
// The aux loss is like a "fairness penalty" on a load balancer
function computeAuxLoss(routingStats: ExpertStats[]): number {
  const N = routingStats.length;

  // f_i: what fraction of requests each server actually handled
  const fractions = routingStats.map(s => s.requestsHandled / totalRequests);

  // P_i: what fraction of routing probability each server got
  const meanProbs = routingStats.map(s => s.averageRoutingScore);

  // Penalty: high when one server gets both lots of requests AND
  // high routing probability (rich-get-richer signal)
  const penalty = N * fractions.reduce(
    (sum, f, i) => sum + f * meanProbs[i], 0
  );

  return 0.01 * penalty; // Small weight relative to main objective
}
```

---

## 6. Our Implementation: `MoEFFN` and `ExpertRouter`

### File: `src/cola_coder/features/moe_layer.py`

The full MoE layer is a drop-in replacement for the standard `SwiGLUFFN`. It
maintains the same interface: takes `(batch, seq_len, dim)` in, returns
`(batch, seq_len, dim)` out.

### `MoEFFN` constructor

```python
MoEFFN(
    dim=512,            # Model dimension (must match transformer)
    hidden_dim=1376,    # SwiGLU hidden dim (each expert gets this)
    num_experts=8,      # Total expert count
    top_k=2,            # Active experts per token
    dropout=0.0,        # Dropout rate
    capacity_factor=1.25,  # Max tokens per expert = capacity_factor * (N/E)
)
```

**Default numbers with our configs:**

| Config | dim | hidden_dim | num_experts | top_k | Total FFN params | Active FFN params |
|--------|-----|-----------|-------------|-------|-----------------|-------------------|
| tiny   | 512 | 1376      | 8           | 2     | 16.9M           | 4.2M              |
| small  | 768 | 2048      | 8           | 2     | 37.7M           | 9.4M              |
| medium | 1024| 2816      | 8           | 2     | 69.2M           | 17.3M             |

### The capacity factor

The capacity factor limits how many tokens any single expert can process:

```python
capacity = int(self.capacity_factor * num_tokens / self.num_experts)
# With 1024 tokens, 8 experts, capacity_factor=1.25:
# capacity = int(1.25 * 1024 / 8) = 160 tokens per expert

if len(token_indices) > capacity:
    token_indices = token_indices[:capacity]  # Drop excess tokens
```

This is a hard safety net against collapse. Even if the router tries to send
500 tokens to one expert, the capacity cap truncates it to 160. The dropped
tokens still get processed by their second-choice expert (since we do top-2).

**Why 1.25?** With perfectly uniform routing, each expert gets `1024/8 = 128`
tokens. The factor of 1.25 allows 25% headroom (160 tokens) for natural
variation. Higher values are more permissive (let the router be more uneven),
lower values enforce stricter balance.

```
capacity_factor = 1.0:   Perfectly even (128 each). Drops tokens too aggressively.
capacity_factor = 1.25:  25% headroom (160 each). Good default.
capacity_factor = 1.5:   50% headroom (192 each). More flexible.
capacity_factor = 2.0:   Very permissive. Close to no cap for balanced routing.
```

### The forward pass: step by step

Here is what happens when a batch enters `MoEFFN.forward()`:

```
1. Flatten: (batch=4, seq=256, dim=512) -> (1024, 512)
            Treat all tokens equally regardless of which sequence they came from.

2. Route:   ExpertRouter produces probabilities (1024, 8)
            Each token has a score for each expert.

3. Top-k:   Select top-2 experts per token.
            (1024, 2) indices + (1024, 2) weights, renormalized.

4. Aux loss: Compute load balancing penalty from routing stats.

5. Dispatch: For each of the k=2 expert slots:
               For each of the 8 experts:
                 - Find which tokens were assigned to this expert
                 - Apply capacity cap
                 - Run those tokens through the expert's SwiGLU FFN
                 - Multiply by the routing weight
                 - Add to the output tensor

6. Reshape:  (1024, 512) -> (4, 256, 512)
```

```
  Batch of tokens (B*S = 1024 tokens, dim = 512)
  ────────────────────────────────────────────────
          │
          v
  ┌─────────────────┐
  │  ExpertRouter    │  Linear(512, 8) + softmax
  │  4,096 params    │  Output: (1024, 8) probabilities
  └────────┬────────┘
           │
           v
  ┌─────────────────┐
  │  Top-K Selection │  Pick 2 experts per token
  │  + Renormalize   │  (1024, 2) indices & weights
  └────────┬────────┘
           │
    ┌──────┼──────┐  (for each of k=2 slots)
    │      │      │
    v      v      v
  Slot 0 tokens   Slot 1 tokens
    │               │
    │  ┌─────────── │ ─────────────────────────┐
    │  │  Dispatch to experts based on index    │
    │  │                                        │
    v  v                                        v
  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────┐ ┌──────┐
  │ E0   │ │ E1   │ │ E2   │ ...  │ E6   │ │ E7   │
  │130tk │ │142tk │ │125tk │      │118tk │ │137tk │
  │SwiGLU│ │SwiGLU│ │SwiGLU│      │SwiGLU│ │SwiGLU│
  └──┬───┘ └──┬───┘ └──┬───┘      └──┬───┘ └──┬───┘
     │        │        │              │        │
     v        v        v              v        v
  weight * output, accumulated into output tensor
  ────────────────────────────────────────────────
          │
          v
  Reshape to (batch, seq_len, dim)
```

### `_SwiGLUExpert`: each expert's internals

Each expert is a standard SwiGLU FFN, identical to what a dense model uses:

```python
class _SwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # "gate"
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)  # "value"
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # project back

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # SiLU activation on gate
        value = self.up_proj(x)             # Linear projection
        return self.down_proj(gate * value) # Element-wise multiply, project down
```

This is the same `SwiGLU` activation used in LLaMA, Mistral, and our standard
FFN. The only difference is that we have 8 copies of this, and each one only
processes a subset of tokens.

### `MoEConfig`: controlling MoE from config

```python
class MoEConfig:
    num_experts: int = 8          # How many expert FFNs
    top_k: int = 2                # How many active per token
    capacity_factor: float = 1.25 # Token capacity headroom
    aux_loss_weight: float = 0.01 # Load balancing strength
    moe_layers: list[int] | None = None  # Which layers get MoE (None = all)
```

The `moe_layers` field is useful for hybrid architectures: you might use MoE
only in the later layers (where the model does more "reasoning") and keep
the early layers dense (where it does more "feature extraction"). A common
pattern is every-other-layer MoE:

```python
MoEConfig(moe_layers=[1, 3, 5, 7, 9, 11])  # MoE on even-indexed layers
```

---

## 7. MoE vs Multi-Agent Specialization: When to Use Which

Cola-coder has two "experts" systems that solve different problems:

| Aspect | MoE (this doc) | Multi-Agent (discrete specialists) |
|--------|----------------|-----------------------------------|
| Granularity | Per-token routing inside one model | Per-prompt routing to separate models |
| Experts are | FFN sub-networks sharing attention | Completely independent models |
| Router is | A learned linear layer (4K params) | A trained classifier (<5M params) |
| Routing happens | Every layer, every forward pass | Once, before generation starts |
| Expert count | 8 (typical), all in one GPU | 6-7 separate models, swap in/out |
| VRAM for all experts | All loaded simultaneously | Only 1-2 loaded at a time |
| What experts learn | Low-level token patterns | High-level domain knowledge |
| Training | Joint (all experts train together) | Independent (each specialist alone) |
| Adding an expert | Requires retraining from scratch | Train a new specialist, register it |

### When MoE wins

- **Small to medium models (50M-350M)**: MoE gives you more capacity without
  more compute. A 125M model with 8 experts has the capacity of a ~500M model
  but runs at 125M speed.

- **Diverse token-level patterns**: Code has very different patterns for
  different token types (keywords vs identifiers vs literals vs operators).
  MoE naturally learns to specialize at this granularity.

- **Single-model deployment**: One model file, one loading step, no routing
  latency. Simpler inference infrastructure.

### When multi-agent wins

- **Deep domain knowledge**: A specialist trained exclusively on React code
  for 2 days knows React patterns far more deeply than one of 8 MoE experts
  that shares attention layers with everything else.

- **Modular updates**: You can retrain just the Prisma specialist when Prisma 6
  comes out, without touching anything else. MoE requires full retraining.

- **Memory-constrained inference**: Multi-agent only loads one specialist at a
  time (~200MB). MoE loads all experts (~1.6GB for an equivalent model).

- **Interpretability**: With multi-agent, you know exactly which specialist
  generated the code. With MoE, tokens are blended across experts in ways
  that are harder to inspect.

### The hybrid approach

The ideal cola-coder architecture could use both:

```
User prompt
    │
    v
[Domain Router] ──> picks "react" specialist
    │
    v
[React Specialist Model]  <── this model internally uses MoE layers
    │                          so each token goes to specialized
    v                          sub-experts within the React domain
Generated React code
```

A React specialist with MoE layers gets both domain-level specialization
(it only knows React) and token-level specialization (different experts for
JSX vs hooks vs event handlers).

---

## 8. The Domain Router Model: MLP vs Transformer

The domain router is a completely separate model from the MoE token-level
router. It decides which *specialist model* to invoke, not which *FFN expert*
to use. But the training pipeline and concepts overlap, so we cover it here.

### File: `src/cola_coder/features/router_model.py`

Two architectures are available:

### MLP Router (~1M params)

```
Input token IDs (256 tokens)
        │
        v
  ┌──────────────┐
  │  Embedding    │  32768 vocab -> 128 dim
  │  (4.2M)       │
  └──────┬───────┘
         │
         v
  Mean pooling (bag of embeddings)
  Treats input as bag-of-words.
  Order doesn't matter — imports
  are the main signal anyway.
         │
         v
  ┌──────────────┐
  │  Linear(128,  │
  │  256) + ReLU  │
  │  + Dropout    │
  ├──────────────┤
  │  Linear(256,  │
  │  256) + ReLU  │
  │  + Dropout    │
  ├──────────────┤
  │  Linear(256,  │
  │  7)           │  7 domains
  └──────┬───────┘
         │
         v
  Domain logits (7)
```

**Speed:** ~100 microseconds per inference. Essentially free.

**Why bag-of-words works:** For domain classification, the most important
signal is *which tokens appear*, not their order. If the input contains
`import`, `React`, `useState`, `jsx`, the domain is React regardless of
how those tokens are arranged. Mean-pooling captures this efficiently.

### Transformer Router (~3-5M params)

```
Input token IDs (256 tokens)
        │
        v
  ┌──────────────┐
  │  Embedding    │  32768 vocab -> 128 dim
  │  + Position   │  256 positions -> 128 dim
  │  + Dropout    │
  └──────┬───────┘
         │
         v
  ┌──────────────┐
  │  Transformer  │  2 layers, 4 heads
  │  Encoder      │  FFN dim: 256
  │  Layers       │
  └──────┬───────┘
         │
         v
  Mean pooling
         │
         v
  ┌──────────────┐
  │  Linear(128,  │
  │  7)           │
  └──────┬───────┘
         │
         v
  Domain logits (7)
```

**Speed:** ~1 millisecond per inference. Still negligible.

**When to use it:** The transformer router captures token *order* and
*context*, which helps distinguish ambiguous cases. For example,
`import { z } from 'zod'` followed by `import { describe } from 'vitest'`
could be either Zod or Testing — the transformer can weigh which import
appears first and what follows.

### Configuration

```python
RouterConfig(
    vocab_size=32768,    # Must match your tokenizer
    embed_dim=128,       # Small — this is just a classifier
    hidden_dim=256,      # MLP hidden or transformer FFN dim
    num_domains=7,       # react, nextjs, graphql, prisma, zod, testing, general
    max_seq_len=256,     # Only first 256 tokens needed for classification
    dropout=0.1,
    architecture="mlp",  # or "transformer"
    num_layers=2,        # Transformer-specific
    num_heads=4,         # Transformer-specific
)
```

### The `DomainRouter` wrapper

The `DomainRouter` class adds practical features around the raw model:

- **Confidence threshold** (default 0.5): If the model's confidence is below
  this, fall back to heuristic detection.
- **Heuristic fallback**: Uses the `domain_detector` feature to classify by
  import patterns and keywords when the model is uncertain.
- **Graceful degradation**: If the model fails for any reason, returns
  `("general", 1.0)` — the safe default.

```python
router = DomainRouter(model=mlp_router, confidence_threshold=0.5)
domain, confidence = router.route("import { useState } from 'react';")
# -> ("react", 0.92)
```

---

## 9. Training the Router: Data Generation and Pipeline

### Step 1: Generate labeled data

**File: `scripts/generate_router_data.py`**

You need `(code, domain)` pairs. Three sources are available:

#### Option A: From existing training data (.npy)

```bash
python scripts/generate_router_data.py \
  --source data/processed/train_data.npy \
  --tokenizer tokenizer.json \
  --output data/router_training_data.jsonl \
  --max-samples 50000
```

This decodes tokenized chunks from your existing training data and uses the
heuristic domain detector to auto-label them. The `min_confidence=0.3` filter
ensures only reasonably confident labels are kept.

#### Option B: From source code directories

```bash
python scripts/generate_router_data.py \
  --source-dir ./repos/ \
  --output data/router_training_data.jsonl
```

Scans `.ts`, `.tsx`, `.js`, `.jsx` files recursively. Good if you have cloned
repositories organized by framework.

#### Option C: Synthetic bootstrap data

```bash
python scripts/generate_router_data.py \
  --synthetic \
  --output data/router_training_data_synthetic.jsonl
```

Generates template-based examples for each domain. Useful for initial
experiments but not sufficient for production quality. Templates cover basic
patterns like React components with `useState`, Next.js `getServerSideProps`,
GraphQL queries, Prisma client calls, Zod schemas, and Vitest tests.

### Data format

Output is JSONL with one sample per line:

```json
{"code": "import { useState } from 'react';\n...", "domain": "react", "confidence": 0.85, "filename": "App.tsx"}
```

### Balance enforcement

The `RouterDataGenerator` enforces balance through `max_samples_per_domain`:

```python
RouterDataGenerator(
    min_confidence=0.3,          # Reject low-confidence labels
    max_samples_per_domain=7142, # 50000 / 7 domains
    min_code_length=50,          # Skip trivially short files
    max_code_length=2000,        # Truncate long files
)
```

This prevents the "general" domain from dominating (since most code is
general-purpose). Equal representation matters for a balanced router.

### Step 2: Train the router model

**File: `scripts/train_router.py`**

```bash
# MLP router (fast training, good baseline)
python scripts/train_router.py \
  --data data/router_training_data.jsonl \
  --arch mlp \
  --epochs 20 \
  --lr 1e-3 \
  --batch-size 64 \
  --device cuda

# Transformer router (better quality, slightly slower training)
python scripts/train_router.py \
  --data data/router_training_data.jsonl \
  --arch transformer \
  --epochs 20 \
  --lr 1e-3
```

Or combine generation and training in one command:

```bash
python scripts/train_router.py \
  --generate-data \
  --source data/processed/train_data.npy \
  --arch mlp
```

### Training details

- **Optimizer:** AdamW with weight_decay=0.01
- **Scheduler:** Cosine annealing over the full epoch count
- **Gradient clipping:** max_norm=1.0
- **Data split:** 80/20 train/validation
- **Checkpointing:** Saves best model (by val accuracy) and final model
- **Expected accuracy:** 85-95% on 7-domain classification (depends on data quality)

### Output files

```
checkpoints/router/
  best_router.pt          # Best validation accuracy weights
  router_final.pt         # Final epoch weights
  router_config.json      # RouterConfig for reproducibility
```

---

## 10. Practical Considerations: VRAM, Throughput, and Small Models

### VRAM impact

MoE multiplies the FFN parameters by `num_experts` but leaves attention
unchanged. Since FFN is roughly 2/3 of a transformer's parameters:

```
Dense 125M model:     ~250MB (fp16)
  Attention portion:  ~83MB
  FFN portion:        ~167MB

MoE 125M-active model (8 experts, top-2):
  Attention:          ~83MB  (unchanged)
  FFN:                ~167MB * 4 = ~668MB  (8 experts, but each is same
                                            size as original FFN)
  Total:              ~751MB (fp16)

Ratio: 3x more VRAM for the FFN-heavy part, ~3x total.
```

On your RTX 4080 (16GB), this is very manageable. Even a medium (350M active)
MoE model with 8 experts would use ~5-6GB, well within budget.

### Throughput

MoE does not slow down *computation* significantly (same FLOPs), but it does
add overhead from:

1. **Router computation:** Negligible (~0.1% of total FLOPs).
2. **Token dispatch:** Sorting tokens to experts, gathering results. This is
   memory-bound, not compute-bound. Expect ~5-15% overhead.
3. **Expert utilization:** GPUs are most efficient with large, uniform
   matrix multiplies. MoE breaks the FFN into smaller per-expert batches,
   which can underutilize tensor cores. Impact: ~10-20% throughput hit.

Net effect: expect **10-25% slower wall-clock time** compared to a dense model
with the same active parameters. You trade speed for capacity.

### When MoE helps small models

MoE is most beneficial when:

- **You are parameter-limited but not compute-limited.** If your 50M model
  underfits (train loss still dropping, more capacity would help), MoE gives
  you 4x effective capacity at ~1.3x the compute.

- **Your data is diverse.** Code models see wildly different token patterns
  (Python vs TypeScript, imports vs logic, strings vs types). Experts can
  specialize on these clusters.

- **You can afford the memory.** If you are already VRAM-constrained with
  gradient checkpointing and mixed precision, adding MoE might push you
  over the limit.

MoE is less helpful when:

- **Your model already fits the data well.** If train loss is already low,
  more capacity does not help.

- **You are training on a single narrow domain.** If all your data is React
  TypeScript, there is less diversity for experts to specialize on.

- **Inference latency matters more than quality.** The dispatch overhead is
  a constant tax on every forward pass.

### Quick VRAM estimates for cola-coder MoE

| Config | Dense VRAM | MoE VRAM (8 experts) | Active params |
|--------|-----------|---------------------|---------------|
| tiny   | ~3.6 GB   | ~6.5 GB             | 50M (same speed) |
| small  | ~6.5 GB   | ~12 GB              | 125M (same speed) |
| medium | ~8.2 GB   | ~15 GB              | 350M (same speed) |

These estimates include optimizer states for training. Inference-only is
roughly 3x less.

---

## 11. Configuration: Enabling MoE in Cola-Coder

### Toggle in `configs/features.yaml`

MoE is disabled by default (it is experimental):

```yaml
moe_layer: false    # Experimental — disabled by default
```

Set to `true` to enable. When enabled, MoE layers replace the standard
SwiGLU FFN in the transformer blocks specified by `moe_layers` in the
MoE config.

### MoE config defaults

The `MoEConfig` class holds all MoE-specific settings:

```python
MoEConfig(
    num_experts=8,            # 8 expert FFNs per MoE layer
    top_k=2,                  # 2 active per token
    capacity_factor=1.25,     # 25% headroom over even distribution
    aux_loss_weight=0.01,     # Load balancing loss coefficient
    moe_layers=None,          # None = all layers; [1,3,5] = specific layers
)
```

### Recommended starting points

**For experimentation (tiny config, RTX 4080):**

```python
MoEConfig(
    num_experts=4,       # Start small: fewer experts = faster iteration
    top_k=2,
    capacity_factor=1.25,
    aux_loss_weight=0.01,
    moe_layers=None,     # All layers
)
```

4 experts is enough to see if MoE helps your setup. You can scale to 8 later.

**For production (small/medium config):**

```python
MoEConfig(
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,
    aux_loss_weight=0.01,
    moe_layers=[2, 4, 6, 8, 10],  # Every other layer after layer 1
)
```

Every-other-layer MoE is a common pattern. The early layers do feature
extraction (dense is fine), and the later layers do specialization (MoE helps).

### Monitoring during training

Key metrics to watch:

1. **Aux loss**: Should stay between 0.5 and 2.0. If it spikes above 3.0,
   experts are collapsing. Consider increasing `aux_loss_weight`.

2. **Expert utilization**: Log the fraction of tokens per expert. All
   experts should handle between 5% and 25% of tokens (for 8 experts).
   If one expert handles >40%, increase `aux_loss_weight` or decrease
   `capacity_factor`.

3. **Main loss vs dense baseline**: MoE should match dense loss within the
   first 10% of training and beat it from there. If it is consistently
   worse, the aux loss weight might be too high (sacrificing quality for
   balance).

### End-to-end: adding MoE to a training run

```bash
# 1. Enable MoE in features config
#    Edit configs/features.yaml: moe_layer: true

# 2. Run training (MoE is applied automatically when enabled)
python scripts/train.py --config configs/small.yaml

# 3. Monitor expert utilization in logs
#    Look for aux_loss in training output

# 4. Generate with the MoE model (no special flags needed)
python scripts/generate.py --checkpoint checkpoints/small/latest
```

The MoE layer is designed as a drop-in replacement. When `moe_layer` is
`true` in features.yaml, the model builder swaps `SwiGLUFFN` for `MoEFFN`
in the specified layers. Checkpoints save and load all expert weights
automatically. No changes to the training script or generation pipeline
are needed.

---

## Further Reading

- **Switch Transformer** (Fedus et al., 2022): The paper that made MoE
  practical for transformers. Introduced top-1 routing and capacity factors.
- **Mixtral 8x7B** (Mistral AI, 2024): The model that proved MoE works for
  open-source LLMs. 8 experts, top-2 routing, 12.9B total / 3.5B active.
- **DeepSeek-V2** (2024): Pushed MoE further with fine-grained expert
  segmentation and shared experts.
- **Our multi-agent deep dive**: `docs/deep-dives/multi-agent-specialization.md`
  for the discrete specialist approach and how it compares.
