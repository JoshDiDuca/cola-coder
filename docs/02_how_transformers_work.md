# How Transformers Work

A practical guide for TypeScript developers building a code generation model.

No PhD required. If you can follow a `.map().reduce()` chain, you can follow this.

---

## 1. The Big Picture

### What Does a Language Model Actually Do?

One thing. It predicts the next token given all previous tokens.

```
Input:  "function add(a, b) { return a +"
Output: " b"   (with high probability)
```

That's it. Every impressive thing you've seen an LLM do — writing code, answering
questions, debugging — is just next-token prediction applied repeatedly.

### The Training Loop in One Paragraph

Take a huge pile of code. Chop it into sequences of tokens. For each sequence,
ask the model to predict every next token. Compare the model's predictions to
the actual next tokens. Compute how wrong it was (the "loss"). Use calculus
(backpropagation) to nudge every parameter in the model so it would be slightly
less wrong next time. Repeat this billions of times. That's training.

### Tokens: Not Words, Not Characters — Subword Pieces

Tokenization splits text into pieces the model works with. These are NOT words
and NOT single characters. They are subword units learned from the training data.

Here's how a BPE tokenizer might split some TypeScript:

```
Source code:       "function getData() {\n  return fetch(url);\n}"

Tokens:            ["function", " get", "Data", "()", " {", "\n ", " return",
                    " fetch", "(", "url", ");", "\n", "}"]

Token IDs:         [1547, 1078, 2891, 3451, 892, 207, 1189, 3802, 9, 4511, 568, 207, 890]
```

Notice:
- Common words like `function` and `return` are single tokens
- `getData` splits into `get` + `Data` (camelCase boundary)
- `()` can be one token (very common in code)
- Whitespace and newlines are explicit tokens
- Each token maps to an integer ID

A typical code vocabulary has 32,000 to 50,000 tokens.

### Why Decoder-Only?

There are three transformer architectures:

```
Encoder-only:    BERT        — reads all tokens at once, good for classification
Encoder-Decoder: T5          — reads input, then generates output
Decoder-only:    GPT/LLaMA   — generates left-to-right, one token at a time
```

We use decoder-only because code generation is inherently left-to-right.
You write `function add(` before you write `a, b)`. The model should only
see what came before, never what comes after. Decoder-only enforces this
naturally with causal masking (explained in Section 3).

---

## 2. Embeddings: From Token IDs to Vectors

### The Problem

The model can't do math on token ID `1547`. It needs a richer representation —
a list of numbers that captures the meaning and relationships of each token.

### Token Embedding: A Lookup Table

Think of the embedding layer as a giant `Map<number, number[]>`:

```typescript
// Conceptually, the embedding layer is:
const embeddings: Map<number, number[]> = new Map();
embeddings.set(0,    [0.012, -0.034, 0.091, ..., 0.005]);  // 512 numbers
embeddings.set(1,    [0.045,  0.021, -0.078, ..., 0.033]);
// ... one row for every token in the vocabulary
embeddings.set(31999, [-0.011, 0.088, 0.012, ..., -0.041]);

function embed(tokenId: number): number[] {
  return embeddings.get(tokenId)!;  // simple lookup, nothing fancy
}
```

Each token gets a vector (a list of numbers). That's it. No computation — just
a table lookup.

### What Do These Numbers Represent?

Honestly? We don't fully know. The model learns them during training. But after
training, you'll find that tokens with similar meanings end up with similar
vectors. Tokens like `for`, `while`, and `loop` will be close together in this
high-dimensional space, while `for` and `banana` will be far apart.

Think of it like this: each number is a "feature dial." One might loosely
correspond to "how much is this a control flow keyword," another to "how much
is this related to string operations." But we don't set these — the model
discovers whatever features are useful for predicting the next token.

### Dimension = How Many Numbers Per Token

The "dimension" (often called `d_model` or just `dim`) is how many numbers
each vector has:

```
Small model:   dim = 512    (each token → list of 512 numbers)
Medium model:  dim = 768    (each token → list of 768 numbers)
Large model:   dim = 1024   (each token → list of 1024 numbers)
```

More dimensions = more capacity to represent nuance, but more parameters
and more compute.

### The Embedding Step Visualized

```
Token IDs:        [1547,        1078,        2891        ]
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
Embedding     │ Look up  │ │ Look up  │ │ Look up  │
Table         │ row 1547 │ │ row 1078 │ │ row 2891 │
(vocab_size   └────┬─────┘ └────┬─────┘ └────┬─────┘
 x dim)            │             │             │
                   ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
Vectors       │ [0.01,   │ │ [0.04,   │ │ [-0.02,  │
(dim=512)     │  -0.03,  │ │   0.02,  │ │   0.08,  │
              │  0.09,   │ │  -0.07,  │ │   0.01,  │
              │  ...     │ │  ...     │ │  ...     │
              │  0.005]  │ │   0.03]  │ │  -0.04]  │
              └──────────┘ └──────────┘ └──────────┘

Result: a matrix of shape [sequence_length, dim]
        3 tokens x 512 dimensions = 3 x 512 matrix
```

After this step, every token is a 512-dimensional vector (or whatever your
dim is). The rest of the transformer operates entirely on these vectors.

---

## 3. Attention: The Core Innovation

### The Intuition

Consider this code:

```typescript
const result = items.filter(item => item.price > threshold);
```

When predicting the token after `threshold`, the model needs to know:
- `result` is a `const` binding (syntax context)
- `items` is what we're filtering (data flow)
- `price` is a property we're comparing (semantic context)
- `threshold` is the variable being referenced (immediate context)

Attention is the mechanism that lets each token "look at" other tokens and
pull in relevant information. For each token, it asks: "which other tokens
should I pay attention to, and how much?"

### Query, Key, Value: The Analogy

Think of attention like a search engine:

```
Q (Query):   "What am I looking for?"
K (Key):     "What do I contain?" (the label/tag on each token)
V (Value):   "Here's my actual information" (the content to retrieve)
```

Concretely, for each token position, we compute three vectors by multiplying
the token's embedding by three different weight matrices:

```typescript
// For each token vector x:
const q = matmul(x, W_q);  // "I'm looking for..."
const k = matmul(x, W_k);  // "I contain..."
const v = matmul(x, W_v);  // "My information is..."
```

Each token broadcasts its Key ("here's what I am") and its Value ("here's my
content"). Then each token uses its Query to search through all the Keys,
finds the most relevant ones, and retrieves a weighted mix of their Values.

### How Attention Scores Are Computed

Step by step:

```
1. Compute similarity:  score(i, j) = dot_product(Q[i], K[j])

   The dot product measures how similar two vectors are.
   High dot product = token i's query matches token j's key = "pay attention here"

2. Scale:              score(i, j) = score(i, j) / sqrt(head_dim)

   (We'll explain why in a moment)

3. Mask:               If j > i, set score(i, j) = -infinity

   (Causal mask — can't look at future tokens)

4. Normalize:          weights = softmax(scores)

   Softmax converts raw scores into percentages that sum to 1.
   If token 3 has scores [2.1, 0.5, 1.8] for tokens [0, 1, 2],
   softmax gives roughly [0.48, 0.10, 0.42] — "48% attention on token 0,
   10% on token 1, 42% on token 2"

5. Retrieve:           output[i] = sum(weights[i][j] * V[j] for all j)

   Weighted average of the value vectors.
```

In pseudocode:

```typescript
function attention(Q: Matrix, K: Matrix, V: Matrix, mask: Matrix): Matrix {
  // Q, K, V shapes: [seq_len, head_dim]

  // Step 1: compute all pairwise similarities
  const scores = matmul(Q, transpose(K));  // [seq_len, seq_len]

  // Step 2: scale down
  const scaled = scores.map(s => s / Math.sqrt(headDim));

  // Step 3: apply causal mask (set future positions to -Infinity)
  const masked = scaled.map((s, i, j) => j > i ? -Infinity : s);

  // Step 4: normalize each row to probabilities
  const weights = masked.mapRows(row => softmax(row));

  // Step 5: weighted sum of values
  const output = matmul(weights, V);  // [seq_len, head_dim]

  return output;
}
```

### Causal Masking: Can Only Look Backwards

This is the "decoder-only" constraint. When processing token at position 5,
it can only attend to tokens at positions 0, 1, 2, 3, 4 — not 6, 7, 8, etc.

```
The causal mask (1 = can attend, 0 = blocked):

              Key positions
              0   1   2   3   4   5
           ┌───┬───┬───┬───┬───┬───┐
      0    │ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │  Token 0 can only see itself
Query 1    │ 1 │ 1 │ 0 │ 0 │ 0 │ 0 │  Token 1 sees tokens 0-1
pos.  2    │ 1 │ 1 │ 1 │ 0 │ 0 │ 0 │  Token 2 sees tokens 0-2
      3    │ 1 │ 1 │ 1 │ 1 │ 0 │ 0 │  Token 3 sees tokens 0-3
      4    │ 1 │ 1 │ 1 │ 1 │ 1 │ 0 │  Token 4 sees tokens 0-4
      5    │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │  Token 5 sees tokens 0-5
           └───┴───┴───┴───┴───┴───┘

The 0 positions are set to -infinity before softmax,
which makes them contribute exactly 0% attention weight.
```

This ensures the model can never "cheat" by looking at the answer.

### Multi-Head Attention

Instead of one big attention operation, we split into multiple "heads" that
attend independently, then combine the results.

```
                    Input vector (dim=512)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Q (512)      K (512)      V (512)
              │            │            │
     ┌────┬───┴──┬────┐   ...         ...
     ▼    ▼      ▼    ▼
   Head  Head  Head  Head    (8 heads, each with head_dim = 64)
    0     1     2     7
     │    │      │    │
     │    │      │    │      Each head does its own attention
     ▼    ▼      ▼    ▼
   [64]  [64]  [64]  [64]   Each produces a 64-dim output
     │    │      │    │
     └────┴───┬──┴────┘
              ▼
        Concatenate → [512]
              │
              ▼
        Linear (W_o) → [512]   Project back to model dimension
              │
              ▼
           Output (dim=512)
```

Why multiple heads? Each head can learn to attend to different things:

```
Head 0: "What's the nearest opening bracket?"    (syntax matching)
Head 1: "Where was this variable defined?"       (variable tracking)
Head 2: "What type context am I in?"             (type inference)
Head 3: "What's the indentation pattern?"        (formatting)
...
```

We don't explicitly program these roles — the model learns whatever
specializations help it predict the next token.

### Scaled Dot-Product: Why Divide by sqrt(head_dim)?

Quick version: dot products between high-dimensional vectors tend to produce
large numbers. Large numbers fed into softmax create extremely peaked
distributions (one token gets 99.9% attention, everything else gets ~0%).
This makes gradients tiny and training slow.

Dividing by `sqrt(head_dim)` keeps the scores in a reasonable range where
softmax produces useful, spread-out distributions.

```
head_dim = 64
sqrt(64) = 8

Without scaling: scores might be [-20, 45, 12, -8, ...]
  softmax → [~0, ~1.0, ~0, ~0, ...]  ← too peaked, gradient ≈ 0

With scaling (divide by 8): scores become [-2.5, 5.6, 1.5, -1.0, ...]
  softmax → [0.01, 0.82, 0.13, 0.01, ...]  ← useful distribution
```

In plain English: dividing by `sqrt(head_dim)` is just a normalizing trick
to keep the numbers from getting too extreme. The formula:

```
attention_score = (Q * K^T) / sqrt(head_dim)

Where:
  Q        = query vector for the current token
  K^T      = transposed key vectors for all tokens we can attend to
  head_dim = dimension of each attention head (e.g., 64)
  sqrt()   = square root function
```

---

## 4. RoPE: Position Encoding

### The Problem

Attention computes dot products between token vectors. Dot products don't care
about order — they're "set operations." Without position information:

```
"x = y + z"   and   "z = y + x"
```

...would produce identical attention patterns, because the same tokens are
present — just in different positions. But position matters enormously in code!

### How RoPE Works (Intuition)

RoPE (Rotary Position Embedding) encodes position by literally rotating the
Query and Key vectors.

Think of it this way: imagine each pair of numbers in your Q/K vector as
a point on a 2D plane. RoPE rotates that point by an angle that depends on
the token's position in the sequence.

```
Position 0:  rotate by 0 degrees
Position 1:  rotate by θ degrees
Position 2:  rotate by 2θ degrees
Position 3:  rotate by 3θ degrees
...

Different pairs of dimensions use different base angles (θ),
so the rotation pattern is unique for each position.
```

The key insight: when you compute the dot product of two rotated vectors,
the result depends on the *difference* in their positions, not their absolute
positions. Token at position 10 attending to token at position 7 looks the
same as token at position 100 attending to token at position 97 — both are
3 positions apart.

```
            Dimension pairs in Q/K vector (dim=8 example)
            ┌─────────┬─────────┬─────────┬─────────┐
            │ (d0,d1) │ (d2,d3) │ (d4,d5) │ (d6,d7) │
            └────┬────┴────┬────┴────┬────┴────┬────┘
                 │         │         │         │
                 ▼         ▼         ▼         ▼
            Rotate by   Rotate by  Rotate by  Rotate by
            pos * θ₁    pos * θ₂   pos * θ₃   pos * θ₄

            θ₁ is fast-changing (high frequency)
            θ₄ is slow-changing (low frequency)

            This creates a unique "fingerprint" for each position.
```

### The Rotation Formula

For each pair of dimensions (2i, 2i+1) at position `pos`:

```
θ_i = 1 / (base ^ (2i / dim))

Where:
  base = 10000 (a constant, sometimes 500000 for longer contexts)
  i    = which dimension pair (0, 1, 2, ...)
  dim  = head dimension

Then apply:
  q_rotated[2i]     = q[2i] * cos(pos * θ_i) - q[2i+1] * sin(pos * θ_i)
  q_rotated[2i+1]   = q[2i] * sin(pos * θ_i) + q[2i+1] * cos(pos * θ_i)

(Same for K vectors)
```

If that looks like a 2D rotation matrix — it is. Each dimension pair gets
rotated by an angle that increases with position.

### In Pseudocode

```typescript
function applyRoPE(
  vec: number[],   // Q or K vector, length = head_dim
  position: number // token's position in the sequence
): number[] {
  const result = new Array(vec.length);

  for (let i = 0; i < vec.length; i += 2) {
    const dimIndex = i / 2;
    const theta = 1.0 / Math.pow(10000, (2 * dimIndex) / vec.length);
    const angle = position * theta;

    const cos_a = Math.cos(angle);
    const sin_a = Math.sin(angle);

    // 2D rotation
    result[i]     = vec[i] * cos_a - vec[i + 1] * sin_a;
    result[i + 1] = vec[i] * sin_a + vec[i + 1] * cos_a;
  }

  return result;
}
```

### Why RoPE is Better Than Learned Positional Embeddings

Learned positional embeddings (the original transformer approach) have a fixed
table: position 0 gets vector A, position 1 gets vector B, etc. If you trained
with max 2048 positions, the model has never seen position 2049 — it can't
generalize.

RoPE uses a mathematical formula, not a lookup table. The rotation angles are
computed on the fly. This means the model can handle sequence lengths it has
never seen during training (with some techniques like RoPE scaling). For code,
where files can be very long, this matters.

### No Learned Parameters

RoPE adds zero trainable parameters to the model. The rotation angles are
purely mathematical — computed from the position and dimension index. The model
learns to work *with* these rotations through its Q and K weight matrices.

---

## 5. Grouped Query Attention (GQA)

### Standard Multi-Head Attention Review

In standard MHA, every attention head has its own Q, K, and V projections:

```
Standard MHA (8 heads):

  Q heads: Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇     (8 query heads)
  K heads: K₀  K₁  K₂  K₃  K₄  K₅  K₆  K₇     (8 key heads)
  V heads: V₀  V₁  V₂  V₃  V₄  V₅  V₆  V₇     (8 value heads)

  Each Q head pairs with its own K and V:
  Q₀↔K₀,V₀   Q₁↔K₁,V₁   Q₂↔K₂,V₂   ...
```

### GQA: Sharing K and V

With Grouped Query Attention, multiple Q heads share the same K and V head:

```
GQA (8 query heads, 4 KV heads):

  Q heads: Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇     (8 query heads)
  K heads: K₀       K₁       K₂       K₃          (4 key heads)
  V heads: V₀       V₁       V₂       V₃          (4 value heads)

  Grouping (2 Q heads per KV group):
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Q₀  Q₁  │ │ Q₂  Q₃  │ │ Q₄  Q₅  │ │ Q₆  Q₇  │
  │  K₀ V₀  │ │  K₁ V₁  │ │  K₂ V₂  │ │  K₃ V₃  │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘
   Group 0      Group 1     Group 2     Group 3
```

Q₀ and Q₁ both compute attention using the same K₀ and V₀. They still
produce different attention patterns (because their Q vectors are different),
but they search through the same "database" of keys and values.

### Why GQA?

During inference, the KV-cache stores all past K and V vectors so we don't
recompute them. This cache is often the biggest memory bottleneck:

```
KV-cache size comparison for a 2048-token sequence, dim=768, head_dim=64:

Standard MHA (8 KV heads):
  2048 tokens * 8 heads * 64 dims * 2 (K+V) * 2 bytes = 4 MB per layer

GQA (4 KV heads):
  2048 tokens * 4 heads * 64 dims * 2 (K+V) * 2 bytes = 2 MB per layer

With 24 layers:
  MHA:  96 MB
  GQA:  48 MB  ← half the VRAM for KV-cache
```

Quality loss is minimal. Research shows GQA matches MHA quality while being
significantly more memory-efficient.

### Special Cases

```
n_kv_heads = n_heads        →  Standard MHA (no sharing)
n_kv_heads = 1              →  Multi-Query Attention (MQA, maximum sharing)
1 < n_kv_heads < n_heads    →  GQA (the sweet spot)
```

Our configurations:

```
Small:   n_heads=8,  n_kv_heads=4  (2 Q heads per KV group)
Medium:  n_heads=12, n_kv_heads=4  (3 Q heads per KV group)
```

---

## 6. Feed-Forward Network (SwiGLU)

### What the FFN Does

Attention is all about gathering information from other tokens. The FFN is
about *processing* that gathered information. Think of attention as "reading"
and FFN as "thinking."

Every token goes through the same FFN independently (no token-to-token
interaction here — that's attention's job).

### Standard FFN (for comparison)

A basic FFN is two matrix multiplications with an activation function between:

```typescript
function standardFFN(x: Vector): Vector {
  const hidden = relu(matmul(x, W1));  // project up: dim → 4*dim
  const output = matmul(hidden, W2);   // project down: 4*dim → dim
  return output;
}
```

### SwiGLU: The Gated Version

SwiGLU is a more effective variant. It uses three projections instead of two,
with a gating mechanism:

```typescript
function swigluFFN(x: Vector): Vector {
  const gate = silu(matmul(x, W_gate));  // gate projection: dim → hidden_dim
  const up   = matmul(x, W_up);         // up projection:   dim → hidden_dim
  const gated = gate * up;               // element-wise multiply (gating)
  const output = matmul(gated, W_down);  // down projection: hidden_dim → dim
  return output;
}
```

The gate decides "how much of each feature to let through." It's like an
if-statement for each feature dimension — some features are amplified,
others are suppressed.

```
        Input x (dim=512)
            │
     ┌──────┼──────┐
     ▼      │      ▼
   W_gate   │    W_up
     │      │      │
     ▼      │      ▼
   SiLU     │   (linear)
     │      │      │
     ▼      │      ▼
   gate     │     up
     │      │      │
     └───┬──┘──────┘
         │
         ▼
    gate * up   (element-wise multiply)
         │
         ▼
       W_down
         │
         ▼
    Output (dim=512)
```

### SiLU Activation Function

SiLU (also called "Swish") is a smooth version of ReLU:

```
ReLU:   if x > 0: output = x,  else: output = 0
SiLU:   output = x * sigmoid(x)

ASCII plot:

  output
    │          SiLU          ReLU
  3 │                      ╱    ╱
    │                    ╱    ╱
  2 │                  ╱    ╱
    │                ╱    ╱
  1 │             ╱     ╱
    │           ╱     ╱
  0 │─────────╱─────╱──────
    │       ╱     ╱
 -1 │~---~╱      │
    │             │
    └──────────────────── input
   -4  -2   0   2   4

SiLU is smooth everywhere (no sharp corner at 0).
It allows small negative values, unlike ReLU.
```

The formula:

```
SiLU(x) = x * sigmoid(x) = x * (1 / (1 + e^(-x)))

Where:
  x       = input value
  sigmoid = squishes any number to the range (0, 1)
  e       = Euler's number (~2.718)
```

In words: "multiply the input by how much the sigmoid 'approves' of it."
Positive inputs get mostly passed through. Negative inputs get mostly
squashed to zero, but not entirely.

### Why the 2.667x Hidden Dimension?

Standard FFN has 2 weight matrices, each of size `dim * 4*dim`.
Total parameters: `2 * dim * 4*dim = 8 * dim^2`.

SwiGLU has 3 weight matrices. To keep roughly the same parameter count:
`3 * dim * hidden_dim = 8 * dim^2`, so `hidden_dim = 8/3 * dim ≈ 2.667 * dim`.

```
Standard FFN:  hidden = 4 * dim     →  2 matrices  →  8 * dim^2 params
SwiGLU FFN:    hidden = 2.667 * dim →  3 matrices  →  8 * dim^2 params (same!)
```

In practice, we round `hidden_dim` to a multiple of 256 for GPU efficiency.

---

## 7. RMSNorm: Keeping Things Stable

### The Problem

Neural networks do a lot of matrix multiplications. Each one can make numbers
slightly bigger or smaller. After dozens of layers, values can explode to
millions or collapse to near-zero. When this happens, training breaks down —
gradients become useless.

```
Layer 1 output:  [0.5, -0.3, 0.8, ...]       ← reasonable
Layer 12 output: [150, -89, 234, ...]          ← getting big
Layer 24 output: [45000, -28000, 67000, ...]   ← exploded, training fails
```

### RMSNorm: The Solution

RMSNorm is simple: compute the "root mean square" of the vector, divide by it
(so the vector has RMS = 1), then apply a learnable scale.

```typescript
function rmsNorm(x: number[], weight: number[]): number[] {
  // Step 1: compute root mean square
  const meanSquare = x.reduce((sum, v) => sum + v * v, 0) / x.length;
  const rms = Math.sqrt(meanSquare + 1e-6);  // epsilon for numerical stability

  // Step 2: normalize
  const normalized = x.map(v => v / rms);

  // Step 3: apply learnable scale (element-wise)
  return normalized.map((v, i) => v * weight[i]);
}
```

The formula:

```
RMSNorm(x) = (x / RMS(x)) * γ

Where:
  x      = input vector (e.g., 512 numbers)
  RMS(x) = sqrt(mean(x_i^2))  = sqrt((x_0^2 + x_1^2 + ... + x_n^2) / n)
  γ      = learnable scale parameters (one per dimension, initialized to 1)
```

In words: "Compute how big the numbers are on average (the RMS). Divide
everything by that so the average magnitude is 1. Then let the model learn
per-dimension scaling factors."

### RMSNorm vs LayerNorm

LayerNorm also centers the values (subtracts the mean) and has a bias term.
RMSNorm skips both — just normalizes the scale. Research shows this works
just as well and is faster to compute.

```
LayerNorm:  normalize(x - mean(x)) * γ + β    (center + scale + shift)
RMSNorm:    normalize(x) * γ                   (scale only)
```

### Pre-Norm vs Post-Norm

Where you put the norm matters:

```
Post-norm (original transformer):      Pre-norm (modern, what we use):
  x → Attention → Add → Norm             x → Norm → Attention → Add
  x → FFN       → Add → Norm             x → Norm → FFN       → Add
```

Pre-norm is more stable for training deep networks. The gradient flows more
smoothly through the residual connections (next section) when the norm comes
first. Nearly all modern LLMs use pre-norm.

---

## 8. Residual Connections: The Highway

### The Problem

In a deep network, gradients must flow backwards through every layer during
training. With 24+ layers, gradients can shrink to near-zero by the time they
reach the early layers ("vanishing gradients"). This means early layers
effectively stop learning.

### The Solution: Residual (Skip) Connections

The idea is dead simple: add the input of each sub-layer to its output.

```typescript
// Without residual connection:
const output = attention(x);

// With residual connection:
const output = x + attention(x);
//              ^
//              └── the input goes straight through (the "skip" connection)
```

Think of it like a highway with exits. Information from earlier layers can
flow directly to later layers on the "highway." Each layer's attention/FFN
is an optional "exit" that can add refinements, but the original signal
never gets lost.

```
Without residual:                 With residual:

  Input                            Input ─────────────────┐
    │                                │                     │
    ▼                                ▼                     │
  Layer 1                          Layer 1                 │
    │                                │                     │
    ▼                                ▼                     │
  Layer 2                          + (add) ◄───────────────┘
    │                                │
    ▼                                │──────────────────┐
  Layer 3                          Layer 2               │
    │                                │                   │
    ▼                                ▼                   │
  Output                           + (add) ◄────────────┘
  (gradient                          │
   vanishes)                         ▼
                                   Output
                                   (gradient flows
                                    on the highway)
```

### Why It Works

During backpropagation, the gradient at the `+` operation splits into two
paths: one through the sub-layer (may shrink) and one directly to the previous
layer (stays full strength). The direct path guarantees gradients reach early
layers.

```
Gradient flow at the "+" (residual add):

  gradient_in ──────────►─────────────────►  gradient to previous layer
                         │                   (FULL strength, always)
                         ▼
                    sub-layer
                    gradient
                    (may shrink)
```

This is why transformers can stack 24, 48, even 100+ layers without gradient
problems. Each layer is a small refinement on the highway.

---

## 9. Putting It All Together: One Transformer Block

Here's the complete architecture of a single transformer block, using all the
components from the previous sections:

```
               Input (shape: [seq_len, dim])
                 │
                 ▼
          ┌──────────────┐
          │   RMSNorm     │   Normalize before attention
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  GQA Attention│   Multi-head attention with grouped KV
          │  + RoPE       │   Position encoding via rotation
          │  + Causal Mask│   Can only look at past tokens
          └──────┬───────┘
                 │
                 ▼
               (+)────────────── Residual connection (add input)
                 │
                 ▼
          ┌──────────────┐
          │   RMSNorm     │   Normalize before FFN
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  SwiGLU FFN   │   Process gathered context
          └──────┬───────┘
                 │
                 ▼
               (+)────────────── Residual connection (add input)
                 │
                 ▼
              Output (shape: [seq_len, dim])
```

### In Pseudocode

```typescript
function transformerBlock(
  x: Matrix,          // [seq_len, dim]
  block: BlockWeights,
  positions: number[] // token positions for RoPE
): Matrix {
  // --- Attention sub-layer ---
  let residual = x;
  x = rmsNorm(x, block.attn_norm_weight);

  const q = matmul(x, block.wq);  // [seq_len, n_heads * head_dim]
  const k = matmul(x, block.wk);  // [seq_len, n_kv_heads * head_dim]
  const v = matmul(x, block.wv);  // [seq_len, n_kv_heads * head_dim]

  // Apply RoPE to queries and keys
  const q_rotated = applyRoPE(q, positions);
  const k_rotated = applyRoPE(k, positions);

  // Grouped query attention with causal mask
  const attn_output = groupedQueryAttention(q_rotated, k_rotated, v);
  const projected = matmul(attn_output, block.wo);

  x = residual + projected;  // residual connection

  // --- FFN sub-layer ---
  residual = x;
  x = rmsNorm(x, block.ffn_norm_weight);

  const gate = silu(matmul(x, block.w_gate));
  const up   = matmul(x, block.w_up);
  const ffn_output = matmul(gate * up, block.w_down);

  x = residual + ffn_output;  // residual connection

  return x;
}
```

### The Full Model

Stack N of these blocks together, and you get the full model:

```
Token IDs: [1547, 1078, 2891, ...]
     │
     ▼
┌──────────────────┐
│  Token Embedding  │   Lookup table: token_id → vector
└────────┬─────────┘
         │
         ▼              ┐
┌──────────────────┐    │
│ Transformer Block │    │
│      (Layer 0)    │    │
└────────┬─────────┘    │
         │               │
         ▼               │
┌──────────────────┐    │
│ Transformer Block │    ├── N blocks (e.g., 12 or 24)
│      (Layer 1)    │    │
└────────┬─────────┘    │
         │               │
         ▼               │
        ...              │
         │               │
         ▼               │
┌──────────────────┐    │
│ Transformer Block │    │
│    (Layer N-1)    │    │
└────────┬─────────┘    ┘
         │
         ▼
┌──────────────────┐
│    Final RMSNorm  │   One last normalization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Linear (lm_head) │   Map dim → vocab_size
└────────┬─────────┘
         │
         ▼
   Logits: [vocab_size]   Raw scores for each token
```

In pseudocode:

```typescript
function transformer(tokenIds: number[]): number[][] {
  // Embed tokens
  let x = tokenIds.map(id => embeddingTable[id]);  // [seq_len, dim]

  // Positions for RoPE
  const positions = tokenIds.map((_, i) => i);

  // Pass through all transformer blocks
  for (let layer = 0; layer < numLayers; layer++) {
    x = transformerBlock(x, blocks[layer], positions);
  }

  // Final normalization
  x = rmsNorm(x, finalNormWeight);

  // Project to vocabulary
  const logits = matmul(x, lmHeadWeight);  // [seq_len, vocab_size]

  return logits;
}
```

---

## 10. From Logits to Text: The Output

### What Are Logits?

The final linear layer produces a vector of `vocab_size` numbers for each
token position. These are called "logits" — raw, unnormalized scores.

```
For the last token position, the logits might look like:

  Token ID    Token          Logit (raw score)
  ─────────────────────────────────────────────
  0           <pad>          -12.3
  1           <eos>          -5.1
  ...
  892         " {"           8.7      ← high score
  1189        " return"      11.2     ← highest score
  2891        "Data"         -2.4
  3802        " fetch"       6.1
  ...
  31999       "██"           -15.0

The model thinks " return" is the most likely next token.
```

### Softmax: Logits to Probabilities

Softmax converts raw scores to probabilities that sum to 1:

```
probability(token_i) = e^(logit_i) / sum(e^(logit_j) for all j)

Where:
  e          = Euler's number (~2.718)
  logit_i    = raw score for token i
  the sum    = over ALL tokens in vocabulary (normalizing constant)
```

In words: "raise e to the power of each logit (makes everything positive),
then divide by the total so they all add up to 1."

```
Logits:         [-5.1,  8.7,  11.2,  6.1]
After e^x:      [0.006, 5990, 73130, 446]
Sum:            79566
Probabilities:  [0.00,  0.08,  0.92,  0.01]

" return" gets 92% probability. That's our prediction.
```

### Training: Cross-Entropy Loss

During training, we know the actual next token. The loss function measures
how wrong the model's probability distribution is:

```
loss = -log(probability of the correct token)

If the model assigns 92% to the correct token:
  loss = -log(0.92) = 0.083  ← small loss, good prediction

If the model assigns 5% to the correct token:
  loss = -log(0.05) = 3.0    ← large loss, bad prediction
```

The optimizer adjusts all model parameters to make this loss smaller.
Over billions of examples, the model learns to assign high probability
to the correct next token.

### Inference: Sampling from the Distribution

At inference time, we don't always pick the highest-probability token.
That would be "greedy decoding" and tends to produce repetitive, boring text.
Instead, we sample from the distribution with some controls:

**Temperature** — controls randomness:
```
adjusted_logits = logits / temperature

temperature = 0.0  →  always pick the top token (deterministic)
temperature = 0.7  →  mostly pick likely tokens, some variety
temperature = 1.0  →  sample from the raw distribution
temperature = 2.0  →  very random, picks unlikely tokens often
```

**Top-k** — only consider the top k most likely tokens:
```
top_k = 50: zero out all but the 50 highest-probability tokens, renormalize
```

**Top-p (nucleus sampling)** — only consider tokens whose cumulative
probability reaches p:
```
top_p = 0.9: sort tokens by probability, keep adding tokens until their
cumulative probability reaches 90%, zero out the rest, renormalize
```

For code generation, lower temperature (0.2-0.7) usually works best.
Code has stricter constraints than natural language — there are fewer
"correct" next tokens.

---

## 11. Weight Tying

### The Observation

Look at the two ends of the model:

```
Input:  token_id  →  embedding matrix  →  vector    (dim=512)
Output: vector    →  lm_head matrix    →  logit scores (vocab_size=32000)
```

The embedding matrix has shape `[vocab_size, dim]` = `[32000, 512]`.
The lm_head matrix has shape `[dim, vocab_size]` = `[512, 32000]`.

These are the same shape, transposed! And they do conceptually inverse
operations: one maps from "token space" to "vector space," the other maps
back from "vector space" to "token space."

### Sharing the Weights

Weight tying means we use the same matrix for both:

```typescript
// Without weight tying:
const embeddingMatrix = new Matrix(vocabSize, dim);  // 32000 * 512 = 16.4M params
const lmHeadMatrix    = new Matrix(dim, vocabSize);  // 512 * 32000 = 16.4M params
// Total: 32.8M parameters

// With weight tying:
const sharedMatrix    = new Matrix(vocabSize, dim);  // 32000 * 512 = 16.4M params
const embeddingMatrix = sharedMatrix;
const lmHeadMatrix    = transpose(sharedMatrix);
// Total: 16.4M parameters  ← saved 16.4M parameters!
```

### Why This Works

The embedding matrix learns that similar tokens have similar vectors. If
`for` and `while` have similar embeddings, they share semantic features.

The output layer should do the reverse: if the current hidden state looks
like a "loop keyword" vector, it should assign high scores to both `for`
and `while`. Using the transposed embedding matrix does exactly this.

### Impact

For small models, this is a significant saving:

```
Model with dim=512, vocab=32000:
  Saved parameters:  512 * 32000 = 16.4M
  Total model params: ~50M
  Savings: ~33% of total parameters!

Larger model with dim=1024, vocab=32000:
  Saved parameters:  1024 * 32000 = 32.8M
  Total model params: ~200M
  Savings: ~16% of total parameters
```

Most modern LLMs use weight tying. It's free parameters savings with no
quality loss (and sometimes even helps because it acts as a regularizer).

---

## 12. How Our Model Compares

### Architecture Comparison Table

```
┌──────────────────┬──────────┬──────────┬───────────────┬──────────────┬────────────┐
│ Component        │ Ours     │ LLaMA 3  │ DeepSeek-Coder│ StarCoder 2  │ Mistral    │
│                  │ (small)  │ (8B)     │ (6.7B)        │ (7B)         │ (7B)       │
├──────────────────┼──────────┼──────────┼───────────────┼──────────────┼────────────┤
│ Architecture     │ Decoder  │ Decoder  │ Decoder       │ Decoder      │ Decoder    │
│ Attention        │ GQA      │ GQA      │ MHA           │ GQA          │ GQA        │
│ Position enc.    │ RoPE     │ RoPE     │ RoPE          │ RoPE         │ RoPE       │
│ FFN type         │ SwiGLU   │ SwiGLU   │ SwiGLU        │ SwiGLU       │ SwiGLU     │
│ Normalization    │ RMSNorm  │ RMSNorm  │ RMSNorm       │ RMSNorm      │ RMSNorm    │
│ Norm placement   │ Pre-norm │ Pre-norm │ Pre-norm      │ Pre-norm     │ Pre-norm   │
│ Weight tying     │ Yes      │ No       │ No            │ No           │ No         │
│ Window attention │ No       │ No       │ No            │ Yes (SW)     │ Yes (SW)   │
├──────────────────┼──────────┼──────────┼───────────────┼──────────────┼────────────┤
│ dim              │ 512-1024 │ 4096     │ 4096          │ 4608         │ 4096       │
│ n_layers         │ 12-24    │ 32       │ 32            │ 32           │ 32         │
│ n_heads          │ 8-12     │ 32       │ 32            │ 36           │ 32         │
│ n_kv_heads       │ 4        │ 8        │ 32 (MHA)      │ 4            │ 8          │
│ head_dim         │ 64       │ 128      │ 128           │ 128          │ 128        │
│ hidden_dim (FFN) │ 1365-    │ 14336    │ 11008         │ 17920        │ 14336      │
│                  │  2730    │          │               │              │            │
│ vocab_size       │ 32000    │ 128256   │ 32256         │ 49152        │ 32000      │
│ context_length   │ 2048-    │ 8192     │ 16384         │ 16384        │ 8192-      │
│                  │  4096    │          │               │              │  32768     │
│ Parameters       │ ~50-200M │ 8B       │ 6.7B          │ 7B           │ 7B         │
└──────────────────┴──────────┴──────────┴───────────────┴──────────────┴────────────┘
```

### Key Takeaways

The architecture is the same. Read that again: **our model uses the exact same
architecture as LLaMA 3, Mistral, and the other state-of-the-art models.**

The differences are:
1. **Scale** — they have 30-100x more parameters
2. **Training data** — they trained on trillions of tokens (we'll train on millions)
3. **Training compute** — they used thousands of GPUs for weeks/months
4. **Vocabulary size** — they have larger vocabularies for broader language support
5. **Context length** — they support longer sequences

But the fundamental building blocks — decoder-only transformer, GQA, RoPE,
SwiGLU, RMSNorm, pre-norm residual connections — are identical. What we're
building is a miniature version of these production models.

### What We Get By Using the Same Architecture

- **Proven design** — these choices have been validated at massive scale
- **Tooling compatibility** — same architecture means we can use the same
  inference engines (llama.cpp, vLLM, etc.) with minimal adaptation
- **Transfer learning** — if we later want to fine-tune a larger model,
  the architecture is already familiar
- **Community knowledge** — debugging and optimization techniques from
  the LLaMA/Mistral community apply directly to our model

### What We Don't Have (and That's OK)

Some features present in larger models that we skip:

- **Sliding window attention** (Mistral, StarCoder 2): more memory-efficient
  for very long sequences. Our context is short enough that full attention
  is fine.
- **Mixture of Experts** (Mixtral, DeepSeek-V2): routes tokens to specialized
  sub-networks. Only needed at very large scale.
- **Speculative decoding**, **KV-cache quantization**: inference optimizations
  that matter at deployment scale, not during learning.

---

## Summary: The Full Data Flow

One final look at everything together — follow a single token through the
entire model:

```
"return" (source code)
    │
    ▼
Tokenizer: "return" → token_id 1189
    │
    ▼
Embedding lookup: id 1189 → vector of 512 floats
    │
    ▼
╔═══════════════════════════════════════════════════════╗
║  Transformer Block (repeated N times)                  ║
║                                                        ║
║  1. RMSNorm (normalize the vector)                     ║
║  2. Compute Q, K, V projections                        ║
║  3. Apply RoPE rotations to Q and K                    ║
║  4. GQA: compute attention scores with causal mask     ║
║  5. Weighted sum of V vectors → attention output       ║
║  6. Project output back to model dimension             ║
║  7. Add residual (step 0 input + step 6 output)        ║
║  8. RMSNorm (normalize again)                          ║
║  9. SwiGLU FFN (gate + up + down projections)          ║
║  10. Add residual (step 7 output + step 9 output)      ║
║                                                        ║
╚═══════════════════════════════════════════════════════╝
    │
    ▼
Final RMSNorm
    │
    ▼
Linear projection: 512-dim vector → 32000 logits
    │
    ▼
Softmax → probability distribution over vocabulary
    │
    ▼
Sample next token (e.g., " result" with 23% probability)
    │
    ▼
Append to sequence → repeat for next token
```

Everything in this document — attention, RoPE, GQA, SwiGLU, RMSNorm,
residual connections, weight tying — exists to make that next-token
prediction as accurate as possible.

The architecture is solved. What remains is engineering: building it
efficiently, preparing good training data, and running the training loop.
That's what the rest of this guide covers.
