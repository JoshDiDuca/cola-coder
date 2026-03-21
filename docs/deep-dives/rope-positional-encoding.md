# Rotary Position Embeddings (RoPE) and Theta Tuning

A deep dive into how cola-coder encodes token position using rotations,
why the `rope_theta` parameter matters enormously, and what the research
says about choosing the right value for code models.

No linear algebra degree required. If you can visualize a clock hand spinning,
you can follow this.

---

## 1. The Position Problem

### Why Transformers Are Position-Blind

An RNN processes tokens one at a time, left to right. Position is baked into the
computation itself — by the time it reads token 5, it has already run through
tokens 0-4 sequentially. Position is implicit in the architecture.

Transformers are different. Attention computes all pairwise similarities at once,
in parallel. This is what makes them fast on GPUs — but it also means they have
**zero concept of order**.

```
Input A:  "x = y + z"
Input B:  "z = y + x"

Attention sees:  {x, =, y, +, z}   for BOTH inputs
                 (same tokens, same attention scores)
```

Without position information, the model literally cannot tell these apart. It sees
a bag of tokens, not a sequence. This is a catastrophic limitation for code, where
`a = b` and `b = a` have completely different semantics.

### The TypeScript Analogy

Think of it like this. Suppose you had an `Array.map()` that received every element
but not its index:

```typescript
// Standard map — you know position
["a", "b", "c"].map((item, index) => processWithPosition(item, index));

// Transformer attention without position encoding — no index!
["a", "b", "c"].map((item) => processWithoutPosition(item));
// Can't distinguish ["a", "b", "c"] from ["c", "b", "a"]
```

Position encoding is how we give the transformer its "index parameter" back.

---

## 2. A Brief History of Position Encodings

### Absolute Sinusoidal (Vaswani et al., 2017)

The original "Attention Is All You Need" paper added fixed sine/cosine waves to
the token embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Each position gets a unique vector added directly to the embedding. Simple, but flawed:

- **Absolute**: the model sees position 42 as a fixed identity, not "42 positions from
  the start." In code, the function name at position 5 vs position 500 should behave
  similarly — absolute encoding makes this hard.
- **No extrapolation**: trained on sequences of length 512? The model has never seen
  position 513 and has no idea what to do with it.

### Learned Absolute (GPT-2 / BERT)

Replace the sine functions with learned vectors — one trainable vector per position.
Better in practice but inherits the same fundamental problems: absolute identity,
hard maximum length, no extrapolation.

### Relative Position Encodings (Shaw et al., 2018)

The key insight: what matters in language (and code) is not *where* a token is, but
*how far apart* two tokens are. The relationship between a function name and its
opening `{` should be the same whether they're at positions 10-15 or 200-205.

Relative encodings compute a bias based on the *distance* between query and key
positions. This is architecturally cleaner but the early implementations (T5-style
relative bias) added overhead and complexity.

### ALiBi (Press et al., 2022)

Attention with Linear Biases. Adds a simple linear penalty proportional to distance.
Elegant, but sacrifices some expressiveness — distance always reduces attention,
which isn't always what you want in code (think of a variable used 100 lines below
its declaration).

### RoPE (Su et al., 2021)

Rotary Position Embeddings. The approach used by virtually every top code model today:
LLaMA, Mistral, DeepSeek-Coder, Yi-Coder, Qwen, CodeLlama, StarCoder 2.

RoPE is relative, efficient, and — critically — it can extrapolate to longer
sequences than it was trained on. This is what cola-coder uses.

---

## 3. What RoPE Actually Does

### The Core Idea: Position as Rotation

RoPE encodes position by **rotating** the query and key vectors in 2D planes.

Take the query vector Q and key vector K at each attention head. Instead of adding
a position vector, RoPE *rotates* Q and K by an angle proportional to their
position in the sequence.

When you compute the dot product of two rotated vectors (the attention score), the
result depends only on the *angle difference* — which is the *distance* between
positions. Rotation naturally gives you relative position encoding.

### The 2D Rotation Intuition

Let's start with the simplest case: two dimensions.

Imagine you have a 2D vector (like an arrow pointing somewhere on a flat surface).
Rotating it by angle alpha means spinning it counterclockwise by alpha degrees:

```
                    BEFORE rotation              AFTER rotation by α
                    (position 0)                 (position p)

                         ↑ y                          ↑ y
                         |                            |   ╱ rotated
                         |  ╱ original                |  ╱   vector
                         | ╱                          | ╱  ←── angle = p × freq
                         |╱                           |╱
                    ─────┼──────→ x              ─────┼──────→ x
```

The rotation matrix for angle alpha is:

```
    ┌                    ┐   ┌     ┐
    │  cos(α)   -sin(α)  │   │  x  │
    │  sin(α)    cos(α)  │ × │  y  │
    └                    ┘   └     ┘
```

For a token at position `p`, the rotation angle is `alpha = p * frequency`.
Different positions get different rotation angles, so every position is unique.

### Scaling to High Dimensions: Multiple Clock Hands

A real attention head doesn't have 2 dimensions — it has `head_dim` dimensions
(64 in our 4080_max config). RoPE handles this by pairing dimensions:

```
head_dim = 64

Pair 0:  dims [0, 1]     →  rotate by  p × freq_0   (fastest rotation)
Pair 1:  dims [2, 3]     →  rotate by  p × freq_1
Pair 2:  dims [4, 5]     →  rotate by  p × freq_2
  ...
Pair 31: dims [62, 63]   →  rotate by  p × freq_31  (slowest rotation)
```

Each pair of dimensions is an independent 2D rotation plane, each spinning at a
different frequency. This is the "multiple clock hands" analogy:

```
    FAST CLOCK (pair 0)          MEDIUM CLOCK (pair 16)        SLOW CLOCK (pair 31)
    Completes full rotation      Completes full rotation       Completes full rotation
    in ~6 positions              in ~812 positions             in ~628K positions (θ=10K)

         12                           12                            12
        ╱                            |                             |
      9 ─── 3                     9 ─── 3                      9 ─── 3
        ╲                            |                             |
         6                            6                             6

    Hand position after            Hand barely moved             Hand hasn't moved
    5 tokens: ~~~270°              after 5 tokens: ~2.2°         perceptibly
```

The fast-spinning hands let the model distinguish adjacent tokens (is this `(` right
next to `)` or 3 tokens away?). The slow-spinning hands encode coarse, long-range
position (is this variable reference in the same function or 500 lines away?).

Together, they create a unique "fingerprint" for every position.

---

## 4. The Math, Made Concrete

### Frequency Computation

From our `rope.py`, the frequency for dimension pair `i` is:

```python
freqs[i] = 1.0 / (theta ** (2*i / dim))
```

With `theta = 10,000` and `head_dim = 64`:

```
Pair  0:  freq = 1 / 10000^(0/64)  = 1 / 10000^0.000   = 1.000000
Pair  1:  freq = 1 / 10000^(2/64)  = 1 / 10000^0.03125  = 0.7498
Pair  2:  freq = 1 / 10000^(4/64)  = 1 / 10000^0.0625   = 0.5623
...
Pair 16:  freq = 1 / 10000^(32/64) = 1 / 10000^0.5      = 0.01
...
Pair 31:  freq = 1 / 10000^(62/64) = 1 / 10000^0.96875  = 0.00001334
```

### Wavelength: How Many Positions for a Full Rotation

The *wavelength* is the number of positions it takes for that dimension pair to
complete one full 360-degree rotation (2*pi radians):

```
wavelength = 2π / frequency
```

```
                 theta = 10,000                    theta = 500,000

Pair  0:     λ = 2π / 1.0     =        6.28       λ =        6.28
Pair  1:     λ = 2π / 0.7498  =        8.38       λ =        7.56
Pair  2:     λ = 2π / 0.5623  =       11.18       λ =        8.58
  ...
Pair  8:     λ = 2π / 0.1     =       62.8        λ =       17.0
  ...
Pair 16:     λ = 2π / 0.01    =      628          λ =       62.8
  ...
Pair 24:     λ = 2π / 0.001   =    6,283          λ =      628
  ...
Pair 31:     λ = 2π / 0.0000133 = 471,239         λ =  23,562,149
```

Look at what's happening. With `theta = 10K`:
- Pair 0 completes a full rotation every ~6 tokens (good for local patterns)
- Pair 16 completes a full rotation every ~628 tokens
- Pair 31 completes a full rotation every ~471K tokens

With `theta = 500K`, the wavelengths are spread *much* further apart. We'll come
back to why this matters in Section 6.

### Complex Number Trick

In our implementation, we don't use rotation matrices directly. Instead, we use a
clever mathematical equivalence:

**Rotating a 2D vector = multiplying a complex number by e^(i*angle)**

```python
# From rope.py — this is the key line:
q_complex = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
q_rotated = q_complex * rope_freqs  # complex multiplication = rotation!
```

In TypeScript terms, imagine treating consecutive pairs as (real, imaginary):

```typescript
// Conceptual TypeScript version of complex rotation
interface Complex { re: number; im: number; }

function rotate(v: Complex, angle: number): Complex {
  // This IS the rotation matrix, just written as complex multiplication
  return {
    re: v.re * Math.cos(angle) - v.im * Math.sin(angle),
    im: v.re * Math.sin(angle) + v.im * Math.cos(angle),
  };
}

// Apply to each pair of dimensions
function applyRope(headVector: number[], position: number, freqs: number[]): number[] {
  const result = [...headVector];
  for (let i = 0; i < freqs.length; i++) {
    const angle = position * freqs[i];
    const pair: Complex = { re: result[2*i], im: result[2*i + 1] };
    const rotated = rotate(pair, angle);
    result[2*i] = rotated.re;
    result[2*i + 1] = rotated.im;
  }
  return result;
}
```

PyTorch's `view_as_complex` and complex multiplication do exactly this, but
fused into highly optimized GPU kernels.

---

## 5. Frequency Bands: Local vs. Long-Range

### The Radio Frequency Analogy

Think of RoPE's dimension pairs like radio stations broadcasting at different
frequencies:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FREQUENCY SPECTRUM (dimension pairs)                               │
│                                                                     │
│  HIGH FREQUENCY (pairs 0-7)         "AM Radio — Local News"        │
│  ████████████████                                                   │
│  Wavelength: 6-30 positions                                        │
│  Purpose: Is this ( right next to ) or 5 tokens away?              │
│           Is this ; ending THIS statement or the next one?          │
│                                                                     │
│  MEDIUM FREQUENCY (pairs 8-23)      "FM Radio — Regional"          │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                                 │
│  Wavelength: 30-6000 positions                                      │
│  Purpose: Is this variable in the same function?                    │
│           Is this closing brace matching the opening brace?         │
│                                                                     │
│  LOW FREQUENCY (pairs 24-31)        "Long Wave — International"    │
│  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                 │
│  Wavelength: 6000-471K positions                                    │
│  Purpose: Is this import at the top of the file?                    │
│           Are we in the same class/module?                          │
│           How far from the start of the file are we?                │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Multiple Frequencies Matter for Code

Code has structure at many scales simultaneously:

```python
# Position 0: File-level (need long-range encoding)
import torch
import torch.nn as nn

# Position 50: Class-level (medium range)
class MyModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Position 80: Statement-level (short range)
        self.linear = nn.Linear(dim, dim)  # ( matches ) — need exact local position
        # Position 95: the "dim" argument here refers to the parameter
        # defined at position 60 — medium-range dependency
```

Each frequency band contributes a different kind of positional signal. The model
learns to use the right bands for the right tasks during training.

---

## 6. The Theta Parameter

### What Theta Controls

`theta` (also written as theta or base) is the single most important hyperparameter
in RoPE. It sets the base of the geometric series of frequencies:

```python
freqs[i] = 1.0 / (theta ** (2*i / dim))
```

Larger theta means:
- **Lower starting frequency** for the slow-rotating pairs
- **More spread** between adjacent frequency bands
- **Longer maximum wavelength** — the slowest pair takes more positions to complete
  a full rotation

### Concrete Comparison

Let's see what happens to the frequency spectrum with different theta values.
All numbers for `head_dim = 64` (32 dimension pairs):

```
                  Fastest pair      Middle pair       Slowest pair
                  (pair 0)          (pair 16)         (pair 31)
theta = 10K:      freq = 1.0        freq = 0.01       freq = 0.0000134
                  λ = 6.28          λ = 628           λ = 471K

theta = 500K:     freq = 1.0        freq = 0.00447    freq = 8.95e-8
                  λ = 6.28          λ = 1,405         λ = 70.2M

theta = 10M:      freq = 1.0        freq = 0.00316    freq = 3.16e-9
                  λ = 6.28          λ = 1,988         λ = 1.99B
```

Notice: pair 0 always has `freq = 1.0` (because `theta^0 = 1` regardless of theta).
The local, fine-grained position encoding is unchanged. What changes is how the
medium and low-frequency bands behave.

### The Wavelength Collision Problem

Here's the key problem with low theta values. With `theta = 10K` and a sequence
length of 4096:

```
Pair 31:  wavelength = 471,239 positions
          At position 4096, rotation angle = 4096 / 471239 × 360° = 3.1°

Pair 30:  wavelength = 88,898 positions
          At position 4096, rotation angle = 4096 / 88898 × 360° = 16.6°

Pair 29:  wavelength = 16,767 positions
          At position 4096, rotation angle = 4096 / 16767 × 360° = 87.9°
```

Pairs 30 and 31 have barely rotated by position 4096. Their rotation angles are
so small that they're practically indistinguishable from each other. These
dimensions are **wasted** — they carry almost no positional information within
the actual sequence length.

This is wavelength collision: when multiple frequency bands produce nearly
identical rotation angles across the range of positions that actually appear
during training.

```
  WAVELENGTH COLLISION (theta=10K, seq_len=4096)

  Rotation angle after 4096 positions:

  Pair  0: ████████████████████████████████████ 360° × many  (wraps ~650 times)
  Pair  5: █████████████████████░░░░░░░░░░░░░░░ ~200°
  Pair 10: ███████████░░░░░░░░░░░░░░░░░░░░░░░░░ ~108°
  Pair 15: ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  33°
  Pair 20: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.6°
  Pair 25: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.39°
  Pair 30: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.046°  ← pairs 25-31
  Pair 31: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.031°     nearly identical
           └─ effectively 7 wasted dimensions
```

```
  WITH theta=500K (same seq_len=4096)

  Rotation angle after 4096 positions:

  Pair  0: ████████████████████████████████████ 360° × many  (wraps ~650 times)
  Pair  5: ███████████████████░░░░░░░░░░░░░░░░░ ~178°
  Pair 10: ████████████░░░░░░░░░░░░░░░░░░░░░░░░ ~118°
  Pair 15: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  79°
  Pair 20: █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  53°
  Pair 25: ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  35°
  Pair 30: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  16°    ← much better spread!
  Pair 31: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  12°       every pair is distinct
           └─ ALL 32 pairs carry useful signal
```

With `theta = 500K`, the frequency bands are more evenly distributed across the
useful range. Every dimension pair contributes meaningful positional information.

---

## 7. Why theta=10K Was the Original Choice

The value `theta = 10,000` comes from the original transformer paper (Vaswani et al.,
2017). At the time:

- Sequence lengths were 512 tokens (not 4096+)
- Models were much smaller (65M parameters in the base model)
- The original paper used sinusoidal encodings, not RoPE, but the base frequency
  of 10,000 carried over when Su et al. introduced RoPE

For 512-token sequences, `theta = 10K` is actually quite reasonable — the wavelength
collision problem doesn't manifest because the slow-rotating dimensions still rotate
enough to be distinguishable over just 512 positions.

The problem emerged when the community pushed to 2K, 4K, 8K, and eventually 128K+
context lengths. At those lengths, the frequency utilization with `theta = 10K`
becomes increasingly inefficient.

---

## 8. Why theta=500K-1M Is Better for Code

### Code Has Unique Long-Range Dependencies

Code is unusual among text modalities because it has very precise long-range
dependencies. Natural language has a "locality bias" — most references are to
nearby words. Code does not:

```typescript
// Position 0: import statement
import { DatabaseClient } from './database';    // ← defined here

// ... 200 lines of other code ...

// Position 3800: the reference
async function getUser(id: string) {
  const db = new DatabaseClient();              // ← used 3800 tokens later
  // The model must recall the exact interface from the import
}
```

To handle these dependencies well, the model needs dimension pairs that are
**distinguishable** across distances of 1000-4000+ tokens. With `theta = 10K`,
many of the slow-rotating pairs have practically collapsed to the same angle
at those distances. With `theta = 500K`, they're all distinct.

### Research Results

Every major code-focused LLM has moved away from `theta = 10K`:

| Model              | theta        | Context Length | Notes                        |
|--------------------|-------------|----------------|------------------------------|
| Original (2017)    | 10,000      | 512            | The default that stuck       |
| LLaMA 2 (2023)     | 10,000      | 4,096          | Still using the old value    |
| Code Llama (2023)   | 1,000,000   | 16,384+        | Massive jump for code tasks  |
| LLaMA 3 (2024)     | 500,000     | 8,192          | Adopted high-theta for all   |
| Mistral (2023)     | 10,000      | 8,192          | (with sliding window attn)   |
| Yi-Coder (2024)    | 10,000,000  | 128,000        | Highest theta in production  |
| DeepSeek-V2 (2024) | 10,000*     | 128K           | Uses YaRN extension instead  |
| Qwen 2.5 (2024)    | 1,000,000   | 32,768+        | High theta for code variant  |

The trend is unmistakable: higher theta for longer contexts, especially for code.

### What Improvement to Expect

The improvement from theta tuning is most visible in:

1. **Long-range copy/reference accuracy**: Can the model correctly complete
   `self.linear` when `self.linear = nn.Linear(...)` was defined 2000 tokens ago?
   With `theta = 10K`, accuracy drops off steeply past ~500 tokens. With
   `theta = 500K`, it degrades much more gracefully.

2. **Brace/bracket matching**: Code has strict structural requirements — every `{`
   needs a `}`. At long distances, models with low theta start hallucinating extra
   or missing closing braces.

3. **Training loss on long sequences**: You'll see a lower loss on the later
   positions in a sequence (tokens 2000-4096) with `theta = 500K` vs `10K`,
   because the model has better positional resolution at those distances.

Typical impact: 0.05-0.15 improvement in perplexity on long-context evaluation,
with the biggest gains on structural accuracy tasks. Not enormous on average
metrics, but very noticeable on the failure modes that matter most for code
generation quality.

---

## 9. Context Length Extrapolation

### What Is Extrapolation?

Extrapolation means the model performs reasonably on sequences *longer* than it
was trained on. If you train with `max_seq_len = 4096`, can the model handle
6,000 tokens at inference time?

With absolute position encodings: absolutely not. Position 5000 was never seen,
and the model has no representation for it.

With RoPE: it depends on theta.

### Why Higher Theta Helps Extrapolation

The key insight is that extrapolation fails when rotation angles exceed what the
model saw during training. If the slowest dimension pair rotated at most 3 degrees
during training (because the sequence was only 4096 tokens long), seeing 10 degrees
at inference means the model is in uncharted territory.

Higher theta means *slower rotation rates* for the low-frequency pairs. This means
the rotation angles at longer-than-training positions are less extreme:

```
Training on 4096 tokens, inference at 8192 tokens:

theta = 10K:
  Pair 20 at pos 4096: 3.6° (seen in training)
  Pair 20 at pos 8192: 7.2° (2x what it saw — risky!)

theta = 500K:
  Pair 20 at pos 4096: 53° (seen in training)
  Pair 20 at pos 8192: 106° (2x what it saw — but still within a natural range)
```

The model trained with `theta = 500K` has already seen a wide range of angles
during training. Doubling them is a gentle extrapolation. The model with `theta = 10K`
saw only tiny angles for the slow pairs — doubling something barely above zero
is proportionally a much bigger change.

### Practical Rule of Thumb

With `theta = 500K` and training on 4096 tokens, you can reasonably expect
decent performance up to ~8K-12K tokens at inference (2-3x training length) with
some degradation. Beyond that, you'll want YaRN or similar extensions.

---

## 10. YaRN and Other RoPE Extensions

### Why Extensions Exist

Even with high theta, there are limits to how far you can extrapolate. If you
train on 4K tokens and want to run inference at 128K, you need more than just
a theta change. Three approaches have emerged.

### Position Interpolation (PI, Chen et al., 2023)

The simplest approach. Instead of extrapolating, compress: scale down all position
indices so that a 128K-token sequence maps to the 0-4K range the model was trained on.

```
Standard:                positions [0, 1, 2, ..., 131071]
Position interpolation:  positions [0, 0.03125, 0.0625, ..., 4095]
                         (divide all positions by 32)
```

Cheap and easy, but compresses ALL positions — even nearby tokens that worked
fine. This hurts local pattern recognition. Requires some fine-tuning to work well.

### NTK-Aware Scaling (bloc97, 2023)

The insight: we don't need to compress ALL frequency bands equally.
High-frequency pairs (local patterns) should stay unchanged. Only low-frequency
pairs (long-range) need adjustment.

NTK-aware scaling modifies theta itself, effectively only stretching the
slow-rotating dimensions:

```python
# NTK-aware scaling
alpha = target_length / training_length  # e.g., 128K / 4K = 32
theta_ntk = theta * alpha ** (dim / (dim - 2))
```

This preserves local resolution while extending range. A significant improvement
over naive interpolation.

### YaRN (Yet another RoPE extensioN, Peng et al., 2023)

The state of the art. Combines NTK-aware scaling with a temperature scaling
factor and an attention scaling correction. The idea:

1. **NTK scaling** for the base frequencies (as above)
2. **Temperature correction**: scale the attention logits to compensate for the
   changed magnitude of position encodings
3. **Per-dimension interpolation**: smoothly transition between "no change" for
   high-frequency pairs and "full interpolation" for low-frequency pairs

DeepSeek-V2 uses YaRN to extend from 4K training length to 128K inference.
It's the most principled approach but requires more careful implementation.

### What This Means for cola-coder

For our 4080_max config with 4K training length and `theta = 500K`:

- **No extension needed** for 4K-8K inference: the high theta already handles this
- **NTK scaling** would be sufficient for 8K-32K inference with minimal fine-tuning
- **YaRN** would be the way to go for 32K+ inference

Since we're training from scratch (not extending a pre-trained model), choosing
`theta = 500K` upfront is the simplest and best approach. Extensions like YaRN are
primarily useful for adapting already-trained models to longer contexts.

---

## 11. Our Implementation: Walking Through the Code

### File: `src/cola_coder/model/rope.py`

The implementation has two functions. Let's trace through them step by step.

#### `precompute_rope_freqs` — The Frequency Table

```python
def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
```

This runs **once** at model initialization and creates a lookup table of all
rotation values. No parameters are learned — this is a fixed, deterministic table.

**Step 1: Compute the frequency for each dimension pair**

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
```

`torch.arange(0, dim, 2)` produces `[0, 2, 4, ..., dim-2]`. Dividing by `dim`
and raising theta to that power gives the geometric series of frequencies.

With `dim = 64` and `theta = 500K`:
```
freqs = [1.0, 0.826, 0.683, 0.564, ..., 0.0000000895]
         ↑ pair 0                        ↑ pair 31
         fastest                         slowest
```

**Step 2: Create the position index**

```python
positions = torch.arange(max_seq_len, device=device).float()
```

Just `[0.0, 1.0, 2.0, ..., max_seq_len - 1]`.

**Step 3: Compute all rotation angles**

```python
angles = torch.outer(positions, freqs)
```

Outer product: every position times every frequency. The result is a 2D table
of shape `(max_seq_len, dim // 2)` where `angles[pos][pair]` is the rotation
angle for position `pos` at dimension pair `pair`.

```
         pair 0    pair 1    pair 2    ...    pair 31
pos 0  [  0.000,    0.000,    0.000,  ...     0.000  ]
pos 1  [  1.000,    0.826,    0.683,  ...     0.000  ]
pos 2  [  2.000,    1.653,    1.366,  ...     0.000  ]
pos 3  [  3.000,    2.479,    2.049,  ...     0.000  ]
  ...
pos 4095 [ 4095.0, 3383.5,   2796.2, ...     0.0004 ]
```

**Step 4: Convert to complex exponentials**

```python
return torch.polar(torch.ones_like(angles), angles)
```

`torch.polar(magnitude, angle)` creates complex numbers: `e^(i*angle)`.
The magnitude is 1 (unit rotation — we rotate without scaling).

The result: `cos(angle) + i*sin(angle)` for every (position, pair).
This is the precomputed frequency table.

In TypeScript terms, this function is like building a lookup `Map`:

```typescript
// Conceptual equivalent
type RopeTable = Map<number, Map<number, { cos: number; sin: number }>>;

function precomputeRopeFreqs(dim: number, maxSeqLen: number, theta: number): RopeTable {
  const table: RopeTable = new Map();
  for (let pos = 0; pos < maxSeqLen; pos++) {
    const posMap = new Map();
    for (let pair = 0; pair < dim / 2; pair++) {
      const freq = 1.0 / Math.pow(theta, (2 * pair) / dim);
      const angle = pos * freq;
      posMap.set(pair, { cos: Math.cos(angle), sin: Math.sin(angle) });
    }
    table.set(pos, posMap);
  }
  return table;
}
```

#### `apply_rope` — Applying the Rotations

```python
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
```

This runs on **every forward pass**. It takes Q and K tensors and rotates them
using the precomputed frequency table.

**Step 1: Slice the frequency table**

```python
rope_freqs = freqs[start_pos : start_pos + seq_len]
rope_freqs = rope_freqs.unsqueeze(0).unsqueeze(2)
```

During training, `start_pos = 0` and we take the full sequence length.
During inference with KV-cache, `start_pos` is the current position and
`seq_len = 1` (just the new token). The unsqueeze adds batch and head dimensions
for broadcasting.

**Step 2: Reinterpret as complex numbers**

```python
q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
```

This takes the last dimension (head_dim = 64) and reshapes it as 32 complex
numbers. No data is copied — it's a reinterpretation. Dimensions `[0, 1]` become
the first complex number, `[2, 3]` become the second, and so on.

**Step 3: Rotate via complex multiplication**

```python
q_rotated = q_complex * rope_freqs
```

Complex multiplication IS rotation. If `z = a + bi` and `r = cos(theta) + i*sin(theta)`,
then `z * r` rotates `z` by angle theta. This single line is the entire RoPE operation.

**Step 4: Convert back to real**

```python
q_out = torch.view_as_real(q_rotated).reshape(q.shape)
return q_out.type_as(q), k_out.type_as(k)
```

Unpack the complex numbers back to pairs of real numbers and cast back to the
original dtype (bf16 during training).

### Where RoPE Gets Called

In `attention.py`, line 139:

```python
q, k = apply_rope(q, k, rope_freqs, start_pos)
```

Notice: only Q and K are rotated, never V. This is by design — position
information enters through the attention scores (which depend on Q and K dot
products), not through the values that get aggregated.

---

## 12. Practical Impact: What You'd See in Training

### theta=10K vs theta=500K on cola-coder

If you trained two identical 455M models with `max_seq_len = 4096`, one with
`theta = 10K` and one with `theta = 500K`, here's what to expect:

**Early training (steps 0-5000):**
- Nearly identical loss curves. Position encoding barely matters when the model
  is still learning basic token distributions.

**Mid training (steps 5000-50000):**
- The `theta = 500K` model starts pulling ahead on sequences with long-range
  dependencies. The gap shows up mainly on later tokens in the sequence
  (positions 2000-4096).
- Average loss might differ by only 0.02-0.05 at this stage.

**Late training (steps 50000+):**
- The `theta = 500K` model shows clearer advantages:
  - Better closing brace prediction for deeply nested structures
  - More accurate variable references to distant declarations
  - Fewer "forgot the import" style errors
  - Perplexity improvement of ~0.1-0.15 on long-context evaluation sets

**Qualitative code generation differences:**

```python
# theta=10K model at inference (4K context):
def process_data(self):
    # After 2000+ tokens of context...
    result = self.transformer(x)    # might hallucinate wrong attribute name
    return result.logits             # because positional signal is degraded

# theta=500K model at inference (same 4K context):
def process_data(self):
    # After 2000+ tokens of context...
    result = self.transformer(x)    # correctly references attribute from
    return result.logits             # class definition 2000 tokens earlier
```

The difference is subtle in aggregate metrics but very noticeable in practice,
especially for code completion tasks that require tracking state across long files.

---

## 13. When to Change Theta

### Scenarios Where You'd Modify rope_theta

**Starting a new training run from scratch:**
- Use `theta = 500K` for `max_seq_len >= 2048` (our 4080_max config)
- Use `theta = 10K` for `max_seq_len <= 512` (the classic value is fine here)
- Use `theta = 10K-50K` for `max_seq_len = 1024` (a moderate middle ground)

**Extending context length of a trained model:**
- If you trained at 4K and want to fine-tune at 8K, consider increasing theta
  proportionally: `new_theta = old_theta * (new_seq_len / old_seq_len)`
- Or use NTK-aware scaling (see Section 10)
- Requires some fine-tuning — you can't just change theta and expect it to work

**Fine-tuning a pre-trained model:**
- Keep the same theta as the pre-trained model used
- Only change theta if you're also extending the context length
- Changing theta without purpose will hurt the model — it learned position
  patterns based on the original theta

**Switching between configs:**
- Our tiny/small/medium configs use `theta = 10K` (default) with shorter
  sequence lengths
- The 4080_max config uses `theta = 500K` with 4096 sequence length
- These are independent training runs — no need for consistency across configs

### The Rule of Thumb

```
theta ≈ 10000 * (max_seq_len / 512)

seq_len =  512  →  theta ≈  10,000
seq_len = 1024  →  theta ≈  20,000
seq_len = 2048  →  theta ≈  40,000
seq_len = 4096  →  theta ≈  80,000-500,000   (higher for code)
seq_len = 8192  →  theta ≈ 500,000-1,000,000
```

This is a rough starting point. Code models benefit from being on the higher end
of this range because code has more precise long-range dependencies than natural
language.

The research community has converged on `theta = 500K` as a strong default for
4K-8K context code models, and `theta = 1M-10M` for 32K-128K context models.
Our choice of `theta = 500,000` in `configs/4080_max.yaml` follows this consensus.

---

## 14. Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                     RoPE AT A GLANCE                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  WHAT:   Encode position by rotating Q/K vectors in 2D planes     │
│  WHY:    Gives relative position encoding — attention scores       │
│          depend on DISTANCE between tokens, not absolute position  │
│  HOW:    Pair up dimensions, each pair rotates at different speed  │
│          Complex multiplication = rotation (fast on GPU)           │
│                                                                    │
│  THETA:  Controls the frequency spread                             │
│          Low theta (10K): frequencies cluster, wasted dimensions   │
│          High theta (500K): frequencies spread, all dims useful    │
│                                                                    │
│  OUR CHOICE:                                                       │
│          configs/4080_max.yaml → rope_theta: 500000.0              │
│          Follows LLaMA 3, Code Llama, Yi-Coder research            │
│          Optimal for 4096-token code sequences                     │
│                                                                    │
│  CODE:                                                             │
│          rope.py → precompute_rope_freqs() + apply_rope()          │
│          ~60 lines total. Precomputed table, complex multiply.     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Further Reading

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021 (the original RoPE paper)
- [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) — Chen et al., 2023
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — Peng et al., 2023
- [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) — bloc97, 2023
- [LLaMA 3 Technical Report](https://arxiv.org/abs/2407.21783) — Meta, 2024 (discusses theta=500K choice)
