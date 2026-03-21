# Fill-in-the-Middle (FIM): Teaching Your Model to Complete Code, Not Just Continue It

A standard language model can only generate code left-to-right. Type a function
signature, and it'll write the body. But what happens when a developer puts
their cursor *in the middle* of existing code and hits Tab? The model sees code
**above** *and* **below** the cursor — and a vanilla left-to-right model has
no idea what to do with that.

Fill-in-the-Middle (FIM) fixes this. It's the single technique that makes
code models useful in real IDEs.

This guide covers the theory, the research, and exactly how cola-coder
implements it.

---

## Table of Contents

1. [The Problem: Left-to-Right Isn't How Developers Write Code](#1-the-problem-left-to-right-isnt-how-developers-write-code)
2. [What FIM Actually Is](#2-what-fim-actually-is)
3. [PSM Format (Prefix-Suffix-Middle)](#3-psm-format-prefix-suffix-middle)
4. [SPM Format (Suffix-Prefix-Middle)](#4-spm-format-suffix-prefix-middle)
5. [Why Mixing PSM and SPM Matters](#5-why-mixing-psm-and-spm-matters)
6. [Special Tokens and How the Model Learns Them](#6-special-tokens-and-how-the-model-learns-them)
7. [The Training Signal: Why "Code With a Hole" Works](#7-the-training-signal-why-code-with-a-hole-works)
8. [FIM Rate: How Much Is Enough?](#8-fim-rate-how-much-is-enough)
9. [Our Implementation in Detail](#9-our-implementation-in-detail)
10. [Real-World Use: From Cursor Position to FIM Query](#10-real-world-use-from-cursor-position-to-fim-query)
11. [Configuration in the Cola-Coder Pipeline](#11-configuration-in-the-cola-coder-pipeline)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Key Takeaways](#13-key-takeaways)

---

## 1. The Problem: Left-to-Right Isn't How Developers Write Code

Think about how you actually write TypeScript. You don't start at line 1 and
type sequentially to line 500. You:

- Write a function signature, skip the body, write the next function, then
  come back and fill in the first body
- Add a parameter to a function, then jump to every call site to pass it
- Insert a new line between two existing lines
- Place your cursor inside a half-written expression and hit autocomplete

All of these are **middle-of-document** edits. A standard autoregressive model
(GPT-style, left-to-right) literally cannot do this. It only knows how to
predict the *next* token given all *previous* tokens. It has never seen the
pattern "here's what comes before AND after — fill in the gap."

```
Standard LM sees:

  function greet(name: string) {
    const message = "Hello, " + █
                                 ↑ cursor
  Only knows what's ABOVE. The closing } below? Invisible.


What IDE autocomplete needs:

  function greet(name: string) {
    const message = "Hello, " + █ + "!";
                                 ↑ cursor
  Sees ABOVE ("Hello, " +) AND BELOW (+ "!";)
  Can infer: name
```

Without FIM training, a model asked to complete that gap would generate
something like `"World";\n}\n\nfunction ...` — it would keep going past the
cursor, duplicating the code that already exists below.

---

## 2. What FIM Actually Is

FIM is a **data transformation** applied during training. You take a normal
code sequence, cut it into three pieces (prefix, middle, suffix), rearrange
them with special separator tokens, and train the model on the rearranged
version.

The key insight from the paper ("Efficient Training of Language Models to Fill
in the Middle," Bavarian et al., 2022) is that this transformation is **free**.
You don't need extra data. You don't need a different architecture. You just
rearrange existing training sequences, and the model learns a new capability
without losing its left-to-right ability.

Here's the TS analogy. Imagine you have an array:

```typescript
const original = ["a", "b", "c", "d", "e", "f", "g", "h"];
//                  └─ prefix ─┘  └─ middle ─┘  └─ suffix ─┘
```

FIM rearranges it to:

```typescript
const fim = ["<PREFIX>", "a", "b",           // prefix
             "<SUFFIX>", "f", "g", "h",      // suffix
             "<MIDDLE>", "c", "d", "e"];      // middle (the "answer")
```

The model trains on this sequence left-to-right. When it reaches `<MIDDLE>`,
it has already "seen" both the prefix and suffix in its context window, so it
can predict `c, d, e` conditioned on both. The loss is computed on the
**entire** sequence — including the prefix and suffix sections — but the
critical learning happens on the middle tokens, where the model must
synthesize both contexts.

---

## 3. PSM Format (Prefix-Suffix-Middle)

PSM is the most intuitive FIM format. It presents the code in a natural
reading order: prefix first, suffix second, then the middle that goes between
them.

### The transformation step by step

Given this TypeScript source:

```typescript
// Original source code
interface User {
  name: string;
  email: string;
  age: number;
}

function validateUser(user: User): boolean {
  return user.name.length > 0 && user.email.includes("@");
}
```

Suppose we randomly split it into three pieces at line boundaries:

```
PREFIX (lines 1-4):
  interface User {
    name: string;
    email: string;

MIDDLE (lines 5-6):
    age: number;
  }

SUFFIX (lines 7-10):

  function validateUser(user: User): boolean {
    return user.name.length > 0 && user.email.includes("@");
  }
```

The PSM transformation produces:

```
<|fim_prefix|>interface User {
  name: string;
  email: string;
<|fim_suffix|>
function validateUser(user: User): boolean {
  return user.name.length > 0 && user.email.includes("@");
}
<|fim_middle|>  age: number;
}
```

### What the model sees during training

```
┌─────────────────────────────────────────────────────────────┐
│ <|fim_prefix|>  interface User {\n  name: string;\n  ...    │
│                 ↑                                           │
│                 "Here's what comes BEFORE the hole"         │
│                                                             │
│ <|fim_suffix|>  \nfunction validateUser(user: User)...      │
│                 ↑                                           │
│                 "Here's what comes AFTER the hole"          │
│                                                             │
│ <|fim_middle|>    age: number;\n}\n                         │
│                 ↑                                           │
│                 "Now generate what goes IN the hole"        │
└─────────────────────────────────────────────────────────────┘

Autoregressive prediction (left to right):
  ──────────────────────────────────────────────>
  prefix tokens → suffix tokens → MIDDLE tokens
                                  ^^^^^^^^^^^^^^
                                  This is where FIM "pays off."
                                  The model predicts these tokens
                                  with BOTH prefix and suffix in
                                  its attention window.
```

The cross-entropy loss is computed on the entire sequence, but the
middle section is where the model really demonstrates its FIM
capability — it must use bidirectional context to predict what fills
the gap.

---

## 4. SPM Format (Suffix-Prefix-Middle)

SPM flips the order of prefix and suffix:

```
<|fim_suffix|>  [suffix tokens]
<|fim_prefix|>  [prefix tokens]
<|fim_middle|>  [middle tokens]
```

Using the same example:

```
<|fim_suffix|>
function validateUser(user: User): boolean {
  return user.name.length > 0 && user.email.includes("@");
}
<|fim_prefix|>interface User {
  name: string;
  email: string;
<|fim_middle|>  age: number;
}
```

### Why does SPM exist?

The suffix-first ordering has a subtle benefit: it forces the model to be
**order-agnostic** about how context arrives. In PSM, the model always sees
prefix first, which could create a bias toward "reading forward." SPM breaks
that habit.

Think of it like this in TS terms:

```typescript
// PSM is like: "I'll tell you what came before, then what came after."
// SPM is like: "I'll tell you what came after, then what came before."

// A robust model should handle EITHER ordering.
// It's the same information — just presented differently.
```

In practice, SPM slightly improves the model's ability to use suffix context.
Without SPM training, models tend to "forget" the suffix and over-rely on the
prefix — because in PSM, the prefix tokens are always more recent in the
attention window (they're adjacent to the middle tokens via the suffix
in-between).

---

## 5. Why Mixing PSM and SPM Matters

The research is clear: **mixing PSM and SPM at roughly 50/50 yields the best
results.** The original FIM paper and follow-up work (including from
BigCode/StarCoder) show a roughly **+5 point improvement** on infilling
benchmarks when you use a mix vs. PSM-only.

```
Benchmark scores (approximate, from FIM paper):

  PSM only:     62% on infill benchmarks
  SPM only:     60% on infill benchmarks
  50/50 mix:    67% on infill benchmarks   ← +5 pts over PSM-only
                                             +7 pts over SPM-only
```

Why does mixing help more than either format alone?

1. **Prevents format overfitting.** If the model only sees PSM, it develops a
   rigid "prefix always comes first" prior. Mixing forces it to learn a more
   general "I have prefix and suffix in some order — find the boundary tokens
   and use both."

2. **Better suffix attention.** In PSM, the suffix sits between the prefix and
   middle. The model's attention to suffix tokens can get "diluted" by the
   intervening prefix tokens. In SPM, the suffix is at the start — it gets
   strong positional priority. Training on both makes the model robust.

3. **Matches real-world variance.** Different IDE plugins and inference
   backends may format FIM queries differently. A model trained on both
   formats handles more deployment scenarios.

Cola-coder defaults to `psm_rate=0.5`, meaning each FIM sample has a 50%
chance of being PSM and 50% chance of being SPM.

---

## 6. Special Tokens and How the Model Learns Them

FIM uses three special tokens that are part of the tokenizer vocabulary:

| Token | ID Attribute | Purpose |
|-------|-------------|---------|
| `<\|fim_prefix\|>` | `tokenizer.fim_prefix_id` | "What follows is the code BEFORE the hole" |
| `<\|fim_suffix\|>` | `tokenizer.fim_suffix_id` | "What follows is the code AFTER the hole" |
| `<\|fim_middle\|>` | `tokenizer.fim_middle_id` | "Now generate what goes IN the hole" |

These tokens are **never split** by the tokenizer — each is a single token ID.
They act as structural markers, like XML tags that the model learns to parse.

### How does the model learn what these tokens mean?

It's pure pattern recognition from training data. Consider hundreds of
thousands of training examples:

```
Example 1: <|fim_prefix|> import React from  <|fim_suffix|>  export default App;  <|fim_middle|> "react";\n\nfunction App() {\n  return <div>Hello</div>;\n}\n\n
Example 2: <|fim_prefix|> const sum = (a: number, b: number)  <|fim_suffix|>  console.log(sum(1, 2));  <|fim_middle|> : number => a + b;\n\n
Example 3: <|fim_prefix|> class Dog {\n  constructor(  <|fim_suffix|>  }\n  bark() { return "woof"; }\n}  <|fim_middle|> public name: string) {\n    this.name = name;\n
```

After seeing enough examples, the model learns:

- After `<|fim_prefix|>`: expect code that ends abruptly mid-thought
- After `<|fim_suffix|>`: expect code that starts abruptly mid-thought
- After `<|fim_middle|>`: generate code that bridges the prefix and suffix

It's analogous to how TypeScript's type system works. The special tokens are
like generic type parameters — they don't inherently mean anything, but from
their consistent usage patterns, the compiler (model) learns their role:

```typescript
// The model "learns" something like this type signature:
type FIMSequence<P extends Code, S extends Code, M extends Code> = {
  prefix: P;       // after <|fim_prefix|>
  suffix: S;       // after <|fim_suffix|>
  middle: M;       // after <|fim_middle|> — the part to generate
  constraint: P + M + S === OriginalCode;  // must form valid code
};
```

### Token setup in cola-coder

The `setup_fim_tokenizer()` function in `fim.py` ensures these tokens exist in
the vocabulary and caches their IDs:

```python
# From src/cola_coder/data/fim.py
_FIM_TOKENS = {
    "fim_prefix": "<|fim_prefix|>",
    "fim_suffix": "<|fim_suffix|>",
    "fim_middle": "<|fim_middle|>",
}

# Each token is verified to be in the vocabulary.
# If missing (e.g., old tokenizer), it's added.
# The IDs are cached as: tokenizer.fim_prefix_id, etc.
```

---

## 7. The Training Signal: Why "Code With a Hole" Works

The core question: why does shuffling sequence order teach the model anything
new? After all, the tokens are the same — just rearranged.

### Standard left-to-right training

```
Input:   t1  t2  t3  t4  t5  t6  t7  t8
Target:  t2  t3  t4  t5  t6  t7  t8  <eos>

Each token is predicted from ALL previous tokens (causal mask).
t5 sees: t1, t2, t3, t4
t5 does NOT see: t6, t7, t8
```

The model learns: "given everything before position i, predict position i."
This is great for generating code forward from a prompt, but it means the
model **never practices** predicting a token using future context.

### FIM training

```
PSM input:   <P>  t1  t2  <S>  t6  t7  t8  <M>  t3  t4  t5
             └─prefix─┘  └───suffix───┘    └──middle──┘

When predicting t3 (first middle token):
  t3 sees: <P>, t1, t2, <S>, t6, t7, t8, <M>
  That means t3 is predicted using BOTH prefix (t1,t2) AND suffix (t6,t7,t8)!
```

The causal attention mask is unchanged — the model still only looks "left"
in the sequence. But because we've physically moved the suffix tokens to
appear *before* the middle tokens, "looking left" now includes both prefix
and suffix context.

```
Standard LM attention for predicting t5:

  t1 ──→ t2 ──→ t3 ──→ t4 ──→ [t5]
  ↑      ↑      ↑      ↑
  can    can    can    can
  see    see    see    see

FIM attention for predicting t3 (which was originally t3 in the source):

  <P> ──→ t1 ──→ t2 ──→ <S> ──→ t6 ──→ t7 ──→ t8 ──→ <M> ──→ [t3]
  ↑       ↑      ↑      ↑       ↑      ↑      ↑      ↑
  can     can    can    can     can    can    can    can
  see     see    see    see     see    see    see    see
                                ^^^^^^^^^^^^^^^^^^
                                suffix context is now visible!
```

This is the elegant trick: **no architecture changes needed**. The standard
causal transformer, with its standard left-to-right attention mask, learns
bidirectional infilling purely through data transformation.

### The "free lunch" property

The FIM paper's most surprising finding: **FIM training doesn't hurt
left-to-right performance.** When 50% of training samples use FIM and 50%
are standard left-to-right, the model's performance on standard benchmarks
(HumanEval, etc.) is essentially unchanged — while it gains a whole new
infilling capability.

This is because:
- The non-FIM samples maintain left-to-right skill
- The FIM samples still train on valid code tokens (just reordered)
- The prefix/suffix sections in FIM samples also provide left-to-right
  signal (the model still predicts them autoregressively)

---

## 8. FIM Rate: How Much Is Enough?

`fim_rate` controls what fraction of training samples get the FIM
transformation. The rest stay as standard left-to-right sequences.

### Research findings

| FIM Rate | Left-to-Right Perf | Infill Perf | Recommendation |
|----------|-------------------|-------------|----------------|
| 0%       | Baseline          | 0 (can't do it) | No FIM capability |
| 10-30%   | ~Baseline         | Moderate    | Too little FIM practice |
| **50%**  | **~Baseline**     | **Good**    | **Sweet spot (default)** |
| 70%      | Slight drop       | Very good   | Aggressive but viable |
| 90%+     | Noticeable drop   | Slightly better | Too much — hurts L2R |
| 100%     | Significant drop  | Best infill | Never do this |

The research consensus (FIM paper, StarCoder, Code Llama) is that **50-70%
is optimal**. At this range:

- Left-to-right generation is preserved (within ~0.5% of no-FIM baseline)
- Infilling capability is near-maximum
- The model sees enough of both formats to generalize

### Why not 100%?

If every sample is FIM, the model never practices raw left-to-right generation.
That matters because:

1. **Many code tasks ARE left-to-right.** "Write a function that..." is pure
   left-to-right generation. You don't want to degrade this.

2. **FIM adds noise to the training signal.** The special tokens and
   rearrangement mean some training capacity goes to learning the FIM
   protocol rather than code patterns. At 100%, that overhead dominates.

3. **Diminishing returns.** Going from 0% to 50% FIM is a massive infilling
   improvement. Going from 50% to 100% barely helps infilling but
   measurably hurts everything else.

### Cola-coder defaults

```yaml
# In training config or collator settings:
fim_rate: 0.5    # 50% of samples → FIM, 50% → standard L2R
psm_rate: 0.5    # Of FIM samples: 50% PSM, 50% SPM
```

This means in any given batch of 32 samples:
- ~16 are standard left-to-right
- ~8 are PSM format
- ~8 are SPM format

---

## 9. Our Implementation in Detail

Cola-coder implements FIM in two places, for two stages of the pipeline.

### FIMTransform (`src/cola_coder/data/fim.py`)

This is the main transformation class with two APIs:

**Token-level API** — `apply(token_ids, tokenizer)`:
Used during training on already-tokenized data. Operates on lists of integer
token IDs.

```python
# Simplified flow of apply():
def apply(self, token_ids, tokenizer):
    # 1. Random skip: with probability (1 - fim_rate), return unchanged
    if self._rng.random() >= self.fim_rate:
        return token_ids

    # 2. Reserve 3 slots for special tokens (to keep length constant)
    content = token_ids[: n - 3]

    # 3. Pick two random split points (constrained to 10%-90% of length)
    split1 = random.randint(min_idx, max_idx - 1)
    split2 = random.randint(split1 + 1, max_idx)

    prefix = content[:split1]
    middle = content[split1:split2]
    suffix = content[split2:]

    # 4. Assemble in PSM or SPM format
    if random.random() < self.psm_rate:
        result = [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle
    else:
        result = [fim_suffix_id] + suffix + [fim_prefix_id] + prefix + [fim_middle_id] + middle

    # 5. Truncate to original length
    return result[:n]
```

**Text-level API** — `apply_to_text(text)`:
Used during data preparation on raw source strings. Critically, this API
**splits at line boundaries** when possible:

```python
# Simplified flow of apply_to_text():
def apply_to_text(self, text):
    lines = text.splitlines(keepends=True)

    if len(lines) >= 3:
        # Split at LINE boundaries (clean splits)
        split1 = random_line_index(...)
        split2 = random_line_index(...)
        prefix = "".join(lines[:split1])
        middle = "".join(lines[split1:split2])
        suffix = "".join(lines[split2:])
    else:
        # Fall back to character boundaries for short texts
        # (still safe — just less clean)
```

Line-boundary splitting is important because it avoids cutting identifiers
or keywords in half. More on this in [Common Pitfalls](#12-common-pitfalls).

### The split constraints

The `FIMTransform` enforces that the middle section is between 10% and 90%
of the total length:

```
MIN_MIDDLE_FRAC = 0.10
MAX_MIDDLE_FRAC = 0.90
```

This prevents degenerate cases:

```
BAD: middle is 1 token, prefix is 500 tokens, suffix is 500 tokens
     → The model barely practices infilling (trivial middle)

BAD: middle is 990 tokens, prefix is 5 tokens, suffix is 5 tokens
     → The model barely has any context to work with

GOOD: middle is 100-400 tokens out of 512
      → Substantial context AND substantial infill target
```

```
Visualized:

  |<──────────── sequence length ──────────────>|
  |   10%   |                              | 10%|
  |         |     valid split range        |    |
  | min     |<─────────────────────────────>|   | max
  | bound   |  split points chosen here    |   | bound
```

### FIMCollator (`src/cola_coder/data/collator.py`)

The collator applies FIM during the batching step — right before data enters
the model. This is useful when you want to apply FIM dynamically (different
random splits each epoch) rather than pre-computing FIM during data prep.

```python
class FIMCollator:
    def __call__(self, examples):
        batch = []
        for ex in examples:
            tokens = ex["input_ids"]
            if random.random() < self.fim_rate:
                tokens = self._apply_fim(tokens)
            batch.append(tokens)
        return {"input_ids": torch.stack(batch)}
```

The collator's `_apply_fim` method uses a simpler split strategy than
`FIMTransform` — it picks one split region and constructs PSM format, then
truncates/pads to maintain a constant sequence length.

Key difference from `FIMTransform`: the collator works directly with
PyTorch tensors (not Python lists), and it constructs the FIM sequence using
`torch.cat`. This makes it efficient for the training hot path.

---

## 10. Real-World Use: From Cursor Position to FIM Query

Here's how FIM training translates to actual IDE autocomplete at inference
time.

### Step 1: User places cursor

```typescript
// File open in VS Code:
function calculateDiscount(price: number, tier: "gold" | "silver" | "bronze") {
  const rates = { gold: 0.3, silver: 0.2, bronze: 0.1 };
  █                                          ← cursor is here
  return price * (1 - discount);
}
```

### Step 2: IDE extension creates FIM prompt

The extension splits the file content at the cursor position:

```typescript
// In the IDE extension (TypeScript):
const fileContent = editor.document.getText();
const cursorOffset = editor.document.offsetAt(editor.selection.active);

const prefix = fileContent.substring(0, cursorOffset);
const suffix = fileContent.substring(cursorOffset);

// Construct FIM prompt:
const fimPrompt = `<|fim_prefix|>${prefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
```

The resulting prompt sent to the model:

```
<|fim_prefix|>function calculateDiscount(price: number, tier: "gold" | "silver" | "bronze") {
  const rates = { gold: 0.3, silver: 0.2, bronze: 0.1 };
  <|fim_suffix|>
  return price * (1 - discount);
}
<|fim_middle|>
```

### Step 3: Model generates the middle

The model, having trained on thousands of FIM examples, generates:

```
const discount = rates[tier];
```

It knows to:
- Declare `discount` (because the suffix uses it in `price * (1 - discount)`)
- Use `rates[tier]` (because the prefix defines `rates` and the function takes `tier`)
- End with a newline (because the suffix starts with `\n  return`)

### Step 4: IDE inserts the completion

```typescript
function calculateDiscount(price: number, tier: "gold" | "silver" | "bronze") {
  const rates = { gold: 0.3, silver: 0.2, bronze: 0.1 };
  const discount = rates[tier];              ← inserted by model
  return price * (1 - discount);
}
```

### The full flow as an ASCII diagram

```
  ┌─────────────────────────────────────────────────────┐
  │  IDE / Editor                                       │
  │                                                     │
  │  1. User types or places cursor                     │
  │  2. Extension captures:                             │
  │     • prefix = everything before cursor             │
  │     • suffix = everything after cursor              │
  │  3. Sends FIM-formatted prompt to model             │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Model (cola-coder inference)                       │
  │                                                     │
  │  4. Tokenize the FIM prompt                         │
  │  5. Run autoregressive generation                   │
  │     • Attends to prefix AND suffix tokens           │
  │     • Generates tokens after <|fim_middle|>         │
  │  6. Stop at <|eos|> or max length                   │
  │  7. Return generated middle tokens                  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  IDE / Editor                                       │
  │                                                     │
  │  8. Decode tokens to text                           │
  │  9. Insert at cursor position                       │
  │  10. User sees inline suggestion (ghost text)       │
  └─────────────────────────────────────────────────────┘
```

---

## 11. Configuration in the Cola-Coder Pipeline

### Where FIM is configured

FIM settings can appear in several places:

**In the collator** (dynamic FIM at training time):

```python
from cola_coder.data.collator import FIMCollator

collator = FIMCollator(
    fim_rate=0.5,          # 50% of samples get FIM
    fim_prefix_id=4,       # <|fim_prefix|> token ID
    fim_middle_id=5,       # <|fim_middle|> token ID
    fim_suffix_id=6,       # <|fim_suffix|> token ID
)
```

**In the FIMTransform** (pre-computed FIM during data prep, or dynamic):

```python
from cola_coder.data.fim import FIMTransform

transform = FIMTransform(
    fim_rate=0.5,          # 50% FIM probability
    psm_rate=0.5,          # 50% PSM, 50% SPM
    truncate_or_pad=True,  # Keep sequence length constant
    seed=42,               # Reproducible for testing
)
```

**Token IDs are set up automatically:**

```python
from cola_coder.data.fim import setup_fim_tokenizer

# This verifies/adds FIM tokens and caches IDs on the tokenizer:
ids = setup_fim_tokenizer(tokenizer)
# ids = {"fim_prefix": 4, "fim_suffix": 6, "fim_middle": 5}
# Also sets: tokenizer.fim_prefix_id, tokenizer.fim_suffix_id, tokenizer.fim_middle_id
```

### Dynamic vs. pre-computed FIM

There are two strategies, and cola-coder supports both:

| Strategy | When FIM is applied | Pros | Cons |
|----------|-------------------|------|------|
| **Dynamic** (collator) | Each batch, at training time | Different FIM splits every epoch; more data diversity | Slightly slower training loop |
| **Pre-computed** (data prep) | Once, during `prepare_data.py` | Zero overhead at training time | Same splits every epoch; less diversity |

For small-scale training (our use case), **dynamic FIM via the collator is
recommended**. The overhead is negligible on modern hardware, and the
diversity benefit of seeing different FIM splits each epoch is significant
for small models.

---

## 12. Common Pitfalls

### Pitfall 1: Splitting mid-token

If you split raw text at an arbitrary character position, you can break a
multi-character token in half:

```
Original text: "function calculateTotal("
                         ↑ split here
Prefix: "function calc"
Middle: "ulateTotal("

Problem: "calc" and "ulate" are not natural token boundaries.
The tokenizer will create DIFFERENT tokens for "calc" vs "calculate".
The model trains on garbage token sequences.
```

**Our solution:** `apply_to_text()` splits at **line boundaries** when there
are 3+ lines. This guarantees we never split mid-identifier. For the
token-level API (`apply()`), this isn't an issue because we split between
already-tokenized integer IDs — every split point is by definition a token
boundary.

### Pitfall 2: Tiny prefix or suffix

```
Prefix:  "f"
Middle:  "unction calculateTotal(price: number): number {\n  return price * 1.1;\n}"
Suffix:  ""

This is basically standard left-to-right generation — the "prefix" provides
almost no context, and there's no suffix at all.
```

**Our solution:** The `MIN_MIDDLE_FRAC = 0.10` and `MAX_MIDDLE_FRAC = 0.90`
constraints ensure the split points fall in the 10%-90% range. This
guarantees at least 10% of the content appears in both the prefix and suffix
sides.

### Pitfall 3: FIM on non-code data

FIM is designed for structured code. Applying it to natural language (docs,
comments, READMEs) or data files (JSON configs, CSV) is wasteful:

- **Natural language** doesn't have the structural constraints that make FIM
  useful. There's no "correct" way to fill a gap in a paragraph — many
  continuations work.
- **Data files** are often repetitive. The model learns to infill JSON commas
  and CSV columns instead of code logic.

**Recommendation:** Apply FIM only to actual source code files during data
prep. If your pipeline includes mixed data, filter FIM application by file
extension or content type.

### Pitfall 4: Forgetting to add FIM tokens to the tokenizer

If you train a tokenizer without reserving FIM special tokens and then try
to add them later, the token IDs may conflict with existing vocabulary. The
`setup_fim_tokenizer()` function handles this gracefully — it checks if the
tokens exist and adds them if needed — but it's better to include them from
the start.

### Pitfall 5: Sequence length budget

FIM adds 3 extra tokens (the special markers) to every FIM sample. If you
don't account for this, your effective content per sample shrinks:

```
Without FIM:  512 content tokens
With FIM:     509 content tokens + 3 special tokens = 512 total

That's a 0.6% reduction — negligible.
```

Our implementation handles this with `truncate_or_pad=True`: it reserves 3
token slots from the content before splitting, so the output length exactly
matches the input length. No content is silently lost.

### Pitfall 6: Inconsistent FIM format between training and inference

If you train with PSM format but your inference server sends SPM format (or
vice versa), the model will produce garbage. Make sure your inference pipeline
uses the same FIM format the model was trained on.

Since we train with **both** PSM and SPM (50/50 mix), our model handles
either format at inference time. Most IDE extensions use PSM (it's more
intuitive), and that works perfectly.

---

## 13. Key Takeaways

```
┌────────────────────────────────────────────────────────────────┐
│  FIM in one sentence:                                          │
│                                                                │
│  Rearrange training sequences so the model learns to fill      │
│  gaps using both prefix AND suffix context, with zero          │
│  architecture changes and zero performance cost.               │
└────────────────────────────────────────────────────────────────┘
```

**The essentials:**

- FIM is a **data transformation**, not an architecture change
- Use **50% FIM rate** (our default) — backed by research as optimal
- Mix **PSM and SPM at 50/50** — gains ~5 points on infilling benchmarks
- Split at **line boundaries** to avoid breaking tokens
- **No performance cost** on left-to-right benchmarks at 50% FIM rate
- FIM is what makes IDE autocomplete possible — without it, a code model
  can only append, never insert

**Our implementation:**

- `src/cola_coder/data/fim.py` — `FIMTransform` class with token-level
  and text-level APIs, PSM/SPM mixing, line-boundary splits
- `src/cola_coder/data/collator.py` — `FIMCollator` for dynamic FIM
  during training batching
- Special tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`
  — verified and cached by `setup_fim_tokenizer()`

**Further reading:**

- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255) — the original FIM paper (Bavarian et al., 2022)
- [StarCoder paper](https://arxiv.org/abs/2305.06161) — FIM at scale with PSM/SPM mixing
- [Code Llama paper](https://arxiv.org/abs/2308.12950) — FIM with long-context infilling
