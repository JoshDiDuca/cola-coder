# The Training Pipeline

A practical guide to training our code generation model from scratch.

If you have ever tuned hyperparameters on an API rate limiter or cache eviction policy,
you already have the intuition. Training a model is the same loop: measure error,
adjust, repeat. The math is just gradient descent instead of manual knob-turning.

---

## Table of Contents

1. [What Training Actually Does](#1-what-training-actually-does)
2. [The Training Loop Step by Step](#2-the-training-loop-step-by-step)
3. [The Loss Function: Cross-Entropy](#3-the-loss-function-cross-entropy)
4. [Backpropagation](#4-backpropagation)
5. [The Optimizer: AdamW](#5-the-optimizer-adamw)
6. [Learning Rate Schedule](#6-learning-rate-schedule)
7. [Mixed Precision Training](#7-mixed-precision-training)
8. [Gradient Accumulation](#8-gradient-accumulation)
9. [Gradient Checkpointing](#9-gradient-checkpointing)
10. [Gradient Clipping](#10-gradient-clipping)
11. [Checkpointing: Saving Progress](#11-checkpointing-saving-progress)
12. [Monitoring Training](#12-monitoring-training)
13. [When to Stop Training](#13-when-to-stop-training)
14. [Common Problems and Fixes](#14-common-problems-and-fixes)
15. [Running Training with Our Project](#15-running-training-with-our-project)

---

## 1. What Training Actually Does

A transformer model is a giant function with millions of numbers called **weights**
(or parameters). Before training, these weights are random. The model outputs garbage.

Training adjusts these weights so the model gets better at one task: **predicting the
next token**. That is the entire objective. There is no separate "understand code" step
or "learn syntax" step. Next-token prediction, applied to enough code, produces all of
those capabilities as side effects.

```
Before training:
  Input:  "function add(a, b) { return a +"
  Output: "zz %%% AAAA q q q"   (random garbage)

After training:
  Input:  "function add(a, b) { return a +"
  Output: " b"                   (correct with high probability)
```

The model has around 50 million to 1 billion weights depending on the config you
choose. Training touches every single one of them, thousands of times, nudging each
one in the direction that makes the model's predictions slightly less wrong.

---

## 2. The Training Loop Step by Step

Here is what happens on every training step. This repeats tens of thousands of times.

```
FOR each training step:
  1. LOAD a batch of token sequences from the dataset
  2. FORWARD PASS: feed tokens into the model, get predictions
  3. COMPUTE LOSS: measure how wrong the predictions were
  4. BACKWARD PASS: compute gradients (which direction to adjust each weight)
  5. CLIP gradients (prevent explosions)
  6. OPTIMIZER STEP: actually update the weights
  7. SCHEDULER STEP: adjust the learning rate
  8. LOG metrics (loss, perplexity, throughput)
  9. CHECKPOINT periodically (save model to disk)
```

In our codebase, this lives in `src/codeformer/training/trainer.py`. Here is the
core loop, simplified:

```python
for step in range(max_steps):
    # 1. Zero out old gradients
    optimizer.zero_grad(set_to_none=True)

    # 2. Get a batch of training data
    batch = next(data_iter)
    input_ids = batch["input_ids"].to("cuda")

    # 3. Forward pass + loss (in mixed precision)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model.compute_loss(input_ids)

    # 4. Backward pass (compute gradients)
    loss.backward()

    # 5. Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 6. Update weights
    optimizer.step()

    # 7. Adjust learning rate
    scheduler.step()
```

For a TS dev: this is the equivalent of a server's main event loop, except instead
of handling HTTP requests, each iteration processes a batch of training data and
nudges the model to be slightly less wrong.

---

## 3. The Loss Function: Cross-Entropy

### "How surprised was the model?"

The loss function measures how wrong the model's predictions were. We use
**cross-entropy loss**, which has a clean intuition: it measures the model's
**surprise** when it sees the actual next token.

```
Model's prediction (probability distribution over 32,768 vocabulary tokens):
  " b"     -> 0.02   (2% chance)
  " c"     -> 0.01
  "return"  -> 0.003
  ...32,765 other tokens with tiny probabilities...

Actual next token: " b"

Cross-entropy = -log(0.02) = 3.91    (high loss -- the model was very surprised)
```

After some training:

```
Model's prediction:
  " b"     -> 0.85   (85% chance)
  " c"     -> 0.03
  "return"  -> 0.001
  ...

Actual next token: " b"

Cross-entropy = -log(0.85) = 0.16    (low loss -- the model expected this)
```

The math: `loss = -log(probability the model assigned to the correct token)`.

- If the model was confident and right: loss is near 0 (good).
- If the model was uncertain or wrong: loss is high (bad).
- Random guessing across 32,768 tokens: loss = -log(1/32768) = 10.4.

Some reference points for the loss values you will see during training:

```
loss = 10.4    Random guessing (untrained model starts here)
loss = 8.0     Model learned that some tokens are more common than others
loss = 5.0     Model learned basic syntax patterns
loss = 3.0     Model generates plausible-looking code
loss = 2.0     Model generates correct, working code fairly often
loss = 1.5     Quite good -- approaching the limits of the model size
```

In our code (`src/codeformer/model/transformer.py`):

```python
def compute_loss(self, token_ids):
    logits = self.forward(token_ids)

    # Shift: position i predicts position i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss
```

The shift is important: `logits[i]` predicts `token_ids[i+1]`. Given everything up
to position i, predict what comes at position i+1. This is the core next-token
prediction objective.

---

## 4. Backpropagation

### Error flows backward, telling each weight how to change

After the forward pass gives us a loss value (say 3.91), we need to figure out how
each of the model's millions of weights contributed to that error. That is
**backpropagation** -- short for "backward propagation of errors."

The intuition:

```
Forward pass (left to right):
  tokens -> embedding -> block1 -> block2 -> ... -> blockN -> output -> loss = 3.91

Backward pass (right to left):
  loss = 3.91 -> output -> blockN -> ... -> block2 -> block1 -> embedding
                    |          |                |         |
                 gradient   gradient         gradient  gradient
```

Each weight gets a **gradient**: a number that says "if you increase this weight
slightly, the loss would change by this much." The gradient tells us both the
direction (should this weight go up or down?) and the magnitude (by how much?).

For a TS dev: think of it like a call stack trace, but instead of tracking function
calls, it tracks how each computation contributed to the final error. The chain rule
from calculus makes this efficient -- you compute the contribution of each layer from
the output back to the input, reusing intermediate results.

In PyTorch, one line does it all:

```python
loss.backward()  # Computes gradients for every parameter in the model
```

After this call, every parameter `p` in the model has a `p.grad` tensor that says
how to adjust it. The optimizer then uses these gradients to actually move the weights.

---

## 5. The Optimizer: AdamW

### Smarter than basic gradient descent

The simplest approach would be: `weight -= learning_rate * gradient`. This is vanilla
gradient descent. It works, but it is slow and fragile.

**AdamW** is the standard optimizer for transformer training. It is smarter in three
ways:

1. **Momentum (first moment):** Keeps a running average of recent gradients. This
   smooths out noise -- if the gradient has been pointing "left" for the last 10
   steps, keep going left even if one step says "right." Like a ball rolling downhill
   that does not reverse direction from a small bump.

2. **Adaptive step size (second moment):** Keeps a running average of squared
   gradients per-weight. Weights with consistently large gradients get smaller steps
   (they are already moving fast). Weights with small, noisy gradients get larger
   steps (they need a push). Each weight effectively gets its own learning rate.

3. **Weight decay:** Gradually shrinks weights toward zero. This is regularization --
   it prevents any single weight from becoming too large, which reduces overfitting.
   The "W" in AdamW means it applies weight decay correctly (decoupled from the
   gradient, unlike the original Adam).

```python
# What AdamW roughly does per weight (simplified pseudocode):
m = 0.9 * m + 0.1 * gradient            # Momentum: smoothed direction
v = 0.95 * v + 0.05 * gradient**2       # Adaptive: smoothed magnitude
adjusted_grad = m / (sqrt(v) + 1e-8)    # Scale by inverse magnitude
weight -= lr * (adjusted_grad + weight_decay * weight)  # Step + decay
```

In our code (`src/codeformer/training/optimizer.py`):

```python
def create_optimizer(model, learning_rate=3e-4, weight_decay=0.1):
    # Key detail: don't apply weight decay to biases and norm weights
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.dim() <= 1 or "norm" in name:
            no_decay_params.append(param)   # Biases, norms: no decay
        else:
            decay_params.append(param)       # Weight matrices: decay

    return AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=(0.9, 0.95),  # Momentum and second moment coefficients
    )
```

The `betas=(0.9, 0.95)` control how "long the memory" is:
- `0.9` for momentum: roughly a 10-step moving average.
- `0.95` for second moment: roughly a 20-step moving average.

These are the same values used by LLaMA, GPT, and most modern models. Do not change
them unless you have a specific reason.

**Important detail:** weight decay is NOT applied to bias parameters or normalization
weights. These are 1D tensors that do not benefit from regularization. Applying decay
to them hurts training quality. This is a pattern you will see in every serious
transformer training codebase.

---

## 6. Learning Rate Schedule

The learning rate controls how big of a step the optimizer takes. Too big and the
model overshoots and diverges (loss goes to infinity). Too small and training takes
forever.

The standard approach: **linear warmup, then cosine decay**.

```
Learning Rate
     ^
     |
peak |         .--------.
     |        /          \
     |       /             \
     |      /                \_________  min_lr
     |     /
     |    /
     |   /
     |  /
     | /
  0  +--------------------------------------> training step
     0   warmup              max_steps
         steps

     |-----|--------------------------------|
     Phase 1:      Phase 2:
     Linear        Cosine Decay
     Warmup
```

**Phase 1 -- Warmup:** Start with a tiny learning rate and linearly increase it to
the peak over `warmup_steps`. Why? At the start, the model's weights are random and
the gradients are large and noisy. A big learning rate here would cause instability.
The warmup lets the model "get its bearings" before taking big steps.

**Phase 2 -- Cosine decay:** Gradually decrease the learning rate following a cosine
curve from `peak_lr` to `min_lr`. Why? As training progresses, the model is closer
to a good solution. Smaller steps prevent overshooting. The cosine shape is smoother
than a linear decay and works slightly better in practice.

Our configs use these values:

| Config | Peak LR  | Min LR   | Warmup Steps | Max Steps |
|--------|----------|----------|--------------|-----------|
| tiny   | 3.0e-4   | 3.0e-5   | 500          | 20,000    |
| small  | 6.0e-4   | 6.0e-5   | 1,000        | 100,000   |
| medium | 3.0e-4   | 3.0e-5   | 2,000        | 200,000   |
| large  | 3.0e-4   | 3.0e-5   | 4,000        | 500,000   |

Rule of thumb: smaller models can handle higher learning rates. The small model uses
`6e-4` while the large model uses `3e-4`.

---

## 7. Mixed Precision Training

### Use half-precision numbers for speed, same quality

By default, PyTorch uses 32-bit floats (`float32`) for all computations. That is more
precision than we need. Mixed precision uses 16-bit floats for most operations, which:

- Uses **half the VRAM** for activations.
- Runs **roughly 2x faster** on modern GPUs (they have dedicated half-precision cores).
- Produces **the same model quality** when done correctly.

There are two 16-bit formats:

```
float32:  1 sign bit | 8 exponent bits | 23 mantissa bits  (standard)
float16:  1 sign bit | 5 exponent bits | 10 mantissa bits  (older GPUs)
bfloat16: 1 sign bit | 8 exponent bits |  7 mantissa bits  (newer GPUs)
```

**bfloat16** (bf16) is preferred because it has the same exponent range as float32.
This means it can represent very large and very small numbers without overflow or
underflow. Available on RTX 4080, 4090, A100, H100, and newer.

**float16** (fp16) is for older GPUs like the RTX 3080. It has a smaller exponent
range, so you need a **GradScaler** to prevent gradient underflow (numbers rounding
to zero because they are too small to represent in fp16).

In our trainer:

```python
# bf16 path (RTX 4080+): simpler, no scaler needed
with autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model.compute_loss(input_ids)
loss.backward()
optimizer.step()

# fp16 path (RTX 3080): needs GradScaler to prevent underflow
scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.float16):
    loss = model.compute_loss(input_ids)
scaler.scale(loss).backward()        # Scale loss up before backward
scaler.unscale_(optimizer)            # Unscale gradients before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                # Unscales gradients, then steps
scaler.update()                       # Adjust scale factor for next step
```

**Which to use:** Check your GPU. Set `precision: "bf16"` for RTX 40-series, A100,
H100. Set `precision: "fp16"` for RTX 30-series or older.

```python
# Quick check: does my GPU support bf16?
import torch
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}")
print(f"bf16 support: {props.major >= 8}")  # Ampere (sm_80+) and newer
```

---

## 8. Gradient Accumulation

### Process small batches, accumulate gradients, update less frequently

The effective batch size matters for training stability. Larger batches give smoother
gradient estimates (less noise). But a large batch might not fit in GPU memory.

**Gradient accumulation** solves this: process several small "micro-batches,"
accumulate the gradients, and only update the weights after all micro-batches.

```
Without gradient accumulation (batch_size=32):
  [batch of 32] -> forward -> backward -> UPDATE weights

With gradient accumulation (batch_size=8, accumulation=4):
  [micro-batch of 8] -> forward -> backward -> accumulate gradients
  [micro-batch of 8] -> forward -> backward -> accumulate gradients
  [micro-batch of 8] -> forward -> backward -> accumulate gradients
  [micro-batch of 8] -> forward -> backward -> UPDATE weights

  Effective batch size = 8 x 4 = 32 (same result, 4x less VRAM for activations)
```

In our trainer:

```python
optimizer.zero_grad(set_to_none=True)  # Clear old gradients

for micro_step in range(gradient_accumulation):
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model.compute_loss(input_ids)
        scaled_loss = loss / gradient_accumulation  # Average across micro-batches

    scaled_loss.backward()  # Gradients ACCUMULATE (they add up in .grad)

# Only update weights after all micro-batches
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The key insight: `loss.backward()` **adds to** existing gradients by default. By
calling it multiple times before `optimizer.step()`, the gradients accumulate.
Dividing the loss by `gradient_accumulation` ensures the total is an average, not
a sum.

Our configs:

| Config | Micro Batch | Accumulation | Effective Batch |
|--------|-------------|--------------|-----------------|
| tiny   | 32          | 1            | 32              |
| small  | 8           | 4            | 32              |
| medium | 2           | 16           | 32              |
| large  | 8           | 8            | 64              |

Notice how the medium model uses a micro batch of just 2 (to fit in 16GB VRAM)
but accumulates 16 times to reach an effective batch of 32.

---

## 9. Gradient Checkpointing

### Forget intermediate values during forward pass, recompute during backward

During the forward pass, PyTorch stores every intermediate result (called
**activations**) because they are needed later in the backward pass to compute
gradients. For a deep model, these activations consume a lot of VRAM.

**Gradient checkpointing** says: throw away most activations during the forward pass.
When the backward pass needs them, **recompute** them on the fly.

```
Normal (no checkpointing):
  Forward:  save A1, A2, A3, A4, A5, ... A24  (all 24 layers' activations)
  Backward: use  A24, A23, ... A2, A1          (read from memory)
  VRAM:     HIGH (stores all activations)
  Speed:    FAST (no extra work)

With gradient checkpointing:
  Forward:  save A1, ___, A3, ___, A5, ...     (save every other layer)
  Backward: recompute A24 from A23,
            recompute A22 from A21, ...         (recompute what was discarded)
  VRAM:     ~HALF
  Speed:    ~30% SLOWER (extra forward passes during backward)
```

This is a pure tradeoff: less memory, more compute. The model and the final result
are identical. Only the training speed changes.

When to use it:
- **Required** for the medium model (350M) on 16GB GPUs.
- **Required** for the large model (1B+) even on 24GB GPUs.
- **Not needed** for tiny or small models on 16GB GPUs.

In the config:

```yaml
# configs/medium.yaml
training:
  gradient_checkpointing: true  # REQUIRED at this size on 16GB
```

In the code, enabling it is one call:

```python
if config.training.gradient_checkpointing:
    model.enable_gradient_checkpointing()
    print("Gradient checkpointing enabled (saves VRAM, ~30% slower)")
```

---

## 10. Gradient Clipping

### Prevent exploding gradients

Sometimes, a single bad batch causes extremely large gradients. If the optimizer
applies these unchecked, the weights change drastically, the model forgets everything
it learned, and training diverges (loss shoots to infinity or becomes NaN).

**Gradient clipping** caps the total gradient magnitude. If the gradients are "too
big," they get scaled down proportionally so the total norm equals the clip value.

```python
# Clip gradients so their total norm does not exceed 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This does NOT change the direction of the gradients -- only their magnitude. If the
total gradient norm is 5.0 and the clip threshold is 1.0, every gradient gets
multiplied by 1.0/5.0 = 0.2. The direction of the update is preserved.

Our configs all use `grad_clip: 1.0`. The reasoning config uses `grad_clip: 0.5`
(tighter clipping because RL gradients tend to be noisier).

Think of it as a safety valve: during normal training it rarely triggers, but it
prevents catastrophic weight updates when an unusual batch comes through.

---

## 11. Checkpointing: Saving Progress

Training takes days or weeks. You need to save progress regularly so you can:

1. **Resume after crashes** (power loss, GPU error, OOM kill).
2. **Load the model for inference** (actually use it to generate code).
3. **Fine-tune later** (e.g., add reasoning capabilities on top of a base model).

Each checkpoint saves three files:

```
checkpoints/small/step_00005000/
  model.safetensors     # Model weights (safe binary format)
  training_state.pt     # Optimizer state, scheduler state, RNG state
  metadata.json         # Step number, loss value, config (human-readable)
```

We use **safetensors** instead of PyTorch's default pickle format for model weights.
Pickle can execute arbitrary code when loading (like `eval()`). Safetensors is a
simple binary format that only stores tensors (like `JSON.parse()`).

The "latest" pointer file lets you find the most recent checkpoint quickly:

```python
# Resume training from where you left off
trainer = Trainer(config, resume_from="./checkpoints/small/latest")
```

Old checkpoints are automatically cleaned up to save disk space. The
`max_checkpoints` setting in the config controls how many to keep.

**Why save optimizer state?** The optimizer's momentum buffers and adaptive step sizes
take thousands of steps to build up. If you resume training without them, the first
few thousand steps perform poorly -- the optimizer is "cold" again. Saving and
restoring these lets training continue smoothly.

---

## 12. Monitoring Training

### What to watch and what it means

Every 100 steps, the trainer logs a line like this:

```
step    100 | loss 8.2451 | ppl 3808.12 | lr 6.00e-05 | tok/s 45,230
step    200 | loss 7.1023 | ppl 1214.73 | lr 1.20e-04 | tok/s 44,891
step    500 | loss 5.8934 | ppl  362.40 | lr 3.00e-04 | tok/s 45,102
step   1000 | loss 4.2156 | ppl   67.72 | lr 3.00e-04 | tok/s 44,970
step   5000 | loss 3.1204 | ppl   22.65 | lr 2.85e-04 | tok/s 45,055
step  20000 | loss 2.4891 | ppl   12.06 | lr 1.20e-04 | tok/s 44,988
```

**Loss:** The cross-entropy loss. Should decrease over time. A trained small model
should reach roughly 2.0-2.5 loss.

**Perplexity (ppl):** `exp(loss)`. Intuitively, the number of tokens the model is
"choosing between" at each position. A perplexity of 12 means the model is roughly
as uncertain as picking uniformly from 12 options. Good code models reach 8-15.

| Perplexity | What It Means                              |
|------------|--------------------------------------------|
| 32768      | Random guessing (untrained, = vocab_size)  |
| 100-500    | Learning basic patterns                    |
| 20-50      | Decent -- generates plausible code         |
| 8-15       | Good -- generates working code often       |
| 3-8        | Very good (needs a large model to get here)|

**Learning rate (lr):** Should follow the warmup-then-decay schedule. If it is stuck
at 0 or a constant, something is wrong with the scheduler.

**Throughput (tok/s):** Tokens processed per second. Should be roughly constant. A
sudden drop means something is wrong (CPU bottleneck, thermal throttling, etc.).

### What a healthy training run looks like

```
Loss curve:

  loss
   ^
 9 |x
   |x
 7 | x
   |  x
 5 |   xx
   |     xx
 3 |       xxxxx
   |            xxxxxxx
 2 |                   xxxxxxxxxxxxxxxxxxx
   +-----------------------------------------> step
   0        5k       10k       15k       20k
```

The loss drops quickly at first (learning basic syntax and common patterns), then
gradually slows down (learning more subtle patterns). This curve shape is normal
and expected.

### Using Weights and Biases (wandb)

For a web dashboard with real-time graphs of all metrics:

```bash
# Install (one time)
pip install wandb

# Login (one time -- creates account or links existing)
wandb login

# Train with wandb logging enabled
python scripts/train.py --config configs/small.yaml --wandb
```

This gives you a dashboard at wandb.ai with charts of loss, perplexity, learning
rate, and throughput. You can monitor training from your phone. Free for personal use.

---

## 13. When to Stop Training

### Signs that training is done

1. **Loss plateaus:** The loss has not decreased meaningfully in the last 20% of
   training steps. "Meaningfully" means less than 0.05 change over several thousand
   steps.

2. **Perplexity stops improving:** Same as loss, but easier to interpret. If
   perplexity has been around 12 for the last 5,000 steps, you are probably done.

3. **Evaluation metrics plateau:** If you run HumanEval periodically, the pass@1
   score stops improving.

4. **You hit your step budget:** The configs are set for a reasonable number of
   steps for each model size. If you have finished `max_steps`, check the metrics.
   If loss is still decreasing, you can extend training -- but diminishing returns
   kick in fast.

### Signs that something is wrong (not just "done")

- **Loss increases:** The model is getting worse. Likely learning rate is too high.
- **Loss is stuck from the start:** Has not decreased for thousands of steps. Check
  that data is actually being loaded (`num_workers` > 0, data path is correct).
- **Loss is NaN:** Something exploded numerically. See the troubleshooting section.

---

## 14. Common Problems and Fixes

### Loss spikes

```
Normal:  2.45, 2.44, 2.43, 2.42
Spike:   2.45, 2.44, 5.87, 2.50, 2.48, 2.46
                      ^^^^
```

**Cause:** A bad batch of data (very unusual code, corrupted file, binary data mixed
in) caused large gradients. Gradient clipping limits the damage.

**Fix:** Usually nothing -- the model recovers within a few steps. If spikes are
frequent (every few hundred steps), reduce `grad_clip` from 1.0 to 0.5. If the model
does not recover after a spike, reduce the learning rate.

### Loss becomes NaN

**Cause:** Numerical overflow or underflow. A computation produced infinity or
division by zero.

**Fixes (try in order):**
1. If using fp16, make sure `GradScaler` is enabled. Better yet, switch to bf16 if
   your GPU supports it.
2. Reduce the learning rate by 2-3x.
3. Check the data for corrupted files (binary data, extremely long lines, etc.).
4. Reduce `grad_clip` to 0.5.

### Slow convergence (loss decreases very slowly)

**Fixes:**
1. Increase the learning rate (try 2x what you have).
2. Make sure warmup is not too long (should be about 1-5% of total steps).
3. Verify the data pipeline is actually shuffling. If the model sees the same files
   over and over, it overfits to those and stops generalizing.
4. Check effective batch size -- too small (< 16) leads to noisy gradients.

### Out of memory (OOM)

**Fixes (in order of least disruptive):**
1. Reduce `batch_size` by half. Increase `gradient_accumulation` to compensate.
2. Enable `gradient_checkpointing: true`.
3. Reduce `max_seq_len` (e.g., from 2048 to 1024).
4. Use a smaller model config.

### Training is very slow

**Fixes:**
1. Make sure you are training on GPU, not CPU. Run `nvidia-smi` -- GPU utilization
   should be above 90%.
2. Increase `num_workers` in the data config (try 4 or 8).
3. Make sure mixed precision is enabled (`precision: "bf16"` or `"fp16"`).
4. Try `torch.compile(model)` for a 10-20% speedup on PyTorch 2.0+.

---

## 15. Running Training with Our Project

### First-time setup

```bash
# Create virtual environment and install dependencies
make setup

# Download and preprocess training data
make prepare

# Train the BPE tokenizer on the downloaded data
make tokenizer
```

### Start training

```bash
# Recommended first run: tiny model (verifies everything works)
make train-tiny

# Or run directly with more control
python scripts/train.py --config configs/tiny.yaml

# Enable wandb logging
python scripts/train.py --config configs/tiny.yaml --wandb

# Resume from a checkpoint after a crash or interruption
python scripts/train.py --config configs/tiny.yaml \
    --resume checkpoints/tiny/latest
```

### Training progression (recommended order)

```
Step 1: Train tiny (50M params, ~2 days on RTX 4080)
  make train-tiny
  - Verifies the full pipeline works end to end.
  - Tests data loading, loss computation, checkpointing, everything.
  - Target: loss ~2.5-3.0 by the end.

Step 2: Train small (125M params, ~1 week on RTX 4080)
  make train-small
  - The primary target for local development.
  - Produces usable (if basic) code completions.
  - Target: loss ~2.0-2.5 by the end.

Step 3: Train medium (350M params, ~2 weeks on RTX 4080)
  make train-medium
  - Best results achievable on a single consumer GPU.
  - Requires gradient checkpointing on 16GB cards.
  - Target: loss ~1.8-2.2 by the end.

Step 4 (cloud only): Train large (1B+ params)
  make train-large
  - Requires A100/H100 or multi-GPU setup.
  - See docs/05_hardware_guide.md for cloud options.
```

### Verifying training is working

After starting training, check these within the first few minutes:

```bash
# 1. Is the GPU being used?
nvidia-smi
# GPU utilization should be >90%, VRAM usage should be substantial

# 2. Check the training log output
# You should see loss around 8-10 at step 0 (random model)
# Loss should drop to ~5-6 within the first 500 steps
# If loss is not decreasing at all, something is wrong
```

### Config reference

All training hyperparameters live in `configs/*.yaml`:

```yaml
training:
  batch_size: 8               # Micro-batch size (per accumulation step)
  gradient_accumulation: 4    # Micro-batches per weight update
  learning_rate: 6.0e-4       # Peak learning rate
  min_lr: 6.0e-5              # Floor of the cosine schedule
  warmup_steps: 1000          # Linear warmup duration
  max_steps: 100000           # Total training steps
  weight_decay: 0.1           # L2 regularization strength
  grad_clip: 1.0              # Gradient norm clip threshold
  precision: "bf16"           # "bf16" for RTX 4080+, "fp16" for RTX 3080
  gradient_checkpointing: false  # Enable for medium/large models
```

**Key relationships between these numbers:**

```
effective_batch_size = batch_size * gradient_accumulation
tokens_per_step = effective_batch_size * max_seq_len
total_tokens = tokens_per_step * max_steps
training_time ~ total_tokens / tokens_per_second
```

For the small model with defaults:
- Effective batch: 8 * 4 = 32 sequences per step
- Tokens per step: 32 * 2048 = 65,536 tokens
- Total tokens: 65,536 * 100,000 = ~6.5 billion tokens of code
