# Checkpoint Safety: Protecting Your Training Investment

Your 68-hour training run is not just time. It is electricity, GPU wear, opportunity cost,
and the accumulated knowledge of billions of gradient updates. A corrupted checkpoint turns
all of that into nothing. This document explains every layer of defense we have built into
the checkpoint system, why each one exists, and what happens when they break.

If you have ever lost an unsaved file to a power outage, multiply that feeling by 68 hours
and a few hundred dollars in compute. That is what a broken checkpoint feels like.

---

## Table of Contents

1. [Why Checkpoints Matter](#1-why-checkpoints-matter)
2. [Pickle vs Safetensors](#2-pickle-vs-safetensors)
3. [What's in a Checkpoint Directory](#3-whats-in-a-checkpoint-directory)
4. [Weight Tying — The Most Critical Invariant](#4-weight-tying--the-most-critical-invariant)
5. [torch.compile Prefix Handling](#5-torchcompile-prefix-handling)
6. [Atomic Saves](#6-atomic-saves)
7. [Auto-Resume and the Latest Pointer](#7-auto-resume-and-the-latest-pointer)
8. [Checkpoint Cleanup](#8-checkpoint-cleanup)
9. [The Checkpoint Test Suite](#9-the-checkpoint-test-suite)
10. [Recovery Scenarios](#10-recovery-scenarios)

---

## 1. Why Checkpoints Matter

A training run is a long, stateful computation. At step 50,000, the model's weights encode
everything it has learned from every gradient update since step 0. There is no shortcut to
reconstruct that state — you cannot "fast-forward" to step 50,000. You must replay all
50,000 steps, processing the same data, in the same order.

Losing a checkpoint means restarting from your last saved point. If you save every 1,000
steps and lose the checkpoint at step 50,000, you go back to step 49,000. That is annoying
but survivable. If your *only* checkpoint is corrupt, you restart from zero.

But checkpoints are not just the model weights. A resumable checkpoint includes:

- **Model weights** — the learned parameters (the "brain")
- **Optimizer state** — AdamW's momentum buffers and variance estimates (the "memory" of
  how each weight has been trending). Without these, the optimizer has to re-learn the
  gradient landscape. Resuming without optimizer state often causes a loss spike that takes
  thousands of steps to recover from.
- **Scheduler state** — where you are in the learning rate schedule (warmup done? how far
  into cosine decay?)
- **RNG state** — so data ordering is reproducible across resumes
- **Step number** — so you know where to resume
- **Current loss** — for monitoring continuity

Think of it like saving a game. The model weights are your character's stats. The optimizer
state is the game's internal AI memory. The scheduler is the difficulty curve position. If
you load a save but lose the AI memory, enemies suddenly behave erratically for a while.

---

## 2. Pickle vs Safetensors

### The Problem with Pickle

PyTorch's default serialization uses Python's `pickle` format. Pickle can serialize
*arbitrary Python objects*, including executable code. When you `torch.load()` a pickle
file, Python's pickle deserializer runs — and it can execute arbitrary code embedded in
the file.

**TS analogy:** Pickle is `eval()`. Loading a pickle file is like running
`eval(fileContents)`. You are trusting that whoever created the file did not embed
malicious code. Safetensors is `JSON.parse()` — it can only produce data, never execute
code.

This is not theoretical. Researchers have demonstrated malicious pickle files that open
reverse shells, exfiltrate data, or silently modify model weights on load. Every model
checkpoint you download from the internet and load with `torch.load()` has full code
execution access to your machine.

### Why Safetensors

Safetensors is a simple binary format designed by Hugging Face specifically for storing
tensors. It has a strict specification:

- **Header:** JSON metadata (tensor names, shapes, dtypes, byte offsets)
- **Body:** Raw tensor bytes, concatenated, no gaps

That is it. No code, no objects, no custom deserializers. The parser reads the header,
computes byte offsets, and maps raw memory into tensors.

| Property | Pickle | Safetensors |
|----------|--------|-------------|
| Code execution on load | Yes (arbitrary) | No (impossible) |
| Load speed (1GB file) | ~2-5 seconds | ~50-200ms |
| Memory mapping | No (full load) | Yes (zero-copy) |
| Type safety | No (any Python object) | Yes (tensors only) |
| Corruption detection | None | Header validation |
| Shared tensors | Implicit (pickle magic) | Explicit refusal |

The speed difference comes from memory mapping. When you load a safetensors file, the OS
maps the file into virtual memory. Tensor data is read directly from disk pages on demand —
no copying, no deserialization. This is the same mechanism that makes `mmap` fast in C or
memory-mapped files fast in Node.js.

### What We Use Where

```
model.safetensors  → safetensors (safe, fast, memory-mapped)
training_state.pt  → pickle (PyTorch limitation — optimizer state dicts are complex
                     nested Python objects that safetensors cannot represent)
metadata.json      → plain JSON (human-readable, inspectable with any text editor)
```

The `training_state.pt` file is still pickle, but we load it with `weights_only=True`:

```python
torch.load(training_state_path, map_location=device, weights_only=True)
```

This tells PyTorch to reject any non-tensor objects in the pickle stream. It is not as
safe as safetensors (the parser still runs), but it blocks the most common attack vectors.
The optimizer state contains only numbers — momentum buffers, variance estimates, scalar
hyperparameters — so there is no reason for it to contain code.

---

## 3. What's in a Checkpoint Directory

Every checkpoint is a directory named `step_NNNNNNNN` (zero-padded to 8 digits):

```
checkpoints/tiny/
  step_00010000/
    model.safetensors    # Model weights (safetensors format)
    training_state.pt    # Optimizer + scheduler + RNG state (pickle)
    metadata.json        # Step number, loss, config (human-readable JSON)
  step_00020000/
    ...
  latest                 # Text file containing the path to the most recent checkpoint
  training_manifest.yaml # Full provenance info (optional, written when manifest_info provided)
```

### model.safetensors

Contains every learned parameter in the model *except* `output.weight` (which is excluded
because of weight tying — more on this in Section 4). Keys are clean parameter names with
no framework prefixes:

```
tok_emb.weight              # Token embedding matrix [vocab_size, dim]
blocks.0.attn_norm.weight   # RMSNorm for attention in block 0
blocks.0.attention.wq.weight  # Query projection
blocks.0.attention.wk.weight  # Key projection
blocks.0.attention.wv.weight  # Value projection
blocks.0.attention.wo.weight  # Output projection
blocks.0.ffn_norm.weight    # RMSNorm for FFN in block 0
blocks.0.ffn.w1.weight      # SwiGLU gate projection
blocks.0.ffn.w2.weight      # SwiGLU down projection
blocks.0.ffn.w3.weight      # SwiGLU up projection
... (repeat for each block)
final_norm.weight           # Final RMSNorm before output
```

Note: `output.weight` is **not** in this list. This is intentional. See Section 4.

### training_state.pt

Contains everything needed to resume training at exactly the right point:

```python
{
    "optimizer": optimizer.state_dict(),   # AdamW m and v buffers for every parameter
    "scheduler": scheduler.state_dict(),   # LR schedule position
    "step": step,                          # Current training step
    "rng_state": torch.random.get_rng_state(),  # For reproducible data ordering
}
```

The optimizer state is the largest part. For AdamW, every parameter gets two buffers:
- `exp_avg` (m) — first moment estimate (moving average of gradients)
- `exp_avg_sq` (v) — second moment estimate (moving average of squared gradients)

This means the optimizer state is roughly **2x the model size**. For a 50M parameter model,
the optimizer state is ~100M parameters worth of floats. This is why training uses ~3x the
VRAM of inference.

### metadata.json

Human-readable JSON for quick inspection without loading any tensors:

```json
{
  "step": 10000,
  "loss": 3.245,
  "config": {
    "model": { "dim": 512, "n_layers": 8 },
    "training": { "batch_size": 32, "learning_rate": 3e-4 }
  }
}
```

You can inspect this with any text editor or `cat` to check training progress without
loading the model. The `config` field stores the full training configuration so you can
verify that the checkpoint matches your current config.

### latest

A plain text file containing the absolute path to the most recent checkpoint directory.
On Linux this would be a symlink, but Windows does not reliably support symlinks without
admin privileges, so we use a text file:

```
C:\Users\josh\ai research\cola-coder\checkpoints\tiny\step_00010000
```

This is what `--auto-resume` reads to find where to continue training.

---

## 4. Weight Tying — The Most Critical Invariant

This is the single most important thing to understand about our checkpoint system. Getting
weight tying wrong produces a model that *appears* to load successfully but generates
complete garbage.

### What Weight Tying Is

In `transformer.py`, line 121-122:

```python
self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
self.output.weight = self.tok_emb.weight  # Weight tying
```

After this line, `self.output.weight` and `self.tok_emb.weight` are **the same tensor in
memory**. Not a copy — the same object. One tensor, two names.

**TS analogy:**

```typescript
const embeddingWeights = new Float32Array(vocab_size * dim);
const model = {
  tok_emb: { weight: embeddingWeights },
  output:  { weight: embeddingWeights },  // Same reference, not a copy
};

// Modifying one modifies the other:
model.tok_emb.weight[0] = 42;
console.log(model.output.weight[0]); // 42 — same underlying memory
```

This is exactly `const a = {}; const b = a;` — two variables, one object.

### Why We Do It

The token embedding layer maps token IDs to vectors: `token_id → float[dim]`.
The output layer maps vectors back to token scores: `float[dim] → logits[vocab_size]`.

These are inverse operations. The embedding asks "what does this token look like as a
vector?" and the output asks "which token does this vector look like?" Sharing the same
weight matrix enforces the constraint that these two operations are consistent — a token's
embedding vector is literally what the output layer uses to recognize that token.

Practically, it cuts the embedding parameter count in half. For our tiny model
(`vocab_size=32000, dim=512`), that is 32000 * 512 = **16.4M parameters** saved. For a
50M parameter model, that is a third of all parameters. The savings are enormous.

### The Save Trick

When we save a checkpoint, we must handle the fact that `tok_emb.weight` and
`output.weight` point to the same memory. Safetensors *refuses* to save two keys that
reference the same underlying tensor — it raises `RuntimeError: Some tensors share memory`.
This is actually a safety feature: safetensors is strict about what it stores.

Our solution in `checkpoint.py` (lines 80-88):

```python
raw_state = model.state_dict()
state_dict = {}
for k, v in raw_state.items():
    clean_key = k.removeprefix("_orig_mod.")
    if clean_key == "output.weight":
        continue  # Skip — it's the same tensor as tok_emb.weight
    state_dict[clean_key] = v.contiguous()
save_file(state_dict, str(tmp_dir / "model.safetensors"))
```

We iterate over every key in the state dict and **skip** `output.weight` entirely. It is
not saved because it does not need to be — it is the same data as `tok_emb.weight`, which
*is* saved. We save the data once under the name `tok_emb.weight` and discard the alias.

### The Load Trick

When we load a checkpoint, we call `model.load_state_dict(state_dict, strict=False)`.

The `strict=False` is critical. Without it, PyTorch would raise an error because
`output.weight` is missing from the checkpoint — it expects every key in the model to have
a corresponding key in the file. With `strict=False`, it loads what it can and ignores
missing keys.

But here is the key insight: **we do not need to explicitly re-tie the weights after
loading.** The model's `__init__` method already sets `self.output.weight = self.tok_emb.weight`.
When `load_state_dict` loads `tok_emb.weight`, it writes into that tensor's memory. Since
`output.weight` points to the same memory, it automatically gets the loaded values too.

The weight tying is established by the model constructor. The checkpoint load just fills in
the data. The sharing relationship is never broken because `load_state_dict` writes *into*
existing tensors rather than replacing them (when `strict=False` with matching keys).

### What Breaks If You Get This Wrong

If weight tying is broken after a load, `output.weight` contains its randomly initialized
values from `_init_weights()` instead of the trained embedding values. The model will:

1. Appear to load without errors
2. Report reasonable-looking loss values (the loss function still works)
3. Generate **complete garbage** during inference — because the output layer is using
   random weights to score tokens

This is an insidious bug because it looks like a model quality problem, not a loading bug.
You might think "the model just needs more training" when in reality the output layer is
disconnected from what the model learned.

The test that catches this (`test_checkpoint.py`, line 121):

```python
assert model2.output.weight.data_ptr() == model2.tok_emb.weight.data_ptr(), (
    "Weight tying broken after checkpoint load"
)
```

`data_ptr()` returns the raw memory address of the tensor's data. If these two pointers
are equal, the tensors share memory. If they differ, weight tying is broken.

---

## 5. torch.compile Prefix Handling

### The Problem

When you wrap a model with `torch.compile()`, PyTorch creates a wrapper object. The
original model becomes an attribute called `_orig_mod`. This means every key in the state
dict gets prefixed:

```
Before compile:  tok_emb.weight, blocks.0.attn_norm.weight, ...
After compile:   _orig_mod.tok_emb.weight, _orig_mod.blocks.0.attn_norm.weight, ...
```

If you save a checkpoint from a compiled model and try to load it into an uncompiled model
(or vice versa), every single key will mismatch. PyTorch will raise errors about
"unexpected keys" and "missing keys." Your checkpoint is not actually incompatible — the
weights are identical. The keys just have different names.

**TS analogy:** Imagine you have an object `{ name: "Josh" }` and you wrap it:
`{ _orig_mod: { name: "Josh" } }`. The data is the same, but `obj.name` is now
`obj._orig_mod.name`. The checkpoint system needs to normalize this.

### Our Solution

On save (`checkpoint.py`, line 84):

```python
clean_key = k.removeprefix("_orig_mod.")
```

Every key is stripped of the `_orig_mod.` prefix before saving. Checkpoints *always* store
clean keys, regardless of whether the model was compiled.

On load (`checkpoint.py`, lines 183-184):

```python
if hasattr(model, "_orig_mod"):
    state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
```

If the model being loaded into is compiled (detected by the presence of `_orig_mod`), we
add the prefix back to every key. If it is not compiled, we use the clean keys as-is.

This means checkpoints are **portable** across compilation states:

| Saved from | Loaded into | Works? |
|-----------|-------------|--------|
| Compiled | Compiled | Yes |
| Compiled | Uncompiled | Yes |
| Uncompiled | Compiled | Yes |
| Uncompiled | Uncompiled | Yes |

The test suite verifies all four combinations in `TestTorchCompileCheckpoint`.

### What Breaks Without This

Without prefix stripping, a checkpoint saved from a compiled model contains keys like
`_orig_mod.tok_emb.weight`. Loading into an uncompiled model fails with:

```
RuntimeError: Error(s) in loading state_dict:
    Unexpected key(s) in state_dict: "_orig_mod.tok_emb.weight", ...
    Missing key(s) in state_dict: "tok_emb.weight", ...
```

Your training run saves a checkpoint at step 40,000. You stop training to make a code
change. You restart. But now `torch.compile` is enabled (or disabled) and every checkpoint
is unloadable. 40,000 steps — gone. Unless you know to manually rename the keys, which is
exactly the kind of panic-driven debugging you should not have to do at 3 AM.

---

## 6. Atomic Saves

### The Problem

Saving a checkpoint writes multiple files (model.safetensors, training_state.pt,
metadata.json). Each file write takes time — the model.safetensors file for a 50M model
is ~200MB. If power fails, the process crashes, or you hit Ctrl+C mid-save, you get a
partially written checkpoint:

- `model.safetensors` might be truncated (half the weights written)
- `training_state.pt` might not exist yet (optimizer state gone)
- `metadata.json` might be empty

This partial checkpoint *looks* like a valid checkpoint directory. The auto-resume system
finds it, tries to load it, and crashes. Now you have lost both the in-progress checkpoint
*and* the ability to auto-resume.

### Our Solution: Write-Then-Rename

This is the same pattern used by databases (write-ahead logging) and text editors (save to
temp, then rename):

```python
# checkpoint.py, lines 64-113

# Step 1: Write to a temporary directory
final_dir = Path(output_dir) / f"step_{step:08d}"
tmp_dir = Path(output_dir) / f".tmp_step_{step:08d}"

# Step 2: Clean up any previous failed temp dir
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)

# Step 3: Write all files into the temp directory
save_file(state_dict, str(tmp_dir / "model.safetensors"))
torch.save({...}, tmp_dir / "training_state.pt")
(tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

# Step 4: Atomic rename — this is the critical moment
if final_dir.exists():
    shutil.rmtree(final_dir)
tmp_dir.rename(final_dir)  # On most filesystems, rename is atomic
```

**TS analogy:** This is like writing to a `.tmp` file and then doing `fs.renameSync()`.
The rename operation is atomic on most filesystems — it either completes fully or not at
all. There is no state where the directory is half-renamed.

The invariant: a directory named `step_NNNNNNNN` (without a `.tmp_` prefix) is *always*
a complete, valid checkpoint. If the process crashes during write, only the `.tmp_` directory
is corrupted, and it is cleaned up on the next save attempt.

The test that verifies this (`test_checkpoint.py`, line 515-526):

```python
def test_atomic_save_cleans_tmp_dir(self, tmp_path):
    """No .tmp_* directories left after successful save."""
    ...
    tmp_dirs = [d for d in tmp_path.iterdir() if d.name.startswith(".tmp_")]
    assert len(tmp_dirs) == 0, f"Temp dirs remain: {tmp_dirs}"
```

### What Breaks Without This

Without atomic saves, a crash during save can produce:

```
step_00050000/
  model.safetensors    # 97MB out of 200MB — truncated
  training_state.pt    # Does not exist
  metadata.json        # Does not exist
```

The next training start sees `step_00050000/`, tries to load it, fails because
`model.safetensors` is truncated, and raises a confusing error. If the previous checkpoint
was already deleted by cleanup, you have **no valid checkpoint** and must restart from zero.

---

## 7. Auto-Resume and the Latest Pointer

### How It Works

When you run training with `--auto-resume`, the system calls `detect_latest_checkpoint()`:

```python
# checkpoint.py, lines 266-321

def detect_latest_checkpoint(checkpoints_dir="checkpoints"):
    # 1. Scan for "latest" text files in each size subdirectory
    for size_dir in base.iterdir():
        latest_file = size_dir / "latest"
        if latest_file.is_file():
            info = get_checkpoint_info(str(latest_file))
            # Track the highest step number across all sizes

    # 2. Fallback: if no "latest" files, scan for step_* directories directly
    for size_dir in base.iterdir():
        step_dirs = sorted(size_dir.glob("step_*"), ...)
        # Take the highest step dir
```

The function scans across all model sizes (`tiny/`, `small/`, `medium/`, etc.) and returns
the checkpoint with the highest step number. This means you can switch between model sizes
and auto-resume will still find the right checkpoint.

### The latest Pointer

Every `save_checkpoint` call updates the `latest` file:

```python
# checkpoint.py, lines 128-132
latest_path = Path(output_dir) / "latest"
if latest_path.exists() or latest_path.is_symlink():
    latest_path.unlink()
latest_path.write_text(str(ckpt_dir))
```

This is a simple text file containing the absolute path to the most recent checkpoint.
On Linux, you would use a symlink. On Windows, symlinks require admin privileges, so we
use a text file that `load_checkpoint` reads:

```python
# checkpoint.py, lines 163-164
if ckpt_dir.name == "latest" and ckpt_dir.is_file():
    ckpt_dir = Path(ckpt_dir.read_text().strip())
```

### The Two-Phase Discovery

The `detect_latest_checkpoint` function has a fallback. If no `latest` files exist (for
example, if they were accidentally deleted), it falls back to scanning `step_*` directories
and sorting by step number. This means you can always recover even if the `latest` pointer
is corrupted or missing — the actual checkpoint directories are the source of truth.

---

## 8. Checkpoint Cleanup

### Why Cleanup Exists

Each checkpoint for a 50M model is ~200MB (model) + ~400MB (optimizer state) + a few KB
(metadata). At 5 checkpoints, that is 3GB. For larger models, it scales linearly. Without
cleanup, a long training run fills the disk.

### How It Works

```python
# checkpoint.py, lines 324-337

def _cleanup_old_checkpoints(output_dir, max_checkpoints):
    ckpt_dirs = sorted(
        [d for d in Path(output_dir).iterdir()
         if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    while len(ckpt_dirs) > max_checkpoints:
        old_dir = ckpt_dirs.pop(0)
        for f in old_dir.iterdir():
            f.unlink()
        old_dir.rmdir()
```

Checkpoint directories are sorted by step number. The oldest ones are deleted first,
keeping only the `max_checkpoints` most recent. The default is 5.

### Why You Want At Least 3

- **1 checkpoint:** If the latest checkpoint is corrupted (rare but possible, e.g., disk
  error during rename), you have nothing. Total loss.
- **2 checkpoints:** If the latest is corrupt, you fall back to the previous one. But if
  the *previous* one was already deleted and the latest is also bad, you are stuck.
- **3 checkpoints:** Current + fallback + safety margin. If the latest two are somehow
  both bad (extremely unlikely), you still have one more. This is the minimum safe number.

For a 68-hour training run, the cost of disk space for 3-5 checkpoints is trivial compared
to the cost of losing the run. Do not set `max_checkpoints=1` to save disk space. Set it
to at least 3. If disk space is genuinely tight, move old checkpoints to another drive
rather than deleting them.

---

## 9. The Checkpoint Test Suite

The test file at `tests/test_checkpoint.py` is not a nice-to-have. It is a pre-flight
checklist. Every test exists because the bug it catches has either happened to us or would
be catastrophic if it did.

**The rule: if these tests fail, DO NOT start training.**

### Test-by-Test Breakdown

#### TestCheckpointRoundTrip

| Test | What It Catches |
|------|----------------|
| `test_save_creates_expected_files` | Missing files = incomplete checkpoint. If model.safetensors is not created, there is a bug in save_file or the state dict processing. |
| `test_weight_fidelity` | Loaded weights must be **bitwise identical** to saved weights. Even a single flipped bit means the model diverges from where it was. This catches dtype conversion bugs, endianness issues, and tensor contiguity problems. |
| `test_weight_tying_preserved_after_load` | The most critical test. Verifies that `output.weight.data_ptr() == tok_emb.weight.data_ptr()` after a load cycle. If this fails, the model generates garbage. |
| `test_optimizer_state_restored` | Verifies scheduler epoch survives the round trip. If optimizer state is lost, the model experiences a "learning rate shock" on resume — the optimizer forgets its momentum estimates and the learning rate may jump to the wrong point in the schedule. |
| `test_step_number_round_trip` | If the step number is wrong, the scheduler computes the wrong learning rate, logging is off, and checkpoint naming is broken. |

#### TestWeightTyingSafetensors

| Test | What It Catches |
|------|----------------|
| `test_output_weight_not_in_saved_keys` | If `output.weight` is in the saved state dict, safetensors will crash with `RuntimeError: Some tensors share memory`. This would prevent *any* checkpoint from being saved. |
| `test_no_orig_mod_prefix_in_saved_keys` | If `_orig_mod.` prefixes leak into the saved file, the checkpoint is not portable between compiled and uncompiled models. |

#### TestTorchCompileCheckpoint

| Test | What It Catches |
|------|----------------|
| `test_save_compiled_model` | Verifies that saving a compiled model does not crash. This is the exact bug that originally caused the shared-tensor RuntimeError. |
| `test_load_into_compiled_model` | Round-trip through a compiled model. |
| `test_cross_load_compiled_to_uncompiled` | Save from compiled, load into plain. Must work. |
| `test_cross_load_uncompiled_to_compiled` | Save from plain, load into compiled. Must work. |

These four tests form a complete matrix of compiled/uncompiled combinations. All four must
pass or you cannot safely use `torch.compile` during training.

#### TestLoadModelOnly

| Test | What It Catches |
|------|----------------|
| `test_load_model_only_matches_original` | The inference loading path must produce the same weights as the training loading path. |
| `test_load_model_only_weight_tying` | Weight tying must survive the inference load path too, not just the training load path. |

#### TestMetadataAndLatest

| Test | What It Catches |
|------|----------------|
| `test_metadata_json` | Metadata must be correctly written and parseable. If metadata is corrupt, `get_checkpoint_info` returns empty and auto-resume cannot find the checkpoint. |
| `test_latest_pointer` / `test_latest_pointer_updates` | The latest pointer must always point to the most recent checkpoint. If it is stale, auto-resume loads an old checkpoint and re-does work. |
| `test_load_via_latest` | The entire chain (latest pointer -> checkpoint directory -> loaded model) must work end to end. |

#### TestCheckpointCleanup

| Test | What It Catches |
|------|----------------|
| `test_max_checkpoints_enforced` | Disk does not fill up during long runs. |
| `test_newest_checkpoints_kept` | The cleanup algorithm deletes the *oldest* checkpoints, not the newest. Getting this wrong = deleting your best checkpoint. |

#### TestDetectLatestCheckpoint

| Test | What It Catches |
|------|----------------|
| `test_no_checkpoints` | Returns None gracefully instead of crashing. |
| `test_finds_latest_across_sizes` | Auto-discovery across multiple model size directories. |

#### TestEdgeCases

| Test | What It Catches |
|------|----------------|
| `test_overwrite_existing_checkpoint` | Saving to the same step twice (e.g., after a manual retry) must not corrupt the checkpoint. |
| `test_atomic_save_cleans_tmp_dir` | No orphaned `.tmp_` directories after a successful save. |
| `test_forward_pass_after_load` | The loaded model produces valid (non-NaN) logits. Catches dtype mismatches and weight tying failures. |
| `test_loss_computation_after_load` | Full training step (forward + backward + optimizer step) works after loading. This is the ultimate end-to-end test: if this passes, training can resume. |

---

## 10. Recovery Scenarios

### Corrupted Checkpoint

**Symptoms:** `load_checkpoint` raises an error (truncated file, invalid safetensors header,
missing files).

**Recovery:**
1. Delete the corrupted checkpoint directory
2. Update the `latest` file to point to the previous checkpoint, or just delete `latest`
   (the fallback scanner in `detect_latest_checkpoint` will find the next most recent one)
3. Restart with `--auto-resume`

If `max_checkpoints >= 3`, you always have at least two fallback checkpoints.

### Mismatched Config

**Symptoms:** Model architecture differs between checkpoint and current config (different
`dim`, `n_layers`, `n_heads`, etc.). `load_state_dict` raises errors about tensor shape
mismatches.

**Recovery:** You cannot load a checkpoint into a model with a different architecture. The
shapes must match exactly. Options:
1. Use the same config as the checkpoint (check `metadata.json` for the original config)
2. Start a new training run with the new config
3. For specific changes (like adding layers), write a custom migration script

The `metadata.json` file stores the config used when the checkpoint was saved. Compare it
with your current config to identify mismatches.

### Interrupted Save (Power Loss, Ctrl+C, Crash)

**Symptoms:** A `.tmp_step_*` directory exists in the checkpoint folder. The real
`step_*` directory either does not exist (the rename had not happened yet) or is intact
(the rename completed before the crash).

**Recovery:** Nothing to do. The atomic save pattern handles this automatically:
1. If the rename completed, the `step_*` directory is valid
2. If the rename did not complete, the `step_*` directory still has the previous
   checkpoint's data (or does not exist)
3. On the next save, the `.tmp_*` directory is cleaned up automatically (line 71-72):
   ```python
   if tmp_dir.exists():
       shutil.rmtree(tmp_dir)
   ```

### After Code Changes

**When to re-run tests:**
- Any change to `checkpoint.py` — obviously
- Any change to `transformer.py` — especially `__init__` (weight tying) or `state_dict`
  related code
- Any change to `model/config.py` — config changes can alter model architecture
- Upgrading PyTorch or safetensors versions

**The command:**
```bash
.venv/Scripts/pytest tests/test_checkpoint.py -v
```

Run this before every training start. It takes a few seconds. It could save you 68 hours.

### Incomplete Checkpoint Detection

The load function validates that `model.safetensors` exists before attempting to load:

```python
# checkpoint.py, lines 169-175
model_path = ckpt_dir / "model.safetensors"
if not model_path.exists():
    raise FileNotFoundError(
        f"Incomplete checkpoint at {ckpt_dir} — model.safetensors is missing. "
        f"This usually means a previous save crashed mid-write. "
        f"Delete this directory and resume from an earlier checkpoint."
    )
```

This catches the case where a `.tmp_` directory was manually renamed (bad idea) or where
files were partially deleted. The error message tells you exactly what to do.

---

## Summary: The Invariants

These are the properties that must **always** hold. If any of them break, training is at
risk:

1. **Weight tying:** `output.weight` is not saved. After loading, `output.weight` and
   `tok_emb.weight` share the same `data_ptr()`.

2. **Clean keys:** Saved keys never contain `_orig_mod.` prefix. The prefix is stripped on
   save and restored on load.

3. **Atomic writes:** A `step_*` directory either contains a complete, valid checkpoint or
   does not exist. Partial writes only happen in `.tmp_*` directories.

4. **Latest pointer:** Always points to the most recent *complete* checkpoint. Updated only
   after the atomic rename.

5. **Cleanup safety:** At least `max_checkpoints` most recent checkpoints are preserved.
   Oldest are deleted first.

If you remember nothing else from this document, remember this: **run the checkpoint tests
before starting training.** Every time. No exceptions.

```bash
.venv/Scripts/pytest tests/test_checkpoint.py -v
```

Your 68-hour run depends on it.
