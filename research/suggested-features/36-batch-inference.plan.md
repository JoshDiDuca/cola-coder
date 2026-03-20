# Feature 36: Batch Inference

## Overview

Batch inference processes multiple prompts simultaneously by padding them to the same
length and running a single forward pass through the model. This achieves 5–10× higher
throughput compared to sequential single-prompt generation, making it ideal for:
- Running HumanEval or other benchmarks quickly
- Generating multiple completions for a single prompt (best-of-N sampling)
- Offline data generation pipelines

Status: OPTIONAL — enable via `--feature batch-inference` or CLI menu toggle.

---

## Motivation

- GPU utilization during single-sample generation is typically < 20% (compute is
  bottlenecked by memory bandwidth at batch_size=1).
- Batching increases arithmetic intensity, pushing utilization toward 80%+.
- HumanEval at 20 problems × 1 completion = 20 sequential calls. Batch size 8 reduces
  wall-clock time by ~5×.
- Best-of-N: generate N completions, pick the one that passes tests. Requires fast
  batch generation (N=10, 20, 50).

---

## Architecture / Design

### Left-Padding Convention

Decoder-only models use causal attention, so all prompts in a batch must be aligned
to the RIGHT (generation happens at the rightmost position). Shorter prompts are
padded on the LEFT with a `[PAD]` token.

```
prompt_1: [PAD] [PAD] def add(a, b):
prompt_2: def multiply(a, b):
prompt_3: [PAD] [PAD] [PAD] x = 5
```

The attention mask marks padding positions as 0 so they are ignored.

### Attention Mask

```python
def create_left_padded_batch(
    prompts: list[str],
    tokenizer,
    device: str = "cuda",
    max_length: int | None = None,
) -> dict:
    encodings = [tokenizer.encode(p) for p in prompts]
    max_len = max_length or max(len(e) for e in encodings)

    input_ids = []
    attention_mask = []
    for enc in encodings:
        pad_len = max_len - len(enc)
        input_ids.append([tokenizer.pad_token_id] * pad_len + enc)
        attention_mask.append([0] * pad_len + [1] * len(enc))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }
```

### Batch KV-Cache

Each sequence in the batch maintains its own KV-cache slice. The KV-cache is pre-allocated
as `(batch_size, n_heads, max_seq_len, head_dim)` tensors.

```python
# cola_coder/generator.py  (batch extension)

import torch
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class BatchGenerationState:
    input_ids: torch.Tensor          # (B, T_prompt)
    attention_mask: torch.Tensor     # (B, T_prompt)
    past_key_values: list | None     # KV cache, grows each step
    done: torch.Tensor               # (B,) bool — which seqs finished
    output_ids: list[list[int]]      # collected tokens per sequence
    stop_token_ids: set[int]


class CodeGenerator:
    # ... existing code ...

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_tokens: list[str] | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Generate completions for all prompts using batched inference.
        Returns list of completions in the same order as prompts.
        """
        results: list[str] = [""] * len(prompts)

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_results = self._generate_batch_chunk(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_tokens,
            )
            for i, result in enumerate(batch_results):
                results[batch_start + i] = result

        return results

    def _generate_batch_chunk(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_tokens: list[str] | None,
    ) -> list[str]:
        stop_ids = {self.tokenizer.eos_token_id}
        if stop_tokens:
            for s in stop_tokens:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.add(ids[0])

        batch = create_left_padded_batch(prompts, self.tokenizer, self.config.device)
        B = len(prompts)

        state = BatchGenerationState(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            done=torch.zeros(B, dtype=torch.bool, device=self.config.device),
            output_ids=[[] for _ in range(B)],
            stop_token_ids=stop_ids,
        )

        current_input = state.input_ids

        for step in range(max_new_tokens):
            if state.done.all():
                break

            with torch.no_grad():
                outputs = self.model(
                    current_input if step == 0 else next_tokens.unsqueeze(-1),
                    attention_mask=state.attention_mask if step == 0 else None,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]    # (B, V)
            state.past_key_values = outputs.past_key_values

            # Per-sequence sampling
            next_token_ids = self._batch_sample(logits, temperature, top_k, top_p)
            next_tokens = next_token_ids  # (B,)

            # Record new tokens for non-done sequences
            for b in range(B):
                if not state.done[b]:
                    tid = next_token_ids[b].item()
                    state.output_ids[b].append(tid)
                    if tid in stop_ids:
                        state.done[b] = True

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in state.output_ids
        ]

    def _batch_sample(
        self,
        logits: torch.Tensor,   # (B, V)
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:          # (B,) int64
        """Sample one token per sequence in the batch."""
        from .sampling import apply_temperature, apply_top_k, apply_top_p
        logits = apply_temperature(logits, temperature)
        logits = apply_top_k(logits, top_k)
        logits = apply_top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### Dynamic Batching (Group by Length)

For efficient GPU utilization, sort prompts by length before batching to minimize padding waste:

```python
def dynamic_batch_by_length(
    prompts: list[str],
    tokenizer,
    max_batch_size: int = 8,
    max_tokens_per_batch: int = 4096,
) -> list[list[str]]:
    """
    Group prompts into batches where:
    - batch_size <= max_batch_size
    - total tokens (including padding) <= max_tokens_per_batch
    """
    lengths = [(i, len(tokenizer.encode(p))) for i, p in enumerate(prompts)]
    lengths.sort(key=lambda x: x[1])  # sort by length

    batches: list[list[str]] = []
    current_batch_indices: list[int] = []
    current_max_len = 0

    for idx, length in lengths:
        new_max = max(current_max_len, length)
        projected_tokens = new_max * (len(current_batch_indices) + 1)

        if (len(current_batch_indices) >= max_batch_size
                or projected_tokens > max_tokens_per_batch):
            if current_batch_indices:
                batches.append([prompts[i] for i in current_batch_indices])
            current_batch_indices = [idx]
            current_max_len = length
        else:
            current_batch_indices.append(idx)
            current_max_len = new_max

    if current_batch_indices:
        batches.append([prompts[i] for i in current_batch_indices])

    return batches
```

### Best-of-N Sampling

```python
def best_of_n(
    prompt: str,
    generator: "CodeGenerator",
    n: int = 10,
    scorer: callable = None,  # fn(completion: str) -> float
    **gen_kwargs,
) -> str:
    """Generate N completions, return the one with highest score."""
    prompts = [prompt] * n
    completions = generator.generate_batch(prompts, **gen_kwargs)
    if scorer is None:
        return completions[0]  # return first if no scorer
    scored = [(scorer(c), c) for c in completions]
    return max(scored, key=lambda x: x[0])[1]
```

---

## Implementation Steps

1. **Add `create_left_padded_batch()`** to a new `cola_coder/batch_utils.py`.

2. **Extend `CodeGenerator`** with `generate_batch()` and `_generate_batch_chunk()`.

3. **Add `_batch_sample()`** — vectorized sampling across the batch dimension.
   Verify `sampling.py` functions can handle 2D input (B, V).

4. **Add `dynamic_batch_by_length()`** to `batch_utils.py`.

5. **Extend HumanEval benchmark** to use `generate_batch()`:
   ```python
   # benchmarks/humaneval.py
   all_prompts = [p["prompt"] for p in problems]
   all_completions = generator.generate_batch(all_prompts, batch_size=8)
   ```

6. **Add `--batch-size` CLI argument** to benchmark runner.

7. **Memory guard**: before running batch, estimate memory usage and warn if exceeding
   available VRAM:
   ```python
   estimated_tokens = batch_size * (max_prompt_len + max_new_tokens)
   estimated_bytes = estimated_tokens * n_layers * 2 * n_heads * head_dim * 2  # fp16 KV
   if estimated_bytes > available_vram * 0.8:
       print(f"Warning: batch may OOM. Reduce batch_size or max_new_tokens.")
   ```

8. **CLI menu**: "Run benchmark (batched)" option with configurable batch size.

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Add `generate_batch()`, `_generate_batch_chunk()`, `_batch_sample()` |
| `sampling.py` | Verify/fix top-k/top-p to handle (B, V) input |
| `benchmarks/humaneval.py` | Use `generate_batch()` for all problems |
| `cli/menu.py` | Add batch inference options |
| `cola_coder/batch_utils.py` | New file — padding and dynamic batching |
| `config.py` | Add `BatchConfig` with `batch_size`, `max_tokens_per_batch` |

---

## Testing Strategy

```python
# tests/test_batch_inference.py

def test_batch_matches_single():
    """Batch output should match single-sample output (same seeds)."""
    gen = build_test_generator()
    torch.manual_seed(42)
    single = gen.generate("def add(a, b):", max_new_tokens=20)
    torch.manual_seed(42)
    batch = gen.generate_batch(["def add(a, b):"], max_new_tokens=20)
    assert batch[0] == single

def test_batch_different_lengths():
    """Batch should handle prompts of different lengths."""
    gen = build_test_generator()
    prompts = [
        "x",
        "def very_long_function_name(argument_one, argument_two):",
        "# short",
    ]
    results = gen.generate_batch(prompts, max_new_tokens=20)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)

def test_dynamic_batching_groups_correctly():
    prompts = ["a"] * 5 + ["b" * 100] * 3   # mix of short and long
    batches = dynamic_batch_by_length(prompts, tokenizer, max_batch_size=4,
                                       max_tokens_per_batch=512)
    # Short prompts should be in one batch, long in another
    assert len(batches) >= 2

def test_batch_throughput_improvement():
    """Batch should be faster than sequential for multiple prompts."""
    import time
    gen = build_test_generator()
    prompts = ["def f():\n    return " for _ in range(8)]

    t0 = time.perf_counter()
    for p in prompts:
        gen.generate(p, max_new_tokens=30)
    sequential_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen.generate_batch(prompts, max_new_tokens=30, batch_size=8)
    batch_time = time.perf_counter() - t0

    speedup = sequential_time / batch_time
    print(f"Batch speedup: {speedup:.2f}x")
    assert speedup > 2.0   # expect at least 2x speedup
```

---

## Performance Considerations

- **Padding waste**: with naive batching, a batch of short+long prompts pads all to
  the longest. Dynamic batching minimizes this overhead.
- **Stop token asymmetry**: when one sequence in the batch finishes (hits EOS) before
  others, it must be masked out but the batch continues. Use `done` mask to zero out
  logit gradients for finished sequences.
- **Flash Attention and variable lengths**: FA2 supports variable-length batches via
  `cu_seqlens` (packed format). This eliminates padding entirely but requires custom
  attention mask handling.
- **Memory scaling**: KV-cache grows as `B × T × n_layers × 2 × heads × head_dim`.
  For batch_size=16, 512 tokens, 24 layers, 16 heads, 64 head_dim: ~3 GB in fp16.
- **Optimal batch size**: typically batch_size=8–16 saturates GPU at model sizes < 500 M
  parameters. Larger batches give diminishing returns.

---

## Dependencies

```
torch>=2.2.0    # base requirement
```

No new dependencies. All functionality is built on existing PyTorch primitives.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Left-padding batch creation | 1 hour |
| `generate_batch_chunk()` | 3 hours |
| `_batch_sample()` | 1 hour |
| Dynamic batching | 2 hours |
| Benchmark integration | 2 hours |
| Memory guard | 1 hour |
| Tests | 2 hours |
| **Total** | **~12 hours** |

Complexity rating: **Medium** — the core loop is straightforward; trickiest parts are
handling variable finish times and memory estimation.

---

## 2026 Best Practices

- **Continuous batching (iteration-level batching)**: instead of processing all prompts
  in a batch together until all finish, use iteration-level scheduling. When one sequence
  finishes, immediately insert a new prompt into that slot. This is the approach used
  by vLLM and TGI for high-throughput serving.
- **PagedAttention**: manage KV-cache in fixed-size pages instead of contiguous buffers.
  Eliminates memory fragmentation for variable-length batches. vLLM's core innovation.
- **Speculative batch decoding**: verify multiple speculative tokens per step in batch
  mode. The acceptance/rejection can be done in parallel across the batch.
- **Flash Attention 2 + packed sequences**: eliminate padding entirely by concatenating
  all sequences into one long sequence with a position mask. `flash_attn_varlen_func`
  handles this natively.
- **Chunked prefill**: split long prompts into chunks processed separately, interleaving
  with decode steps from other sequences. Reduces time-to-first-token for concurrent
  requests.
