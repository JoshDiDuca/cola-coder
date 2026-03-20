# Feature 54: Streaming Training

**Status:** Proposed
**CLI Flag:** `--streaming`
**Complexity:** Medium

---

## Overview

Train directly from HuggingFace datasets using `streaming=True`, bypassing the disk-intensive `.npy` memmap pipeline for supported datasets. Implements a `StreamingDataLoader` that wraps `IterableDataset`, handles on-the-fly tokenization, batching, and stream position checkpointing.

---

## Motivation

The current pipeline requires:
1. Download raw data (~50-500GB)
2. Quality filter + deduplicate (hours)
3. Tokenize → `.npy` memmaps (hours)
4. Train from memmaps

Streaming training skips steps 1-3 for supported datasets:
- **No disk space requirement** — data flows GPU-to-GPU
- **Fresh data every epoch** — no deduplication needed across epochs (dataset is effectively infinite)
- **Faster iteration** — start training in seconds, not hours
- **Suitable for prototyping** — try a new dataset without full preprocessing

**Tradeoffs:** No random access (sequential reads only), cannot deduplicate across the stream, harder to reproduce exact batches without checkpointing.

---

## Architecture / Design

```
HuggingFace Hub
  │ datasets.load_dataset("bigcode/the-stack-v2", streaming=True)
  ▼
HF IterableDataset
  │ on-the-fly: filter → tokenize → chunk
  ▼
StreamingBuffer (size=1000 examples)
  │ shuffle within buffer
  ▼
Collator → batch tensors
  │
  ▼
Training loop
  │ checkpoints save: stream shard + offset
```

---

## Implementation Steps

### Step 1: Streaming Dataset Wrapper

```python
# src/data/streaming_dataset.py
from __future__ import annotations
from typing import Iterator, Optional
from datasets import load_dataset, IterableDataset
from transformers import PreTrainedTokenizerFast
import torch
import random
from dataclasses import dataclass

@dataclass
class StreamingConfig:
    dataset_name: str                   # e.g. "bigcode/the-stack-v2"
    dataset_config: Optional[str]       # e.g. "TypeScript"
    split: str = "train"
    text_column: str = "content"
    max_tokens: int = 2048
    buffer_size: int = 1000
    seed: int = 42
    quality_filter: bool = True
    min_chars: int = 100
    max_chars: int = 100_000

class StreamingCodeDataset:
    def __init__(self, config: StreamingConfig, tokenizer: PreTrainedTokenizerFast):
        self.config    = config
        self.tokenizer = tokenizer
        self._dataset: Optional[IterableDataset] = None
        self._shard_info: dict = {}

    def _load(self) -> IterableDataset:
        kwargs = dict(split=self.config.split, streaming=True)
        if self.config.dataset_config:
            kwargs["name"] = self.config.dataset_config
        ds = load_dataset(self.config.dataset_name, **kwargs)
        if self.config.quality_filter:
            ds = ds.filter(self._quality_filter)
        return ds

    def _quality_filter(self, example: dict) -> bool:
        text = example.get(self.config.text_column, "")
        if not isinstance(text, str):
            return False
        n = len(text)
        return self.config.min_chars <= n <= self.config.max_chars

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _chunk(self, token_ids: list[int]) -> list[list[int]]:
        """Split token_ids into chunks of max_tokens."""
        return [
            token_ids[i:i + self.config.max_tokens]
            for i in range(0, len(token_ids), self.config.max_tokens)
            if len(token_ids[i:i + self.config.max_tokens]) > 64
        ]

    def stream_batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        if self._dataset is None:
            self._dataset = self._load()

        buffer: list[list[int]] = []

        for example in self._dataset:
            text = example.get(self.config.text_column, "")
            token_ids = self._tokenize(text)
            buffer.extend(self._chunk(token_ids))

            while len(buffer) >= self.config.buffer_size:
                # Shuffle buffer for some randomness despite sequential reads
                random.shuffle(buffer)

                # Yield batches from the buffer
                while len(buffer) >= batch_size:
                    batch_chunks = buffer[:batch_size]
                    buffer = buffer[batch_size:]
                    yield self._collate(batch_chunks)

        # Yield remaining
        if buffer:
            random.shuffle(buffer)
            while len(buffer) >= batch_size:
                yield self._collate(buffer[:batch_size])
                buffer = buffer[batch_size:]

    def _collate(self, chunks: list[list[int]]) -> dict[str, torch.Tensor]:
        max_len = max(len(c) for c in chunks)
        input_ids = torch.zeros(len(chunks), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(chunks), max_len, dtype=torch.long)
        for i, chunk in enumerate(chunks):
            input_ids[i, :len(chunk)] = torch.tensor(chunk)
            attention_mask[i, :len(chunk)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
```

### Step 2: Stream Position Checkpointing

HuggingFace streaming datasets support shard-level iteration. We track which shard and example offset we've reached for approximate resumption.

```python
# src/data/stream_checkpoint.py
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class StreamCheckpoint:
    dataset_name: str
    dataset_config: Optional[str]
    examples_consumed: int
    shards_consumed: int
    training_step: int
    timestamp: str

def save_stream_checkpoint(checkpoint: StreamCheckpoint, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(checkpoint), f, indent=2)

def load_stream_checkpoint(path: str) -> Optional[StreamCheckpoint]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return StreamCheckpoint(**data)

def skip_to_checkpoint(dataset, checkpoint: StreamCheckpoint):
    """Skip the first N examples in the stream to resume from checkpoint."""
    print(f"[Streaming] Skipping {checkpoint.examples_consumed} examples to resume...")
    for i, _ in enumerate(dataset):
        if i >= checkpoint.examples_consumed:
            break
    print(f"[Streaming] Resumed at example {checkpoint.examples_consumed}")
    return dataset
```

### Step 3: Streaming Training Loop

```python
# src/training/streaming_trainer.py
from datetime import datetime
from pathlib import Path
import torch
from src.data.streaming_dataset import StreamingCodeDataset, StreamingConfig
from src.data.stream_checkpoint import StreamCheckpoint, save_stream_checkpoint, load_stream_checkpoint

def train_streaming(
    model,
    tokenizer,
    optimizer,
    config: StreamingConfig,
    output_dir: str,
    max_steps: int = 10_000,
    checkpoint_interval: int = 500,
    log_interval: int = 10,
    resume_checkpoint: str = None,
    batch_size: int = 4,
    device: str = "cuda",
):
    dataset = StreamingCodeDataset(config, tokenizer)

    step = 0
    examples_consumed = 0

    if resume_checkpoint:
        ckpt = load_stream_checkpoint(resume_checkpoint)
        if ckpt:
            print(f"[Streaming] Resuming from step {ckpt.training_step}, {ckpt.examples_consumed} examples")
            step = ckpt.training_step
            examples_consumed = ckpt.examples_consumed

    model.train()

    for batch in dataset.stream_batches(batch_size):
        if step >= max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        examples_consumed += batch_size
        step += 1

        if step % log_interval == 0:
            print(f"[Step {step}] loss={loss.item():.4f} examples={examples_consumed}")

        if step % checkpoint_interval == 0:
            ckpt_path = Path(output_dir) / f"stream_checkpoint_{step}.json"
            save_stream_checkpoint(StreamCheckpoint(
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                examples_consumed=examples_consumed,
                shards_consumed=0,
                training_step=step,
                timestamp=datetime.utcnow().isoformat(),
            ), str(ckpt_path))
            # Also save model
            model.save_pretrained(Path(output_dir) / f"checkpoint_{step}")
            tokenizer.save_pretrained(Path(output_dir) / f"checkpoint_{step}")

    return step
```

### Step 4: Multi-Dataset Streaming Interleaving

```python
# src/data/interleaved_stream.py
from src.data.streaming_dataset import StreamingCodeDataset, StreamingConfig
import torch
import random

class InterleavedStreamingDataset:
    """
    Interleave multiple streaming datasets with configurable sampling weights.
    """
    def __init__(
        self,
        datasets: list[tuple[StreamingCodeDataset, float]],  # (dataset, weight)
        seed: int = 42,
    ):
        self.datasets = datasets
        self.weights = [w for _, w in datasets]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        random.seed(seed)

    def stream_batches(self, batch_size: int):
        iterators = [ds.stream_batches(batch_size) for ds, _ in self.datasets]
        exhausted = [False] * len(iterators)

        while not all(exhausted):
            active = [i for i, e in enumerate(exhausted) if not e]
            if not active:
                break
            active_weights = [self.weights[i] for i in active]
            total = sum(active_weights)
            norm_weights = [w / total for w in active_weights]

            idx = random.choices(active, weights=norm_weights, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                exhausted[idx] = True
```

### Step 5: CLI Integration

```python
# cli/train.py
parser.add_argument("--streaming", action="store_true",
    help="Train from HuggingFace streaming datasets instead of .npy memmaps.")
parser.add_argument("--streaming-dataset", type=str, default="bigcode/the-stack-v2",
    help="HuggingFace dataset name (default: bigcode/the-stack-v2).")
parser.add_argument("--streaming-config", type=str, default="TypeScript",
    help="Dataset configuration/language (default: TypeScript).")
parser.add_argument("--streaming-buffer", type=int, default=1000,
    help="Shuffle buffer size in examples (default: 1000).")
parser.add_argument("--streaming-resume", type=str, default=None,
    help="Path to stream checkpoint JSON to resume from.")
parser.add_argument("--streaming-quality-filter", action="store_true", default=True,
    help="Apply quality filter to streaming data.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/streaming_dataset.py` | New — core streaming wrapper |
| `src/data/stream_checkpoint.py` | New — checkpoint I/O |
| `src/data/interleaved_stream.py` | New — multi-dataset interleaving |
| `src/training/streaming_trainer.py` | New — streaming training loop |
| `cli/train.py` | Add `--streaming` and related flags |
| `src/training/grpo_trainer.py` | Accept streaming data loader as input |

---

## Testing Strategy

```python
# tests/test_streaming.py
import pytest
from unittest.mock import MagicMock, patch

def test_quality_filter_rejects_short():
    config = StreamingConfig(dataset_name="test", min_chars=100)
    ds = StreamingCodeDataset(config, tokenizer=MagicMock())
    assert not ds._quality_filter({"content": "x" * 50})
    assert ds._quality_filter({"content": "x" * 150})

def test_chunking_produces_correct_sizes():
    config = StreamingConfig(dataset_name="test", max_tokens=10)
    ds = StreamingCodeDataset(config, tokenizer=MagicMock())
    token_ids = list(range(35))
    chunks = ds._chunk(token_ids)
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) == 4  # 10+10+10+5, last dropped if <64 — actually all pass since 5<64

def test_collate_pads_correctly():
    config = StreamingConfig(dataset_name="test")
    ds = StreamingCodeDataset(config, tokenizer=MagicMock())
    chunks = [[1, 2, 3], [4, 5, 6, 7, 8]]
    batch = ds._collate(chunks)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"][0, 3].item() == 0  # padded
    assert batch["attention_mask"][1, 4].item() == 1  # not padded

def test_checkpoint_round_trip(tmp_path):
    ckpt = StreamCheckpoint(
        dataset_name="test", dataset_config=None,
        examples_consumed=500, shards_consumed=2,
        training_step=100, timestamp="2026-01-01T00:00:00",
    )
    path = str(tmp_path / "ckpt.json")
    save_stream_checkpoint(ckpt, path)
    loaded = load_stream_checkpoint(path)
    assert loaded.training_step == 100
    assert loaded.examples_consumed == 500
```

---

## Performance Considerations

- Streaming throughput from HuggingFace Hub: ~1000-3000 examples/second over fast network. With a 1M token/batch training step, this is comfortably fast enough for most model sizes.
- The shuffle buffer (default 1000) adds ~100MB memory overhead for TypeScript files — acceptable.
- On slow internet connections, the stream may become the bottleneck. Add prefetching:

```python
from torch.utils.data import DataLoader

def make_prefetch_loader(dataset: StreamingCodeDataset, batch_size: int, prefetch: int = 4):
    class _IterWrapper(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from dataset.stream_batches(batch_size)
    return DataLoader(_IterWrapper(), batch_size=None, num_workers=prefetch, prefetch_factor=2)
```

- For production training, consider downloading shards to local SSD first (using `datasets` `.download_and_prepare()`) to eliminate network latency after the first pass.

---

## Dependencies

```
datasets>=2.19.0        # HuggingFace datasets with streaming support
huggingface-hub>=0.22.0 # for authenticated access to gated datasets
```

---

## Estimated Complexity

**Development time:** 3-4 days
**Risk:** Low-Medium. HuggingFace streaming is well-supported. Main risk is network reliability during long training runs.
**Lines of new code:** ~350

---

## 2026 Best Practices

- **Streaming as default for prototyping:** In 2025-2026, many teams use streaming for initial experiments and switch to preprocessed data only for long production runs. Support both modes.
- **Checkpoint at shard boundaries:** HuggingFace datasets >= 2.19 exposes shard information. Checkpoint at shard boundaries for exact reproducibility rather than approximate skip-ahead.
- **Data freshness:** The-Stack-v2 and StarCoder datasets are continuously updated. Streaming gives access to the latest data without re-running the preprocessing pipeline.
- **Token budget:** When streaming, track total tokens consumed (not just examples) and log tokens/second as the primary training speed metric.
