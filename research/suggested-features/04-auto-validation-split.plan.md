# 04 - Auto Validation Split

## Overview

During `prepare_data`, automatically hold out 5% of the source files as a validation set before tokenization and chunking. The split is seeded for reproducibility and is saved as separate `train_data.npy` and `val_data.npy` files. The training manifest is updated with split metadata. An optional stratified split divides by file size or estimated complexity.

The critical design constraint: **split at the file level before tokenization**, not at the token/chunk level after tokenization. This prevents data leakage from context windows that straddle chunk boundaries.

---

## Motivation

Training without a validation set is flying blind. Without a held-out validation loss:
- You cannot detect overfitting (train loss → 0 but model memorizes, does not generalize)
- You cannot compare checkpoints objectively
- You cannot tune learning rate or dropout effectively

Most small training projects skip validation because it "wastes" data. For a 50M model learning TypeScript, the quality of 5% validation is worth far more than the 5% extra training examples.

**Why split before tokenization?**

Suppose you have a file `utils.ts`. If you split at the token/chunk level, it's possible that:
- Chunk 47 (tokens 9400-9599) lands in train
- Chunk 48 (tokens 9600-9799) lands in val

These two chunks share overlapping context (the model trained on chunk 47 has seen the beginning of the function body that chunk 48 tests). This is **data leakage**: the validation loss is artificially low.

By splitting at the file level, no file appears in both train and val.

---

## Architecture / Design

### Split Strategy

```
Input: list of source .ts/.js files
       │
       ├─ Shuffle with seed
       │
       ├─ Split: 95% train files / 5% val files
       │
       ├─ Tokenize train files → train_tokens.npy
       ├─ Tokenize val files   → val_tokens.npy
       │
       └─ Save manifest with split info
```

### File Manifest Format

```json
{
  "split_seed": 42,
  "split_ratio": 0.05,
  "total_files": 12847,
  "train_files": 12205,
  "val_files": 642,
  "train_tokens": 48200000,
  "val_tokens": 2400000,
  "train_data_path": "data/processed/train_data.npy",
  "val_data_path": "data/processed/val_data.npy",
  "created_at": "2026-03-20T14:23:01",
  "stratified": false
}
```

### Stratified Split Option

When `--stratified` is used, files are grouped into 3 buckets by token count:
- Small: < 50 tokens (utility functions, exports)
- Medium: 50-500 tokens (most files)
- Large: > 500 tokens (complex modules)

Then 5% is sampled from each bucket independently, ensuring the val set has the same size distribution as the train set.

---

## Implementation Steps

### Step 1: File collection with metadata

```python
# src/data/splitter.py
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

@dataclass
class FileRecord:
    path: str
    byte_size: int
    estimated_tokens: int  # rough estimate: byte_size / 4

    @property
    def size_bucket(self) -> Literal["small", "medium", "large"]:
        if self.estimated_tokens < 50:
            return "small"
        elif self.estimated_tokens < 500:
            return "medium"
        return "large"

def collect_files(source_dir: Path, extensions: list[str] = None) -> list[FileRecord]:
    if extensions is None:
        extensions = [".ts", ".tsx", ".js", ".jsx"]
    records = []
    for ext in extensions:
        for fpath in source_dir.rglob(f"*{ext}"):
            try:
                size = fpath.stat().st_size
                records.append(FileRecord(
                    path=str(fpath),
                    byte_size=size,
                    estimated_tokens=max(1, size // 4),
                ))
            except OSError:
                continue
    return records
```

### Step 2: Split with reproducible seed

```python
# src/data/splitter.py (continued)

def split_files(
    records: list[FileRecord],
    val_ratio: float = 0.05,
    seed: int = 42,
    stratified: bool = False,
) -> tuple[list[FileRecord], list[FileRecord]]:
    """
    Split file records into train/val sets.
    Returns (train_records, val_records).
    """
    rng = random.Random(seed)

    if not stratified:
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        return shuffled[n_val:], shuffled[:n_val]

    # Stratified: split per bucket
    buckets: dict[str, list[FileRecord]] = {"small": [], "medium": [], "large": []}
    for r in records:
        buckets[r.size_bucket].append(r)

    train_all, val_all = [], []
    for bucket_name, bucket_records in buckets.items():
        rng.shuffle(bucket_records)
        n_val = max(1 if bucket_records else 0, int(len(bucket_records) * val_ratio))
        val_all.extend(bucket_records[:n_val])
        train_all.extend(bucket_records[n_val:])

    return train_all, val_all
```

### Step 3: Tokenize each split and save to .npy

```python
# src/data/splitter.py (continued)
import numpy as np
from tqdm import tqdm

def tokenize_and_save(
    file_records: list[FileRecord],
    tokenizer,
    output_path: Path,
    chunk_size: int = 512,
    overlap: int = 0,
) -> int:
    """
    Tokenize all files, concatenate tokens into chunks of chunk_size,
    save as memmap-compatible .npy file.
    Returns total token count.
    """
    all_tokens = []
    for record in tqdm(file_records, desc=f"Tokenizing -> {output_path.name}"):
        try:
            text = Path(record.path).read_text(encoding="utf-8", errors="replace")
            token_ids = tokenizer.encode(text)
            all_tokens.extend(token_ids)
        except Exception:
            continue

    # Chunk into fixed-size windows
    chunks = []
    stride = chunk_size - overlap
    for i in range(0, len(all_tokens) - chunk_size + 1, stride):
        chunks.append(all_tokens[i : i + chunk_size])

    if not chunks:
        raise ValueError(f"No chunks produced from {len(file_records)} files")

    arr = np.array(chunks, dtype=np.uint16)
    np.save(output_path, arr)
    return len(all_tokens)
```

### Step 4: Updated prepare_data entry point

```python
# src/prepare_data.py (modified)
import typer
from pathlib import Path
import json
from datetime import datetime
from src.data.splitter import collect_files, split_files, tokenize_and_save
from src.tokenizer import load_tokenizer
from src.cli import header, confirm
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

@app.command()
def main(
    source_dir: Path = typer.Argument(..., help="Directory of .ts/.js source files"),
    output_dir: Path = typer.Option(Path("data/processed"), help="Output directory"),
    val_ratio: float = typer.Option(0.05, help="Fraction of files for validation (0.0 to disable)"),
    seed: int = typer.Option(42, help="Random seed for reproducible split"),
    stratified: bool = typer.Option(False, help="Use stratified split by file size"),
    chunk_size: int = typer.Option(512, help="Token chunk size"),
):
    header("Prepare Data", f"Source: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Scanning source files...[/cyan]")
    records = collect_files(source_dir)
    console.print(f"Found [bold]{len(records)}[/bold] source files")

    if val_ratio > 0:
        train_records, val_records = split_files(records, val_ratio, seed, stratified)
        console.print(f"Split: [green]{len(train_records)}[/green] train, [yellow]{len(val_records)}[/yellow] val")
    else:
        train_records = records
        val_records = []
        console.print("[yellow]Warning: val_ratio=0, no validation split created[/yellow]")

    tokenizer = load_tokenizer()

    train_path = output_dir / "train_data.npy"
    val_path = output_dir / "val_data.npy"

    train_tokens = tokenize_and_save(train_records, tokenizer, train_path, chunk_size)

    val_tokens = 0
    if val_records:
        val_tokens = tokenize_and_save(val_records, tokenizer, val_path, chunk_size)

    # Save manifest
    manifest = {
        "split_seed": seed,
        "split_ratio": val_ratio,
        "total_files": len(records),
        "train_files": len(train_records),
        "val_files": len(val_records),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_data_path": str(train_path),
        "val_data_path": str(val_path) if val_records else None,
        "chunk_size": chunk_size,
        "stratified": stratified,
        "created_at": datetime.utcnow().isoformat(),
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _render_summary_table(manifest)

def _render_summary_table(m: dict) -> None:
    table = Table(title="Data Split Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total files", str(m["total_files"]))
    table.add_row("Train files", str(m["train_files"]))
    table.add_row("Val files", str(m["val_files"]))
    table.add_row("Train tokens", f"{m['train_tokens']:,}")
    table.add_row("Val tokens", f"{m['val_tokens']:,}")
    table.add_row("Split seed", str(m["split_seed"]))
    table.add_row("Stratified", str(m["stratified"]))
    console.print(table)

if __name__ == "__main__":
    app()
```

### Step 5: Update trainer.py to load val set

```python
# src/trainer.py (additions)
import numpy as np
from pathlib import Path
import json

def load_val_dataset(data_dir: Path) -> np.ndarray | None:
    """Load val_data.npy if it exists, return None otherwise."""
    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    val_path = manifest.get("val_data_path")
    if not val_path or not Path(val_path).exists():
        return None
    return np.load(val_path)

# In Trainer.train():
def compute_val_loss(self, val_data: np.ndarray, n_batches: int = 20) -> float:
    """Compute validation loss on N random batches from val_data."""
    self.model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            idx = random.randint(0, len(val_data) - 1)
            batch = torch.tensor(val_data[idx:idx+1], dtype=torch.long).to(self.device)
            x, y = batch[:, :-1], batch[:, 1:]
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
    self.model.train()
    return total_loss / n_batches

# Call in training loop:
# if step % config.val_interval == 0 and val_data is not None:
#     val_loss = self.compute_val_loss(val_data)
#     logger.log({"step": step, "val_loss": val_loss})
```

### Step 6: Config additions

```yaml
# configs/small.yaml (additions)
data:
  val_ratio: 0.05           # 0.0 to disable
  split_seed: 42
  stratified_split: false   # true = by file size bucket

training:
  val_interval: 500         # compute val loss every N steps
  val_batches: 20           # number of random val batches per eval
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/prepare_data.py` | Add split logic, separate tokenization paths, manifest writing |
| `src/data/splitter.py` | New: collect_files, split_files, tokenize_and_save |
| `src/trainer.py` | Add val data loading, `compute_val_loss()`, val logging |
| `configs/*.yaml` | Add `data.val_ratio`, `training.val_interval` |
| `runs/manifest.json` | Add val_loss entries alongside train_loss |

---

## Testing Strategy

- **Determinism test**: Run `split_files(records, seed=42)` twice, verify identical output
- **No overlap test**: Assert `set(train_paths) & set(val_paths) == set()` — no file in both sets
- **Ratio test**: With 1000 files and val_ratio=0.05, verify val has 50 files
- **Stratified test**: With 100 small, 400 medium, 500 large files, verify each bucket is split 5%
- **Integration test**: Run full prepare_data on a small directory, verify both .npy files created, load and shape-check them
- **Leakage audit** (manual): Pick a random val file, verify no token substring from it appears in any train chunk

---

## Performance Considerations

- Tokenization is the bottleneck: ~100K tokens/second on CPU. 50M tokens = 500 seconds (~8 min). This is unchanged from current behavior.
- File collection with `rglob` on 50K+ files takes ~1-2 seconds — acceptable
- Stratified bucket assignment is O(n) — no performance concern
- The .npy files are memory-mapped at training time; loading val_data.npy for 20 random batches per step is O(1) disk seeks

---

## Dependencies

No new Python packages required. All functionality uses:
- `numpy` (already installed)
- `pathlib`, `random`, `json` (stdlib)

---

## Estimated Complexity

**Low** — 1 day.

- File collector + splitter: 2 hours
- Updated prepare_data.py: 2 hours
- Trainer modifications: 2 hours
- Config additions: 30 minutes
- Testing: 2 hours

Total: ~8.5 hours

---

## 2026 Best Practices

- **File-level split, not chunk-level**: This is non-negotiable for meaningful validation loss. Chunk-level splits cause data leakage and give falsely optimistic val loss.
- **Seeded splits**: Always seed the split so results are reproducible. Store the seed in the manifest so anyone can recreate the exact same split.
- **Persist the val split**: Save the list of val file paths in `split_manifest.json`. If you regenerate data, use the same seed to get the same files in val — otherwise you're changing the evaluation target.
- **5% is usually enough**: With 50M tokens of training data, 5% = 2.5M tokens for validation. This is more than sufficient for a stable val loss estimate. Don't over-allocate to val.
- **Stratified splits for heterogeneous corpora**: If the corpus has many tiny utility files alongside large framework files, unstratified sampling might put all large files in train. Stratified sampling ensures balanced representation.
