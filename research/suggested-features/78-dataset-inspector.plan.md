# 78 - Dataset Inspector

## Overview

A CLI tool for browsing and inspecting the contents of `.npy` dataset files. Shows N random samples with syntax highlighting, displays token distribution statistics, generates a sequence length histogram, reports vocabulary coverage, and supports searching for samples containing specific tokens or strings. Useful for debugging data quality issues.

**Feature flag:** Standalone CLI tool, always optional. `cola-coder inspect-dataset`

---

## Motivation

After running `prepare.py`, the dataset is stored as a flat `.npy` array of token IDs. Without a way to inspect it, you have to trust that the data preparation was correct. Common data quality bugs that only appear when you look at the data:

- The tokenizer is including special tokens in the middle of code samples
- File encoding issues are producing garbled text (mojibake)
- Some samples are just `\n\n\n\n` repeated (empty files that passed the min-token filter)
- The vocabulary is heavily skewed toward a few tokens (data contamination from one large file)
- Import statements are being duplicated due to a preprocessing bug

The dataset inspector makes these bugs visible in minutes rather than only being discovered after a training run produces garbage.

---

## Architecture / Design

### Data Format

Assumes the dataset is stored as produced by `prepare.py`:

```python
# Option A: flat numpy array of shape (N,) where N = total tokens,
#           with EOS tokens as sequence separators
data = np.load("data/train.npy")  # shape: (total_tokens,)

# Option B: ragged list of sequences stored as object array
data = np.load("data/train.npy", allow_pickle=True)  # shape: (N_sequences,)
```

The inspector handles both formats automatically.

---

## Implementation Steps

### Step 1: Data Loader (`tools/dataset_inspector.py`)

```python
import numpy as np
from pathlib import Path
from typing import Iterator

class DatasetReader:
    def __init__(self, npy_path: Path, eos_token_id: int = 0):
        self.path = npy_path
        self.eos_token_id = eos_token_id
        self._data = None
        self._sequences: list[np.ndarray] = []

    def load(self):
        self._data = np.load(str(self.path), allow_pickle=True, mmap_mode="r")

        if self._data.dtype == object:
            # Ragged array of sequences
            self._sequences = list(self._data)
        else:
            # Flat array: split on EOS tokens
            self._sequences = self._split_on_eos(self._data)

    def _split_on_eos(self, flat: np.ndarray) -> list[np.ndarray]:
        """Split flat token array on EOS token."""
        eos_positions = np.where(flat == self.eos_token_id)[0]
        if len(eos_positions) == 0:
            return [flat]

        sequences = []
        prev = 0
        for pos in eos_positions:
            if pos > prev:
                sequences.append(flat[prev:pos])
            prev = pos + 1
        if prev < len(flat):
            sequences.append(flat[prev:])
        return sequences

    @property
    def n_sequences(self) -> int:
        return len(self._sequences)

    @property
    def total_tokens(self) -> int:
        return sum(len(s) for s in self._sequences)

    def random_sample(self, rng=None) -> np.ndarray:
        if rng is None:
            import random
            idx = random.randint(0, len(self._sequences) - 1)
        else:
            idx = rng.randint(0, len(self._sequences) - 1)
        return self._sequences[idx]

    def random_samples(self, n: int, seed: int = 42) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._sequences), min(n, len(self._sequences)), replace=False)
        return [self._sequences[i] for i in indices]

    def get_token_distribution(self) -> dict:
        """Compute token frequency distribution."""
        all_tokens = np.concatenate(self._sequences)
        unique, counts = np.unique(all_tokens, return_counts=True)
        total = len(all_tokens)
        top_n = np.argsort(-counts)[:50]
        return {
            "total_tokens": total,
            "unique_tokens": len(unique),
            "vocab_coverage": len(unique),
            "top_tokens": [(int(unique[i]), int(counts[i]), counts[i]/total) for i in top_n],
        }

    def get_length_histogram(self, n_bins: int = 10) -> dict:
        lengths = [len(s) for s in self._sequences]
        if not lengths:
            return {"bins": [], "counts": []}
        hist, bin_edges = np.histogram(lengths, bins=n_bins)
        return {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "bins": [int(e) for e in bin_edges],
            "counts": [int(c) for c in hist],
        }

    def search(self, query_token_ids: list[int]) -> list[int]:
        """Return indices of sequences containing all given token IDs."""
        results = []
        query_set = set(query_token_ids)
        for i, seq in enumerate(self._sequences):
            if query_set.issubset(set(seq.tolist())):
                results.append(i)
        return results
```

### Step 2: Display Functions (`tools/dataset_inspector.py`)

```python
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def display_sample(
    tokens: np.ndarray,
    tokenizer,
    sample_idx: int,
    console: Console,
):
    """Display a single decoded sample with syntax highlighting."""
    text = tokenizer.decode(tokens.tolist())
    console.print(Panel(
        Syntax(text[:2000], "typescript", theme="monokai", line_numbers=True, word_wrap=True),
        title=f"[dim]Sample #{sample_idx}[/]  "
              f"[cyan]{len(tokens)} tokens[/]  "
              f"[dim]{len(text)} chars[/]",
        border_style="blue",
    ))

def display_length_histogram(hist: dict, console: Console):
    """Render an ASCII histogram of sequence lengths."""
    bins = hist["bins"]
    counts = hist["counts"]
    if not counts:
        console.print("[dim]No data[/]")
        return

    max_count = max(counts)
    bar_width = 40
    console.print(f"\n[bold]Sequence Length Distribution[/]")
    console.print(f"  min={hist['min']}  max={hist['max']}  "
                  f"mean={hist['mean']:.0f}  median={hist['median']:.0f}\n")

    for i, count in enumerate(counts):
        low = bins[i]
        high = bins[i + 1]
        bar_len = int(count / max_count * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        pct = count / sum(counts) * 100
        console.print(
            f"  [dim]{low:5d}-{high:<5d}[/] [cyan]{bar}[/] "
            f"[dim]{count:6,} ({pct:4.1f}%)[/]"
        )

def display_token_distribution(dist: dict, tokenizer, console: Console, top_n: int = 20):
    """Show top N most frequent tokens."""
    console.print(f"\n[bold]Token Distribution[/]")
    console.print(f"  Total tokens:   [cyan]{dist['total_tokens']:,}[/]")
    console.print(f"  Unique tokens:  [cyan]{dist['unique_tokens']:,}[/] / "
                  f"{tokenizer.vocab_size} vocab ({dist['unique_tokens']/tokenizer.vocab_size*100:.1f}% coverage)\n")

    table = Table(title=f"Top {top_n} Tokens")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Token ID", justify="right")
    table.add_column("Decoded", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Freq %", justify="right")

    for rank, (token_id, count, freq) in enumerate(dist["top_tokens"][:top_n], 1):
        try:
            decoded = repr(tokenizer.decode([token_id]))[:20]
        except Exception:
            decoded = f"<id={token_id}>"
        table.add_row(
            str(rank), str(token_id), decoded,
            f"{count:,}", f"{freq*100:.2f}%"
        )
    console.print(table)

def display_search_results(
    results: list[int],
    sequences: list,
    tokenizer,
    query: str,
    console: Console,
    max_results: int = 5,
):
    if not results:
        console.print(f"[yellow]No samples found containing '{query}'[/]")
        return
    console.print(f"[green]Found {len(results)} samples containing '{query}'[/]")
    for idx in results[:max_results]:
        display_sample(sequences[idx], tokenizer, idx, console)
```

### Step 3: Anomaly Detector (`tools/dataset_inspector.py`)

```python
def detect_anomalies(reader: DatasetReader, tokenizer, console: Console):
    """Auto-detect common data quality issues."""
    console.print("\n[bold]Anomaly Detection[/]\n")
    issues = []

    # Check for near-empty sequences (< 10 non-whitespace tokens)
    whitespace_tokens = set(
        tokenizer.encode("\n " + " " * 20)
    )
    short_samples = 0
    for seq in reader._sequences:
        non_ws = sum(1 for t in seq.tolist() if t not in whitespace_tokens)
        if non_ws < 10:
            short_samples += 1

    if short_samples > reader.n_sequences * 0.05:
        issues.append(
            f"[yellow]Warning:[/] {short_samples:,} samples ({short_samples/reader.n_sequences*100:.1f}%) "
            "have fewer than 10 non-whitespace tokens. Check minimum token filter."
        )

    # Check for token ID 0 appearing mid-sequence (potential EOS contamination)
    eos_mid_count = 0
    for seq in reader._sequences[:1000]:  # sample check
        if 0 in seq[1:-1]:  # EOS appearing not at start/end
            eos_mid_count += 1

    if eos_mid_count > 10:
        issues.append(
            f"[red]Error:[/] EOS token (0) found mid-sequence in {eos_mid_count}+ samples. "
            "This indicates a tokenization bug."
        )

    # Check vocabulary coverage
    dist = reader.get_token_distribution()
    if dist["unique_tokens"] < tokenizer.vocab_size * 0.1:
        issues.append(
            f"[yellow]Warning:[/] Only {dist['unique_tokens']:,} of {tokenizer.vocab_size:,} "
            "vocabulary tokens appear in the dataset. Vocabulary may be under-utilized."
        )

    # Top token frequency (potential repetition bias)
    if dist["top_tokens"]:
        top_id, top_count, top_freq = dist["top_tokens"][0]
        if top_freq > 0.05:
            decoded = repr(tokenizer.decode([top_id]))[:15]
            issues.append(
                f"[yellow]Warning:[/] Token {top_id} ({decoded}) appears in {top_freq*100:.1f}% "
                "of all tokens. This may indicate data bias."
            )

    if issues:
        for issue in issues:
            console.print(f"  {issue}")
    else:
        console.print("  [green]No anomalies detected.[/]")
```

### Step 4: CLI Entry Point (`cli/inspect_cmd.py`)

```python
import click
from pathlib import Path

@click.group("inspect-dataset")
def inspect_cmd():
    """Browse and inspect dataset .npy files."""
    pass

@inspect_cmd.command("show")
@click.argument("npy_path", type=click.Path(exists=True))
@click.option("--n", default=5, help="Number of random samples to show")
@click.option("--seed", default=42)
@click.option("--tokenizer", "tokenizer_path", default="tokenizer/", type=click.Path())
def cmd_show(npy_path, n, seed, tokenizer_path):
    """Show N random samples from the dataset."""
    from rich.console import Console
    tokenizer = load_tokenizer(tokenizer_path)
    reader = DatasetReader(Path(npy_path))
    reader.load()
    console = Console()
    console.print(f"\n[bold]Dataset:[/] {npy_path}  "
                  f"[cyan]{reader.n_sequences:,} sequences[/]  "
                  f"{reader.total_tokens:,} total tokens\n")
    samples = reader.random_samples(n, seed)
    for i, sample in enumerate(samples):
        display_sample(sample, tokenizer, i, console)

@inspect_cmd.command("stats")
@click.argument("npy_path", type=click.Path(exists=True))
@click.option("--tokenizer", "tokenizer_path", default="tokenizer/")
@click.option("--anomalies", is_flag=True, default=True)
def cmd_stats(npy_path, tokenizer_path, anomalies):
    """Show token distribution and sequence length stats."""
    from rich.console import Console
    tokenizer = load_tokenizer(tokenizer_path)
    reader = DatasetReader(Path(npy_path))
    reader.load()
    console = Console()
    display_length_histogram(reader.get_length_histogram(), console)
    display_token_distribution(reader.get_token_distribution(), tokenizer, console)
    if anomalies:
        detect_anomalies(reader, tokenizer, console)

@inspect_cmd.command("search")
@click.argument("npy_path", type=click.Path(exists=True))
@click.argument("query")
@click.option("--tokenizer", "tokenizer_path", default="tokenizer/")
@click.option("--max-results", default=5)
def cmd_search(npy_path, query, tokenizer_path, max_results):
    """Find samples containing a specific string."""
    from rich.console import Console
    tokenizer = load_tokenizer(tokenizer_path)
    reader = DatasetReader(Path(npy_path))
    reader.load()
    query_ids = tokenizer.encode(query)
    results = reader.search(query_ids)
    console = Console()
    display_search_results(results, reader._sequences, tokenizer, query, console, max_results)
```

---

## Key Files to Modify

- `tools/dataset_inspector.py` - New file: core inspector logic
- `cli/inspect_cmd.py` - New file: CLI
- `cli/main.py` - Register `inspect-dataset` command group

---

## Testing Strategy

1. **Sequence splitting test**: create flat array with known EOS positions, assert `_split_on_eos` produces correct sequence boundaries.
2. **Ragged format test**: save an object array of token lists, load with inspector, assert `n_sequences` and `total_tokens` are correct.
3. **Length histogram test**: create 100 sequences of known lengths, assert histogram bins cover the full range.
4. **Token distribution test**: create dataset with known token frequencies, assert top token in distribution matches expected.
5. **Search test**: create a dataset where one sequence contains a known token subsequence, assert search finds it.
6. **Anomaly detection test**: create dataset with 20% near-empty sequences (< 10 non-whitespace tokens), assert anomaly is flagged.
7. **Display test**: call `display_sample` with a known token list, assert no exceptions.

---

## Performance Considerations

- Loading a 50k-sequence dataset into memory: `np.load` with `mmap_mode="r"` avoids loading all tokens at once. For object arrays (ragged format), mmap is not supported—load the full array but it's typically <200MB.
- `get_token_distribution()` calls `np.concatenate` on all sequences. For 30M tokens, this creates a 120MB array (int32). Acceptable.
- `search()` is O(N × avg_sequence_length). For 50k sequences of avg 900 tokens, this is ~45M operations. At numpy speeds, ~0.5s. Acceptable.
- For very large datasets (>500k sequences), add a `--sample-n` option that only inspects a random subset for stats computation.

---

## Dependencies

No new Python dependencies. Uses `numpy`, `rich` (already required), and the project's own tokenizer.

---

## Estimated Complexity

**Low.** Simple numpy operations and Rich display code. The anomaly detector requires knowledge of what to look for (domain-specific), but the implementation is straightforward. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **mmap_mode for large datasets**: always use `np.load(..., mmap_mode="r")` for flat numpy arrays. This prevents loading 500MB into RAM just to inspect 5 samples.
- **Vocabulary coverage as a quality signal**: a dataset that uses only 30% of the vocabulary suggests the data is not diverse enough. Track vocabulary coverage over different dataset versions (connects to plan 62).
- **Token ID 0 is a canary**: if EOS token (typically ID 0) appears mid-sequence, it's almost always a bug. This check should be part of every dataset preparation pipeline's smoke test.
- **Random seed for reproducible inspection**: always show samples from a fixed seed by default. This makes it possible to share "I looked at samples with seed=42 and saw X" in team conversations. Allow `--seed random` to get truly random samples.
- **Connect to data versioning (plan 62)**: ideally, `inspect-dataset` can resolve a dataset version tag and automatically find the correct `.npy` file: `cola-coder inspect-dataset --version baseline-v1`.
