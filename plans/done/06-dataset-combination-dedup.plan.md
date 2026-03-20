# Plan: Dataset Combination & Deduplication

## Problem

As we add more data sources (HuggingFace, GitHub scraper, local files, synthetic), we need:

1. **Combine** multiple .npy datasets into one training file
2. **Deduplicate** across datasets (GitHub code may overlap with StarCoderData)
3. **Mix** with configurable ratios (70% HF + 20% GitHub + 10% synthetic)
4. **Track** provenance through combination

## Architecture

### Dataset Combination

```python
# src/cola_coder/data/combine.py

class DatasetCombiner:
    """Combine multiple .npy token arrays into one training dataset.

    Supports three mixing strategies:
    1. Concatenate: Simply append datasets end-to-end
    2. Interleave: Round-robin chunks from each dataset (better mixing)
    3. Weighted: Sample from each dataset with specified probability
    """

    def combine(
        self,
        datasets: list[DatasetInput],
        strategy: str = "interleave",  # "concat" | "interleave" | "weighted"
        output_path: str = "./data/processed/combined.npy",
        max_tokens: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> CombineResult:
        """Combine datasets into a single training file."""
```

### DatasetInput

```python
@dataclass
class DatasetInput:
    path: str              # Path to .npy file
    weight: float = 1.0    # Relative weight for weighted mixing
    name: str = ""         # Human-readable name for manifest
    max_chunks: int | None = None  # Optional cap per dataset
```

### Mixing Strategies

#### Concatenate
Simple append. Fast, deterministic. Order matters (model sees dataset A first, then B).
Good for curriculum learning (easy data first, hard data second).

#### Interleave
Round-robin: chunk from A, chunk from B, chunk from A, ...
Better mixing, model sees variety throughout training.
Weighted interleave: if A has weight 0.7 and B has weight 0.3, roughly 70% of chunks come from A.

#### Weighted Random
Random sampling with replacement based on weights.
Best mixing quality, but non-deterministic (use seed for reproducibility).
This is what datatrove and dolma use.

### Deduplication Across Datasets

```python
# src/cola_coder/data/dedup.py

class CrossDatasetDeduplicator:
    """Remove near-duplicates across multiple datasets.

    Uses MinHash LSH for scalable near-duplicate detection.
    Can handle millions of documents with ~500 bytes RAM per doc.

    Pipeline:
    1. Build MinHash signatures for all chunks in dataset A
    2. For each chunk in dataset B, query against A's index
    3. Mark duplicates for removal
    4. Output deduplicated dataset

    Also supports exact dedup (hash-based, faster but misses near-dupes).
    """

    def __init__(
        self,
        method: str = "minhash",     # "exact" | "minhash"
        threshold: float = 0.8,       # Jaccard similarity threshold
        num_perm: int = 128,          # MinHash permutations
        ngram_size: int = 5,          # Character n-gram size
    ):
        ...

    def build_index(self, dataset_path: str, tokenizer_path: str) -> int:
        """Index a dataset. Returns number of documents indexed."""

    def find_duplicates(self, candidate_path: str, tokenizer_path: str) -> set[int]:
        """Find chunks in candidate that are duplicates of indexed data.
        Returns set of chunk indices to remove."""

    def deduplicate_pair(
        self,
        primary_path: str,
        secondary_path: str,
        tokenizer_path: str,
        output_path: str,
    ) -> DeduplicationResult:
        """Remove chunks from secondary that duplicate primary.
        Primary dataset is kept intact."""

    def deduplicate_multi(
        self,
        datasets: list[str],
        tokenizer_path: str,
        output_dir: str,
    ) -> list[DeduplicationResult]:
        """Cross-deduplicate across multiple datasets.
        Priority: first dataset wins (later datasets have dupes removed)."""
```

### Exact Deduplication (Fast Mode)

```python
class ExactDeduplicator:
    """Hash-based exact duplicate removal.

    Much faster than MinHash (~100x) but only catches exact copies.
    Good as a first pass before MinHash.

    Uses SHA-256 of normalized content (whitespace-stripped).
    Memory: 32 bytes per document hash.
    10M documents = ~320MB RAM.
    """

    def hash_chunk(self, tokens: np.ndarray) -> str:
        """SHA-256 hash of token array."""
        return hashlib.sha256(tokens.tobytes()).hexdigest()
```

## CLI: combine_datasets.py

```python
# scripts/combine_datasets.py

"""Interactive dataset combination tool.

Usage:
    python scripts/combine_datasets.py --tokenizer tokenizer.json

Menu flow:
    1. Scan data/processed/ for .npy files
    2. Multi-select which datasets to combine
    3. Choose mixing strategy (concat/interleave/weighted)
    4. Optional: set weights per dataset
    5. Optional: cross-deduplicate
    6. Choose output name
    7. Run combination + save manifest
"""
```

### CLI Menu Flow

```
Step 1/4 · Select Datasets
  [✓] train_ts_js_500M.npy          (500M tokens, 2 days ago)
  [✓] github_elite_repos.npy        (120M tokens, 1 hour ago)
  [ ] train_python_200M.npy         (200M tokens, 5 days ago)
  [✓] synthetic_curriculum.npy      (50M tokens, 3 hours ago)

  3 selected — press Enter to confirm

Step 2/4 · Mixing Strategy
  [1] Interleave  — Round-robin chunks for best mixing (recommended)
  [2] Weighted    — Random sampling by weight
  [3] Concatenate — Append in order (for curriculum learning)

Step 3/4 · Weights
  train_ts_js_500M.npy       [0.7] ████████████████████░░░░░░░░░
  github_elite_repos.npy     [0.2] ██████░░░░░░░░░░░░░░░░░░░░░░░
  synthetic_curriculum.npy   [0.1] ███░░░░░░░░░░░░░░░░░░░░░░░░░░

  (arrow keys to adjust, Enter to confirm)

Step 4/4 · Deduplication
  [1] None        — Skip dedup (fastest)
  [2] Exact       — Remove exact duplicates only (~10 sec)
  [3] Near-dedup  — MinHash near-duplicate removal (~5 min) (recommended)

Summary
  Datasets:  3 files (670M tokens total)
  Strategy:  Interleave (weighted)
  Dedup:     MinHash (threshold=0.8)
  Output:    combined_ts_js_github_synth.npy

  Press Enter to start...
```

## Manifest for Combined Datasets

```yaml
version: "1.0"
created: "2026-03-20T15:00:00Z"
tool: "cola-coder/combine_datasets.py"

combination:
  strategy: "interleave"
  shuffle: true
  seed: 42

  sources:
    - name: "train_ts_js_500M"
      path: "data/processed/train_ts_js_500M.npy"
      weight: 0.7
      chunks_contributed: 170800
      manifest: "data/processed/train_ts_js_500M.manifest.yaml"

    - name: "github_elite_repos"
      path: "data/processed/github_elite_repos.npy"
      weight: 0.2
      chunks_contributed: 48800
      manifest: "data/processed/github_elite_repos.manifest.yaml"

    - name: "synthetic_curriculum"
      path: "data/processed/synthetic_curriculum.npy"
      weight: 0.1
      chunks_contributed: 24400
      manifest: "data/processed/synthetic_curriculum.manifest.yaml"

  deduplication:
    method: "minhash"
    threshold: 0.8
    duplicates_removed: 3200
    dedup_rate: 0.013

output:
  file: "combined_ts_js_github_synth.npy"
  total_chunks: 244000
  total_tokens: 499712000
  chunk_size: 2048
```

## Dependencies

```
datasketch>=1.6.0     # MinHash LSH (already in plan 04)
```

## Implementation Files

```
src/cola_coder/data/
  combine.py         # DatasetCombiner class
  dedup.py           # CrossDatasetDeduplicator, ExactDeduplicator
scripts/
  combine_datasets.py  # Interactive CLI tool
```
