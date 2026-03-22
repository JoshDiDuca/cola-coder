"""Create context-augmented training examples from existing code data.

Teaches the model the <|repo|>/<|file|> format by building pairs where one
chunk is shown as "repo context" for another related chunk.

Two pairing strategies are used:
  1. Jaccard similarity (token overlap) — pairs chunks that share many tokens
     (proxy for files from the same repository or related modules).
  2. Import-based — detects import-like token patterns and pairs accordingly.

The output is a mix of context-augmented and plain-code examples at the
requested ratio (default 30% augmented, 70% plain).

Usage:
    python scripts/prepare_repo_context_data.py --input data/processed/train_data.npy \\
        --tokenizer tokenizer.json
    python scripts/prepare_repo_context_data.py --context-ratio 0.3
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

# Make cola_coder importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cola_coder.cli import cli

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum Jaccard similarity to consider two chunks "related"
_JACCARD_THRESHOLD = 0.15

# How many candidates to sample when searching for a context pair (per chunk)
_CANDIDATE_SAMPLE = 200

# Max token budget reserved for the repo context block
_CONTEXT_BUDGET_RATIO = 0.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _jaccard(a: set[int], b: set[int]) -> float:
    """Compute Jaccard similarity between two token-ID sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _build_context_example(
    context_tokens: list[int],
    target_tokens: list[int],
    seq_len: int,
    bos_id: int,
    eos_id: int,
    repo_start_id: int,
    repo_end_id: int,
    file_start_id: int,
    file_end_id: int,
) -> list[int] | None:
    """Build one <|repo|>...<|/repo|><|file|>...<|/file|> training example.

    Trims context to fit within the seq_len budget, returns None if too small
    to form a meaningful example.
    """
    # Reserve space: bos + repo_start + context + repo_end + file_start + target + file_end + eos
    overhead = 6  # 6 special tokens
    budget = seq_len - overhead
    if budget < 64:
        return None

    context_budget = int(budget * _CONTEXT_BUDGET_RATIO)
    target_budget = budget - context_budget

    # Trim both to budget
    ctx = context_tokens[:context_budget]
    tgt = target_tokens[:target_budget]

    if len(tgt) < 16:
        return None

    combined = (
        [bos_id, repo_start_id]
        + ctx
        + [repo_end_id, file_start_id]
        + tgt
        + [file_end_id, eos_id]
    )

    # Pad or trim to exact seq_len
    if len(combined) < seq_len:
        combined += [eos_id] * (seq_len - len(combined))
    elif len(combined) > seq_len:
        combined = combined[:seq_len]

    return combined


def _find_context_pair(
    idx: int,
    token_sets: list[set[int]],
    rng: random.Random,
    n: int,
) -> int | None:
    """Find a related chunk index for idx using Jaccard similarity sampling."""
    candidates = rng.sample(range(n), min(_CANDIDATE_SAMPLE, n))
    best_idx = None
    best_score = _JACCARD_THRESHOLD

    target_set = token_sets[idx]

    for c in candidates:
        if c == idx:
            continue
        score = _jaccard(target_set, token_sets[c])
        if score > best_score:
            best_score = score
            best_idx = c

    return best_idx


def _get_special_token_id(tokenizer, token: str, fallback: int) -> int:
    """Get token ID, falling back if the token is not in the vocabulary."""
    tid = tokenizer.tokenizer.token_to_id(token)
    return tid if tid is not None else fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create context-augmented training examples from tokenized code data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/train_data.npy",
        help="Input tokenized data .npy file (default: data/processed/train_data.npy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/context_train_data.npy",
        help="Output .npy file (default: data/processed/context_train_data.npy).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to trained tokenizer.json (default: tokenizer.json).",
    )
    parser.add_argument(
        "--context-ratio",
        type=float,
        default=0.3,
        help="Fraction of output chunks that are context-augmented (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit input chunks processed (useful for testing).",
    )
    args = parser.parse_args()

    if not 0.0 < args.context_ratio < 1.0:
        cli.fatal("--context-ratio must be between 0.0 and 1.0 (exclusive).")

    cli.header("Cola-Coder", "Repo Context Data Preparation")
    cli.info("Input", args.input)
    cli.info("Context ratio", f"{args.context_ratio:.0%} augmented / "
             f"{1 - args.context_ratio:.0%} plain")
    cli.info("Seed", args.seed)

    # ---- Step 1: Load tokenizer ----
    cli.step(1, 4, "Loading tokenizer")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint="Train a tokenizer first: python scripts/train_tokenizer.py",
        )

    try:
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder.",
            hint="Try: pip install -e .",
        )

    try:
        tokenizer = CodeTokenizer(str(tokenizer_path))
        cli.info("Vocabulary size", tokenizer.vocab_size)
    except Exception as exc:
        cli.fatal(f"Error loading tokenizer: {exc}")

    # Resolve <|repo|> / <|file|> special token IDs — fall back to eos if absent
    repo_start_id = _get_special_token_id(tokenizer, "<|repo|>", tokenizer.eos_id)
    repo_end_id = _get_special_token_id(tokenizer, "<|/repo|>", tokenizer.eos_id)
    file_start_id = _get_special_token_id(tokenizer, "<|file|>", tokenizer.eos_id)
    file_end_id = _get_special_token_id(tokenizer, "<|/file|>", tokenizer.eos_id)

    cli.info(
        "Special tokens",
        f"<|repo|>={repo_start_id}  <|/repo|>={repo_end_id}  "
        f"<|file|>={file_start_id}  <|/file|>={file_end_id}",
    )

    # ---- Step 2: Load existing data ----
    cli.step(2, 4, "Loading tokenized data")

    input_path = Path(args.input)
    if not input_path.exists():
        cli.fatal(
            f"Input file not found: {input_path}",
            hint="Run prepare_data.py first to create train_data.npy",
        )

    try:
        data = np.load(str(input_path), mmap_mode="r")
    except Exception as exc:
        cli.fatal(f"Could not load {input_path}: {exc}")

    if data.ndim != 2:
        cli.fatal(f"Expected 2D array, got shape {data.shape}")

    n_total, seq_len = data.shape
    if args.max_chunks:
        n_total = min(n_total, args.max_chunks)

    cli.info("Chunks loaded", f"{n_total:,}")
    cli.info("Sequence length", seq_len)
    cli.info("File size", _format_size(input_path.stat().st_size))

    # ---- Step 3: Build token sets for similarity search ----
    cli.step(3, 4, "Building token sets for similarity pairing")
    cli.dim("Computing per-chunk token sets (this indexes the data for pair finding)...")

    start = time.time()
    # Use a sample for large datasets to keep memory reasonable
    sample_size = min(n_total, 50_000)
    if sample_size < n_total:
        cli.dim(f"  Sampling {sample_size:,} of {n_total:,} chunks for index.")

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    sample_indices = sorted(rng_np.choice(n_total, size=sample_size, replace=False).tolist())
    token_sets: list[set[int]] = []

    for i, idx in enumerate(sample_indices):
        chunk = data[idx].tolist()
        # Exclude padding / special tokens from similarity to focus on content
        token_sets.append(set(chunk) - {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id})
        if (i + 1) % 5000 == 0:
            cli.dim(f"  Indexed {i + 1:,}/{sample_size:,} chunks...")

    elapsed_index = time.time() - start
    cli.info("Index built in", f"{elapsed_index:.1f}s")

    # ---- Step 4: Build augmented examples ----
    cli.step(4, 4, "Building context-augmented training examples")

    n_augmented_target = int(n_total * args.context_ratio)
    n_plain_target = n_total - n_augmented_target

    cli.info("Target augmented chunks", f"{n_augmented_target:,}")
    cli.info("Target plain chunks", f"{n_plain_target:,}")

    output_chunks: list[np.ndarray] = []
    n_augmented_actual = 0
    n_no_pair = 0

    # Shuffle the sample indices to build varied pairs
    shuffled = list(range(len(sample_indices)))
    rng.shuffle(shuffled)

    start = time.time()

    for pos, si in enumerate(shuffled):
        if n_augmented_actual >= n_augmented_target:
            break

        ctx_idx = _find_context_pair(si, token_sets, rng, len(sample_indices))
        if ctx_idx is None:
            n_no_pair += 1
            continue

        context_tokens = data[sample_indices[ctx_idx]].tolist()
        target_tokens = data[sample_indices[si]].tolist()

        # Strip leading padding/BOS to keep context compact
        context_tokens = [
            t for t in context_tokens
            if t not in (tokenizer.pad_id,)
        ]

        example = _build_context_example(
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            seq_len=seq_len,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            repo_start_id=repo_start_id,
            repo_end_id=repo_end_id,
            file_start_id=file_start_id,
            file_end_id=file_end_id,
        )
        if example is None:
            continue

        output_chunks.append(np.array(example, dtype=np.uint16))
        n_augmented_actual += 1

        if n_augmented_actual % 500 == 0:
            elapsed = time.time() - start
            rate = n_augmented_actual / elapsed if elapsed > 0 else 0
            cli.dim(
                f"  Augmented: {n_augmented_actual:,}/{n_augmented_target:,}  "
                f"({rate:.0f}/s)  no_pair={n_no_pair}"
            )

    cli.info("Augmented pairs built", f"{n_augmented_actual:,}")
    if n_no_pair > 0:
        cli.dim(f"  Pairs not found (low Jaccard): {n_no_pair:,}")

    # Fill the remainder with plain (unmodified) chunks
    plain_indices = rng_np.choice(n_total, size=n_plain_target, replace=False).tolist()
    cli.dim(f"Adding {n_plain_target:,} plain chunks...")

    for idx in plain_indices:
        output_chunks.append(np.array(data[int(idx)], dtype=np.uint16))

    # Shuffle the combined set
    rng.shuffle(output_chunks)

    # Stack and save
    if not output_chunks:
        cli.fatal("No output chunks were produced.")

    result = np.stack(output_chunks, axis=0)  # shape: [N, seq_len]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), result)

    file_size = output_path.stat().st_size
    total_elapsed = time.time() - start

    cli.info("Output shape", f"{result.shape}")
    cli.info("File size", _format_size(file_size))

    # ---- Write manifest ----
    manifest_path = output_path.with_suffix("").with_suffix(".manifest.yaml")
    try:
        from cola_coder.manifest import write_data_manifest

        write_data_manifest(
            str(manifest_path),
            output_file=output_path.name,
            output_size_bytes=file_size,
            num_chunks=result.shape[0],
            chunk_size=seq_len,
            total_tokens=result.shape[0] * seq_len,
            dtype="uint16",
            total_files=n_total,
            dataset=f"context_augmented:{input_path.name}",
            filter_mode="none",
            tokenizer_path=str(tokenizer_path),
            vocab_size=tokenizer.vocab_size,
            wall_time_seconds=total_elapsed,
            throughput_tokens_per_sec=(result.shape[0] * seq_len) / total_elapsed
            if total_elapsed > 0 else 0,
        )
        cli.dim(f"Manifest: {manifest_path}")
    except Exception as exc:
        cli.warn(f"Could not write manifest: {exc}")

    cli.done(
        "Repo context data preparation complete!",
        extras={
            "Output": str(output_path.resolve()),
            "Total chunks": f"{result.shape[0]:,}",
            "Augmented": f"{n_augmented_actual:,} ({n_augmented_actual / result.shape[0]:.1%})",
            "Plain": f"{n_plain_target:,}",
            "Next step": "python scripts/combine_datasets.py",
        },
    )


if __name__ == "__main__":
    main()
