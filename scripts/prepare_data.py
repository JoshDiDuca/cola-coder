"""Download and preprocess training data.

Streams code data from HuggingFace, tokenizes it with a trained BPE tokenizer,
and saves chunked token arrays ready for training.

Usage:
    python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json
    python scripts/prepare_data.py --tokenizer tokenizer.json --max-tokens 1000000
    python scripts/prepare_data.py --tokenizer tokenizer.json --workers 8  # parallel
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def _format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _scan_datasets(output_dir: str) -> list[dict]:
    """Scan for existing .npy dataset files and return metadata."""
    out_path = Path(output_dir)
    if not out_path.exists():
        return []

    datasets = []
    for f in sorted(out_path.glob("*.npy")):
        if f.name.endswith("_tmp.npy"):
            continue  # Skip temp files from in-progress runs
        stat = f.stat()
        # Read shape from npy header without loading into memory
        try:
            arr = np.load(str(f), mmap_mode="r")
            chunks, seq_len = arr.shape
            token_count = chunks * seq_len
            detail = f"{chunks:,} chunks x {seq_len} tokens = {token_count:,} total"
        except Exception:
            detail = "unknown format"

        datasets.append({
            "name": f.stem,
            "path": str(f),
            "size": _format_size(stat.st_size),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "detail": detail,
        })
    return datasets


def _auto_name(output_dir: str, languages: list[str], max_tokens: int | None) -> str:
    """Generate an auto-name like train_ts_js_500M."""
    lang_tag = "_".join(lang[:2] for lang in languages)  # typescript -> ts
    if max_tokens:
        if max_tokens >= 1_000_000_000:
            tok_tag = f"{max_tokens / 1e9:.0f}B"
        elif max_tokens >= 1_000_000:
            tok_tag = f"{max_tokens / 1e6:.0f}M"
        else:
            tok_tag = f"{max_tokens / 1e3:.0f}K"
        base = f"train_{lang_tag}_{tok_tag}"
    else:
        base = f"train_{lang_tag}_full"

    # Ensure unique: add _2, _3 etc if name already exists
    out_path = Path(output_dir)
    if not (out_path / f"{base}.npy").exists():
        return base
    i = 2
    while (out_path / f"{base}_{i}.npy").exists():
        i += 1
    return f"{base}_{i}"


def _resolve_output(
    output_dir: str,
    languages: list[str],
    max_tokens: int | None,
) -> str:
    """Interactive output file selection. Returns the output name (no .npy)."""
    existing = _scan_datasets(output_dir)

    if not existing:
        # No existing datasets — auto-name and go
        name = _auto_name(output_dir, languages, max_tokens)
        cli.info("Output", f"{name}.npy (new dataset)")
        return name

    # Show existing datasets
    cli.file_table("Existing Datasets", existing)

    options = [
        {"label": "Create new dataset", "detail": "Auto-named, won't touch existing data"},
    ]
    for ds in existing:
        options.append({
            "label": f"Overwrite {ds['name']}.npy",
            "detail": f"{ds['size']}  |  {ds['date']}",
        })

    choice = cli.choose("What would you like to do?", options, allow_cancel=True)

    if choice is None:
        cli.dim("Cancelled.")
        sys.exit(0)
    elif choice == 0:
        # Create new
        name = _auto_name(output_dir, languages, max_tokens)
        cli.info("Output", f"{name}.npy (new dataset)")
        return name
    else:
        # Overwrite existing
        ds = existing[choice - 1]
        if not cli.confirm(
            f"Overwrite {ds['name']}.npy ({ds['size']})? This cannot be undone."
        ):
            cli.dim("Cancelled.")
            sys.exit(0)
        cli.info("Output", f"{ds['name']}.npy (overwriting)")
        return ds["name"]


def main():
    storage = get_storage_config()
    storage.apply_hf_cache()

    parser = argparse.ArgumentParser(
        description="Download and preprocess code data for training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (used to read data/language settings).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=storage.tokenizer_path,
        help="Path to trained tokenizer.json file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to process (default: no limit).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(storage.data_dir) / "processed"),
        help="Output directory for processed data (default: ./data/processed).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (without .npy). If not set, interactive chooser is shown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for quality filtering. "
             "Default: all CPU cores (up to 16). Use 1 for sequential.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of files to tokenize at once (default: 256). Higher = faster "
             "but more memory. The Rust tokenizer parallelizes within each batch.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use slow HTTP streaming instead of bulk download. Only use this "
             "if you can't fit the dataset on disk (~50GB for 3 languages). "
             "Default: download to cache first, then process locally (MUCH faster).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Override languages (e.g. --languages typescript javascript). "
             "Also makes the filter language-aware (skips irrelevant syntax checks).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Compute quality scores for each chunk and save as a .weights.npy "
             "sidecar file. When this file exists, training automatically uses "
             "quality-weighted loss (high-quality code contributes more). "
             "Adds ~30%% to preprocessing time.",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable quality filtering (use raw data as-is).",
    )
    filter_group.add_argument(
        "--filter-strict",
        action="store_true",
        help="Use strict quality filtering (rejects mediocre code, not just bad code). "
             "Keeps only clearly high-quality files. Typical rejection rate: 60-75%%.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Data Preparation")

    # Default workers: all CPU cores, capped at 16
    if args.workers is None:
        args.workers = max(1, min(os.cpu_count() or 4, 16))

    # ---- Validate inputs ----
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        cli.fatal(
            f"Tokenizer file not found: {tokenizer_path}",
            hint="Train a tokenizer first with: python scripts/train_tokenizer.py",
        )

    # ---- Load config if provided ----
    dataset_name = "bigcode/starcoderdata"
    languages = ["typescript", "javascript"]
    max_seq_len = 2048

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            cli.fatal(f"Config file not found: {config_path}")

        cli.dim(f"Loading config from {config_path}...")
        try:
            from cola_coder.model.config import Config
            config = Config.from_yaml(str(config_path))
            dataset_name = config.data.dataset
            languages = config.data.languages
            max_seq_len = config.model.max_seq_len
            cli.info("Dataset", dataset_name)
            cli.info("Languages", ", ".join(languages))
            cli.info("Chunk size", max_seq_len)
        except Exception as e:
            cli.fatal(f"Error loading config: {e}")

    # CLI --languages overrides config
    if args.languages:
        languages = args.languages
        cli.info("Languages (override)", ", ".join(languages))

    # ---- Resolve output file ----
    if args.output_name:
        output_name = args.output_name
        cli.info("Output", f"{output_name}.npy")
    else:
        output_name = _resolve_output(args.output_dir, languages, args.max_tokens)

    # ---- Step 1: Load tokenizer ----
    cli.step(1, 2, "Loading tokenizer")
    cli.dim(f"Source: {tokenizer_path}")

    try:
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        tokenizer = CodeTokenizer(str(tokenizer_path))
        cli.info("Vocabulary size", tokenizer.vocab_size)
    except Exception as e:
        cli.fatal(f"Error loading tokenizer: {e}")

    # ---- Step 2: Load and preprocess data ----
    cli.step(2, 2, "Processing data")
    if args.stream:
        cli.dim(f"Streaming data from {dataset_name} (slow HTTP mode)...")
    else:
        cli.dim(f"Downloading data from {dataset_name} (bulk download -> local processing)...")
    if args.max_tokens:
        cli.info("Token limit", f"{args.max_tokens:,}")

    try:
        from cola_coder.data.download import stream_code_data
        from cola_coder.data.preprocess import tokenize_and_chunk
        from cola_coder.data.quality_filter import (
            filtered_stream, parallel_filtered_stream,
            FilterStats, FilterMode,
        )
    except ImportError:
        cli.fatal("Could not import data modules.")

    try:
        data_stream = stream_code_data(
            dataset_name=dataset_name,
            languages=languages,
            streaming=args.stream,
        )

        # Apply quality filtering (parallel when workers > 1)
        if args.no_filter:
            cli.warn("Quality filtering: DISABLED")
        elif args.filter_strict:
            cli.info("Quality filtering", "STRICT (only keeping high-quality code)")
            cli.info("Expected rejection rate", "60-75%")
            cli.info("Workers", args.workers)
            stats = FilterStats()
            if args.workers > 1:
                data_stream = parallel_filtered_stream(
                    data_stream, mode=FilterMode.STRICT,
                    stats=stats, num_workers=args.workers,
                    languages=languages,
                )
            else:
                data_stream = filtered_stream(
                    data_stream, mode=FilterMode.STRICT, stats=stats,
                    languages=languages,
                )
        else:
            cli.info("Quality filtering", "CONSERVATIVE (use --filter-strict for stricter)")
            cli.info("Workers", args.workers)
            stats = FilterStats()
            if args.workers > 1:
                data_stream = parallel_filtered_stream(
                    data_stream, mode=FilterMode.CONSERVATIVE,
                    stats=stats, num_workers=args.workers,
                    languages=languages,
                )
            else:
                data_stream = filtered_stream(
                    data_stream, mode=FilterMode.CONSERVATIVE, stats=stats,
                    languages=languages,
                )

        cli.dim(f"Batch size: {args.batch_size}  |  Tokenization: producer-consumer + Rust batch")

        output_file = tokenize_and_chunk(
            text_iterator=data_stream,
            tokenizer=tokenizer,
            chunk_size=max_seq_len,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            output_name=output_name,
        )
    except KeyboardInterrupt:
        cli.warn("Interrupted by user. Partial data may have been saved.")
        sys.exit(1)
    except Exception as e:
        cli.fatal(f"Error processing data: {e}")

    # ---- Quality scoring (optional) ----
    if args.score:
        cli.rule("Quality Scoring")
        cli.info("Scoring mode", "Computing quality scores for weighted training")
        try:
            from cola_coder.features.code_scorer import is_enabled as scorer_enabled
            if scorer_enabled():
                from cola_coder.features.code_scorer import CodeScorer
                scorer = CodeScorer()

                # Load the data we just saved and score each chunk
                data = np.load(output_file, mmap_mode="r")
                num_chunks = data.shape[0]
                weights = np.zeros(num_chunks, dtype=np.float32)

                cli.info("Chunks to score", f"{num_chunks:,}")

                # We need the tokenizer to decode chunks back to text for scoring
                from tqdm import tqdm as scoring_tqdm
                for i in scoring_tqdm(range(num_chunks), desc="Scoring"):
                    # Decode chunk back to text
                    chunk_ids = data[i].tolist()
                    text = tokenizer.decode(chunk_ids)
                    # Score and convert to training weight
                    result = scorer.score(text)
                    weights[i] = scorer.score_to_weight(result)

                # Save weights file
                weights_path = str(Path(output_file).with_suffix(".weights.npy"))
                np.save(weights_path, weights)

                # Stats
                mean_weight = weights.mean()
                excellent = (weights >= 1.8).sum()
                good = ((weights >= 1.3) & (weights < 1.8)).sum()
                average = ((weights >= 0.8) & (weights < 1.3)).sum()
                poor = ((weights >= 0.1) & (weights < 0.8)).sum()
                reject = (weights < 0.1).sum()

                cli.kv_table({
                    "Mean weight": f"{mean_weight:.3f}",
                    "Excellent (2.0x)": f"{excellent:,} ({excellent/num_chunks*100:.1f}%)",
                    "Good (1.5x)": f"{good:,} ({good/num_chunks*100:.1f}%)",
                    "Average (1.0x)": f"{average:,} ({average/num_chunks*100:.1f}%)",
                    "Poor (0.3x)": f"{poor:,} ({poor/num_chunks*100:.1f}%)",
                    "Reject (0.0x)": f"{reject:,} ({reject/num_chunks*100:.1f}%)",
                }, title="Quality Score Distribution")

                cli.success(f"Quality weights saved to {weights_path}")
            else:
                cli.warn("Code scorer feature is disabled. Enable in configs/features.yaml")
        except ImportError:
            cli.warn("Code scorer feature not available. Skipping quality scoring.")
        except Exception as e:
            cli.warn(f"Quality scoring failed: {e}. Training will use uniform weights.")

    # ---- Summary ----
    extras = {
        "Output": str(Path(output_file).resolve()),
        "Next step": f"python scripts/train.py --data {output_file}",
    }
    if args.score:
        weights_path = str(Path(output_file).with_suffix(".weights.npy"))
        if Path(weights_path).exists():
            extras["Weights"] = weights_path
            extras["Note"] = "Training will auto-detect weights and use quality-weighted loss"
    cli.done("Data preprocessing complete!", extras=extras)


if __name__ == "__main__":
    main()
