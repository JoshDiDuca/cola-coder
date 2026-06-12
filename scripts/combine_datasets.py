"""Interactive dataset combination tool for Cola-Coder.

Walks you through combining multiple .npy datasets into one training file:
  1. Scan data/processed/ for .npy files, show multi-select
  2. Choose mixing strategy (interleave, weighted, concat)
  3. Set weights per dataset (if weighted/interleave)
  4. Choose dedup method (none, exact, minhash)
  5. Show summary, confirm
  6. Run combination

Usage:
    python scripts/combine_datasets.py
    python scripts/combine_datasets.py --data-dir ./data/processed
    python scripts/combine_datasets.py --tokenizer tokenizer.json
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    elif nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    elif nbytes >= 1e3:
        return f"{nbytes / 1e3:.1f} KB"
    return f"{nbytes} B"


def _format_tokens(n: int) -> str:
    """Human-readable token count."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B tokens"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M tokens"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K tokens"
    return f"{n} tokens"


def scan_datasets(data_dir: str) -> list[dict]:
    """Scan a directory for .npy dataset files and return metadata.

    Returns list of dicts with: path, name, file_size, chunks, chunk_size,
    tokens, modified.
    """
    import numpy as np

    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    results = []
    for npy_file in sorted(data_path.glob("*.npy")):
        try:
            arr = np.load(str(npy_file), mmap_mode="r")
            if arr.ndim != 2:
                continue
            chunks, chunk_size = arr.shape
            tokens = chunks * chunk_size
            stat = npy_file.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            age = datetime.now() - modified
            if age.days > 0:
                age_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                age_str = f"{age.seconds // 3600}h ago"
            else:
                age_str = f"{age.seconds // 60}m ago"

            results.append({
                "path": str(npy_file),
                "name": npy_file.stem,
                "file_size": stat.st_size,
                "file_size_str": _format_size(stat.st_size),
                "chunks": chunks,
                "chunk_size": chunk_size,
                "tokens": tokens,
                "tokens_str": _format_tokens(tokens),
                "modified": modified,
                "age_str": age_str,
            })
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Menu definitions
# ---------------------------------------------------------------------------

STRATEGY_OPTIONS = [
    {
        "label": "Interleave",
        "detail": "Round-robin chunks for best mixing",
        "value": "interleave",
        "recommended": True,
    },
    {
        "label": "Weighted",
        "detail": "Random sampling by weight",
        "value": "weighted",
    },
    {
        "label": "Concatenate",
        "detail": "Append in order (for curriculum learning)",
        "value": "concat",
    },
]

DEDUP_OPTIONS = [
    {
        "label": "None",
        "detail": "Skip deduplication (fastest)",
        "value": "none",
    },
    {
        "label": "Exact",
        "detail": "Remove exact duplicate chunks only (~10 sec)",
        "value": "exact",
    },
    {
        "label": "Near-dedup (MinHash)",
        "detail": "MinHash near-duplicate removal (requires datasketch)",
        "value": "minhash",
        "recommended": True,
    },
]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_menu(data_dir: str, tokenizer_path: str | None = None) -> dict:
    """Run the interactive menu and return settings."""

    # Step 1: Scan for datasets
    datasets = scan_datasets(data_dir)
    if not datasets:
        cli.header("Cola-Coder", "Dataset Combiner")
        cli.print()
        cli.print(f"  [bold red]No .npy datasets found in {data_dir}[/bold red]")
        cli.print("  [dim]Run prepare_data.py first to create training data.[/dim]")
        sys.exit(1)

    # Build options for multi-select
    ds_options = []
    for ds in datasets:
        ds_options.append({
            "label": ds["name"],
            "detail": f"{ds['tokens_str']} \u2022 {ds['file_size_str']} \u2022 {ds['age_str']}",
        })

    selected_indices = cli.multi_select(
        "Select datasets to combine:",
        ds_options,
        preselected=list(range(len(datasets))),  # All preselected
    )
    selected_datasets = [datasets[i] for i in selected_indices]

    # Step 2: Mixing strategy
    strategy_idx = cli.choose(
        "How should datasets be mixed?",
        STRATEGY_OPTIONS,
    )
    strategy = STRATEGY_OPTIONS[strategy_idx]["value"]

    # Step 3: Weights
    if strategy in ("interleave", "weighted"):
        weight_ds = [
            {"label": ds["name"], "detail": ds["tokens_str"]}
            for ds in selected_datasets
        ]
        weights = cli.weight_editor(weight_ds, title="Set weights for each dataset")
    else:
        weights = [1.0 / len(selected_datasets)] * len(selected_datasets)

    # Step 4: Dedup method
    dedup_idx = cli.choose(
        "Deduplication method:",
        DEDUP_OPTIONS,
    )
    dedup_method = DEDUP_OPTIONS[dedup_idx]["value"]

    return {
        "datasets": selected_datasets,
        "strategy": strategy,
        "weights": weights,
        "dedup_method": dedup_method,
        "tokenizer_path": tokenizer_path,
    }


def show_summary(settings: dict, output_path: str) -> bool:
    """Show summary and ask for confirmation."""
    cli.header("Cola-Coder", "Dataset Combiner")
    cli.print()

    # Datasets
    ds_names = [ds["name"] for ds in settings["datasets"]]
    total_tokens = sum(ds["tokens"] for ds in settings["datasets"])

    summary: dict[str, str] = {}
    summary["Datasets"] = f"{len(ds_names)} files ({_format_tokens(total_tokens)} total)"
    for i, ds in enumerate(settings["datasets"]):
        w = settings["weights"][i]
        summary[""] = f"  {ds['name']}  [{w:.0%}]  ({ds['tokens_str']})"

    # Strategy
    summary["Strategy"] = settings["strategy"].capitalize()

    # Dedup
    dedup = settings["dedup_method"]
    if dedup == "none":
        summary["Dedup"] = "None (skip)"
    elif dedup == "exact":
        summary["Dedup"] = "Exact (hash-based)"
    else:
        summary["Dedup"] = "MinHash (near-duplicate)"

    # Output
    summary["Output"] = output_path

    cli.kv_table(summary, title="Summary")
    cli.print()

    return cli.confirm("Start combination?", default=True)


def run_pipeline(settings: dict, output_path: str):
    """Execute the combination pipeline."""
    from cola_coder.data.combine import DatasetCombiner, DatasetInput
    from cola_coder.data.dedup import CrossDatasetDeduplicator

    cli.print()
    cli.print("[bold cyan]Starting dataset combination...[/bold cyan]")
    cli.print()

    datasets = settings["datasets"]
    dedup_method = settings["dedup_method"]
    dedup_removed = 0

    # Step 1: Optional dedup
    paths_to_combine = [ds["path"] for ds in datasets]

    if dedup_method != "none" and len(datasets) > 1:
        cli.print("[dim]Running deduplication...[/dim]")
        t0 = time.time()

        # Both "exact" and "minhash" dedup ACROSS datasets: each secondary is
        # deduped against the primary (kept intact), via CrossDatasetDeduplicator.
        # The earlier "exact" path called ExactDeduplicator.deduplicate_array
        # per-dataset, which only removed WITHIN-dataset dupes (already handled by
        # prepare_data's exact dedup) and left cross-dataset duplicates — exactly
        # what dedup-at-combine is supposed to remove (BUG-104).
        cross_dedup = CrossDatasetDeduplicator(method=dedup_method, threshold=0.8)
        label = "exact" if dedup_method == "exact" else "near-"
        temp_paths = [datasets[0]["path"]]  # Primary kept as-is
        for i in range(1, len(datasets)):
            ds = datasets[i]
            temp_path = str(
                Path(output_path).parent / f"_temp_dedup_{ds['name']}.npy"
            )
            result = cross_dedup.deduplicate_pair(
                primary_path=datasets[0]["path"],
                secondary_path=ds["path"],
                tokenizer_path=settings.get("tokenizer_path"),
                output_path=temp_path,
            )
            dedup_removed += result.duplicates_removed
            temp_paths.append(temp_path)
            cli.print(
                f"  {ds['name']}: {result.duplicates_removed} "
                f"{label}duplicates removed"
            )

        paths_to_combine = temp_paths

        elapsed = time.time() - t0
        cli.print(
            f"  [green]Dedup complete: {dedup_removed} total removed "
            f"({elapsed:.1f}s)[/green]"
        )
        cli.print()

    # Step 2: Combine
    cli.print("[dim]Combining datasets...[/dim]")
    t0 = time.time()

    combiner = DatasetCombiner()
    ds_inputs = []
    for i, path in enumerate(paths_to_combine):
        ds_inputs.append(
            DatasetInput(
                path=path,
                weight=settings["weights"][i],
                name=datasets[i]["name"],
            )
        )

    result = combiner.combine(
        datasets=ds_inputs,
        strategy=settings["strategy"],
        output_path=output_path,
        shuffle=True,
        seed=42,
    )

    elapsed = time.time() - t0
    cli.print(f"  [green]Done in {elapsed:.1f}s[/green]")

    # Clean up temp files
    for path in paths_to_combine:
        if "_temp_dedup_" in path:
            try:
                os.remove(path)
            except OSError:
                pass

    # Per-source mix breakdown — lets the user confirm the realized ratio
    # matches the requested weights (e.g. a 70/20/10 code/text/math mix).
    if result.sources:
        cli.print("\n[bold]Realized mix[/bold] (contributed / requested):")
        for s in result.sources:
            cli.print(
                f"  {s['name']:<16} {s['chunks_contributed']:>9,} chunks  "
                f"{s['fraction'] * 100:5.1f}%  (requested {s['weight'] * 100:.0f}%)"
            )

    # Show result
    dedup_line = f"\nDedup:  {dedup_removed:,} duplicates removed" if dedup_removed else ""
    cli.done(
        "Dataset combination complete!",
        extras={
            "Output": str(Path(result.output_path).resolve()),
            "Chunks": f"{result.total_chunks:,}",
            "Tokens": f"{result.total_tokens:,} ({_format_tokens(result.total_tokens)})",
            **({"Dedup": f"{dedup_removed:,} duplicates removed"} if dedup_removed else {}),
            "Next step": "python scripts/train.py --config configs/tiny.yaml",
        },
    )
    _ = dedup_line  # suppress unused variable


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_weighted_datasets(dataset_args: list[str]) -> tuple[list[str], list[float]]:
    """Parse ``path:weight`` entries from --datasets CLI arg.

    Each element may be ``path`` (weight defaults to 1.0) or ``path:weight``
    (weight is a positive float).  Returns ``(paths, weights)`` where weights
    are the raw values before normalisation — callers are responsible for
    normalising if required.
    """
    paths: list[str] = []
    weights: list[float] = []
    for entry in dataset_args:
        if ":" in entry:
            # Split on the *last* colon so Windows absolute paths (C:\...) work.
            colon_pos = entry.rfind(":")
            path_part = entry[:colon_pos]
            weight_part = entry[colon_pos + 1:]
            try:
                w = float(weight_part)
                if w <= 0:
                    raise ValueError("weight must be positive")
            except ValueError:
                cli.print(
                    f"[yellow]Warning:[/yellow] invalid weight '{weight_part}' "
                    f"for '{path_part}' — defaulting to 1.0"
                )
                w = 1.0
            paths.append(path_part)
            weights.append(w)
        else:
            paths.append(entry)
            weights.append(1.0)
    return paths, weights


def _run_weighted_mix(paths: list[str], weights: list[float], output_path: str) -> None:
    """Load .npy datasets and mix them proportionally to their weights.

    Uses numpy random choice (with replacement) to sample rows from each
    dataset according to normalised weights, then concatenates and saves the
    result.  The total row count is the sum of all input row counts.
    """
    import numpy as np

    # Load and validate all datasets, keeping paths/weights/arrays in lockstep.
    # (The earlier code `continue`-skipped non-2D arrays, which desynced the
    # later zip(paths, arrays, norm_weights, ...) and applied each weight to the
    # wrong dataset.) Weights are renormalised over the datasets actually loaded.
    loaded_paths: list[str] = []
    loaded_weights: list[float] = []
    arrays = []
    chunk_size: int | None = None
    for p, w in zip(paths, weights):
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            cli.print(f"[red]Error:[/red] {p} is not a 2-D array — skipping")
            continue
        if chunk_size is None:
            chunk_size = arr.shape[1]
        elif arr.shape[1] != chunk_size:
            # All inputs must share chunk_size or np.concatenate fails opaquely
            # downstream — mirror DatasetCombiner.combine's explicit check.
            cli.print(
                f"[red]Error:[/red] chunk size mismatch: {p} has "
                f"{arr.shape[1]}, expected {chunk_size}. Aborting."
            )
            sys.exit(1)
        loaded_paths.append(p)
        loaded_weights.append(w)
        arrays.append(arr)

    if not arrays:
        cli.print("[red]No valid datasets loaded. Aborting.[/red]")
        sys.exit(1)

    total_weight = sum(loaded_weights)
    norm_weights = [w / total_weight for w in loaded_weights]

    # Total number of rows in the output == sum of all input row counts.
    total_rows = sum(a.shape[0] for a in arrays)

    # Compute how many rows to sample from each dataset (each contributes >=1).
    row_counts = [max(1, round(total_rows * w)) for w in norm_weights]
    # Reconcile to exactly total_rows. Rounding + the min-1 clamp can over- or
    # under-shoot; absorb the difference into the LARGEST bucket so it never
    # goes negative (the old code forced the LAST bucket, which underflowed to a
    # negative size — and a negative rng.choice size — when many tiny datasets
    # were each clamped up to 1).
    diff = total_rows - sum(row_counts)
    if diff != 0:
        j = max(range(len(row_counts)), key=lambda k: row_counts[k])
        row_counts[j] = max(1, row_counts[j] + diff)

    cli.print(f"\n  [bold]Weighted mixing:[/bold]  {total_rows:,} total rows")
    for p, arr, w, n in zip(loaded_paths, arrays, norm_weights, row_counts):
        cli.print(
            f"    {Path(p).stem:<30}  weight={w:.1%}  "
            f"sample {n:,} / {arr.shape[0]:,} rows"
        )

    rng = np.random.default_rng(seed=42)
    parts = []
    for arr, n in zip(arrays, row_counts):
        idx = rng.choice(arr.shape[0], size=n, replace=(n > arr.shape[0]))
        parts.append(np.array(arr[idx]))

    mixed = np.concatenate(parts, axis=0)
    # Shuffle the concatenated result.
    shuffle_idx = rng.permutation(mixed.shape[0])
    mixed = mixed[shuffle_idx]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), mixed)

    cli.print(
        f"\n  [green]Done.[/green]  Saved {mixed.shape[0]:,} x {mixed.shape[1]} "
        f"to [cyan]{out.resolve()}[/cyan]"
    )


def main():
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Interactive dataset combination tool for Cola-Coder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Model config YAML. When given, resolves --data-dir via DatasetResolver.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing .npy dataset files (default: per-dataset dir from DatasetResolver).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (optional, for MinHash text decoding).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for combined dataset (default: auto-generated).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="PATH[:WEIGHT]",
        help=(
            "Non-interactive weighted mix.  Provide two or more dataset paths, "
            "each optionally suffixed with :weight (e.g. data/code.npy:0.8 "
            "data/docs.npy:0.2).  Weights are normalised automatically.  "
            "Bypasses the interactive TUI."
        ),
    )
    args = parser.parse_args()

    if args.data_dir is None:
        if args.config is not None:
            from cola_coder.data.dataset_resolver import DatasetResolver
            args.data_dir = str(DatasetResolver.get_dataset_dir(config_path=args.config))
        else:
            args.data_dir = str(Path(storage.data_dir) / "processed")

    # ── Non-interactive weighted mix (--datasets supplied) ────────────────
    if args.datasets:
        if len(args.datasets) < 1:
            cli.print("[red]Error:[/red] --datasets requires at least one path.")
            sys.exit(1)

        paths, weights = _parse_weighted_datasets(args.datasets)

        # Validate files exist
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            for m in missing:
                cli.print(f"[red]Error:[/red] file not found: {m}")
            sys.exit(1)

        # Auto-generate output path if not provided
        if args.output:
            output_path = args.output
        else:
            stems = [Path(p).stem for p in paths]
            combined_name = "_".join(stems) if len(stems) <= 3 else f"combined_{len(stems)}ds"
            output_path = str(Path(args.data_dir) / f"{combined_name}_weighted.npy")

        cli.header("Cola-Coder", "Dataset Combiner")
        _run_weighted_mix(paths, weights, output_path)
        return

    # ── Interactive TUI flow ───────────────────────────────────────────────
    try:
        settings = run_menu(args.data_dir, args.tokenizer)

        # Generate output path
        if args.output:
            output_path = args.output
        else:
            names = [ds["name"] for ds in settings["datasets"]]
            if len(names) <= 3:
                combined_name = "_".join(names)
            else:
                combined_name = f"combined_{len(names)}ds"
            output_path = str(
                Path(args.data_dir) / f"{combined_name}_combined.npy"
            )

        if show_summary(settings, output_path):
            run_pipeline(settings, output_path)
        else:
            cli.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
