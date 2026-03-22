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

from cola_coder.model.config import get_storage_config

# ---------------------------------------------------------------------------
# Menu UI helpers using rich (matching prepare_data_interactive.py style)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("Error: 'rich' is required. Install with: pip install rich")
    sys.exit(1)

try:
    from cola_coder.cli import cli as _cli
except ImportError:
    _cli = None

console = Console()

# ANSI escape for reading raw keypresses on Windows
if sys.platform == "win32":
    import msvcrt

    def _read_key() -> str:
        """Read a single keypress on Windows. Returns arrow names or characters."""
        key = msvcrt.getwch()
        if key == "\xe0" or key == "\x00":  # Arrow/special key prefix
            key2 = msvcrt.getwch()
            return {
                "H": "up",
                "P": "down",
                "M": "right",
                "K": "left",
            }.get(key2, "")
        if key == "\r":
            return "enter"
        if key == "\x1b":
            return "escape"
        if key == " ":
            return "space"
        return key
else:
    import tty
    import termios

    def _read_key() -> str:
        """Read a single keypress on Unix."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                        ch3, ""
                    )
            if ch == "\r" or ch == "\n":
                return "enter"
            if ch == " ":
                return "space"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_menu(
    title: str,
    options: list[dict],
    step: str = "",
) -> int:
    """Show an interactive arrow-key menu and return the selected index."""
    selected = 0
    for i, opt in enumerate(options):
        if opt.get("recommended"):
            selected = i
            break

    while True:
        console.clear()
        _draw_header()

        console.print()
        console.print(f"  [bold cyan]{step}[/bold cyan]" if step else "")
        console.print(f"  [bold]{title}[/bold]")
        console.print(
            "  [dim]Use [bold]arrow keys[/bold] to navigate, "
            "[bold]Enter[/bold] to select[/dim]"
        )
        console.print()

        for i, opt in enumerate(options):
            is_selected = i == selected
            prefix = "  [bold green]>[/bold green] " if is_selected else "    "
            label = opt["label"]
            desc = opt.get("description", "")
            rec = "  [yellow](recommended)[/yellow]" if opt.get("recommended") else ""

            if is_selected:
                console.print(
                    f"{prefix}[bold white on blue] {label} [/bold white on blue]"
                    f"  [dim]{desc}[/dim]{rec}"
                )
            else:
                console.print(
                    f"{prefix}[white]{label}[/white]"
                    f"  [dim]{desc}[/dim]{rec}"
                )

        key = _read_key()
        if key == "up":
            selected = (selected - 1) % len(options)
        elif key == "down":
            selected = (selected + 1) % len(options)
        elif key == "enter":
            return selected
        elif key == "escape" or key == "\x03":
            console.print("\n[red]Cancelled.[/red]")
            sys.exit(0)


def multi_select_menu(
    title: str,
    options: list[dict],
    step: str = "",
    preselected: set[int] | None = None,
) -> list[int]:
    """Show a multi-select menu. Space toggles, Enter confirms."""
    cursor = 0
    checked = set(preselected or set())

    while True:
        console.clear()
        _draw_header()

        console.print()
        console.print(f"  [bold cyan]{step}[/bold cyan]" if step else "")
        console.print(f"  [bold]{title}[/bold]")
        console.print(
            "  [dim]Use [bold]arrow keys[/bold] to navigate, "
            "[bold]Space[/bold] to toggle, "
            "[bold]Enter[/bold] to confirm[/dim]"
        )
        console.print()

        for i, opt in enumerate(options):
            is_cursor = i == cursor
            is_checked = i in checked
            check = (
                "[bold green]\u2713[/bold green]"
                if is_checked
                else "[dim]\u2717[/dim]"
            )
            prefix = "  [bold green]>[/bold green] " if is_cursor else "    "
            label = opt["label"]
            desc = opt.get("description", "")

            if is_cursor:
                console.print(
                    f"{prefix}{check}  [bold white on blue] {label} [/bold white on blue]"
                    f"  [dim]{desc}[/dim]"
                )
            else:
                console.print(
                    f"{prefix}{check}  [white]{label}[/white]  [dim]{desc}[/dim]"
                )

        console.print()
        count = len(checked)
        console.print(f"  [dim]{count} selected \u2014 press Enter to confirm[/dim]")

        key = _read_key()
        if key == "up":
            cursor = (cursor - 1) % len(options)
        elif key == "down":
            cursor = (cursor + 1) % len(options)
        elif key == "space":
            if cursor in checked:
                checked.discard(cursor)
            else:
                checked.add(cursor)
        elif key == "enter":
            if not checked:
                continue
            return sorted(checked)
        elif key == "escape" or key == "\x03":
            console.print("\n[red]Cancelled.[/red]")
            sys.exit(0)


def weight_editor(
    datasets: list[dict],
    step: str = "",
) -> list[float]:
    """Interactive weight editor. Left/right to adjust, up/down to navigate.

    Args:
        datasets: List of dicts with "label" and "description" keys.

    Returns:
        List of normalized weights (sum to 1.0).
    """
    n = len(datasets)
    weights = [1.0 / n] * n
    cursor = 0
    increment = 0.05

    while True:
        console.clear()
        _draw_header()

        console.print()
        console.print(f"  [bold cyan]{step}[/bold cyan]" if step else "")
        console.print("  [bold]Set weights for each dataset[/bold]")
        console.print(
            "  [dim]Use [bold]up/down[/bold] to navigate, "
            "[bold]left/right[/bold] to adjust weight, "
            "[bold]Enter[/bold] to confirm[/dim]"
        )
        console.print()

        for i, ds in enumerate(datasets):
            is_cursor = i == cursor
            w = weights[i]
            bar_len = int(w * 30)
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            prefix = "  [bold green]>[/bold green] " if is_cursor else "    "
            label = ds["label"]

            if is_cursor:
                console.print(
                    f"{prefix}[bold white on blue] {label:<30} [/bold white on blue]"
                    f"  [{w:.2f}] {bar}"
                )
            else:
                console.print(
                    f"{prefix}[white]{label:<30}[/white]"
                    f"  [{w:.2f}] [dim]{bar}[/dim]"
                )

        console.print()
        total = sum(weights)
        console.print(f"  [dim]Total: {total:.2f} (will be normalized to 1.0)[/dim]")

        key = _read_key()
        if key == "up":
            cursor = (cursor - 1) % n
        elif key == "down":
            cursor = (cursor + 1) % n
        elif key == "right":
            weights[cursor] = min(1.0, weights[cursor] + increment)
        elif key == "left":
            weights[cursor] = max(0.05, weights[cursor] - increment)
        elif key == "enter":
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]
            return weights
        elif key == "escape" or key == "\x03":
            console.print("\n[red]Cancelled.[/red]")
            sys.exit(0)


def _draw_header():
    """Draw the app header."""
    if _cli:
        _cli.header("Cola-Coder", "Dataset Combiner")
    else:
        header = Text()
        header.append("Cola-Coder", style="bold cyan")
        header.append("  Dataset Combiner", style="bold white")
        console.print(
            Panel(
                header,
                box=box.DOUBLE,
                style="cyan",
                padding=(0, 2),
            )
        )


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
        "description": "Round-robin chunks for best mixing",
        "value": "interleave",
        "recommended": True,
    },
    {
        "label": "Weighted",
        "description": "Random sampling by weight",
        "value": "weighted",
    },
    {
        "label": "Concatenate",
        "description": "Append in order (for curriculum learning)",
        "value": "concat",
    },
]

DEDUP_OPTIONS = [
    {
        "label": "None",
        "description": "Skip deduplication (fastest)",
        "value": "none",
    },
    {
        "label": "Exact",
        "description": "Remove exact duplicate chunks only (~10 sec)",
        "value": "exact",
    },
    {
        "label": "Near-dedup (MinHash)",
        "description": "MinHash near-duplicate removal (requires datasketch)",
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
        console.clear()
        _draw_header()
        console.print()
        console.print(
            f"  [bold red]No .npy datasets found in {data_dir}[/bold red]"
        )
        console.print(
            "  [dim]Run prepare_data.py first to create training data.[/dim]"
        )
        sys.exit(1)

    # Build options for multi-select
    ds_options = []
    for ds in datasets:
        ds_options.append({
            "label": ds["name"],
            "description": f"{ds['tokens_str']} \u2022 {ds['file_size_str']} \u2022 {ds['age_str']}",
        })

    selected_indices = multi_select_menu(
        title="Select datasets to combine:",
        options=ds_options,
        step="Step 1/4 \u2022 Select Datasets",
        preselected=set(range(len(datasets))),  # All preselected
    )
    selected_datasets = [datasets[i] for i in selected_indices]

    # Step 2: Mixing strategy
    strategy_idx = select_menu(
        title="How should datasets be mixed?",
        options=STRATEGY_OPTIONS,
        step="Step 2/4 \u2022 Mixing Strategy",
    )
    strategy = STRATEGY_OPTIONS[strategy_idx]["value"]

    # Step 3: Weights
    if strategy in ("interleave", "weighted"):
        weight_ds = [
            {"label": ds["name"], "description": ds["tokens_str"]}
            for ds in selected_datasets
        ]
        weights = weight_editor(
            weight_ds,
            step="Step 3/4 \u2022 Weights",
        )
    else:
        weights = [1.0 / len(selected_datasets)] * len(selected_datasets)

    # Step 4: Dedup method
    dedup_idx = select_menu(
        title="Deduplication method:",
        options=DEDUP_OPTIONS,
        step="Step 4/4 \u2022 Deduplication",
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
    console.clear()
    _draw_header()
    console.print()

    table = Table(
        show_header=False,
        box=box.SIMPLE_HEAVY,
        padding=(0, 2),
        title="[bold]Summary[/bold]",
        title_style="bold white",
    )
    table.add_column("Setting", style="cyan", width=16)
    table.add_column("Value", style="white")

    # Datasets
    ds_names = [ds["name"] for ds in settings["datasets"]]
    total_tokens = sum(ds["tokens"] for ds in settings["datasets"])
    table.add_row(
        "Datasets",
        f"{len(ds_names)} files ({_format_tokens(total_tokens)} total)",
    )
    for i, ds in enumerate(settings["datasets"]):
        w = settings["weights"][i]
        table.add_row("", f"  {ds['name']}  [{w:.0%}]  ({ds['tokens_str']})")

    # Strategy
    table.add_row("Strategy", settings["strategy"].capitalize())

    # Dedup
    dedup = settings["dedup_method"]
    if dedup == "none":
        table.add_row("Dedup", "None (skip)")
    elif dedup == "exact":
        table.add_row("Dedup", "Exact (hash-based)")
    else:
        table.add_row("Dedup", "MinHash (near-duplicate)")

    # Output
    table.add_row("Output", output_path)

    console.print(table)
    console.print()
    console.print(
        "  [bold green]Press Enter to start[/bold green]"
        "  [dim]or[/dim]  [bold red]Escape to cancel[/bold red]"
    )

    while True:
        key = _read_key()
        if key == "enter":
            return True
        if key == "escape" or key == "\x03":
            return False


def run_pipeline(settings: dict, output_path: str):
    """Execute the combination pipeline."""
    import numpy as np
    from cola_coder.data.combine import DatasetCombiner, DatasetInput
    from cola_coder.data.dedup import ExactDeduplicator, CrossDatasetDeduplicator

    console.print()
    console.print("[bold cyan]Starting dataset combination...[/bold cyan]")
    console.print()

    datasets = settings["datasets"]
    dedup_method = settings["dedup_method"]
    dedup_removed = 0

    # Step 1: Optional dedup
    paths_to_combine = [ds["path"] for ds in datasets]

    if dedup_method != "none" and len(datasets) > 1:
        console.print("[dim]Running deduplication...[/dim]")
        t0 = time.time()

        if dedup_method == "exact":
            # Exact dedup: load all, hash, remove dupes across datasets
            dedup = ExactDeduplicator()
            temp_paths = []
            for i, ds in enumerate(datasets):
                arr = np.load(ds["path"], mmap_mode="r")
                clean, removed = dedup.deduplicate_array(np.array(arr))
                dedup_removed += removed
                # Save cleaned version to temp file
                temp_path = str(
                    Path(output_path).parent / f"_temp_dedup_{ds['name']}.npy"
                )
                np.save(temp_path, clean)
                temp_paths.append(temp_path)
                console.print(
                    f"  {ds['name']}: {removed} exact duplicates removed"
                )
            paths_to_combine = temp_paths

        elif dedup_method == "minhash":
            cross_dedup = CrossDatasetDeduplicator(
                method="minhash", threshold=0.8,
            )
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
                console.print(
                    f"  {ds['name']}: {result.duplicates_removed} "
                    f"near-duplicates removed"
                )

            paths_to_combine = temp_paths

        elapsed = time.time() - t0
        console.print(
            f"  [green]Dedup complete: {dedup_removed} total removed "
            f"({elapsed:.1f}s)[/green]"
        )
        console.print()

    # Step 2: Combine
    console.print("[dim]Combining datasets...[/dim]")
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
    console.print(f"  [green]Done in {elapsed:.1f}s[/green]")

    # Clean up temp files
    for path in paths_to_combine:
        if "_temp_dedup_" in path:
            try:
                os.remove(path)
            except OSError:
                pass

    # Show result
    console.print()
    console.print(
        Panel(
            f"[bold green]Dataset combination complete![/bold green]\n\n"
            f"Output: [cyan]{Path(result.output_path).resolve()}[/cyan]\n"
            f"Chunks: {result.total_chunks:,}\n"
            f"Tokens: {result.total_tokens:,} ({_format_tokens(result.total_tokens)})\n"
            + (f"Dedup:  {dedup_removed:,} duplicates removed\n"
               if dedup_removed else "")
            + "\nNext step:\n"
            "  [dim]python scripts/train.py --config configs/tiny.yaml[/dim]",
            title="[bold]Complete[/bold]",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


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
                console.print(
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

    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    arrays = []
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            console.print(f"[red]Error:[/red] {p} is not a 2-D array — skipping")
            continue
        arrays.append(arr)

    if not arrays:
        console.print("[red]No valid datasets loaded. Aborting.[/red]")
        sys.exit(1)

    # Total number of rows in the output == sum of all input row counts.
    total_rows = sum(a.shape[0] for a in arrays)

    # Compute how many rows to sample from each dataset.
    row_counts = [max(1, round(total_rows * w)) for w in norm_weights]
    # Adjust last dataset to hit exact total.
    row_counts[-1] = total_rows - sum(row_counts[:-1])

    console.print(f"\n  [bold]Weighted mixing:[/bold]  {total_rows:,} total rows")
    for p, arr, w, n in zip(paths, arrays, norm_weights, row_counts):
        console.print(
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

    console.print(
        f"\n  [green]Done.[/green]  Saved {mixed.shape[0]:,} x {mixed.shape[1]} "
        f"to [cyan]{out.resolve()}[/cyan]"
    )


def main():
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Interactive dataset combination tool for Cola-Coder.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(storage.data_dir) / "processed"),
        help="Directory containing .npy dataset files (default: ./data/processed).",
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

    # ── Non-interactive weighted mix (--datasets supplied) ────────────────
    if args.datasets:
        if len(args.datasets) < 1:
            console.print("[red]Error:[/red] --datasets requires at least one path.")
            sys.exit(1)

        paths, weights = _parse_weighted_datasets(args.datasets)

        # Validate files exist
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            for m in missing:
                console.print(f"[red]Error:[/red] file not found: {m}")
            sys.exit(1)

        # Auto-generate output path if not provided
        if args.output:
            output_path = args.output
        else:
            stems = [Path(p).stem for p in paths]
            combined_name = "_".join(stems) if len(stems) <= 3 else f"combined_{len(stems)}ds"
            output_path = str(Path(args.data_dir) / f"{combined_name}_weighted.npy")

        _draw_header()
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
            console.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
