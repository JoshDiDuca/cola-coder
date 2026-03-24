"""Interactive data preparation menu for Cola-Coder.

A nice CLI menu that walks you through data preparation options:
  1. Data size (light → full)
  2. Quality filter (off / conservative / strict)
  3. Languages to include

Uses the same pipeline as prepare_data.py but with a friendly interface.

Usage:
    python scripts/prepare_data_interactive.py --tokenizer tokenizer.json
"""

import argparse
import os
import sys
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Menu definitions
# ---------------------------------------------------------------------------

SIZE_OPTIONS = [
    {
        "label": "Light",
        "detail": "10M tokens \u2022 ~10 sec \u2022 pipeline testing",
        "max_tokens": 10_000_000,
    },
    {
        "label": "Medium",
        "detail": "500M tokens \u2022 ~5 min \u2022 good for tiny model (50M params)",
        "max_tokens": 500_000_000,
        "recommended": True,
    },
    {
        "label": "Large",
        "detail": "2B tokens \u2022 ~15 min \u2022 Chinchilla-optimal for small model (125M)",
        "max_tokens": 2_000_000_000,
    },
    {
        "label": "Full",
        "detail": "No limit \u2022 ~2-3 hours \u2022 process entire dataset",
        "max_tokens": None,
    },
]

FILTER_OPTIONS = [
    {
        "label": "Off",
        "detail": "No filtering \u2022 raw data as-is",
        "mode": None,
    },
    {
        "label": "Conservative",
        "detail": "Reject clearly bad code only \u2022 ~48% rejection",
        "mode": "conservative",
        "recommended": True,
    },
    {
        "label": "Strict",
        "detail": "Keep only high-quality code \u2022 ~65% rejection",
        "mode": "strict",
    },
]

LANGUAGE_OPTIONS = [
    {"label": "TypeScript", "value": "typescript"},
    {"label": "JavaScript", "value": "javascript"},
    {"label": "Python", "value": "python"},
    {"label": "Java", "value": "java"},
    {"label": "Go", "value": "go"},
    {"label": "Rust", "value": "rust"},
]

LANGUAGE_PRESETS = [
    {
        "label": "TypeScript + JavaScript",
        "detail": "Focused TS/JS model",
        "languages": ["typescript", "javascript"],
        "recommended": True,
    },
    {
        "label": "Python only",
        "detail": "Python-focused model",
        "languages": ["python"],
    },
    {
        "label": "Python + TypeScript + JavaScript",
        "detail": "Three most popular languages",
        "languages": ["python", "typescript", "javascript"],
    },
    {
        "label": "All 6 languages",
        "detail": "Python, TS, JS, Java, Go, Rust",
        "languages": ["python", "typescript", "javascript", "java", "go", "rust"],
    },
    {
        "label": "Custom...",
        "detail": "Pick individual languages",
        "languages": None,  # Triggers multi-select
    },
]


# ---------------------------------------------------------------------------
# Type-check quality scoring options (Advanced)
# ---------------------------------------------------------------------------

def _tsc_available() -> bool:
    """Check if TypeScript compiler is available for type-check scoring."""
    try:
        from cola_coder.reasoning.rewards.type_check import TypeCheckReward
        return TypeCheckReward.is_available()
    except ImportError:
        return False


TSC_SCORING_OPTIONS = [
    {
        "label": "Off",
        "detail": "Don't use tsc scoring",
        "mode": None,
    },
    {
        "label": "Score",
        "detail": "Score TS files with tsc, add quality_score to metadata",
        "mode": "score",
        "recommended": True,
    },
    {
        "label": "Filter",
        "detail": "Reject TS files that don't type-check (strict quality gate)",
        "mode": "filter",
    },
]

MIXING_STRATEGY_OPTIONS = [
    {
        "label": "Equal",
        "detail": "Equal weights across all sources",
        "preset": "equal",
    },
    {
        "label": "TS-focused",
        "detail": "50% TypeScript, 25% JS, 15% Python, 10% other",
        "preset": "typescript_focused",
        "recommended": True,
    },
    {
        "label": "Balanced",
        "detail": "Distributed across all languages",
        "preset": "balanced_code",
    },
    {
        "label": "Quality-tier",
        "detail": "Weight by data quality (verified > tested > raw)",
        "preset": "quality_tiers",
    },
    {
        "label": "Custom...",
        "detail": "Set weights manually per language",
        "preset": None,
    },
]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def _get_custom_mixing_weights(languages: list[str], step: str) -> dict[str, float]:
    """Prompt user to enter custom mixing weights for each language.

    Falls back to equal weights if input is invalid.
    """
    cli.header("Cola-Coder", "Interactive Data Prep")
    cli.print()
    cli.print(f"  [bold cyan]{step}[/bold cyan]")
    cli.print("  [bold]Set custom mixing weights[/bold]")
    cli.print("  [dim]Enter a weight for each language (weights will be normalized).[/dim]")
    cli.print()

    weights: dict[str, float] = {}
    for lang in languages:
        cli.print(f"  Weight for [cyan]{lang}[/cyan]: ", end="")
        try:
            raw = input().strip()
            weights[lang] = float(raw) if raw else 1.0
        except (ValueError, EOFError):
            weights[lang] = 1.0

    return weights


def run_menu() -> dict:
    """Run the interactive menu and return the collected settings."""

    # Detect tsc for step count
    has_tsc = _tsc_available()
    total_steps = 5 if has_tsc else 4

    # Step 1: Data size
    cli.header("Cola-Coder", "Interactive Data Prep")
    size_idx = cli.choose(
        f"Step 1/{total_steps} \u2022 Data Size — How much data do you want to prepare?",
        SIZE_OPTIONS,
    )
    if size_idx is None:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)
    size = SIZE_OPTIONS[size_idx]

    # Step 2: Quality filter
    filter_idx = cli.choose(
        f"Step 2/{total_steps} \u2022 Quality Filter — What quality filter do you want to use?",
        FILTER_OPTIONS,
    )
    if filter_idx is None:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)
    filter_opt = FILTER_OPTIONS[filter_idx]

    # Step 3: Languages
    lang_idx = cli.choose(
        f"Step 3/{total_steps} \u2022 Languages — Which languages should be included?",
        LANGUAGE_PRESETS,
    )
    if lang_idx is None:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)
    lang_preset = LANGUAGE_PRESETS[lang_idx]

    if lang_preset["languages"] is None:
        # Custom: show multi-select
        selected_indices = cli.multi_select(
            f"Step 3/{total_steps} \u2022 Languages (custom) — Select languages to include:",
            LANGUAGE_OPTIONS,
            preselected=[0, 1],  # TS + JS preselected
        )
        languages = [LANGUAGE_OPTIONS[i]["value"] for i in selected_indices]
    else:
        languages = lang_preset["languages"]

    # Step 4: Data Mixing Strategy
    mixing_step = f"Step 4/{total_steps} \u2022 Advanced: Data Mixing Strategy"
    mixing_idx = cli.choose(
        f"{mixing_step} — How should data from different sources be weighted?",
        MIXING_STRATEGY_OPTIONS,
    )
    if mixing_idx is None:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)
    mixing_opt = MIXING_STRATEGY_OPTIONS[mixing_idx]

    mixing_preset = mixing_opt["preset"]
    mixing_weights = None
    if mixing_preset is None:
        # Custom: prompt for per-language weights
        mixing_weights = _get_custom_mixing_weights(languages, mixing_step)
    elif mixing_preset == "equal":
        mixing_weights = {lang: 1.0 / len(languages) for lang in languages}

    # Step 5 (Advanced): Type-check scoring (only if tsc available and TS selected)
    tsc_mode = None
    if has_tsc and "typescript" in languages:
        tsc_idx = cli.choose(
            f"Step 5/{total_steps} \u2022 Advanced: Type-Check Scoring"
            " — Type-Check Quality Scoring (uses tsc --strict)",
            TSC_SCORING_OPTIONS,
        )
        if tsc_idx is None:
            cli.print("\n[red]Cancelled.[/red]")
            sys.exit(0)
        tsc_mode = TSC_SCORING_OPTIONS[tsc_idx]["mode"]

    workers = max(1, min(os.cpu_count() or 4, 16))

    return {
        "size_label": size["label"],
        "max_tokens": size["max_tokens"],
        "filter_label": filter_opt["label"],
        "filter_mode": filter_opt["mode"],
        "languages": languages,
        "mixing_preset": mixing_preset,
        "mixing_weights": mixing_weights,
        "mixing_label": mixing_opt["label"],
        "workers": workers,
        "tsc_scoring": tsc_mode,
    }


def show_summary(settings: dict) -> bool:
    """Show a summary and ask for confirmation. Returns True to proceed."""
    cli.header("Cola-Coder", "Interactive Data Prep")
    cli.print()

    # Size
    max_tok = settings["max_tokens"]
    if max_tok:
        tok_str = f"{max_tok:,} tokens"
    else:
        tok_str = "No limit (full dataset)"

    # Mixing strategy
    mixing_label = settings.get("mixing_label", "Equal")
    mixing_weights = settings.get("mixing_weights")
    if mixing_weights:
        total = sum(mixing_weights.values())
        if total > 0:
            weight_parts = [
                f"{lang}: {w / total:.0%}"
                for lang, w in sorted(mixing_weights.items(), key=lambda x: -x[1])
            ]
            mixing_str = f"{mixing_label}  ({', '.join(weight_parts)})"
        else:
            mixing_str = mixing_label
    else:
        mixing_str = mixing_label

    summary: dict[str, str] = {
        "Data Size": f"{settings['size_label']}  ({tok_str})",
        "Filter": settings["filter_label"],
        "Languages": ", ".join(settings["languages"]),
        "Mixing": mixing_str,
        "Workers": str(settings["workers"]),
    }

    tsc_mode = settings.get("tsc_scoring")
    if tsc_mode:
        summary["TSC Scoring"] = "Score files" if tsc_mode == "score" else "Filter (reject bad)"

    cli.kv_table(summary, title="Summary")
    cli.print()
    cli.print("  [dim]Tip: You can Ctrl+C during processing to save partial data.[/dim]")
    cli.print()

    return cli.confirm("Start data preparation?", default=True)


def run_pipeline(settings: dict, tokenizer_path: str, output_dir: str, batch_size: int):
    """Run the data preparation pipeline with the selected settings."""
    cli.print()
    cli.print("[bold cyan]Starting data preparation...[/bold cyan]")
    cli.print()

    # Import pipeline modules
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    from cola_coder.data.download import stream_code_data
    from cola_coder.data.preprocess import tokenize_and_chunk
    from cola_coder.data.quality_filter import (
        filtered_stream, parallel_filtered_stream,
        FilterStats, FilterMode,
    )

    # Load tokenizer
    cli.print(f"[dim]Loading tokenizer from {tokenizer_path}...[/dim]")
    tokenizer = CodeTokenizer(tokenizer_path)
    cli.print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # Stream data
    cli.print(f"[dim]Loading data for: {', '.join(settings['languages'])}...[/dim]")
    data_stream = stream_code_data(
        dataset_name="bigcode/starcoderdata",
        languages=settings["languages"],
    )

    # Apply filter
    languages = settings["languages"]
    filter_mode = settings["filter_mode"]
    workers = settings["workers"]

    if filter_mode is None:
        cli.print("[yellow]Quality filtering: OFF[/yellow]")
    else:
        mode = FilterMode(filter_mode)
        stats = FilterStats()
        cli.print(
            f"[green]Quality filtering: {filter_mode.upper()}[/green]"
            f"  [dim]({workers} workers)[/dim]"
        )
        if workers > 1:
            data_stream = parallel_filtered_stream(
                data_stream, mode=mode, stats=stats,
                num_workers=workers, languages=languages,
            )
        else:
            data_stream = filtered_stream(
                data_stream, mode=mode, stats=stats,
                languages=languages,
            )

    cli.print()

    # Tokenize and chunk
    output_file = tokenize_and_chunk(
        text_iterator=data_stream,
        tokenizer=tokenizer,
        chunk_size=2048,
        output_dir=output_dir,
        max_tokens=settings["max_tokens"],
        batch_size=batch_size,
    )

    cli.done(
        "Data Preparation Complete",
        extras={
            "Output": str(Path(output_file).resolve()),
            "Next step": "python scripts/train.py --config configs/tiny.yaml",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    storage = get_storage_config()
    storage.apply_hf_cache()

    parser = argparse.ArgumentParser(
        description="Interactive data preparation for Cola-Coder.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=storage.tokenizer_path,
        help="Path to trained tokenizer.json file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(Path(storage.data_dir) / "processed"),
        help="Output directory (default: ./data/processed).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Files per tokenization batch (default: 256).",
    )
    args = parser.parse_args()

    # Validate tokenizer exists
    if not Path(args.tokenizer).exists():
        cli.print(f"[red]Error: Tokenizer not found: {args.tokenizer}[/red]")
        cli.print("[dim]Train one first: python scripts/train_tokenizer.py[/dim]")
        sys.exit(1)

    try:
        settings = run_menu()
        if show_summary(settings):
            run_pipeline(settings, args.tokenizer, args.output_dir, args.batch_size)
        else:
            cli.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
