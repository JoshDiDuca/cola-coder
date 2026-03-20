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

# ---------------------------------------------------------------------------
# Menu UI helpers using rich
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
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3, "")
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
    """Show an interactive arrow-key menu and return the selected index.

    Each option is a dict with:
        - label: str (displayed name)
        - description: str (right-side detail)
        - recommended: bool (optional, shows a tag)
    """
    selected = 0
    # Find first recommended option as default
    for i, opt in enumerate(options):
        if opt.get("recommended"):
            selected = i
            break

    while True:
        # Clear and redraw
        console.clear()
        _draw_header()

        console.print()
        console.print(f"  [bold cyan]{step}[/bold cyan]" if step else "")
        console.print(f"  [bold]{title}[/bold]")
        console.print(f"  [dim]Use [bold]arrow keys[/bold] to navigate, "
                      f"[bold]Enter[/bold] to select[/dim]")
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
        elif key == "escape" or key == "\x03":  # Ctrl+C
            console.print("\n[red]Cancelled.[/red]")
            sys.exit(0)


def multi_select_menu(
    title: str,
    options: list[dict],
    step: str = "",
    preselected: set[int] | None = None,
) -> list[int]:
    """Show a multi-select menu. Space toggles, Enter confirms.

    Returns list of selected indices.
    """
    cursor = 0
    checked = set(preselected or set())

    while True:
        console.clear()
        _draw_header()

        console.print()
        console.print(f"  [bold cyan]{step}[/bold cyan]" if step else "")
        console.print(f"  [bold]{title}[/bold]")
        console.print(f"  [dim]Use [bold]arrow keys[/bold] to navigate, "
                      f"[bold]Space[/bold] to toggle, "
                      f"[bold]Enter[/bold] to confirm[/dim]")
        console.print()

        for i, opt in enumerate(options):
            is_cursor = i == cursor
            is_checked = i in checked
            check = "[bold green]\u2713[/bold green]" if is_checked else "[dim]\u2717[/dim]"
            prefix = "  [bold green]>[/bold green] " if is_cursor else "    "
            label = opt["label"]
            desc = opt.get("description", "")

            if is_cursor:
                console.print(
                    f"{prefix}{check}  [bold white on blue] {label} [/bold white on blue]"
                    f"  [dim]{desc}[/dim]"
                )
            else:
                console.print(f"{prefix}{check}  [white]{label}[/white]  [dim]{desc}[/dim]")

        console.print()
        count = len(checked)
        console.print(f"  [dim]{count} selected — press Enter to confirm[/dim]")

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
                # At least one must be selected
                continue
            return sorted(checked)
        elif key == "escape" or key == "\x03":
            console.print("\n[red]Cancelled.[/red]")
            sys.exit(0)


def _draw_header():
    """Draw the app header (same style as all other scripts)."""
    if _cli:
        _cli.header("Cola-Coder", "Data Preparation (Interactive)")
    else:
        header = Text()
        header.append("Cola-Coder", style="bold cyan")
        header.append("  Data Preparation", style="bold white")
        console.print(Panel(
            header, box=box.DOUBLE, style="cyan", padding=(0, 2),
        ))


# ---------------------------------------------------------------------------
# Menu definitions
# ---------------------------------------------------------------------------

SIZE_OPTIONS = [
    {
        "label": "Light",
        "description": "10M tokens \u2022 ~10 sec \u2022 pipeline testing",
        "max_tokens": 10_000_000,
    },
    {
        "label": "Medium",
        "description": "500M tokens \u2022 ~5 min \u2022 good for tiny model (50M params)",
        "max_tokens": 500_000_000,
        "recommended": True,
    },
    {
        "label": "Large",
        "description": "2B tokens \u2022 ~15 min \u2022 Chinchilla-optimal for small model (125M)",
        "max_tokens": 2_000_000_000,
    },
    {
        "label": "Full",
        "description": "No limit \u2022 ~2-3 hours \u2022 process entire dataset",
        "max_tokens": None,
    },
]

FILTER_OPTIONS = [
    {
        "label": "Off",
        "description": "No filtering \u2022 raw data as-is",
        "mode": None,
    },
    {
        "label": "Conservative",
        "description": "Reject clearly bad code only \u2022 ~48% rejection",
        "mode": "conservative",
        "recommended": True,
    },
    {
        "label": "Strict",
        "description": "Keep only high-quality code \u2022 ~65% rejection",
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
        "description": "Focused TS/JS model",
        "languages": ["typescript", "javascript"],
        "recommended": True,
    },
    {
        "label": "Python only",
        "description": "Python-focused model",
        "languages": ["python"],
    },
    {
        "label": "Python + TypeScript + JavaScript",
        "description": "Three most popular languages",
        "languages": ["python", "typescript", "javascript"],
    },
    {
        "label": "All 6 languages",
        "description": "Python, TS, JS, Java, Go, Rust",
        "languages": ["python", "typescript", "javascript", "java", "go", "rust"],
    },
    {
        "label": "Custom...",
        "description": "Pick individual languages",
        "languages": None,  # Triggers multi-select
    },
]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_menu() -> dict:
    """Run the interactive menu and return the collected settings."""

    # Step 1: Data size
    size_idx = select_menu(
        title="How much data do you want to prepare?",
        options=SIZE_OPTIONS,
        step="Step 1/3 \u2022 Data Size",
    )
    size = SIZE_OPTIONS[size_idx]

    # Step 2: Quality filter
    filter_idx = select_menu(
        title="What quality filter do you want to use?",
        options=FILTER_OPTIONS,
        step="Step 2/3 \u2022 Quality Filter",
    )
    filter_opt = FILTER_OPTIONS[filter_idx]

    # Step 3: Languages
    lang_idx = select_menu(
        title="Which languages should be included?",
        options=LANGUAGE_PRESETS,
        step="Step 3/3 \u2022 Languages",
    )
    lang_preset = LANGUAGE_PRESETS[lang_idx]

    if lang_preset["languages"] is None:
        # Custom: show multi-select
        selected_indices = multi_select_menu(
            title="Select languages to include:",
            options=LANGUAGE_OPTIONS,
            step="Step 3/3 \u2022 Languages (custom)",
            preselected={0, 1},  # TS + JS preselected
        )
        languages = [LANGUAGE_OPTIONS[i]["value"] for i in selected_indices]
    else:
        languages = lang_preset["languages"]

    workers = max(1, min(os.cpu_count() or 4, 16))

    return {
        "size_label": size["label"],
        "max_tokens": size["max_tokens"],
        "filter_label": filter_opt["label"],
        "filter_mode": filter_opt["mode"],
        "languages": languages,
        "workers": workers,
    }


def show_summary(settings: dict) -> bool:
    """Show a summary and ask for confirmation. Returns True to proceed."""
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
    table.add_column("Setting", style="cyan", width=14)
    table.add_column("Value", style="white")

    # Size
    max_tok = settings["max_tokens"]
    if max_tok:
        tok_str = f"{max_tok:,} tokens"
    else:
        tok_str = "No limit (full dataset)"
    table.add_row("Data Size", f"{settings['size_label']}  ({tok_str})")

    # Filter
    table.add_row("Filter", settings["filter_label"])

    # Languages
    table.add_row("Languages", ", ".join(settings["languages"]))

    # Workers
    table.add_row("Workers", str(settings["workers"]))

    console.print(table)
    console.print()
    console.print("  [bold green]Press Enter to start[/bold green]"
                  "  [dim]or[/dim]  [bold red]Escape to cancel[/bold red]")
    console.print()
    console.print("  [dim]Tip: You can Ctrl+C during processing to save partial data.[/dim]")

    while True:
        key = _read_key()
        if key == "enter":
            return True
        if key == "escape" or key == "\x03":
            return False


def run_pipeline(settings: dict, tokenizer_path: str, output_dir: str, batch_size: int):
    """Run the data preparation pipeline with the selected settings."""
    console.print()
    console.print("[bold cyan]Starting data preparation...[/bold cyan]")
    console.print()

    # Import pipeline modules
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    from cola_coder.data.download import stream_code_data
    from cola_coder.data.preprocess import tokenize_and_chunk
    from cola_coder.data.quality_filter import (
        filtered_stream, parallel_filtered_stream,
        FilterStats, FilterMode,
    )

    # Load tokenizer
    console.print(f"[dim]Loading tokenizer from {tokenizer_path}...[/dim]")
    tokenizer = CodeTokenizer(tokenizer_path)
    console.print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # Stream data
    console.print(f"[dim]Loading data for: {', '.join(settings['languages'])}...[/dim]")
    data_stream = stream_code_data(
        dataset_name="bigcode/starcoderdata",
        languages=settings["languages"],
    )

    # Apply filter
    languages = settings["languages"]
    filter_mode = settings["filter_mode"]
    workers = settings["workers"]

    if filter_mode is None:
        console.print("[yellow]Quality filtering: OFF[/yellow]")
    else:
        mode = FilterMode(filter_mode)
        stats = FilterStats()
        console.print(
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

    console.print()

    # Tokenize and chunk
    output_file = tokenize_and_chunk(
        text_iterator=data_stream,
        tokenizer=tokenizer,
        chunk_size=2048,
        output_dir=output_dir,
        max_tokens=settings["max_tokens"],
        batch_size=batch_size,
    )

    console.print()
    console.print(Panel(
        f"[bold green]Done![/bold green]\n\n"
        f"Output: [cyan]{Path(output_file).resolve()}[/cyan]\n\n"
        f"Next step:\n"
        f"  [dim]python scripts/train.py --config configs/tiny.yaml[/dim]",
        title="[bold]Data Preparation Complete[/bold]",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive data preparation for Cola-Coder.",
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True,
        help="Path to trained tokenizer.json file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/processed",
        help="Output directory (default: ./data/processed).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Files per tokenization batch (default: 256).",
    )
    args = parser.parse_args()

    # Validate tokenizer exists
    if not Path(args.tokenizer).exists():
        console.print(f"[red]Error: Tokenizer not found: {args.tokenizer}[/red]")
        console.print("[dim]Train one first: python scripts/train_tokenizer.py[/dim]")
        sys.exit(1)

    try:
        settings = run_menu()
        if show_summary(settings):
            run_pipeline(settings, args.tokenizer, args.output_dir, args.batch_size)
        else:
            console.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
