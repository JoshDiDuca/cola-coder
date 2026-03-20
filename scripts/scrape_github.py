"""Interactive GitHub scraper for Cola-Coder training data.

A CLI menu that walks you through scraping GitHub repositories:
  1. Mode: Search by language / Search by topic / Import repo list / Clone single repo
  2. Filter preset or custom filters
  3. Max repos to clone
  4. Output directory and name

Then: search -> clone -> extract -> filter -> save as .npy

Usage:
    python scripts/scrape_github.py
    python scripts/scrape_github.py --tokenizer tokenizer.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Menu UI helpers using rich (same approach as prepare_data_interactive.py)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
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


def prompt_input(label: str, default: str = "") -> str:
    """Prompt for text input with a default value."""
    console.print()
    suffix = f" [dim](default: {default})[/dim]" if default else ""
    console.print(f"  [bold]{label}[/bold]{suffix}")
    try:
        raw = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]Cancelled.[/red]")
        sys.exit(0)
    return raw if raw else default


def prompt_int(label: str, default: int) -> int:
    """Prompt for an integer with a default value."""
    while True:
        raw = prompt_input(label, str(default))
        try:
            return int(raw)
        except ValueError:
            console.print("  [red]Please enter a number.[/red]")


def _draw_header():
    """Draw the app header."""
    if _cli:
        _cli.header("Cola-Coder", "GitHub Scraper")
    else:
        header = Text()
        header.append("Cola-Coder", style="bold cyan")
        header.append("  GitHub Scraper", style="bold white")
        console.print(Panel(
            header, box=box.DOUBLE, style="cyan", padding=(0, 2),
        ))


# ---------------------------------------------------------------------------
# Menu definitions
# ---------------------------------------------------------------------------

MODE_OPTIONS = [
    {
        "label": "Search by language",
        "description": "Find repos with a specific primary language",
        "mode": "language",
        "recommended": True,
    },
    {
        "label": "Search by topic",
        "description": "Find repos tagged with specific topics",
        "mode": "topic",
    },
    {
        "label": "Import repo list",
        "description": "Read repo names from a text file (one per line)",
        "mode": "import",
    },
    {
        "label": "Clone single repo",
        "description": "Enter a single owner/repo to clone",
        "mode": "single",
    },
]

PRESET_OPTIONS = [
    {
        "label": "TypeScript Elite",
        "description": ">500 stars, strict TS, has tests, MIT/Apache",
        "preset": "typescript_elite",
    },
    {
        "label": "TypeScript Good",
        "description": ">50 stars, any TS, permissive license",
        "preset": "typescript_good",
        "recommended": True,
    },
    {
        "label": "Python Elite",
        "description": ">500 stars, good Python repos, has tests",
        "preset": "python_elite",
    },
    {
        "label": "Popular (any language)",
        "description": ">1000 stars, any language, permissive license",
        "preset": "popular_any",
    },
    {
        "label": "Custom filters...",
        "description": "Configure your own star/fork/license criteria",
        "preset": None,
    },
]

LANGUAGE_OPTIONS = [
    {"label": "TypeScript", "value": "TypeScript"},
    {"label": "JavaScript", "value": "JavaScript"},
    {"label": "Python", "value": "Python"},
    {"label": "Java", "value": "Java"},
    {"label": "Go", "value": "Go"},
    {"label": "Rust", "value": "Rust"},
    {"label": "C++", "value": "C++"},
    {"label": "C#", "value": "C#"},
    {"label": "Ruby", "value": "Ruby"},
    {"label": "PHP", "value": "PHP"},
]

LICENSE_OPTIONS = [
    {"label": "MIT", "value": "mit"},
    {"label": "Apache 2.0", "value": "apache-2.0"},
    {"label": "BSD 2-Clause", "value": "bsd-2-clause"},
    {"label": "BSD 3-Clause", "value": "bsd-3-clause"},
    {"label": "ISC", "value": "isc"},
    {"label": "GPL 3.0", "value": "gpl-3.0"},
    {"label": "LGPL 2.1", "value": "lgpl-2.1"},
    {"label": "Unlicense", "value": "unlicense"},
]

MAX_REPOS_OPTIONS = [
    {"label": "10 repos", "description": "Quick test", "value": 10},
    {"label": "50 repos", "description": "Small dataset", "value": 50, "recommended": True},
    {"label": "100 repos", "description": "Medium dataset", "value": 100},
    {"label": "250 repos", "description": "Large dataset", "value": 250},
    {"label": "500 repos", "description": "Very large dataset", "value": 500},
    {"label": "Custom...", "description": "Enter a number", "value": None},
]


# ---------------------------------------------------------------------------
# Custom filter builder
# ---------------------------------------------------------------------------

def build_custom_filter() -> "RepoFilter":
    """Walk the user through building a custom RepoFilter."""
    from cola_coder.data.sources.github import RepoFilter

    console.clear()
    _draw_header()
    console.print()
    console.print("  [bold cyan]Custom Filter Builder[/bold cyan]")
    console.print()

    # Primary language
    lang_idx = select_menu(
        title="Primary language?",
        options=LANGUAGE_OPTIONS + [{"label": "Any", "description": "No language filter"}],
        step="Custom Filter - Language",
    )
    if lang_idx < len(LANGUAGE_OPTIONS):
        primary_language = LANGUAGE_OPTIONS[lang_idx]["value"]
    else:
        primary_language = None

    # Stars
    min_stars = prompt_int("Minimum stars?", 50)

    # Licenses
    license_indices = multi_select_menu(
        title="Allowed licenses (select at least one):",
        options=LICENSE_OPTIONS,
        step="Custom Filter - Licenses",
        preselected={0, 1},  # MIT + Apache preselected
    )
    licenses = [LICENSE_OPTIONS[i]["value"] for i in license_indices]

    # Has tests?
    has_tests_idx = select_menu(
        title="Require tests?",
        options=[
            {"label": "Yes", "description": "Only repos with test files"},
            {"label": "No", "description": "Include all repos"},
        ],
        step="Custom Filter - Tests",
    )
    has_tests = has_tests_idx == 0

    return RepoFilter(
        min_stars=min_stars,
        primary_language=primary_language,
        licenses=licenses,
        not_archived=True,
        is_fork=False,
        has_tests=has_tests if has_tests else None,
        pushed_after="2022-01-01",
        max_repo_size_kb=500_000,
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_menu() -> dict:
    """Run the interactive menu and return the collected settings."""

    # Step 1: Mode
    mode_idx = select_menu(
        title="How do you want to find repositories?",
        options=MODE_OPTIONS,
        step="Step 1/4 - Mode",
    )
    mode = MODE_OPTIONS[mode_idx]["mode"]

    settings: dict = {"mode": mode}

    # Mode-specific settings
    if mode == "single":
        console.clear()
        _draw_header()
        repo_name = prompt_input("Enter repository (owner/repo):", "microsoft/TypeScript")
        settings["repos"] = [repo_name]
        settings["filter"] = None
        settings["max_repos"] = 1
    elif mode == "import":
        console.clear()
        _draw_header()
        file_path = prompt_input("Path to repo list file:", "repos.txt")
        if not Path(file_path).exists():
            console.print(f"  [red]File not found: {file_path}[/red]")
            sys.exit(1)
        repos = [
            line.strip() for line in Path(file_path).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        console.print(f"  [green]Found {len(repos)} repos in {file_path}[/green]")
        settings["repos"] = repos
        settings["filter"] = None
        settings["max_repos"] = len(repos)
    else:
        # Search modes: select filter preset
        # Step 2: Filter
        preset_idx = select_menu(
            title="Which filter preset?",
            options=PRESET_OPTIONS,
            step="Step 2/4 - Filter",
        )
        preset_name = PRESET_OPTIONS[preset_idx]["preset"]

        if preset_name is None:
            settings["filter"] = "custom"
            settings["custom_filter"] = build_custom_filter()
        else:
            settings["filter"] = preset_name

        if mode == "topic":
            console.clear()
            _draw_header()
            topics = prompt_input(
                "Enter topics (comma-separated):",
                "react,nextjs",
            )
            settings["topics"] = [t.strip() for t in topics.split(",") if t.strip()]

        # Step 3: Max repos
        max_idx = select_menu(
            title="How many repos to process?",
            options=MAX_REPOS_OPTIONS,
            step="Step 3/4 - Count",
        )
        max_val = MAX_REPOS_OPTIONS[max_idx]["value"]
        if max_val is None:
            max_val = prompt_int("Enter number of repos:", 100)
        settings["max_repos"] = max_val

    # Step 4: Output
    console.clear()
    _draw_header()
    settings["output_dir"] = prompt_input(
        "Output directory:",
        "./data/github_scraped",
    )
    settings["output_name"] = prompt_input(
        "Dataset name:",
        "github_code",
    )

    # Extract languages (for RepoProcessor)
    extract_indices = multi_select_menu(
        title="Which languages to extract from repos?",
        options=LANGUAGE_OPTIONS,
        step="Step 4/4 - Extract Languages",
        preselected={0, 1},  # TypeScript + JavaScript
    )
    settings["extract_languages"] = [LANGUAGE_OPTIONS[i]["value"] for i in extract_indices]

    return settings


def show_summary(settings: dict) -> bool:
    """Show a summary and ask for confirmation."""
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
    table.add_column("Setting", style="cyan", width=18)
    table.add_column("Value", style="white")

    table.add_row("Mode", settings["mode"])

    if settings.get("filter"):
        table.add_row("Filter", settings["filter"])

    if settings.get("repos"):
        repo_str = ", ".join(settings["repos"][:5])
        if len(settings["repos"]) > 5:
            repo_str += f"  (+{len(settings['repos']) - 5} more)"
        table.add_row("Repos", repo_str)

    if settings.get("topics"):
        table.add_row("Topics", ", ".join(settings["topics"]))

    table.add_row("Max repos", str(settings["max_repos"]))
    table.add_row("Extract langs", ", ".join(settings["extract_languages"]))
    table.add_row("Output dir", settings["output_dir"])
    table.add_row("Dataset name", settings["output_name"])

    # Check for GITHUB_TOKEN
    has_token = bool(os.environ.get("GITHUB_TOKEN"))
    token_str = "[green]Set[/green]" if has_token else "[yellow]Not set (60 req/hr limit)[/yellow]"
    table.add_row("GITHUB_TOKEN", token_str)

    console.print(table)
    console.print()
    console.print("  [bold green]Press Enter to start[/bold green]"
                  "  [dim]or[/dim]  [bold red]Escape to cancel[/bold red]")
    console.print()

    if not has_token:
        console.print("  [yellow]Tip: Set GITHUB_TOKEN env var for 5000 requests/hour.[/yellow]")

    while True:
        key = _read_key()
        if key == "enter":
            return True
        if key == "escape" or key == "\x03":
            return False


def run_pipeline(settings: dict, tokenizer_path: str | None):
    """Run the scraping pipeline with the selected settings."""
    from cola_coder.data.sources.github import (
        GitHubClient, GitHubSource, RepoFilter, RepoProcessor,
        FILTER_PRESETS, DataRecord,
    )

    console.print()
    console.print("[bold cyan]Starting GitHub scraper...[/bold cyan]")
    console.print()

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filter
    repo_filter: RepoFilter | None = None
    if settings.get("filter") == "custom":
        repo_filter = settings["custom_filter"]
    elif settings.get("filter"):
        repo_filter = FILTER_PRESETS[settings["filter"]]

    # Add topics to filter if specified
    if settings.get("topics") and repo_filter is not None:
        repo_filter.topics_include = settings["topics"]

    extract_languages = settings["extract_languages"]

    # Collect all records
    records: list[DataRecord] = []

    mode = settings["mode"]

    if mode in ("single", "import"):
        # Direct clone mode
        client = GitHubClient()
        processor = RepoProcessor(languages=extract_languages)
        clone_dir = output_dir / "_clones"

        for repo_name in settings["repos"]:
            console.print(f"  [cyan]Cloning {repo_name}...[/cyan]")
            try:
                repo_path = client.clone_repo(repo_name, clone_dir, shallow=True)
            except Exception as e:
                console.print(f"  [red]Failed to clone {repo_name}: {e}[/red]")
                continue

            try:
                # Get repo info for metadata
                try:
                    info = client.get_repo_info(repo_name)
                    stars = info.get("stargazers_count", 0)
                    html_url = info.get("html_url", "")
                    license_info = info.get("license") or {}
                    spdx = license_info.get("spdx_id", "")
                except Exception:
                    stars, html_url, spdx = 0, "", ""

                # Detect license from files if needed
                if not spdx or spdx == "NOASSERTION":
                    detected = processor.check_license(repo_path)
                    if detected:
                        spdx = detected

                file_count = 0
                for record in processor.extract_files(
                    repo_path,
                    repo_name=repo_name,
                    repo_stars=stars,
                    repo_url=html_url,
                    repo_license=spdx,
                ):
                    records.append(record)
                    file_count += 1

                console.print(
                    f"  [green]Extracted {file_count} files from {repo_name}[/green]"
                )
            finally:
                # Cleanup
                import shutil
                if repo_path.exists():
                    try:
                        shutil.rmtree(repo_path)
                    except OSError:
                        pass

    else:
        # Search mode — use GitHubSource
        if repo_filter is None:
            repo_filter = FILTER_PRESETS["typescript_good"]

        source = GitHubSource(
            filter=repo_filter,
            clone_dir=output_dir / "_clones",
            cache_dir=output_dir / "_cache",
            languages=extract_languages,
            cleanup=True,
        )

        console.print(f"  [dim]Query: {repo_filter.to_github_query()}[/dim]")
        console.print()

        file_count = 0
        for record in source.stream(max_repos=settings["max_repos"]):
            records.append(record)
            file_count += 1
            if file_count % 100 == 0:
                console.print(f"  [dim]  ... {file_count} files extracted so far[/dim]")

    console.print()
    console.print(f"[bold]Total files extracted: {len(records)}[/bold]")

    if not records:
        console.print("[yellow]No files extracted. Check your filters or try a different query.[/yellow]")
        return

    # Save as JSON lines (each line is one record)
    jsonl_path = output_dir / f"{settings['output_name']}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            line = {
                "content": record.content,
                "file_path": record.file_path,
                "language": record.language,
                "repo_name": record.repo_name,
                "repo_stars": record.repo_stars,
                "repo_url": record.repo_url,
                "license": record.license,
                "file_size": record.file_size,
            }
            f.write(json.dumps(line) + "\n")

    console.print(f"  [cyan]Saved JSONL:[/cyan] {jsonl_path.resolve()}")

    # Optionally tokenize to .npy if tokenizer is provided
    if tokenizer_path and Path(tokenizer_path).exists():
        console.print()
        console.print("[dim]Tokenizing to .npy...[/dim]")

        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            from cola_coder.data.preprocess import tokenize_and_chunk

            tokenizer = CodeTokenizer(tokenizer_path)

            def text_stream():
                for r in records:
                    yield r.content

            npy_path = tokenize_and_chunk(
                text_iterator=text_stream(),
                tokenizer=tokenizer,
                chunk_size=2048,
                output_dir=str(output_dir),
                max_tokens=None,
                batch_size=256,
            )

            console.print(f"  [cyan]Saved .npy:[/cyan] {Path(npy_path).resolve()}")
        except ImportError as e:
            console.print(f"  [yellow]Could not tokenize (missing dependency): {e}[/yellow]")
            console.print("  [dim]The JSONL file is still saved and can be tokenized later.[/dim]")
        except Exception as e:
            console.print(f"  [yellow]Tokenization failed: {e}[/yellow]")
            console.print("  [dim]The JSONL file is still saved and can be tokenized later.[/dim]")

    # Summary stats
    console.print()
    lang_counts: dict[str, int] = {}
    total_chars = 0
    for r in records:
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1
        total_chars += len(r.content)

    stats_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="[bold]Dataset Statistics[/bold]",
        title_style="bold white",
    )
    stats_table.add_column("Language", style="white")
    stats_table.add_column("Files", style="green", justify="right")
    stats_table.add_column("% of total", style="dim", justify="right")

    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        stats_table.add_row(lang, f"{count:,}", f"{pct:.1f}%")

    stats_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(records):,}[/bold]",
        "[bold]100%[/bold]",
    )

    console.print(stats_table)
    console.print()

    total_mb = total_chars / 1_000_000
    console.print(f"  [cyan]Total text:[/cyan] {total_mb:.1f} MB ({total_chars:,} chars)")

    console.print()
    console.print(Panel(
        f"[bold green]Done![/bold green]\n\n"
        f"Output: [cyan]{jsonl_path.resolve()}[/cyan]\n\n"
        f"Next steps:\n"
        f"  [dim]1. Review the data: head -5 {jsonl_path}[/dim]\n"
        f"  [dim]2. Tokenize: python scripts/prepare_data.py --tokenizer tokenizer.json[/dim]",
        title="[bold]Scraping Complete[/bold]",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive GitHub scraper for Cola-Coder training data.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Path to trained tokenizer.json (optional, for .npy output).",
    )
    args = parser.parse_args()

    # Validate tokenizer if provided
    if args.tokenizer and not Path(args.tokenizer).exists():
        console.print(f"[red]Error: Tokenizer not found: {args.tokenizer}[/red]")
        console.print("[dim]Train one first: python scripts/train_tokenizer.py[/dim]")
        sys.exit(1)

    try:
        settings = run_menu()
        if show_summary(settings):
            run_pipeline(settings, args.tokenizer)
        else:
            console.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
