"""Interactive GitHub data collector for Cola-Coder training data.

Uses the official GitHub REST API (api.github.com) to discover and download
source code from public repositories. NO HTML scraping — all access is through
the authenticated API with proper rate limiting.

A CLI menu that walks you through collecting from GitHub:
  1. Mode: Search by language / Search by topic / Import repo list / Clone single repo
  2. Filter preset or custom filters
  3. Max repos to clone
  4. Output directory and name

Then: API search -> shallow clone -> extract -> filter -> save as .npy

Requires: GITHUB_TOKEN env var for API authentication (higher rate limits).

Usage:
    python scripts/scrape_github.py
    python scripts/scrape_github.py --tokenizer tokenizer.json
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

from cola_coder.model.config import get_storage_config
from cola_coder.cli import cli
from cola_coder.security.scanner import CompositeMalwareScanner

# ---------------------------------------------------------------------------
# Menu definitions
# ---------------------------------------------------------------------------

MODE_OPTIONS = [
    {
        "label": "Search by language",
        "detail": "Find repos with a specific primary language",
        "mode": "language",
        "recommended": True,
    },
    {
        "label": "Search by topic",
        "detail": "Find repos tagged with specific topics",
        "mode": "topic",
    },
    {
        "label": "Import repo list",
        "detail": "Read repo names from a text file (one per line)",
        "mode": "import",
    },
    {
        "label": "Clone single repo",
        "detail": "Enter a single owner/repo to clone",
        "mode": "single",
    },
]

PRESET_OPTIONS = [
    {
        "label": "TypeScript Elite",
        "detail": ">500 stars, strict TS, has tests, MIT/Apache",
        "preset": "typescript_elite",
    },
    {
        "label": "TypeScript Good",
        "detail": ">50 stars, any TS, permissive license",
        "preset": "typescript_good",
        "recommended": True,
    },
    {
        "label": "Python Elite",
        "detail": ">500 stars, good Python repos, has tests",
        "preset": "python_elite",
    },
    {
        "label": "Popular (any language)",
        "detail": ">1000 stars, any language, permissive license",
        "preset": "popular_any",
    },
    {
        "label": "Custom filters...",
        "detail": "Configure your own star/fork/license criteria",
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
    {"label": "10 repos", "detail": "Quick test", "value": 10},
    {"label": "50 repos", "detail": "Small dataset", "value": 50, "recommended": True},
    {"label": "100 repos", "detail": "Medium dataset", "value": 100},
    {"label": "250 repos", "detail": "Large dataset", "value": 250},
    {"label": "500 repos", "detail": "Very large dataset", "value": 500},
    {"label": "Custom...", "detail": "Enter a number", "value": None},
]


# ---------------------------------------------------------------------------
# Custom filter builder
# ---------------------------------------------------------------------------

def build_custom_filter():
    """Walk the user through building a custom RepoFilter."""
    from cola_coder.data.sources.github import RepoFilter

    cli.header("Cola-Coder", "GitHub Scraper")
    cli.print()
    cli.print("  [bold cyan]Custom Filter Builder[/bold cyan]")
    cli.print()

    # Primary language
    lang_options = LANGUAGE_OPTIONS + [{"label": "Any", "detail": "No language filter"}]
    lang_idx = cli.choose("Primary language?", lang_options)
    if lang_idx is None:
        sys.exit(0)
    if lang_idx < len(LANGUAGE_OPTIONS):
        primary_language = LANGUAGE_OPTIONS[lang_idx]["value"]
    else:
        primary_language = None

    # Stars
    while True:
        try:
            raw = input("  Minimum stars? [50]: ").strip()
            min_stars = int(raw) if raw else 50
            break
        except ValueError:
            cli.print("  [red]Please enter a number.[/red]")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

    # Licenses
    license_indices = cli.multi_select(
        "Allowed licenses (select at least one):",
        LICENSE_OPTIONS,
        preselected=[0, 1],  # MIT + Apache preselected
    )
    licenses = [LICENSE_OPTIONS[i]["value"] for i in license_indices]

    # Has tests?
    has_tests_idx = cli.choose(
        "Require tests?",
        [
            {"label": "Yes", "detail": "Only repos with test files"},
            {"label": "No", "detail": "Include all repos"},
        ],
    )
    if has_tests_idx is None:
        sys.exit(0)
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
    storage = get_storage_config()

    # Step 1: Mode
    mode_idx = cli.choose(
        "Step 1/4 - Mode: How do you want to find repositories?",
        MODE_OPTIONS,
    )
    if mode_idx is None:
        sys.exit(0)
    mode = MODE_OPTIONS[mode_idx]["mode"]

    settings: dict = {"mode": mode}

    # Mode-specific settings
    if mode == "single":
        cli.header("Cola-Coder", "GitHub Scraper")
        try:
            raw = input("  Enter repository (owner/repo) [microsoft/TypeScript]: ").strip()
            repo_name = raw if raw else "microsoft/TypeScript"
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        settings["repos"] = [repo_name]
        settings["filter"] = None
        settings["max_repos"] = 1
    elif mode == "import":
        cli.header("Cola-Coder", "GitHub Scraper")
        try:
            raw = input("  Path to repo list file [repos.txt]: ").strip()
            file_path = raw if raw else "repos.txt"
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if not Path(file_path).exists():
            cli.print(f"  [red]File not found: {file_path}[/red]")
            sys.exit(1)
        repos = [
            line.strip() for line in Path(file_path).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        cli.print(f"  [green]Found {len(repos)} repos in {file_path}[/green]")
        settings["repos"] = repos
        settings["filter"] = None
        settings["max_repos"] = len(repos)
    else:
        # Search modes: select filter preset
        # Step 2: Filter
        preset_idx = cli.choose(
            "Step 2/4 - Filter: Which filter preset?",
            PRESET_OPTIONS,
        )
        if preset_idx is None:
            sys.exit(0)
        preset_name = PRESET_OPTIONS[preset_idx]["preset"]

        if preset_name is None:
            settings["filter"] = "custom"
            settings["custom_filter"] = build_custom_filter()
        else:
            settings["filter"] = preset_name

        if mode == "topic":
            cli.header("Cola-Coder", "GitHub Scraper")
            try:
                raw = input("  Enter topics (comma-separated) [react,nextjs]: ").strip()
                topics = raw if raw else "react,nextjs"
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            settings["topics"] = [t.strip() for t in topics.split(",") if t.strip()]

        # Step 3: Max repos
        max_idx = cli.choose(
            "Step 3/4 - Count: How many repos to process?",
            MAX_REPOS_OPTIONS,
        )
        if max_idx is None:
            sys.exit(0)
        max_val = MAX_REPOS_OPTIONS[max_idx]["value"]
        if max_val is None:
            while True:
                try:
                    raw = input("  Enter number of repos [100]: ").strip()
                    max_val = int(raw) if raw else 100
                    break
                except ValueError:
                    cli.print("  [red]Please enter a number.[/red]")
                except (EOFError, KeyboardInterrupt):
                    sys.exit(0)
        settings["max_repos"] = max_val

    # Step 4: Output
    cli.header("Cola-Coder", "GitHub Scraper")
    default_output = str(Path(storage.data_dir) / "github_scraped")
    try:
        raw = input(f"  Output directory [{default_output}]: ").strip()
        settings["output_dir"] = raw if raw else default_output
        raw = input("  Dataset name [github_code]: ").strip()
        settings["output_name"] = raw if raw else "github_code"
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    # Extract languages (for RepoProcessor)
    extract_indices = cli.multi_select(
        "Step 4/4 - Extract Languages: Which languages to extract from repos?",
        LANGUAGE_OPTIONS,
        preselected=[0, 1],  # TypeScript + JavaScript
    )
    settings["extract_languages"] = [LANGUAGE_OPTIONS[i]["value"] for i in extract_indices]

    return settings


def show_summary(settings: dict) -> bool:
    """Show a summary and ask for confirmation."""
    cli.header("Cola-Coder", "GitHub Scraper")
    cli.print()

    summary: dict[str, str] = {"Mode": settings["mode"]}

    if settings.get("filter"):
        summary["Filter"] = settings["filter"]

    if settings.get("repos"):
        repo_str = ", ".join(settings["repos"][:5])
        if len(settings["repos"]) > 5:
            repo_str += f"  (+{len(settings['repos']) - 5} more)"
        summary["Repos"] = repo_str

    if settings.get("topics"):
        summary["Topics"] = ", ".join(settings["topics"])

    summary["Max repos"] = str(settings["max_repos"])
    summary["Extract langs"] = ", ".join(settings["extract_languages"])
    summary["Output dir"] = settings["output_dir"]
    summary["Dataset name"] = settings["output_name"]

    # Check for GITHUB_TOKEN
    has_token = bool(os.environ.get("GITHUB_TOKEN"))
    token_str = "[green]Set[/green]" if has_token else "[yellow]Not set (60 req/hr limit)[/yellow]"
    summary["GITHUB_TOKEN"] = token_str

    cli.kv_table(summary, title="Summary")
    cli.print()

    if not has_token:
        cli.print("  [yellow]Tip: Set GITHUB_TOKEN env var for 5000 requests/hour.[/yellow]")

    return cli.confirm("Start scraping?", default=True)


def _load_scan_config() -> dict:
    """Load malware_scan config from scoring.yaml."""
    scoring_path = Path("configs/scoring.yaml")
    if not scoring_path.exists():
        return {}
    with open(scoring_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("scoring", {}).get("security", {})


def _scan_cloned_repo(
    clone_dir: Path,
    repo_name: str,
    scan_config: dict,
) -> bool:
    """Scan a cloned repo for malware. Returns True if clean."""
    scanner = CompositeMalwareScanner.from_config(scan_config.get("malware_scan", {}))
    if not scanner.available_scanners:
        return True

    result = scanner.scan_directory(clone_dir)
    if not result.is_clean:
        for t in result.threats:
            logger.warning(
                "MALWARE DETECTED in cloned repo %s [%s/%s]: %s in %s",
                repo_name, t.scanner, t.severity, t.name, t.file_path,
            )
        cli.warn(f"Threats in {repo_name}: {[t.name for t in result.threats]}")
        for t in result.threats:
            cli.error(f"  [{t.severity.upper()}] {t.name}: {Path(t.file_path).name}")
        return False
    return True


def run_pipeline(settings: dict, tokenizer_path: str | None):
    """Run the scraping pipeline with the selected settings."""
    from cola_coder.data.sources.github import (
        GitHubClient, GitHubSource, RepoFilter, RepoProcessor,
        FILTER_PRESETS,
    )
    from cola_coder.data.pipeline import DataRecord

    cli.print()
    cli.print("[bold cyan]Starting GitHub scraper...[/bold cyan]")
    cli.print()

    scan_config = _load_scan_config()

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
            cli.print(f"  [cyan]Cloning {repo_name}...[/cyan]")
            try:
                repo_path = client.clone_repo(repo_name, clone_dir, shallow=True)
            except Exception as e:
                cli.print(f"  [red]Failed to clone {repo_name}: {e}[/red]")
                continue

            # Scan for malware before extraction
            if not _scan_cloned_repo(repo_path, repo_name, scan_config):
                cli.warn(f"  Skipping {repo_name} due to detected threats.")
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
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

                cli.print(
                    f"  [green]Extracted {file_count} files from {repo_name}[/green]"
                )
            finally:
                # Cleanup
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

        cli.print(f"  [dim]Query: {repo_filter.to_github_query()}[/dim]")
        cli.print()

        file_count = 0
        for record in source.stream(max_repos=settings["max_repos"]):
            records.append(record)
            file_count += 1
            if file_count % 100 == 0:
                cli.print(f"  [dim]  ... {file_count} files extracted so far[/dim]")

    cli.print()
    cli.print(f"[bold]Total files extracted: {len(records)}[/bold]")

    if not records:
        cli.print("[yellow]No files extracted. Check your filters or try a different query.[/yellow]")
        return

    # Save as JSON lines (each line is one record)
    jsonl_path = output_dir / f"{settings['output_name']}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            line = {
                "content": record.content,
                "file_path": record.metadata.get("file_path", ""),
                "language": record.metadata.get("language", ""),
                "repo_name": record.metadata.get("repo_name", ""),
                "repo_stars": record.metadata.get("repo_stars", 0),
                "repo_url": record.metadata.get("repo_url", ""),
                "license": record.metadata.get("license", ""),
                "file_size": record.metadata.get("file_size", 0),
            }
            f.write(json.dumps(line) + "\n")

    cli.print(f"  [cyan]Saved JSONL:[/cyan] {jsonl_path.resolve()}")

    # Optionally tokenize to .npy if tokenizer is provided
    if tokenizer_path and Path(tokenizer_path).exists():
        cli.print()
        cli.print("[dim]Tokenizing to .npy...[/dim]")

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

            cli.print(f"  [cyan]Saved .npy:[/cyan] {Path(npy_path).resolve()}")
        except ImportError as e:
            cli.print(f"  [yellow]Could not tokenize (missing dependency): {e}[/yellow]")
            cli.print("  [dim]The JSONL file is still saved and can be tokenized later.[/dim]")
        except Exception as e:
            cli.print(f"  [yellow]Tokenization failed: {e}[/yellow]")
            cli.print("  [dim]The JSONL file is still saved and can be tokenized later.[/dim]")

    # Summary stats
    cli.print()
    lang_counts: dict[str, int] = {}
    total_chars = 0
    for r in records:
        lang = r.metadata.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        total_chars += len(r.content)

    stats = {}
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        stats[lang] = f"{count:,} files ({pct:.1f}%)"
    stats["Total"] = f"{len(records):,} files (100%)"
    cli.kv_table(stats, title="Dataset Statistics")

    cli.print()

    total_mb = total_chars / 1_000_000
    cli.print(f"  [cyan]Total text:[/cyan] {total_mb:.1f} MB ({total_chars:,} chars)")

    cli.done(
        "Scraping Complete",
        extras={
            "Output": str(jsonl_path.resolve()),
            "Next steps": (
                f"1. Review: head -5 {jsonl_path}  "
                "2. Tokenize: python scripts/prepare_data.py --tokenizer tokenizer.json"
            ),
        },
    )


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
        cli.print(f"[red]Error: Tokenizer not found: {args.tokenizer}[/red]")
        cli.print("[dim]Train one first: python scripts/train_tokenizer.py[/dim]")
        sys.exit(1)

    try:
        settings = run_menu()
        if show_summary(settings):
            run_pipeline(settings, args.tokenizer)
        else:
            cli.print("\n[red]Cancelled.[/red]")
    except KeyboardInterrupt:
        cli.print("\n[red]Cancelled.[/red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
