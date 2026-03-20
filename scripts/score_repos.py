"""Standalone CLI tool to score repositories by test quality.

Usage:
    # Score a single repo
    python scripts/score_repos.py /path/to/repo

    # Score all repos in a directory
    python scripts/score_repos.py /path/to/repos/ --all

    # Dry-run mode (detect only, don't execute)
    python scripts/score_repos.py /path/to/repos/ --all --mode dry_run

    # Docker mode (safest)
    python scripts/score_repos.py /path/to/repos/ --all --mode docker

    # Parallel execution
    python scripts/score_repos.py /path/to/repos/ --all --workers 8

    # Output results as JSON
    python scripts/score_repos.py /path/to/repos/ --all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.data.curation.test_runner import TestRunner
from cola_coder.data.curation.test_scorer import RepoScore
from cola_coder.model.config import get_storage_config


def find_repos(base_dir: Path) -> list[Path]:
    """Find all repository-like directories under base_dir.

    A directory is considered a repo if it contains any of:
    package.json, pyproject.toml, setup.py, go.mod, Cargo.toml, .git/
    """
    repos = []
    indicators = [
        "package.json", "pyproject.toml", "setup.py", "setup.cfg",
        "go.mod", "Cargo.toml", ".git",
    ]

    if any((base_dir / ind).exists() for ind in indicators):
        # base_dir itself is a repo
        repos.append(base_dir)
    else:
        # Check immediate subdirectories
        for child in sorted(base_dir.iterdir()):
            if child.is_dir() and any((child / ind).exists() for ind in indicators):
                repos.append(child)

    return repos


def print_results(results: dict[Path, RepoScore], json_output: bool = False) -> None:
    """Print scoring results in a readable format."""
    if json_output:
        out = {}
        for path, score in results.items():
            out[str(path)] = score.to_dict()
        print(json.dumps(out, indent=2))
        return

    # Table format
    tier_colors = {
        "verified": "\033[92m",  # green
        "tested": "\033[93m",    # yellow
        "detected": "\033[94m",  # blue
        "none": "\033[90m",      # gray
    }
    reset = "\033[0m"

    print()
    print(f"{'Repository':<50} {'Tier':<12} {'Score':<8} {'Tests':<15} {'Details'}")
    print("-" * 110)

    for path, score in sorted(results.items(), key=lambda x: -x[1].score):
        name = path.name
        if len(name) > 48:
            name = name[:45] + "..."

        color = tier_colors.get(score.quality_tier, "")
        tier_display = f"{color}{score.quality_tier:<12}{reset}"

        tests_info = ""
        if score.test_result:
            tr = score.test_result
            tests_info = f"{tr.passed}/{tr.total_tests} passed"
            if tr.coverage is not None:
                tests_info += f" ({tr.coverage:.0%} cov)"
        elif score.tests_detected:
            tests_info = "detected"
        else:
            tests_info = "none"

        details = ""
        if score.test_result and score.test_result.framework:
            details = score.test_result.framework
        if score.test_result and score.test_result.error:
            details += f" [{score.test_result.error[:40]}]"

        print(f"  {name:<48} {tier_display} {score.score:<8.3f} {tests_info:<15} {details}")

    print()

    # Summary
    tier_counts = {}
    for score in results.values():
        tier_counts[score.quality_tier] = tier_counts.get(score.quality_tier, 0) + 1

    print("Summary:")
    for tier in ["verified", "tested", "detected", "none"]:
        count = tier_counts.get(tier, 0)
        if count > 0:
            color = tier_colors.get(tier, "")
            print(f"  {color}{tier}{reset}: {count} repos")

    total = len(results)
    avg_score = sum(s.score for s in results.values()) / total if total else 0
    print(f"\n  Total: {total} repos, average score: {avg_score:.3f}")
    print()


def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Score repositories by test quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", type=Path, help="Path to repo or directory of repos")
    parser.add_argument("--all", action="store_true",
                        help="Score all repos in directory (default: treat path as single repo)")
    parser.add_argument("--mode", choices=["subprocess", "docker", "dry_run"],
                        default="dry_run",
                        help="Execution mode (default: dry_run)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Test timeout in seconds (default: 300)")
    parser.add_argument("--install-timeout", type=int, default=120,
                        help="Install timeout in seconds (default: 120)")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(storage.cache_dir) / "test_cache",
                        help="Cache directory (default: data/test_cache/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached results")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find repos
    if args.all:
        repos = find_repos(args.path)
    else:
        repos = [args.path]

    if not repos:
        print(f"No repositories found in {args.path}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"\nScoring {len(repos)} repo(s) in {args.mode} mode...")
        if args.mode == "dry_run":
            print("  (dry_run = detect tests only, no execution)")
        print()

    # Create runner and score
    runner = TestRunner(
        mode=args.mode,
        timeout=args.timeout,
        install_timeout=args.install_timeout,
        cache_dir=args.cache_dir,
    )

    use_cache = not args.no_cache

    if len(repos) == 1:
        score = runner.score_repo(repos[0], use_cache=use_cache)
        results = {repos[0]: score}
    else:
        results = runner.score_repos_parallel(
            repos, max_workers=args.workers, use_cache=use_cache,
        )

    print_results(results, json_output=args.json)


if __name__ == "__main__":
    main()
