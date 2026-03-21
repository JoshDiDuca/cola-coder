"""One-command train → eval → export pipeline for cola-coder.

Runs all pipeline stages in order (or a user-specified subset):
  tokenizer → data_prep → training → smoke_test → evaluation → export

Usage:
    # Run the full pipeline
    python scripts/run_pipeline.py --config configs/tiny.yaml

    # Run specific stages only
    python scripts/run_pipeline.py --config configs/tiny.yaml --stages training,evaluation,export

    # Skip all stages before 'training' (inclusive start at training)
    python scripts/run_pipeline.py --config configs/tiny.yaml --skip-to training

    # Preview what would run without executing
    python scripts/run_pipeline.py --config configs/tiny.yaml --dry-run

    # Export with a specific quantization level
    python scripts/run_pipeline.py --config configs/tiny.yaml --stages export --export-format gguf-q4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project src/ is importable when called directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.pipeline.orchestrator import PipelineOrchestrator, PipelineStage  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Stage name helpers
# ─────────────────────────────────────────────────────────────────────────────

_ALL_STAGES_ORDERED = list(PipelineStage)

_STAGE_BY_NAME: dict[str, PipelineStage] = {s.value: s for s in PipelineStage}


def _parse_stages(stages_str: str) -> list[PipelineStage]:
    """Parse a comma-separated list of stage names into PipelineStage values."""
    parts = [s.strip() for s in stages_str.split(",") if s.strip()]
    result = []
    for name in parts:
        if name not in _STAGE_BY_NAME:
            valid = ", ".join(s.value for s in PipelineStage)
            print(f"[error] Unknown stage: {name!r}. Valid stages: {valid}", file=sys.stderr)
            sys.exit(1)
        result.append(_STAGE_BY_NAME[name])
    return result


def _stages_from_skip_to(skip_to: str) -> list[PipelineStage]:
    """Return all stages starting from (and including) skip_to."""
    if skip_to not in _STAGE_BY_NAME:
        valid = ", ".join(s.value for s in PipelineStage)
        print(f"[error] Unknown stage: {skip_to!r}. Valid stages: {valid}", file=sys.stderr)
        sys.exit(1)

    target = _STAGE_BY_NAME[skip_to]
    idx = _ALL_STAGES_ORDERED.index(target)
    return _ALL_STAGES_ORDERED[idx:]


# ─────────────────────────────────────────────────────────────────────────────
# Rich display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(config_path: str, stages: list[PipelineStage], dry_run: bool) -> None:
    """Print a Rich panel showing the pipeline plan, with plain fallback."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]"
        stage_list = "  →  ".join(s.value for s in stages)
        content = (
            f"Config:  {config_path}\n"
            f"Stages:  {stage_list}\n"
            f"Mode:    {mode}"
        )
        console.print(Panel(content, title="[bold cyan]Cola-Coder Pipeline[/bold cyan]", expand=False))
    except ImportError:
        print("=" * 60)
        print("  Cola-Coder Pipeline")
        print(f"  Config: {config_path}")
        print(f"  Stages: {' → '.join(s.value for s in stages)}")
        if dry_run:
            print("  Mode: DRY RUN")
        print("=" * 60)


def _print_stage_start(stage: PipelineStage) -> None:
    """Print a progress indicator for the current stage."""
    try:
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold cyan]▶ Running stage:[/bold cyan] [bold white]{stage.value}[/bold white]")
    except ImportError:
        print(f"\n>> Running stage: {stage.value}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command train → eval → export pipeline for cola-coder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML (e.g. configs/tiny.yaml).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated list of stages to run. "
            "Valid values: "
            + ", ".join(s.value for s in PipelineStage)
            + ". Default: all stages."
        ),
    )
    parser.add_argument(
        "--skip-to",
        type=str,
        default=None,
        dest="skip_to",
        metavar="STAGE",
        help="Skip all stages before this one and start the pipeline from here.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        dest="skip_existing",
        help="Skip stages whose outputs already exist (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-run all stages even if their outputs exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would run without actually executing anything.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        default=False,
        help="Continue running subsequent stages even if a stage fails.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="gguf-q8",
        choices=["gguf-f16", "gguf-q8", "gguf-q4", "ollama", "quantize"],
        help="GGUF / export format for the export stage (default: gguf-q8).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for per-stage log files (default: pipeline_logs/).",
    )
    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Resolve stages
    if args.stages and args.skip_to:
        print("[error] Cannot use --stages and --skip-to together.", file=sys.stderr)
        return 1

    if args.stages:
        stages = _parse_stages(args.stages)
    elif args.skip_to:
        stages = _stages_from_skip_to(args.skip_to)
    else:
        stages = list(PipelineStage)

    _print_header(str(config_path), stages, args.dry_run)

    # Build and run the orchestrator
    orchestrator = PipelineOrchestrator(
        config_path=str(config_path),
        stages=stages,
        skip_existing=args.skip_existing,
        auto_resume=True,
        continue_on_failure=args.continue_on_failure,
        export_format=args.export_format,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
    )

    results = orchestrator.run()

    # Print final report
    print(orchestrator.format_report())

    # Exit with non-zero code if any stage failed
    any_failed = any(not r.success and not r.skipped for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
