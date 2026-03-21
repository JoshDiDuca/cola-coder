"""Generate a comprehensive quality report for a model checkpoint.

Runs smoke tests, generates sample outputs, and optionally evaluates on
HumanEval. Saves a markdown + JSON report to the output directory.

Usage:
    python scripts/quality_report.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/quality_report.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --eval
    python scripts/quality_report.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --output reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


def _parse_args() -> argparse.Namespace:
    storage_default_config = "configs/tiny.yaml"
    try:
        from cola_coder.model.config import get_storage_config
        storage = get_storage_config()
        # If we can detect the latest checkpoint from storage, use that as hint
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Generate a comprehensive quality report for a model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g. checkpoints/tiny/latest or checkpoints/tiny/step_00020000).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. configs/tiny.yaml). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/",
        help="Output directory for the report files (default: reports/).",
    )
    parser.add_argument(
        "--eval",
        dest="run_eval",
        action="store_true",
        help="Run HumanEval subset and include pass@1 in the report.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of standard-prompt samples to generate (default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda' or 'cpu' (auto-detected by default).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the report but don't save files.",
    )
    return parser.parse_args()


def _auto_detect_checkpoint() -> str | None:
    """Try to auto-detect the latest checkpoint."""
    try:
        from cola_coder.model.config import get_storage_config
        from cola_coder.training.checkpoint import detect_latest_checkpoint

        storage = get_storage_config()
        result = detect_latest_checkpoint(storage.checkpoints_dir)
        if result is None:
            return None
        raw_path, _ = result
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = Path(storage.checkpoints_dir).parent / resolved
        return str(resolved)
    except Exception:
        return None


def _auto_detect_config(checkpoint_path: str) -> str | None:
    """Try to auto-detect the config from the checkpoint path."""
    ckpt_path = Path(checkpoint_path)
    size_name = ckpt_path.parent.name
    candidate = Path("configs") / f"{size_name}.yaml"
    if candidate.exists():
        return str(candidate)
    # Try any yaml in configs/
    configs_dir = Path("configs")
    if configs_dir.exists():
        yamls = sorted(configs_dir.glob("*.yaml"))
        for y in yamls:
            if y.stem not in {"features", "storage", "reasoning", "pipeline", "specialists"}:
                return str(y)
    return None


def main() -> int:
    args = _parse_args()
    cli.header("Cola-Coder", "Quality Report")

    # ── Auto-detect checkpoint ────────────────────────────────────────────
    if args.checkpoint is None:
        cli.print("Auto-detecting latest checkpoint...")
        args.checkpoint = _auto_detect_checkpoint()
        if args.checkpoint is None:
            cli.fatal(
                "No checkpoint found.",
                hint="Pass --checkpoint path/to/checkpoint or train a model first.",
            )
        cli.info("Checkpoint", args.checkpoint)

    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}")

    # ── Auto-detect config ────────────────────────────────────────────────
    if args.config is None:
        args.config = _auto_detect_config(args.checkpoint)
        if args.config is None:
            cli.fatal(
                "Could not auto-detect a config file.",
                hint="Pass --config configs/<size>.yaml explicitly.",
            )
        cli.info("Config", args.config)

    if not Path(args.config).exists():
        cli.fatal(f"Config not found: {args.config}")

    # ── Resolve device ────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    else:
        device = cli.gpu_info()

    cli.info("Device", device)
    cli.info("Output dir", args.output)
    if args.run_eval:
        cli.info("HumanEval", "enabled (subset)")

    # ── Generate report ───────────────────────────────────────────────────
    cli.print("Generating quality report...")

    try:
        from cola_coder.evaluation.quality_report import QualityReportGenerator
    except ImportError as e:
        cli.fatal(f"Import error: {e}", hint="Make sure the package is installed: pip install -e .")

    generator = QualityReportGenerator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
    )

    try:
        report = generator.generate(
            run_eval=args.run_eval,
            num_samples=args.num_samples,
        )
    except Exception as e:
        cli.fatal(f"Report generation failed: {e}")
        return 1  # unreachable but satisfies type checker

    # ── Print summary ─────────────────────────────────────────────────────
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        console.print()
        console.rule("[bold cyan]Quality Report Summary[/bold cyan]")

        # Model info
        from cola_coder.evaluation.quality_report import _human_params
        params_str = _human_params(report.model_params)
        console.print(f"  Parameters: [bold]{params_str}[/bold]")
        console.print(f"  Step: [bold]{report.training_step:,}[/bold]")
        console.print(f"  Loss: [bold]{report.training_loss:.4f}[/bold]")

        # Smoke test table
        smoke_status = "[bold green]PASSED[/bold green]" if report.smoke_test_passed else "[bold red]FAILED[/bold red]"
        num_p = sum(1 for d in report.smoke_test_details if d.get("passed"))
        num_t = len(report.smoke_test_details)
        console.print(f"\n  Smoke test: {smoke_status} ({num_p}/{num_t})")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0, 1))
        table.add_column("Test", style="white")
        table.add_column("Result", width=6, justify="center")
        table.add_column("ms", justify="right", style="dim", width=8)

        for detail in report.smoke_test_details:
            result_str = "[green]PASS[/green]" if detail.get("passed") else "[red]FAIL[/red]"
            table.add_row(
                detail.get("name", "?"),
                result_str,
                f"{detail.get('duration_ms', 0):.0f}",
            )

        console.print(table)

        # Sample outputs
        if report.samples:
            console.rule("[bold cyan]Sample Outputs[/bold cyan]")
            for sample in report.samples[:2]:  # Show first 2 to keep output manageable
                prompt_line = sample["prompt"].split("\n")[0][:50]
                console.print(f"\n[bold]Prompt:[/bold] {prompt_line}...")
                console.print(f"[dim]{sample['output'][:300]}[/dim]")

        if report.humaneval_pass_at_1 is not None:
            console.print(f"\n  HumanEval pass@1: [bold]{report.humaneval_pass_at_1 * 100:.1f}%[/bold]")

        console.print()

    except ImportError:
        # Plain fallback
        print(f"\nStep: {report.training_step}")
        print(f"Loss: {report.training_loss:.4f}")
        smoke_str = "PASSED" if report.smoke_test_passed else "FAILED"
        num_p = sum(1 for d in report.smoke_test_details if d.get("passed"))
        print(f"Smoke test: {smoke_str} ({num_p}/{len(report.smoke_test_details)})")

    # ── Save report ───────────────────────────────────────────────────────
    if not args.no_save:
        generator.save_report(report, output_dir=args.output)
        from pathlib import Path as _P
        from cola_coder.evaluation.quality_report import _human_params as _hp
        out_path = _P(args.output)
        ckpt_name = _P(report.checkpoint_path).name
        step_str = f"step_{report.training_step:07d}"
        base_name = f"quality_report_{ckpt_name}_{step_str}"
        cli.success(f"Report saved to {out_path / base_name}.md / .json")

    return 0 if report.smoke_test_passed else 1


if __name__ == "__main__":
    sys.exit(main())
