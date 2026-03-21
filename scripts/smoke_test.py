"""Run smoke tests against a trained cola-coder model.

Quick validation that a model checkpoint generates reasonable code — runs in
under 30 seconds and exits 0 if everything looks healthy, 1 if any test fails.

Usage:
    python scripts/smoke_test.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/smoke_test.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --json
    python scripts/smoke_test.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cola_coder.cli import cli


def _parse_args() -> argparse.Namespace:
    storage_default = None
    try:
        from cola_coder.model.config import get_storage_config
        storage_default = get_storage_config().tokenizer_path
    except Exception:
        storage_default = "tokenizer.json"

    parser = argparse.ArgumentParser(
        description="Run smoke tests against a trained cola-coder model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (auto-detected from checkpoint if omitted).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=storage_default,
        help=f"Path to tokenizer.json (default: {storage_default}).",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Print machine-readable JSON instead of pretty output.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the 3 fastest tests (generates_tokens, repetition, code_keywords).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda' or 'cpu' (auto-detected by default).",
    )
    return parser.parse_args()


def _print_results(report, output_json: bool) -> None:
    """Pretty-print or JSON-dump the report."""
    if output_json:
        data = {
            "passed": report.passed,
            "summary": report.summary,
            "total_duration_ms": report.total_duration_ms,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in report.results
            ],
        }
        print(json.dumps(data, indent=2))
        return

    # Pretty-print with Rich (or plain fallback)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        # Force UTF-8 on Windows
        if sys.platform == "win32":
            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        console = Console()
        table = Table(
            title="Smoke Test Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Status", width=6, justify="center")
        table.add_column("Test", style="bold white")
        table.add_column("Message", style="dim")
        table.add_column("ms", justify="right", style="dim", width=8)

        for r in report.results:
            status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
            table.add_row(status, r.name, r.message, f"{r.duration_ms:.0f}")

        console.print()
        console.print(table)
        console.print()

        if report.passed:
            console.print(f"[bold green]PASS[/bold green] {report.summary}")
        else:
            console.print(f"[bold red]FAIL[/bold red] {report.summary}")
        console.print()

    except ImportError:
        # Plain fallback
        print()
        width = 70
        print("=" * width)
        print("  Smoke Test Results")
        print("=" * width)
        for r in report.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name:<40} {r.duration_ms:6.0f}ms")
            print(f"         {r.message}")
        print("=" * width)
        print(f"  {report.summary}")
        print("=" * width)
        print()


def main() -> int:
    args = _parse_args()

    if not args.output_json:
        cli.header("Cola-Coder", "Smoke Test")

    # ── Auto-detect checkpoint ────────────────────────────────────────────
    _metadata: dict = {}
    if args.checkpoint is None:
        try:
            from cola_coder.model.config import get_storage_config
            from cola_coder.training.checkpoint import detect_latest_checkpoint

            storage = get_storage_config()
            result = detect_latest_checkpoint(storage.checkpoints_dir)
            if result is None:
                cli.fatal(
                    f"No checkpoint found in {storage.checkpoints_dir}",
                    hint="Pass --checkpoint path/to/ckpt or train a model first",
                )
            raw_path, _metadata = result
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = Path(storage.checkpoints_dir).parent / resolved
            args.checkpoint = str(resolved)
            if not args.output_json:
                cli.info("Auto-detected checkpoint", args.checkpoint)
        except ImportError:
            cli.fatal(
                "Could not import cola_coder. Make sure the package is installed.",
                hint="Try: pip install -e .",
            )

    # ── Auto-detect config ────────────────────────────────────────────────
    if args.config is None:
        ckpt_path = Path(args.checkpoint)
        size_name = ckpt_path.parent.name
        yaml_candidate = Path("configs") / f"{size_name}.yaml"
        if yaml_candidate.exists():
            args.config = str(yaml_candidate)
            if not args.output_json:
                cli.info("Auto-detected config", args.config)
        else:
            configs_dir = Path("configs")
            if configs_dir.exists():
                yamls = sorted(configs_dir.glob("*.yaml"))
                if yamls:
                    args.config = str(yamls[0])
                    if not args.output_json:
                        cli.info("Auto-detected config", args.config)
            if args.config is None:
                cli.fatal(
                    "Could not auto-detect a config file",
                    hint="Pass --config configs/<size>.yaml explicitly",
                )

    # ── Validate paths ────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}")
    if not Path(args.tokenizer).exists():
        cli.fatal(f"Tokenizer not found: {args.tokenizer}")

    # ── Determine device ──────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    elif not args.output_json:
        device = cli.gpu_info()
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # ── Load model ────────────────────────────────────────────────────────
    if not args.output_json:
        cli.print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        config = Config.from_yaml(args.config)
        if not args.output_json:
            cli.info("Model", f"{config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        if not args.output_json:
            cli.info("Tokenizer", f"{tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        if not args.output_json:
            cli.info("Device", device)

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ── Run smoke tests ───────────────────────────────────────────────────
    from cola_coder.evaluation.smoke_test import SmokeTest

    smoke = SmokeTest(generator=generator, tokenizer=tokenizer)

    if args.quick:
        # Run only the 3 fastest tests individually
        if not args.output_json:
            cli.print("Running quick tests (3 fastest)...")
        import time
        from cola_coder.evaluation.smoke_test import SmokeTestReport

        t0 = time.perf_counter()
        results = [
            smoke.test_generates_tokens(),
            smoke.test_repetition(),
            smoke.test_code_keywords(),
        ]
        report = SmokeTestReport(
            results=results,
            total_duration_ms=(time.perf_counter() - t0) * 1000,
        )
    else:
        if not args.output_json:
            cli.print("Running all smoke tests...")
        report = smoke.run_all()

    # ── Print results ─────────────────────────────────────────────────────
    _print_results(report, args.output_json)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
