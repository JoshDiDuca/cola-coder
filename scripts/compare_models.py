"""Compare multiple model checkpoints side-by-side.

Loads each checkpoint (one at a time to save VRAM), generates outputs for the
same set of prompts, and displays a Rich comparison table with performance
metrics.

Usage:
    python scripts/compare_models.py --checkpoints checkpoints/tiny/step_18000 checkpoints/tiny/step_20000 --config configs/tiny.yaml
    python scripts/compare_models.py --checkpoints checkpoints/tiny/step_18000 checkpoints/tiny/step_20000 --configs configs/tiny.yaml configs/tiny.yaml
    python scripts/compare_models.py --checkpoints checkpoints/tiny/step_20000 --config configs/tiny.yaml --prompts "def fib(n):" "class Stack:"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple cola-coder model checkpoints side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint directory paths to compare.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Single YAML config used for all checkpoints.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Per-checkpoint YAML configs (must match --checkpoints count). "
             "Overrides --config if provided.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompts to use. Defaults to 3 standard prompts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens per generation (default: 128).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda' or 'cpu' (auto-detected by default).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison as markdown to this file.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only 3 short prompts and 128 max tokens (fast comparison).",
    )
    return parser.parse_args()


def _auto_detect_config(checkpoint_path: str) -> str | None:
    """Try to find a config yaml for the given checkpoint."""
    ckpt_path = Path(checkpoint_path)
    size_name = ckpt_path.parent.name
    candidate = Path("configs") / f"{size_name}.yaml"
    if candidate.exists():
        return str(candidate)
    configs_dir = Path("configs")
    if configs_dir.exists():
        yamls = sorted(configs_dir.glob("*.yaml"))
        for y in yamls:
            if y.stem not in {"features", "storage", "reasoning", "pipeline", "specialists"}:
                return str(y)
    return None


def _print_rich_comparison(result: "ComparisonResult") -> None:  # noqa: F821
    """Print the comparison using Rich tables."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()

        console.print()
        console.rule("[bold cyan]Model Comparison[/bold cyan]")

        # Model summary table
        model_table = Table(
            title="Models",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        model_table.add_column("#", width=4, justify="right")
        model_table.add_column("Checkpoint")
        model_table.add_column("Params", justify="right")
        model_table.add_column("Step", justify="right")
        model_table.add_column("Loss", justify="right")
        model_table.add_column("Tokens/s", justify="right")

        for i, (m, metric) in enumerate(zip(result.models, result.metrics)):
            from cola_coder.evaluation.quality_report import _human_params
            params_str = m.get("params_human", _human_params(m.get("params", 0)))
            step = m.get("step", "?")
            step_str = f"{step:,}" if isinstance(step, int) else str(step)
            loss = m.get("loss", float("nan"))
            loss_str = f"{loss:.4f}" if isinstance(loss, float) and loss == loss else "N/A"
            tps = metric.get("tokens_per_sec", 0)
            tps_str = f"{tps:.1f}"
            model_table.add_row(
                str(i + 1),
                m.get("name", "?"),
                params_str,
                step_str,
                loss_str,
                tps_str,
            )

        console.print(model_table)
        console.print()

        # Per-prompt output panels
        for p_idx, prompt in enumerate(result.prompts):
            short_prompt = prompt.split("\n")[0][:60]
            console.rule(f"[bold]Prompt {p_idx + 1}:[/bold] {short_prompt}")
            console.print()

            for m_idx, model_info in enumerate(result.models):
                name = model_info.get("name", f"Model {m_idx + 1}")
                output = result.outputs[m_idx][p_idx] if m_idx < len(result.outputs) else ""

                # Truncate very long outputs for display
                display_output = output[:400] + "..." if len(output) > 400 else output

                console.print(Panel(
                    display_output,
                    title=f"[bold cyan]{name}[/bold cyan]",
                    border_style="dim",
                    expand=True,
                ))
            console.print()

        # Performance metrics
        metrics_table = Table(
            title="Performance Metrics",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        metrics_table.add_column("Model")
        metrics_table.add_column("Step", justify="right")
        metrics_table.add_column("Loss", justify="right")
        metrics_table.add_column("Tokens/s", justify="right")
        metrics_table.add_column("Avg Len", justify="right")

        # Determine best (fastest, lowest loss)
        losses = [m.get("loss", float("nan")) for m in result.models]
        valid_losses = [v for v in losses if v == v]  # filter NaN
        best_loss = min(valid_losses) if valid_losses else None

        tpss = [m.get("tokens_per_sec", 0.0) for m in result.metrics]
        best_tps = max(tpss) if tpss else None

        for m_idx, (model_info, metric) in enumerate(zip(result.models, result.metrics)):
            name = model_info.get("name", f"Model {m_idx + 1}")
            step = model_info.get("step", "?")
            step_str = f"{step:,}" if isinstance(step, int) else str(step)
            loss = model_info.get("loss", float("nan"))
            loss_val = loss if isinstance(loss, float) and loss == loss else None
            if loss_val is not None:
                loss_str = f"{loss_val:.4f}"
                if best_loss is not None and abs(loss_val - best_loss) < 1e-8:
                    loss_str = f"[bold green]{loss_str}[/bold green]"
            else:
                loss_str = "N/A"

            tps = metric.get("tokens_per_sec", 0.0)
            tps_str = f"{tps:.1f}"
            if best_tps is not None and abs(tps - best_tps) < 1e-3 and tps > 0:
                tps_str = f"[bold green]{tps_str}[/bold green]"

            avg_len = metric.get("avg_output_len", 0.0)
            metrics_table.add_row(
                name,
                step_str,
                loss_str,
                tps_str,
                f"{avg_len:.0f}",
            )

        console.print(metrics_table)

    except ImportError:
        # Plain fallback
        print("\n=== Model Comparison ===")
        for i, m in enumerate(result.models):
            print(f"Model {i + 1}: {m.get('name')}  step={m.get('step')}  loss={m.get('loss')}")
        print()
        for p_idx, prompt in enumerate(result.prompts):
            print(f"--- Prompt {p_idx + 1}: {prompt[:50]} ---")
            for m_idx, m in enumerate(result.models):
                output = result.outputs[m_idx][p_idx] if m_idx < len(result.outputs) else ""
                print(f"  [{m.get('name')}]: {output[:200]}")
            print()


def main() -> int:
    args = _parse_args()
    cli.header("Cola-Coder", "Model Comparison")

    # ── Resolve configs ───────────────────────────────────────────────────
    if args.configs is not None:
        configs = args.configs
    elif args.config is not None:
        configs = [args.config] * len(args.checkpoints)
    else:
        # Auto-detect per checkpoint
        configs = []
        for ckpt in args.checkpoints:
            c = _auto_detect_config(ckpt)
            if c is None:
                cli.fatal(
                    f"Could not auto-detect config for {ckpt}",
                    hint="Pass --config configs/<size>.yaml explicitly.",
                )
            configs.append(c)

    if len(configs) == 1:
        configs = configs * len(args.checkpoints)

    if len(configs) != len(args.checkpoints):
        cli.fatal(
            f"Number of configs ({len(configs)}) must match checkpoints ({len(args.checkpoints)})"
        )

    # ── Validate paths ────────────────────────────────────────────────────
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            cli.fatal(f"Checkpoint not found: {ckpt}")
    for cfg in configs:
        if not Path(cfg).exists():
            cli.fatal(f"Config not found: {cfg}")

    # ── Device ────────────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    else:
        device = cli.gpu_info()

    cli.info("Device", device)
    cli.info("Checkpoints", str(len(args.checkpoints)))

    # ── Set up prompts ────────────────────────────────────────────────────
    prompts = args.prompts
    if args.quick or prompts is None:
        from cola_coder.evaluation.model_comparison import DEFAULT_COMPARISON_PROMPTS
        prompts = prompts or DEFAULT_COMPARISON_PROMPTS
        max_tokens = 128
    else:
        max_tokens = args.max_tokens

    # ── Run comparison ────────────────────────────────────────────────────
    cli.print("Running comparison (loading models one at a time)...")

    try:
        from cola_coder.evaluation.model_comparison import ModelComparator
    except ImportError as e:
        cli.fatal(f"Import error: {e}", hint="Make sure the package is installed: pip install -e .")

    comparator = ModelComparator(
        checkpoints=args.checkpoints,
        configs=configs,
        device=device,
    )

    try:
        result = comparator.compare(
            prompts=prompts,
            temperature=args.temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        cli.fatal(f"Comparison failed: {e}")
        return 1

    # ── Display results ───────────────────────────────────────────────────
    _print_rich_comparison(result)

    # ── Save markdown ─────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_markdown(), encoding="utf-8")
        cli.success(f"Comparison saved to {args.output}")

    cli.done("Comparison complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
