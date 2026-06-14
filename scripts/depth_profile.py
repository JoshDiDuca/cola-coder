"""Logit-lens per-token DEPTH / convergence profiler (INFER-031).

Decodes every transformer layer through the model's tied output head and reports,
per token, the earliest layer at which the next-token prediction has converged to
the final layer's answer. Aggregated over a few prompts it shows how many layers
THIS model actually needs — input for early-exit / layer-skipping decisions.

Pure analysis over forward passes: no training, no checkpoint writes.

Usage:
    python scripts/depth_profile.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/depth_profile.py --checkpoint ... --config ... --mode entropy --tau 0.5
    python scripts/depth_profile.py --checkpoint ... --config ... --problems all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.evaluation.problem_loader import load_problem_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit-lens per-token depth / convergence profiler (INFER-031).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--problems", default="builtin", choices=["builtin", "extended", "all"],
        help="Problem set whose prompts are profiled (default: builtin)",
    )
    parser.add_argument(
        "--mode", default="argmax", choices=["argmax", "entropy"],
        help="Convergence criterion: argmax-match or entropy<=tau (default: argmax)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.5,
        help="Entropy threshold in nats (entropy mode only; default: 0.5)",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=8,
        help="How many prompts to profile (default: 8)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Truncate each prompt to this many tokens (default: 256)",
    )
    parser.add_argument(
        "--by-difficulty", action="store_true",
        help="Stratify exit depth by each problem's difficulty tier",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    args = parser.parse_args()

    cli.header("Cola-Coder", "Depth / Early-Exit Profile (INFER-031)")

    import torch

    from cola_coder.evaluation.depth_profile import profile_depth
    from cola_coder.inference.loading import load_generator

    try:
        generator, config, tokenizer = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    model = generator.model
    model.eval()

    ps = load_problem_set(source=args.problems)
    problems = list(ps)[: args.max_prompts]

    sequences: list[torch.Tensor] = []
    tiers: list[str] = []
    for p in problems:
        ids = tokenizer.encode(p.prompt, add_bos=True, add_eos=False)[: args.max_tokens]
        if len(ids) < 2:
            continue
        sequences.append(torch.tensor(ids, dtype=torch.long))
        # Map the problem's difficulty onto a DepthReport tier label.
        tiers.append(p.difficulty if p.difficulty in {"easy", "medium", "hard"} else "medium")

    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)
    cli.info("Layers", str(model.n_layers))
    cli.info("Prompts", str(len(sequences)))
    cli.info("Mode", f"{args.mode} (tau={args.tau})" if args.mode == "entropy" else args.mode)

    if not sequences:
        cli.error("No usable prompts (all too short to profile).")
        return

    report = profile_depth(
        model,
        sequences,
        mode=args.mode,
        tau=args.tau,
        difficulty_tiers=tiers if args.by_difficulty else None,
    )

    cli.rule("Results")
    cli.kv_table({
        "Tokens profiled": str(report.n_tokens),
        "Layers": str(report.n_layers),
        "Mean exit depth": f"{report.mean_exit_depth:.2f}",
        "Median exit depth": f"{report.median_exit_depth:.1f}",
        "Mean exit (fraction of depth)": (
            f"{report.mean_exit_depth / max(1, report.n_layers - 1):.0%}"
        ),
    }, title="Depth Profile")

    cli.print("")
    cli.print("  [bold]Cumulative tokens converged by depth:[/bold]")
    for d, frac in enumerate(report.frac_converged_by_depth):
        bar = "#" * int(frac * 30)
        cli.print(f"    L{d:>2}  {frac:5.1%}  {bar}")

    if report.by_tier:
        cli.print("")
        cli.print("  [bold]By difficulty tier:[/bold]")
        for tier, stats in report.by_tier.items():
            cli.print(
                f"    {tier:8s}  mean={stats['mean_exit_depth']:.2f}"
                f"  median={stats['median_exit_depth']:.1f}"
                f"  tokens={stats['n_tokens']}"
            )

    if args.json:
        payload = {
            "mean_exit_depth": report.mean_exit_depth,
            "median_exit_depth": report.median_exit_depth,
            "n_layers": report.n_layers,
            "n_tokens": report.n_tokens,
            "frac_converged_by_depth": report.frac_converged_by_depth,
            "by_tier": report.by_tier,
        }
        cli.print("")
        cli.print(json.dumps(payload, indent=2))

    cli.done("Depth profiling finished")


if __name__ == "__main__":
    main()
