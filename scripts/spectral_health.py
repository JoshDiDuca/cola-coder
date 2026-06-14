"""Spectral-Alignment (SA) training-stability diagnostic (EVAL-035).

For each transformer layer, computes the cosine alignment between the layer's
forward response and ``u1(W)`` (the principal LEFT singular vector of the probed
weight, found by cheap power iteration). A healthy layer's SA distribution is
SIGN-BALANCED (~half positive, half negative); SIGN-COLLAPSE (alignments shifting
all-positive or all-negative) is an EARLY divergence-risk signal that precedes a
loss explosion. The per-layer sign-collapse fraction (0.5 healthy → 1.0 collapsed)
is the scalar to watch; the report flags the worst layer.

DIAGNOSTIC ONLY: forward passes + weight inspection. No training, no checkpoint
writes, no architecture change.

Usage:
    python scripts/spectral_health.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/spectral_health.py --checkpoint ... --config ... --layers q,fc2
    python scripts/spectral_health.py --checkpoint ... --config ... --n-batches 12 --json
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
        description="Spectral-Alignment training-stability diagnostic (EVAL-035).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--problems", default="builtin", choices=["builtin", "extended", "all"],
        help="Problem set whose prompts are profiled (default: builtin)",
    )
    parser.add_argument(
        "--layers", default="q",
        help="Comma-separated probes per block: q,fc2 (default: q)",
    )
    parser.add_argument(
        "--n-batches", type=int, default=8,
        help="How many prompts to feed (default: 8)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Truncate each prompt to this many tokens (default: 256)",
    )
    parser.add_argument(
        "--iters", type=int, default=8,
        help="Power-iteration steps for the principal singular vector (default: 8)",
    )
    parser.add_argument(
        "--by-difficulty", action="store_true",
        help="Stratify sign-collapse by each problem's difficulty tier",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    args = parser.parse_args()

    cli.header("Cola-Coder", "Spectral Health / Divergence Risk (EVAL-035)")

    import torch

    from cola_coder.evaluation.spectral_health import profile_spectral_health
    from cola_coder.inference.loading import load_generator

    probes = tuple(p.strip() for p in args.layers.split(",") if p.strip())
    valid = {"q", "fc2"}
    bad = [p for p in probes if p not in valid]
    if bad or not probes:
        cli.fatal(f"--layers must be a comma list of q,fc2 (got: {args.layers})")

    try:
        generator, config, tokenizer = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    model = generator.model
    model.eval()

    ps = load_problem_set(source=args.problems)
    problems = list(ps)[: args.n_batches]

    sequences: list[torch.Tensor] = []
    tiers: list[str] = []
    for p in problems:
        ids = tokenizer.encode(p.prompt, add_bos=True, add_eos=False)[: args.max_tokens]
        if len(ids) < 1:
            continue
        sequences.append(torch.tensor(ids, dtype=torch.long))
        tiers.append(p.difficulty if p.difficulty in {"easy", "medium", "hard"} else "medium")

    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)
    cli.info("Layers", str(model.n_layers))
    cli.info("Probes", ",".join(probes))
    cli.info("Prompts", str(len(sequences)))

    if not sequences:
        cli.error("No usable prompts (all empty after tokenization).")
        return

    report = profile_spectral_health(
        model,
        sequences,
        probes=probes,
        iters=args.iters,
        by_tier=tiers if args.by_difficulty else None,
    )

    cli.rule("Results")
    health = "HEALTHY" if report.worst_collapse < 0.7 else (
        "WATCH" if report.worst_collapse < 0.85 else "DIVERGENCE RISK"
    )
    cli.kv_table({
        "Layers probed": str(len(report.per_layer)),
        "Worst layer": str(report.worst_layer),
        "Worst sign-collapse": f"{report.worst_collapse:.3f}",
        "Verdict": health,
    }, title="Spectral Health")

    cli.print("")
    cli.print("  [bold]Per-layer sign-collapse (0.50 healthy -> 1.00 collapsed):[/bold]")
    for row in report.per_layer:
        frac = row["sign_collapse"]
        # Bar grows over the [0.5, 1.0] danger band.
        bar = "#" * int(max(0.0, (frac - 0.5) / 0.5) * 30)
        cli.print(
            f"    L{row['layer']:>2}  collapse={frac:5.3f}  "
            f"sa_mean={row['sa_mean']:+.3f}  n={row['n']:<5d}  {bar}"
        )

    if report.by_tier:
        cli.print("")
        cli.print("  [bold]By difficulty tier (worst layer):[/bold]")
        for tier, stats in report.by_tier.items():
            cli.print(
                f"    {tier:8s}  worst_layer={stats['worst_layer']:>2}"
                f"  worst_collapse={stats['worst_collapse']:.3f}"
            )

    if args.json:
        payload = {
            "worst_layer": report.worst_layer,
            "worst_collapse": report.worst_collapse,
            "n_layers": report.n_layers,
            "per_layer": report.per_layer,
            "by_tier": report.by_tier,
        }
        cli.print("")
        cli.print(json.dumps(payload, indent=2))

    cli.done("Spectral health diagnostic finished")


if __name__ == "__main__":
    main()
