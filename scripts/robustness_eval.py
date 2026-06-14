"""Functional robustness evaluation (EVAL-030) — semantically-preserving rewordings.

Applies meaning-preserving transformations to each problem's DOCSTRING ONLY (typos
in prose, whitespace jitter, casing, reordered doctest examples, paraphrasing) and
re-grades the model's solution with the existing sandbox verifier. Reports:

    robust_pass@1   — fraction solved under the WORST rewording
    consistency     — fraction whose pass/fail verdict is invariant
    fragility list  — problems solved clean but failing a mere rewording

Usage:
    python scripts/robustness_eval.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/robustness_eval.py --checkpoint ... --config ... --problems all --ci
    python scripts/robustness_eval.py --checkpoint ... --config ... --kinds typo,paraphrase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.evaluation.perturbations import ALL_KINDS
from cola_coder.evaluation.problem_loader import load_problem_set
from cola_coder.evaluation.robustness_eval import evaluate_robustness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifier-graded functional robustness evaluation (EVAL-030).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--problems", default="builtin", choices=["builtin", "extended", "all"],
        help="Problem set (default: builtin)",
    )
    parser.add_argument(
        "--kinds", default=",".join(ALL_KINDS),
        help=f"Comma-separated perturbation kinds (default: all = {','.join(ALL_KINDS)})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max new tokens per generation (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2 — robustness wants determinism)",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="Attach a bootstrap confidence interval to robust_pass@1",
    )
    args = parser.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in ALL_KINDS]
    if unknown:
        cli.fatal(f"Unknown perturbation kind(s): {unknown}", hint=f"Valid: {list(ALL_KINDS)}")

    cli.header("Cola-Coder", "Functional Robustness Evaluation (EVAL-030)")

    from cola_coder.inference.loading import load_generator

    try:
        generator, config, _ = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    ps = load_problem_set(source=args.problems)
    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)
    cli.info("Problems", str(len(ps)))
    cli.info("Perturbations", ", ".join(kinds))

    def generate_fn(prompt: str) -> str:
        return generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=0.9,
        )

    report = evaluate_robustness(
        generate_fn,
        ps,
        kinds=kinds,
        max_new_tokens=args.max_tokens,
        compute_ci=args.ci,
    )

    cli.rule("Results")
    summary = {
        "robust_pass@1": f"{report.robust_pass_at_1:.1%}",
        "consistency_rate": f"{report.consistency_rate:.1%}",
        "problems": str(report.num_problems),
        "fragile": str(len(report.fragile_task_ids)),
    }
    if report.robust_pass_at_1_ci is not None:
        _, lo, hi = report.robust_pass_at_1_ci
        summary["robust_pass@1 CI"] = f"[{lo:.1%} – {hi:.1%}]"
    cli.kv_table(summary, title="Robustness Metrics")

    if report.fragile_task_ids:
        cli.warn(
            f"{len(report.fragile_task_ids)} problem(s) solved clean but FAILED a "
            "mere rewording:"
        )
        for task_id in report.fragile_task_ids:
            row = next(p for p in report.per_problem if p["task_id"] == task_id)
            broke = [k for k, v in row["verdicts"].items() if k != "clean" and not v]
            cli.warn(f"  {task_id}  (failed on: {', '.join(broke)})")
    else:
        cli.success("No fragility detected — every clean pass survived all rewordings")

    cli.done("Robustness evaluation finished")


if __name__ == "__main__":
    main()
