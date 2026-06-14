"""Verifier-anchored FUNCTION-STEP process-credit profiler (EVAL-034).

A "poor-man's PRM": for each coding problem, generate candidate solution(s),
decompose every candidate into its functions ("steps"), and grade each step with
the SAME sandbox verifier the rest of the project trusts (evaluation.runner.
execute_code). Reports a length-normalized per-candidate ``process_score`` and
the FRAGILE functions — code that is dead / non-executable while the candidate's
overall tests still pass (a latent bug the top-level verifier never exercised).

Pure analysis over forward passes + the real sandbox executor: no training, no
checkpoint writes.

Usage:
    python scripts/process_credit.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/process_credit.py --checkpoint ... --config ... --problems all --best-of 4
    python scripts/process_credit.py --checkpoint ... --config ... --json
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
        description="Verifier-anchored function-step process-credit profiler (EVAL-034).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--problems", default="builtin", choices=["builtin", "extended", "all"],
        help="Problem set to profile (default: builtin)",
    )
    parser.add_argument(
        "--best-of", type=int, default=1,
        help="Candidates per problem (>1 runs sandbox-verified best-of-N; default: 1)",
    )
    parser.add_argument(
        "--max-problems", type=int, default=8,
        help="How many problems to profile (default: 8)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max new tokens per candidate (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="Per-execution sandbox timeout in seconds (default: 10.0)",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    args = parser.parse_args()

    cli.header("Cola-Coder", "Process / Function-Step Credit (EVAL-034)")

    from cola_coder.evaluation.process_credit import profile_candidates
    from cola_coder.evaluation.runner import execute_code
    from cola_coder.inference.best_of_n import generate_best_of_n
    from cola_coder.inference.loading import load_generator

    try:
        generator, config, _ = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    ps = load_problem_set(source=args.problems)
    problems = list(ps)[: args.max_problems]

    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)
    cli.info("Problems", str(len(problems)))
    cli.info("Best-of", str(args.best_of))

    report_rows: list[dict] = []
    all_scores: list[float] = []
    fragile_total = 0

    for i, problem in enumerate(problems, 1):
        cli.step(i, len(problems), problem.task_id)

        # Generate candidate(s). best-of-N gives us multiple completions to profile;
        # best-of-1 is a single greedy-ish sample. We profile every candidate so the
        # report shows the spread, not just the winner.
        result = generate_best_of_n(
            generator,
            problem.prompt,
            num_candidates=max(1, args.best_of),
            language=problem.language,
            tests=problem.test_code,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            execute_fn=execute_code,
        )
        candidates = [c.completion for c in result.candidates]

        profiles = profile_candidates(
            candidates,
            problem.test_code,
            execute_code,
            language=problem.language,
            timeout=args.timeout,
        )

        best = max(profiles, key=lambda p: p["process_score"]) if profiles else None
        if best is None:
            continue
        all_scores.append(best["process_score"])
        fragile = best["fragile_functions"]
        fragile_total += len(fragile)

        cli.info(
            f"  {problem.task_id}",
            f"process_score={best['process_score']:.3f}  "
            f"steps={len(best['steps'])}  "
            f"fragile={len(fragile)}",
        )
        if fragile:
            cli.warn(f"    fragile functions: {', '.join(fragile)}")

        report_rows.append({
            "task_id": problem.task_id,
            "difficulty": problem.difficulty,
            "solved": result.solved,
            "best_process_score": best["process_score"],
            "fragile_functions": fragile,
            "n_candidates": len(candidates),
            "steps": best["steps"],
        })

    cli.rule("Results")
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    cli.kv_table({
        "Problems profiled": str(len(report_rows)),
        "Mean best process_score": f"{mean_score:.3f}",
        "Fragile functions found": str(fragile_total),
    }, title="Process-Credit Summary")

    if args.json:
        cli.print("")
        cli.print(json.dumps({
            "mean_best_process_score": mean_score,
            "fragile_functions_found": fragile_total,
            "problems": report_rows,
        }, indent=2))

    cli.done("Process-credit profiling finished")


if __name__ == "__main__":
    main()
