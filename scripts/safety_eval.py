"""Safety evaluation — generate code from probe prompts and check for issues.

Each suite generates completions from a built-in prompt set and runs
SafetyEvaluator checks on them: compilation, secret leakage, dangerous
patterns, and package/API hallucination.

Usage:
    python scripts/safety_eval.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/safety_eval.py --checkpoint ... --config ... --suite pii
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli

# Probe prompt suites live in the library so they're testable and reusable:
# cola_coder.evaluation.safety_probes
# (basic/extended/pii/license/injection/cwe/all)
from cola_coder.evaluation.safety_probes import SUITES, cwe_probe_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run safety evaluation on generated code.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--suite", default="basic", choices=sorted(SUITES),
        help="Prompt suite (default: basic)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=160,
        help="Max new tokens per completion (default: 160)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", f"Safety Evaluation — {args.suite} suite")

    from cola_coder.evaluation.safety_eval import SafetyEvaluator
    from cola_coder.inference.loading import load_generator

    try:
        generator, config, _ = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    prompts = SUITES[args.suite]
    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)
    cli.info("Prompts", len(prompts))

    # The cwe suite statically scans each completion with CweSecurityScorer
    # (SEC-026) and reports the CWE rate; all other suites use the regex-based
    # SafetyEvaluator. Both are pure static analysis — the model only generates.
    is_cwe = args.suite == "cwe"
    evaluator = SafetyEvaluator()
    flagged = 0
    cwe_family_hits: dict[str, int] = {}

    for i, prompt in enumerate(prompts, 1):
        cli.step(i, len(prompts), prompt.splitlines()[0][:60])
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        completion = result[len(prompt):] if result.startswith(prompt) else result

        if is_cwe:
            cwe_result = cwe_probe_result(completion)
            if not cwe_result.passed:
                flagged += 1
                for family in cwe_result.failed_families:
                    cwe_family_hits[family] = cwe_family_hits.get(family, 0) + 1
                for finding in cwe_result.findings:
                    cli.warn(f"  {finding['cwe']}: {finding['name']}")
        else:
            checks = evaluator.evaluate(completion)
            if checks["issues"]:
                flagged += 1
                for issue in checks["issues"]:
                    cli.warn(f"  {issue}")

    cli.rule("Results")
    if is_cwe:
        total = len(prompts)
        summary: dict[str, str] = {
            "Total prompts": str(total),
            "Completions with a CWE finding": str(flagged),
            "CWE finding rate": f"{flagged / max(total, 1):.1%}",
        }
        for family in sorted(cwe_family_hits):
            summary[family] = str(cwe_family_hits[family])
        cli.kv_table(summary, title="CWE Safety Metrics (cwe)")
    else:
        cli.kv_table(evaluator.metrics.summary(), title=f"Safety Metrics ({args.suite})")

    if flagged:
        cli.warn(f"{flagged}/{len(prompts)} completions raised safety issues")
    else:
        cli.success("No safety issues detected")
    cli.done("Safety evaluation finished")


if __name__ == "__main__":
    main()
