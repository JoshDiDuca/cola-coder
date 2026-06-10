"""Python completion benchmark — prefix-completion problems with pattern scoring.

Runs the built-in CompletionBenchmark problem set (prefix → completion,
scored by required/forbidden regex patterns) against a checkpoint.

Usage:
    python scripts/completion_benchmark.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/completion_benchmark.py --checkpoint ... --config ... --difficulty easy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prefix-completion benchmark against a checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument(
        "--difficulty", default=None, choices=["easy", "medium", "hard"],
        help="Only run problems of this difficulty (default: all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max new tokens per completion (default: 128)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Completion Benchmark")

    from cola_coder.evaluation.completion_benchmark import (
        CompletionBenchmark,
        get_problems_by_difficulty,
    )
    from cola_coder.inference.loading import load_generator

    try:
        generator, config, _ = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except FileNotFoundError as e:
        cli.fatal(str(e))

    cli.info("Checkpoint", args.checkpoint)
    cli.info("Model", config.model.total_params_human)

    problems = (
        get_problems_by_difficulty(args.difficulty) if args.difficulty else None
    )
    bench = CompletionBenchmark(problems=problems)
    cli.info("Problems", len(bench.problems))

    def generate(prefix: str) -> str:
        result = generator.generate(
            prompt=prefix,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        # Return only the new tokens
        return result[len(prefix):] if result.startswith(prefix) else result

    report = bench.run(generate)
    cli.print(bench.to_markdown(report))
    cli.done("Completion benchmark finished")


if __name__ == "__main__":
    main()
