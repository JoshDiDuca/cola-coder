"""Run TypeScript-specific benchmark on a cola-coder model checkpoint.

Usage:
    python scripts/ts_benchmark.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/ts_benchmark.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --category types,react
    python scripts/ts_benchmark.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --json
"""

import argparse
import json
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Evaluate a trained cola-coder model on the TypeScript benchmark."
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
        help="Path to YAML config file (auto-detected from checkpoint if omitted).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=storage.tokenizer_path,
        help=f"Path to tokenizer.json (default: {storage.tokenizer_path}).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Comma-separated list of categories to evaluate (e.g. 'types,react'). "
        "Valid categories: basics, types, react, nextjs, prisma, zod, testing.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per problem (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Print results as JSON instead of human-readable table.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "TypeScript Benchmark")

    # ---- Auto-detect checkpoint ----
    if args.checkpoint is None:
        try:
            from cola_coder.training.checkpoint import detect_latest_checkpoint

            result = detect_latest_checkpoint(storage.checkpoints_dir)
            if result is None:
                cli.fatal(
                    f"No checkpoint found in {storage.checkpoints_dir}",
                    hint="Pass --checkpoint path/to/ckpt or train a model first",
                )
            raw_path, _ = result
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = Path(storage.checkpoints_dir).parent / resolved
            args.checkpoint = str(resolved)
            cli.info("Auto-detected checkpoint", args.checkpoint)
        except ImportError:
            cli.fatal(
                "Could not import cola_coder. Make sure the package is installed.",
                hint="Try: pip install -e .",
            )

    # ---- Auto-detect config ----
    if args.config is None:
        ckpt_path = Path(args.checkpoint)
        size_name = ckpt_path.parent.name
        yaml_candidate = Path("configs") / f"{size_name}.yaml"
        if yaml_candidate.exists():
            args.config = str(yaml_candidate)
            cli.info("Auto-detected config", args.config)
        else:
            configs_dir = Path("configs")
            if configs_dir.exists():
                yamls = sorted(configs_dir.glob("*.yaml"))
                if yamls:
                    args.config = str(yamls[0])
                    cli.info("Auto-detected config", args.config)
            if args.config is None:
                cli.fatal(
                    "Could not auto-detect a config file",
                    hint="Pass --config configs/<size>.yaml explicitly",
                )

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}")
    if not Path(args.tokenizer).exists():
        cli.fatal(f"Tokenizer file not found: {args.tokenizer}")

    # ---- Parse category filter ----
    categories: list[str] | None = None
    if args.category:
        categories = [c.strip() for c in args.category.split(",") if c.strip()]
        valid = {"basics", "types", "react", "nextjs", "prisma", "zod", "testing"}
        unknown = set(categories) - valid
        if unknown:
            cli.fatal(f"Unknown categories: {unknown}", hint=f"Valid: {valid}")

    # ---- Determine device ----
    device = cli.gpu_info()

    # ---- Load model ----
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
        cli.info("Model", f"{config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        cli.info("Tokenizer", f"{tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        cli.info("Checkpoint", args.checkpoint)
        cli.info("Device", device)

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except Exception as exc:
        cli.fatal(f"Loading model: {exc}")

    # ---- Load benchmark ----
    try:
        from cola_coder.evaluation.ts_benchmark import TSBenchmark
    except ImportError:
        cli.fatal(
            "Could not import TSBenchmark. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    benchmark = TSBenchmark(categories=categories)
    problems = benchmark.get_problems()
    cli.success(f"Evaluating on {len(problems)} TypeScript problems")
    if categories:
        cli.info("Categories", ", ".join(categories))
    cli.info("Samples per problem", args.num_samples)
    cli.info("Temperature", args.temperature)
    print()

    # ---- Run benchmark ----
    result = benchmark.run(
        generator=generator,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )

    # ---- Report ----
    if args.output_json:
        print(
            json.dumps(
                {
                    "total_problems": result.total_problems,
                    "solved": result.solved,
                    "pass_rate": result.pass_rate,
                    "by_category": result.by_category,
                    "by_difficulty": result.by_difficulty,
                    "details": result.details,
                },
                indent=2,
            )
        )
    else:
        for detail in result.details:
            status_tag = "[green]PASS[/green]" if detail["passed"] else "[red]FAIL[/red]"
            cli.print(
                f"  {detail['id']:<40} {status_tag}"
                f"  [{detail['num_correct']}/{detail['num_samples']}]"
                f"  ({detail['category']})"
            )

        cli.rule("Results")
        cli.print(f"\n{result.summary()}\n")


if __name__ == "__main__":
    main()
