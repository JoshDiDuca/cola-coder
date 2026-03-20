"""Run evaluation on HumanEval coding problems.

Loads a trained model, generates solutions for each problem, tests them
against provided test cases, and computes pass@k metrics.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint ./checkpoints/step_00010000
    python scripts/evaluate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/evaluate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --num-samples 10
"""

import argparse
from pathlib import Path

from cola_coder.cli import cli


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cola-coder model on HumanEval problems."
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
        help="Path to YAML config file (auto-detected from checkpoint metadata if omitted).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer.json (default: tokenizer.json).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of solutions to generate per problem (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for evaluation (default: 0.2, lower for more deterministic).",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Evaluation")

    # ---- Auto-detect checkpoint ----
    _metadata: dict = {}
    if args.checkpoint is None:
        try:
            from cola_coder.training.checkpoint import detect_latest_checkpoint
            result = detect_latest_checkpoint("checkpoints")
            if result is None:
                cli.fatal(
                    "No checkpoint found in checkpoints/",
                    hint="Pass --checkpoint path/to/ckpt or train a model first",
                )
            raw_path, _metadata = result
            # Resolve relative paths (may be Windows-style relative paths)
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = Path("checkpoints").parent / resolved
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
        # Try to match checkpoints/<size>/step_* -> configs/<size>.yaml
        size_name = ckpt_path.parent.name
        yaml_candidate = Path("configs") / f"{size_name}.yaml"
        if yaml_candidate.exists():
            args.config = str(yaml_candidate)
            cli.info("Auto-detected config", args.config)
        else:
            # Scan configs/ as last resort
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
        cli.fatal(f"Checkpoint not found: {args.checkpoint}", hint="Check the path")

    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}", hint="Check the path")

    if not Path(args.tokenizer).exists():
        cli.fatal(f"Tokenizer file not found: {args.tokenizer}", hint="Check the path")

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
        from cola_coder.evaluation.humaneval import get_all_problems
        from cola_coder.evaluation.runner import evaluate_solution, extract_function
        from cola_coder.evaluation.metrics import ProblemResult, format_results
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
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ---- Run evaluation ----
    problems = get_all_problems()
    cli.success(f"Evaluating on {len(problems)} problems")
    cli.info("Samples per problem", args.num_samples)
    cli.info("Temperature", args.temperature)
    print()

    results = []

    for i, problem in enumerate(problems):
        num_correct = 0

        for sample_idx in range(args.num_samples):
            try:
                # Generate a solution
                generated_text = generator.generate(
                    prompt=problem.prompt,
                    max_new_tokens=256,
                    temperature=args.temperature,
                    top_k=50,
                    top_p=0.9,
                )

                # Extract the function from the generated text
                function_code = extract_function(generated_text, problem.entry_point)

                # Test the solution
                passed, output = evaluate_solution(problem, function_code)
                if passed:
                    num_correct += 1

            except Exception:
                # Count as failed
                pass

        if num_correct > 0:
            cli.print(
                f"  [{i+1}/{len(problems)}] {problem.task_id}..."
                f" [green]PASS[/green] ({num_correct}/{args.num_samples})"
            )
        else:
            cli.print(
                f"  [{i+1}/{len(problems)}] {problem.task_id}..."
                f" [red]FAIL[/red] ({num_correct}/{args.num_samples})"
            )

        results.append(ProblemResult(
            task_id=problem.task_id,
            num_samples=args.num_samples,
            num_correct=num_correct,
        ))

    # ---- Compute and display metrics ----
    k_values = [k for k in [1, 5, 10] if k <= args.num_samples]
    report = format_results(results, k_values=k_values)
    cli.rule("Results")
    cli.print(f"\n{report}")

    # ---- Nano Benchmark (optional) ----
    try:
        from cola_coder.features.nano_benchmark import is_enabled, NanoBenchmark
        if is_enabled():
            cli.rule("Nano Benchmark")
            nano = NanoBenchmark()
            nano.run(generator)
    except Exception:
        pass


if __name__ == "__main__":
    main()
