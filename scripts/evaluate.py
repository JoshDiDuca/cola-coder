"""Run evaluation on HumanEval coding problems.

Loads a trained model, generates solutions for each problem, tests them
against provided test cases, and computes pass@k metrics.

Usage:
    python scripts/evaluate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/evaluate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --num-samples 10
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cola-coder model on HumanEval problems."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (required).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (required).",
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

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    # ---- Determine device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Note: No GPU detected. Running evaluation on CPU (slower).")

    # ---- Load model ----
    print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        from cola_coder.evaluation.humaneval import get_all_problems
        from cola_coder.evaluation.runner import evaluate_solution, extract_function
        from cola_coder.evaluation.metrics import ProblemResult, compute_pass_at_k, format_results
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        config = Config.from_yaml(args.config)
        print(f"  Model: {config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        print(f"  Tokenizer: {tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {device}")

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # ---- Run evaluation ----
    problems = get_all_problems()
    print(f"\nEvaluating on {len(problems)} problems")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Temperature: {args.temperature}")
    print()

    results = []

    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem.task_id}...", end=" ", flush=True)
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

            except Exception as e:
                # Count as failed
                pass

        status = "PASS" if num_correct > 0 else "FAIL"
        print(f"{status} ({num_correct}/{args.num_samples})")

        results.append(ProblemResult(
            task_id=problem.task_id,
            num_samples=args.num_samples,
            num_correct=num_correct,
        ))

    # ---- Compute and display metrics ----
    k_values = [k for k in [1, 5, 10] if k <= args.num_samples]
    report = format_results(results, k_values=k_values)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
