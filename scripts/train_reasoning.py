"""Run GRPO training for reasoning improvement.

Loads a base model checkpoint, adds thinking tokens (<think>...</think>),
and runs Group Relative Policy Optimization on HumanEval problems to
teach the model to reason step-by-step before generating code.

Usage:
    python scripts/train_reasoning.py --config configs/reasoning.yaml --base-checkpoint ./checkpoints/step_00010000
    python scripts/train_reasoning.py --config configs/reasoning.yaml --base-checkpoint ./checkpoints/step_00010000 --epochs 5 --group-size 4
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run GRPO training for reasoning improvement on coding problems."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reasoning.yaml",
        help="Path to reasoning YAML config file (default: configs/reasoning.yaml).",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to base model checkpoint directory (required).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer.json (default: tokenizer.json).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of GRPO training epochs (default: 3).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of solutions to generate per problem per step (default: 8).",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    if not Path(args.base_checkpoint).exists():
        print(f"Error: Base checkpoint not found: {args.base_checkpoint}")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # ---- Device check ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected. GRPO training on CPU will be extremely slow.")
        print("  A CUDA-capable GPU is strongly recommended.")
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")

    # ---- Load config and model ----
    print(f"\nLoading config from {config_path}...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        from cola_coder.reasoning.thinking_tokens import add_thinking_tokens
        from cola_coder.reasoning.grpo import GRPOTrainer
        from cola_coder.evaluation.humaneval import get_all_problems
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        config = Config.from_yaml(str(config_path))
        print(f"  Model: {config.model.total_params_human} parameters")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # ---- Step 1: Load base model ----
    print(f"\nStep 1: Loading base model from {args.base_checkpoint}...")

    try:
        tokenizer = CodeTokenizer(args.tokenizer)
        print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

        model = Transformer(config.model).to(device)
        load_model_only(args.base_checkpoint, model, device=device)
        print(f"  Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # ---- Step 2: Add thinking tokens ----
    print("\nStep 2: Adding thinking tokens...")

    try:
        think_open_id, think_close_id = add_thinking_tokens(tokenizer, model)
        model = model.to(device)
    except Exception as e:
        print(f"Error adding thinking tokens: {e}")
        sys.exit(1)

    # ---- Step 3: Prepare training problems ----
    print("\nStep 3: Preparing HumanEval problems...")

    problems = get_all_problems()
    training_problems = [
        {"prompt": p.prompt, "test_code": p.test_code}
        for p in problems
    ]
    print(f"  {len(training_problems)} problems loaded.")

    # ---- Step 4: Run GRPO training ----
    print(f"\nStep 4: Starting GRPO training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Group size: {args.group_size}")
    print(f"  Device: {device}")

    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            group_size=args.group_size,
            device=device,
        )

        grpo_trainer.train(
            problems=training_problems,
            num_epochs=args.epochs,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nGPU out of memory: {e}")
            print("\nSuggestions:")
            print("  1. Reduce --group-size (e.g., --group-size 4)")
            print("  2. Use a smaller base model")
            sys.exit(1)
        raise

    # ---- Save the reasoning-enhanced model ----
    print("\nSaving reasoning-enhanced model...")

    try:
        from cola_coder.training.checkpoint import save_checkpoint

        output_dir = config.checkpoint.output_dir if hasattr(config, "checkpoint") else "./checkpoints/reasoning"
        save_checkpoint(
            model=model,
            optimizer=grpo_trainer.optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(grpo_trainer.optimizer, lambda step: 1.0),
            step=0,
            loss=0.0,
            config={"model": vars(config.model), "reasoning": True},
            output_dir=output_dir,
        )
        print(f"  Saved to: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")
        print("  The trained model is still in memory but was not persisted.")

    print("\nGRPO reasoning training complete!")


if __name__ == "__main__":
    main()
