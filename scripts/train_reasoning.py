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

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main():
    storage = get_storage_config()

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
        default=storage.tokenizer_path,
        help=f"Path to tokenizer.json (default: {storage.tokenizer_path}).",
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

    cli.header("Cola-Coder", "Reasoning Training (GRPO)")

    # ---- Validate inputs ----
    if not Path(args.base_checkpoint).exists():
        cli.fatal(
            f"Base checkpoint not found: {args.base_checkpoint}",
            hint="Check the path and try again.",
        )

    if not Path(args.tokenizer).exists():
        cli.fatal(
            f"Tokenizer file not found: {args.tokenizer}",
            hint="Run scripts/train_tokenizer.py first.",
        )

    config_path = Path(args.config)
    if not config_path.exists():
        cli.fatal(
            f"Config file not found: {config_path}",
            hint="Check the path or use --config to specify.",
        )

    # ---- Device check ----
    device = cli.gpu_info()

    # ---- Load config and model ----
    cli.step(1, 4, "Loading config and base model")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        from cola_coder.reasoning.thinking_tokens import add_thinking_tokens
        from cola_coder.reasoning.grpo import GRPOTrainer
        from cola_coder.evaluation.humaneval import get_all_problems
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        config = Config.from_yaml(str(config_path))
        cli.info("Config", str(config_path))
        cli.info("Model", f"{config.model.total_params_human} parameters")
    except Exception as e:
        cli.fatal(f"Loading config: {e}")

    try:
        tokenizer = CodeTokenizer(args.tokenizer)
        cli.info("Tokenizer vocab size", tokenizer.vocab_size)

        model = Transformer(config.model).to(device)
        load_model_only(args.base_checkpoint, model, device=device)
        cli.info("Checkpoint", args.base_checkpoint)
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ---- Step 2: Add thinking tokens ----
    cli.step(2, 4, "Adding thinking tokens")

    try:
        think_open_id, think_close_id = add_thinking_tokens(tokenizer, model)
        model = model.to(device)
    except Exception as e:
        cli.fatal(f"Adding thinking tokens: {e}")

    # ---- Step 3: Prepare training problems ----
    cli.step(3, 4, "Preparing HumanEval problems")

    problems = get_all_problems()
    training_problems = [
        {"prompt": p.prompt, "test_code": p.test_code}
        for p in problems
    ]
    cli.info("Problems loaded", len(training_problems))

    # ---- Step 4: Run GRPO training ----
    cli.step(4, 4, "Starting GRPO training")
    cli.info("Epochs", args.epochs)
    cli.info("Group size", args.group_size)
    cli.info("Device", device)

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
        cli.warn("Training interrupted by user.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            cli.error(f"GPU out of memory: {e}")
            cli.dim("Reduce --group-size (e.g., --group-size 4) or use a smaller base model")
            sys.exit(1)
        raise

    # ---- Save the reasoning-enhanced model ----
    try:
        from cola_coder.training.checkpoint import save_checkpoint

        output_dir = (
            config.checkpoint.output_dir
            if hasattr(config, "checkpoint")
            else "./checkpoints/reasoning"
        )
        save_checkpoint(
            model=model,
            optimizer=grpo_trainer.optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                grpo_trainer.optimizer, lambda step: 1.0
            ),
            step=0,
            loss=0.0,
            config={"model": vars(config.model), "reasoning": True},
            output_dir=output_dir,
        )
        cli.info("Saved to", output_dir)
    except Exception as e:
        cli.warn(f"Could not save checkpoint: {e}")
        cli.dim("The trained model is still in memory but was not persisted.")

    cli.success("GRPO reasoning training complete!")


if __name__ == "__main__":
    main()
