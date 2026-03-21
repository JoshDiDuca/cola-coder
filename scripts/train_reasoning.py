"""Run GRPO training for reasoning improvement.

Loads a base model checkpoint, adds thinking tokens (<think>...</think>),
optionally runs SFT warmup on curated CoT examples, then runs Group Relative
Policy Optimization on HumanEval problems to teach the model to reason
step-by-step before generating code.

Usage:
    python scripts/train_reasoning.py --config configs/reasoning.yaml
        --base-checkpoint ./checkpoints/step_00010000
    python scripts/train_reasoning.py --config configs/reasoning.yaml
        --base-checkpoint ./checkpoints/step_00010000 --sft-warmup --sft-epochs 5
    python scripts/train_reasoning.py --config configs/reasoning.yaml
        --base-checkpoint ./checkpoints/step_00010000 --no-sft-warmup
    python scripts/train_reasoning.py --config configs/reasoning.yaml
        --base-checkpoint ./checkpoints/step_00010000 --sft-warmup --sft-synthetic
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
    parser.add_argument(
        "--reward",
        type=str,
        choices=["python_exec", "typescript", "combined"],
        default=None,
        help=(
            "Reward function to use during GRPO training "
            "(default: value from configs/reasoning.yaml, or 'python_exec'). "
            "python_exec: run Python tests via subprocess. "
            "typescript: score generated TypeScript with tsc --strict. "
            "combined: multi-signal reward (type check + syntax + style + completeness)."
        ),
    )

    # SFT warmup flags
    sft_group = parser.add_mutually_exclusive_group()
    sft_group.add_argument(
        "--sft-warmup",
        action="store_true",
        default=None,
        help=(
            "Run SFT warmup on CoT examples before GRPO (recommended). "
            "Overrides the sft_warmup.enabled value in configs/reasoning.yaml."
        ),
    )
    sft_group.add_argument(
        "--no-sft-warmup",
        action="store_true",
        default=False,
        help="Skip SFT warmup and go straight to GRPO training.",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=None,
        help=(
            "Number of SFT warmup epochs (default: from configs/reasoning.yaml or 5). "
            "Keep between 3-10 to avoid overfitting the tiny seed dataset."
        ),
    )
    parser.add_argument(
        "--sft-synthetic",
        action="store_true",
        default=False,
        help=(
            "Generate synthetic CoT examples via self-play augmentation "
            "and include them in SFT warmup training data."
        ),
    )

    # ---- Problem set flags ----
    parser.add_argument(
        "--problems",
        type=str,
        default="builtin",
        choices=["builtin", "extended", "all", "curriculum", "jsonl"],
        help=(
            "Problem set to use for training. "
            "'builtin' = original 20 problems (backward compat), "
            "'extended' / 'all' = all 62 built-in problems, "
            "'curriculum' = all 62 problems sorted easy→medium→hard, "
            "'jsonl' = load from --problems-jsonl path."
        ),
    )
    parser.add_argument(
        "--problems-jsonl",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a JSONL file of custom problems (required when --problems jsonl).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="Cap the problem set to N problems selected randomly (0 = use all).",
    )
    parser.add_argument(
        "--problem-difficulty",
        type=str,
        default="all",
        choices=["all", "easy", "medium", "hard"],
        help="Filter problems by difficulty before training (default: all).",
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
    cli.step(1, 5, "Loading config and base model")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        from cola_coder.reasoning.thinking_tokens import add_thinking_tokens
        from cola_coder.reasoning.grpo import GRPOTrainer
        from cola_coder.evaluation.problem_loader import ProblemSet
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
    cli.step(2, 5, "Adding thinking tokens")

    try:
        think_open_id, think_close_id = add_thinking_tokens(tokenizer, model)
        model = model.to(device)
    except Exception as e:
        cli.fatal(f"Adding thinking tokens: {e}")

    # ---- Step 3: SFT Warmup (optional) ----
    # Resolve whether to run SFT warmup. Priority: CLI flag > config > default (enabled).
    sft_config = getattr(config, "sft_warmup", None) if hasattr(config, "sft_warmup") else None
    config_sft_enabled = True  # default when no config key present
    config_sft_epochs = 5       # default
    config_sft_lr = 5e-5        # default

    if sft_config is not None:
        config_sft_enabled = getattr(sft_config, "enabled", True)
        config_sft_epochs = getattr(sft_config, "epochs", 5)
        config_sft_lr = getattr(sft_config, "learning_rate", 5e-5)

    # Also check reasoning section for flat-key style (reasoning.yaml backwards compat)
    reasoning_cfg = getattr(config, "reasoning", None) if hasattr(config, "reasoning") else None
    if reasoning_cfg is not None:
        config_sft_enabled = getattr(reasoning_cfg, "sft_warmup", config_sft_enabled)
        config_sft_epochs = getattr(reasoning_cfg, "sft_epochs", config_sft_epochs)
        config_sft_lr = getattr(reasoning_cfg, "sft_learning_rate", config_sft_lr)

    run_sft = not args.no_sft_warmup and (args.sft_warmup or config_sft_enabled)
    sft_epochs = args.sft_epochs if args.sft_epochs is not None else config_sft_epochs

    cli.step(3, 5, "SFT warmup" if run_sft else "SFT warmup (skipped)")

    if run_sft:
        try:
            from cola_coder.reasoning.sft_warmup import SFTWarmup
            from cola_coder.reasoning.cot_data import get_cot_training_data

            precision = (
                getattr(config.training, "precision", "bf16")
                if hasattr(config, "training")
                else "bf16"
            )
            sft = SFTWarmup(
                model=model,
                tokenizer=tokenizer,
                device=device,
                learning_rate=config_sft_lr,
                precision=precision,
            )

            sft_examples = get_cot_training_data()

            if args.sft_synthetic:
                cli.info("SFT synthetic", "generating augmented CoT examples...")
                synthetic = sft.generate_synthetic_examples()
                cli.info("SFT synthetic examples", len(synthetic))
                sft_examples = sft_examples + synthetic

            cli.info("SFT examples", len(sft_examples))
            cli.info("SFT epochs", sft_epochs)

            sft.train(examples=sft_examples, num_epochs=sft_epochs)

        except KeyboardInterrupt:
            cli.warn("SFT warmup interrupted -- continuing to GRPO.")
        except Exception as e:
            cli.warn(f"SFT warmup failed: {e}")
            cli.dim("Continuing to GRPO without SFT warmup.")
    else:
        cli.dim("SFT warmup disabled -- skipping to GRPO.")

    # ---- Step 4: Prepare training problems ----
    cli.step(4, 5, "Preparing training problems")

    # Validate jsonl requirement
    if args.problems == "jsonl" and not args.problems_jsonl:
        cli.fatal(
            "--problems-jsonl PATH is required when --problems jsonl",
            hint="Provide a path to your JSONL problem file.",
        )

    # Build problem set from CLI flags
    use_curriculum = args.problems == "curriculum"
    source = "extended" if args.problems in ("curriculum", "all", "extended") else args.problems

    try:
        ps = ProblemSet()
        if source == "jsonl":
            ps.add_from_jsonl(args.problems_jsonl)
        elif source == "builtin":
            ps.add_builtin(extended=False)
        else:
            ps.add_builtin(extended=True)

        if args.problem_difficulty != "all":
            ps = ps.filter_by_difficulty(args.problem_difficulty)

        if use_curriculum:
            ps = ps.curriculum()

        if args.max_problems > 0 and len(ps) > args.max_problems:
            sampled = ps.get_batch(args.max_problems, seed=42)
            ps = ProblemSet(sampled)

        cli.info("Problems loaded", len(ps))
        cli.info("Problem set", ps.summary())
    except Exception as e:
        cli.fatal(f"Loading problems: {e}")

    # ---- Step 5: Run GRPO training ----
    cli.step(5, 5, "Starting GRPO training")
    cli.info("Epochs", args.epochs)
    cli.info("Group size", args.group_size)
    cli.info("Device", device)
    cli.info("Curriculum", use_curriculum)

    # Resolve reward function: CLI flag > config > default
    reward_name: str = args.reward or "python_exec"
    reasoning_cfg = getattr(config, "reasoning", None)
    if args.reward is None and reasoning_cfg is not None:
        cfg_reward = getattr(reasoning_cfg, "reward_function", None)
        if cfg_reward:
            reward_name = str(cfg_reward)
    cli.info("Reward function", reward_name)

    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            group_size=args.group_size,
            device=device,
            reward_fn=reward_name,
        )

        grpo_trainer.train(
            problems=ps,
            num_epochs=args.epochs,
            curriculum=use_curriculum,
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
