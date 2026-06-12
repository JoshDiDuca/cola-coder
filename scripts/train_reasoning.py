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
        default=None,
        help=(
            "Path to base model checkpoint directory "
            "(default: reasoning.base_checkpoint from config)."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-resolved from data sources config if omitted).",
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
        default=None,
        help=(
            "Number of solutions to generate per problem per step "
            "(default: reasoning.group_size from config, or 8)."
        ),
    )
    parser.add_argument(
        "--advantage-norm",
        type=str,
        choices=["std", "mean"],
        default=None,
        help=(
            "Advantage normalization: 'std' (original GRPO) or 'mean' "
            "(Dr. GRPO — avoids over-weighting all-pass/all-fail groups). "
            "Default: reasoning.advantage_norm from config, or 'std'."
        ),
    )
    parser.add_argument(
        "--clip-epsilon-high",
        type=float,
        default=None,
        help=(
            "DAPO clip-higher: separate UPPER PPO clip bound (e.g. 0.28 with "
            "lower bound 0.2) to fight entropy collapse. "
            "Default: reasoning.clip_epsilon_high from config, or symmetric."
        ),
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
        default=None,
        choices=["builtin", "extended", "all", "curriculum", "jsonl"],
        help=(
            "Problem set to use for training. "
            "'builtin' = original 20 problems (backward compat), "
            "'extended' / 'all' = all 62 built-in problems, "
            "'curriculum' = all 62 problems sorted easy→medium→hard, "
            "'jsonl' = load from --problems-jsonl path. "
            "Default: problem_set.source from config, or 'builtin'."
        ),
    )
    parser.add_argument(
        "--problems-jsonl",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSONL file of custom problems (required when --problems "
            "jsonl; default: problem_set.jsonl_path from config)."
        ),
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help=(
            "Cap the problem set to N problems selected randomly (0 = use all; "
            "default: problem_set.max_problems from config, or 0)."
        ),
    )
    parser.add_argument(
        "--problem-difficulty",
        type=str,
        default=None,
        choices=["all", "easy", "medium", "hard"],
        help=(
            "Filter problems by difficulty before training "
            "(default: problem_set.difficulty from config, or 'all')."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "typescript"],
        help=(
            "Language of the coding problems and CoT examples to use. "
            "'python': built-in HumanEval problems + Python CoT examples (default). "
            "'typescript': built-in TypeScript problems + TypeScript CoT examples. "
            "Also auto-selects --reward typescript when not explicitly set."
        ),
    )

    args = parser.parse_args()

    if args.tokenizer is None:
        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            args.tokenizer = (
                str(DatasetResolver.get_tokenizer_path())
                if DatasetResolver.tokenizer_exists()
                else storage.tokenizer_path
            )
        except Exception:
            args.tokenizer = storage.tokenizer_path

    cli.header("Cola-Coder", "Reasoning Training (GRPO)")

    # ---- Validate inputs ----
    # (base checkpoint is validated after config load — it can come from
    # reasoning.base_checkpoint in the YAML when --base-checkpoint is omitted)
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

    # Resolve base checkpoint: CLI flag > reasoning.base_checkpoint in config
    if args.base_checkpoint is None:
        cfg_reasoning = getattr(config, "reasoning", None)
        cfg_ckpt = getattr(cfg_reasoning, "base_checkpoint", None) if cfg_reasoning else None
        if not cfg_ckpt:
            cli.fatal(
                "--base-checkpoint is required "
                "(or set reasoning.base_checkpoint in the config).",
            )
        args.base_checkpoint = str(cfg_ckpt)
    if not Path(args.base_checkpoint).exists():
        cli.fatal(
            f"Base checkpoint not found: {args.base_checkpoint}",
            hint="Check the path and try again.",
        )

    try:
        tokenizer = CodeTokenizer(args.tokenizer)
        cli.info("Tokenizer vocab size", tokenizer.vocab_size)

        # Check checkpoint vocab size — SFT checkpoints have extra ChatML tokens
        ckpt_vocab_size = config.model.vocab_size
        try:
            from safetensors import safe_open
            with safe_open(
                str(Path(args.base_checkpoint) / "model.safetensors"),
                framework="pt",
            ) as f:
                if "tok_emb.weight" in f.keys():
                    ckpt_vocab_size = f.get_tensor("tok_emb.weight").shape[0]
        except Exception:
            pass

        # Resize model to match checkpoint if needed (SFT adds ChatML tokens)
        if ckpt_vocab_size != config.model.vocab_size:
            cli.dim(f"  Checkpoint vocab: {ckpt_vocab_size} (config: {config.model.vocab_size})")
            config.model.vocab_size = ckpt_vocab_size

        # MoE-aware load: flip the config to MoE for upcycled base checkpoints
        # before building the model (no-op for dense), or the load fails on
        # experts.* keys.
        from cola_coder.inference.loading import apply_moe_config_from_checkpoint
        apply_moe_config_from_checkpoint(config, args.base_checkpoint)
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

            sft_examples = get_cot_training_data(language=args.language)

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

    # Resolve problem-set settings: CLI flag > problem_set section in
    # reasoning.yaml > default. (This section was once read by nothing.)
    ps_cfg = getattr(config, "problem_set", None) if hasattr(config, "problem_set") else None

    def _ps_cfg(name: str, default):
        value = getattr(ps_cfg, name, None) if ps_cfg else None
        return default if value in (None, "") else value

    problems_source = args.problems or str(_ps_cfg("source", "builtin"))
    problems_jsonl = args.problems_jsonl or str(_ps_cfg("jsonl_path", "")) or None
    problem_difficulty = args.problem_difficulty or str(_ps_cfg("difficulty", "all"))
    max_problems = (
        args.max_problems if args.max_problems is not None
        else int(_ps_cfg("max_problems", 0))
    )

    # Validate jsonl requirement
    if problems_source == "jsonl" and not problems_jsonl:
        cli.fatal(
            "--problems-jsonl PATH is required when --problems jsonl",
            hint="Provide a path to your JSONL problem file.",
        )

    use_curriculum = (
        problems_source == "curriculum" or bool(_ps_cfg("curriculum_learning", False))
    )
    source = (
        "extended" if problems_source in ("curriculum", "all", "extended")
        else problems_source
    )

    cli.info("Language", args.language)

    try:
        ps = ProblemSet()
        if source == "jsonl":
            ps.add_from_jsonl(problems_jsonl)
        elif args.language == "typescript":
            # TypeScript mode: use TS-specific problems (ignores --problems builtin/extended/all)
            ps.add_typescript()
        elif source == "builtin":
            ps.add_builtin(extended=False)
        else:
            ps.add_builtin(extended=True)

        if problem_difficulty != "all":
            ps = ps.filter_by_difficulty(problem_difficulty)

        if use_curriculum:
            ps = ps.curriculum()

        if max_problems > 0 and len(ps) > max_problems:
            sampled = ps.get_batch(max_problems, seed=42)
            ps = ProblemSet(sampled)

        cli.info("Problems loaded", len(ps))
        cli.info("Problem set", ps.summary())
    except Exception as e:
        cli.fatal(f"Loading problems: {e}")

    # ---- Step 5: Run GRPO training ----
    cli.step(5, 5, "Starting GRPO training")

    # Resolve GRPO hyperparameters: CLI flag > reasoning.yaml > trainer default.
    # Every reasoning.yaml knob MUST reach the trainer — config values that are
    # read nowhere are silent no-ops (this script once ignored
    # parallel_generation/max_thinking_tokens entirely).
    reasoning_cfg = getattr(config, "reasoning", None)

    def _cfg(name: str, default):
        value = getattr(reasoning_cfg, name, None) if reasoning_cfg else None
        return default if value is None else value

    group_size = (
        args.group_size if args.group_size is not None else int(_cfg("group_size", 8))
    )
    advantage_norm = args.advantage_norm or str(_cfg("advantage_norm", "std"))
    clip_epsilon = float(_cfg("clip_epsilon", 0.2))
    cfg_clip_high = _cfg("clip_epsilon_high", None)
    clip_epsilon_high = (
        args.clip_epsilon_high
        if args.clip_epsilon_high is not None
        else (float(cfg_clip_high) if cfg_clip_high is not None else None)
    )
    ppo_epochs = int(_cfg("ppo_epochs", 1))
    max_thinking_tokens = int(_cfg("max_thinking_tokens", 256))
    parallel_generation = bool(_cfg("parallel_generation", False))
    parallel_rewards = bool(_cfg("parallel_rewards", False))
    reward_workers = int(_cfg("reward_workers", 4))
    grpo_lr = (
        float(getattr(config.training, "learning_rate", 1e-5))
        if hasattr(config, "training")
        else 1e-5
    )

    cli.info("Epochs", args.epochs)
    cli.info("Group size", group_size)
    cli.info("Learning rate", grpo_lr)
    cli.info("Advantage norm", advantage_norm)
    cli.info(
        "PPO clip",
        f"{clip_epsilon}"
        + (f" / {clip_epsilon_high} (clip-higher)" if clip_epsilon_high else " (symmetric)"),
    )
    cli.info(
        "PPO epochs",
        f"{ppo_epochs}" + (" (clip inert at 1)" if ppo_epochs <= 1 else " (clip active)"),
    )
    cli.info("Parallel generation", parallel_generation)
    cli.info("Parallel rewards", f"{parallel_rewards} (workers: {reward_workers})")
    cli.info("Device", device)
    cli.info("Curriculum", use_curriculum)

    # Resolve reward function: CLI flag > config > language default > global default
    # When called from the pipeline, --reward is always set from config.data.languages.
    # Language default: typescript → tsc reward, python → python_exec.
    language_default = "typescript" if args.language == "typescript" else "python_exec"
    reward_name: str = args.reward or language_default
    if args.reward is None and reasoning_cfg is not None:
        cfg_reward = getattr(reasoning_cfg, "reward_function", None)
        if cfg_reward:
            reward_name = str(cfg_reward)
    cli.info("Reward function", reward_name)

    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=grpo_lr,
            group_size=group_size,
            clip_epsilon=clip_epsilon,
            clip_epsilon_high=clip_epsilon_high,
            advantage_norm=advantage_norm,
            ppo_epochs=ppo_epochs,
            max_thinking_tokens=max_thinking_tokens,
            device=device,
            reward_fn=reward_name,
            parallel_generation=parallel_generation,
            parallel_rewards=parallel_rewards,
            reward_workers=reward_workers,
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
        # Persist the thinking-token-expanded tokenizer next to the checkpoint.
        # add_thinking_tokens() added <think>/</think> (vocab +2) and the model
        # trained on those ids. Without saving it, inference reloads the BASE
        # tokenizer.json — which lacks those tokens — so the reasoning markers
        # fragment and the trained ids can't be decoded, breaking
        # extract_thinking()/strip_thinking() (same class as BUG-106 for SFT).
        # resolve_tokenizer_path() reads tokenizer_path back from metadata.json.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        reasoning_tokenizer_path = str(Path(output_dir) / "tokenizer.json")
        tokenizer.tokenizer.save(reasoning_tokenizer_path)
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
            tokenizer_path=reasoning_tokenizer_path,
        )
        cli.info("Saved to", output_dir)
        cli.info("Reasoning tokenizer", reasoning_tokenizer_path)
    except Exception as e:
        cli.warn(f"Could not save checkpoint: {e}")
        cli.dim("The trained model is still in memory but was not persisted.")

    cli.success("GRPO reasoning training complete!")


if __name__ == "__main__":
    main()
