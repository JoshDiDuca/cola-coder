"""Full training pipeline orchestrator.

Runs all training stages in sequence with configurable stage selection.
Each stage can be run independently or as part of the full pipeline.

Stages:
1. collect-data     — Download code + text + math + GitHub data
2. prepare-data     — Filter, score, tokenize, mix
3. pretrain         — Mixed pretraining (code+text+math)
4. extend-context   — RoPE scaling + short fine-tune (optional)
5. generate-instructions — Create instruction tuning data
6. instruction-tune — SFT on ChatML instruction data
7. upcycle-moe      — Dense → MoE conversion (optional)
8. train-router     — Train semantic router classifier
9. train-reasoning  — Domain-aware GRPO / self-play
10. evaluate        — Full eval suite + safety

Usage:
    .venv/Scripts/python scripts/full_pipeline.py --config configs/4080_max.yaml
    .venv/Scripts/python scripts/full_pipeline.py --config configs/4080_max.yaml --stages 1,2,3
    .venv/Scripts/python scripts/full_pipeline.py --config configs/4080_max.yaml --start-from 5
    .venv/Scripts/python scripts/full_pipeline.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cola_coder.cli import cli
from cola_coder.model.config import Config


# Stage definitions
STAGES = {
    1: {"name": "collect-data", "description": "Download code, text, math, and GitHub data"},
    2: {"name": "prepare-data", "description": "Filter, score, tokenize, and mix data"},
    3: {"name": "pretrain", "description": "Mixed pretraining (code + text + math)"},
    4: {"name": "extend-context", "description": "Context extension via RoPE scaling"},
    5: {"name": "generate-instructions", "description": "Generate instruction tuning data"},
    6: {"name": "instruction-tune", "description": "SFT on ChatML instruction data"},
    7: {"name": "upcycle-moe", "description": "Convert dense model to MoE"},
    8: {"name": "train-router", "description": "Train semantic router classifier"},
    9: {"name": "train-reasoning", "description": "Domain-aware GRPO reasoning training"},
    10: {"name": "evaluate", "description": "Run full evaluation suite + safety checks"},
}


def run_stage(stage_num: int, config: Config, args: argparse.Namespace) -> bool:
    """Run a single pipeline stage.

    Args:
        stage_num: Stage number (1-10)
        config: Model/training config
        args: CLI arguments

    Returns:
        True if stage completed successfully
    """
    stage = STAGES[stage_num]
    cli.header(f"Stage {stage_num}: {stage['name']}", stage["description"])

    if args.dry_run:
        cli.dim(f"  [DRY RUN] Would run: {stage['name']}")
        return True

    start = time.perf_counter()

    try:
        if stage_num == 1:
            _stage_collect_data(config, args)
        elif stage_num == 2:
            _stage_prepare_data(config, args)
        elif stage_num == 3:
            _stage_pretrain(config, args)
        elif stage_num == 4:
            _stage_extend_context(config, args)
        elif stage_num == 5:
            _stage_generate_instructions(config, args)
        elif stage_num == 6:
            _stage_instruction_tune(config, args)
        elif stage_num == 7:
            _stage_upcycle_moe(config, args)
        elif stage_num == 8:
            _stage_train_router(config, args)
        elif stage_num == 9:
            _stage_train_reasoning(config, args)
        elif stage_num == 10:
            _stage_evaluate(config, args)

        elapsed = time.perf_counter() - start
        cli.success(f"Stage {stage_num} complete ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        cli.error(f"Stage {stage_num} failed ({elapsed:.1f}s)", str(e))
        return False


def _stage_collect_data(config: Config, args: argparse.Namespace) -> None:
    """Stage 1: Collect and prepare code data."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    cmd = [str(venv), "scripts/prepare_data.py", "--config", args.config]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    cmd.append("--score")
    cli.info("Collecting", f"Code data via prepare_data.py (config: {args.config})")
    subprocess.run(cmd, check=True)


def _stage_prepare_data(config: Config, args: argparse.Namespace) -> None:
    """Stage 2: Prepare and mix data."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    cmd = [str(venv), "scripts/prepare_data.py", "--config", args.config]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    subprocess.run(cmd, check=True)


def _stage_pretrain(config: Config, args: argparse.Namespace) -> None:
    """Stage 3: Run base pretraining."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    cmd = [str(venv), "scripts/train.py", "--config", args.config]
    if args.auto_resume:
        cmd.append("--auto-resume")
    subprocess.run(cmd, check=True)


def _stage_extend_context(config: Config, args: argparse.Namespace) -> None:
    """Stage 4: Extend context window via RoPE scaling."""
    rope_type = getattr(config.model.rope_scaling, "type", "none")
    rope_factor = getattr(config.model.rope_scaling, "factor", 1.0)

    if rope_type == "none" or rope_factor <= 1.0:
        cli.dim("RoPE scaling not configured — skipping context extension.")
        return

    import subprocess

    venv = Path(".venv/Scripts/python")
    seq_len = getattr(config.model, "max_seq_len", 2048)
    cli.info("RoPE scaling", f"type={rope_type}, factor={rope_factor}")
    cli.info("Context", f"{seq_len} → {int(seq_len * rope_factor)} tokens")

    cmd = [str(venv), "scripts/train.py", "--config", args.config]
    if args.auto_resume:
        cmd.append("--auto-resume")
    subprocess.run(cmd, check=True)


def _stage_generate_instructions(config: Config, args: argparse.Namespace) -> None:
    """Stage 5: Generate instruction tuning data."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    cmd = [str(venv), "scripts/generate_instructions.py"]
    subprocess.run(cmd, check=True)


def _stage_instruction_tune(config: Config, args: argparse.Namespace) -> None:
    """Stage 6: Run instruction tuning (SFT) via train_sft.py."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    ckpt_dir = Path(config.checkpoint.output_dir)
    latest = ckpt_dir / "latest"

    instruction_data = Path("data/sft/instructions.jsonl")
    if not instruction_data.exists():
        raise FileNotFoundError(
            f"Instruction data not found at {instruction_data}. "
            "Run Stage 5 (generate-instructions) first."
        )

    cmd = [
        str(venv), "scripts/train_sft.py",
        "--data", str(instruction_data),
        "--config", args.config,
        "--checkpoint", str(latest),
        "--epochs", "2",
        "--lr", "2e-5",
    ]
    cli.info("SFT", f"Fine-tuning on {instruction_data}")
    subprocess.run(cmd, check=True)


def _stage_upcycle_moe(config: Config, args: argparse.Namespace) -> None:
    """Stage 7: Convert to MoE."""
    try:
        moe_enabled = config.model.moe.enabled
    except AttributeError:
        moe_enabled = False

    if not moe_enabled:
        cli.dim("MoE not enabled in config — skipping")
        return

    import subprocess

    venv = Path(".venv/Scripts/python")
    config_stem = Path(args.config).stem
    num_experts = getattr(getattr(config.model, "moe", None), "num_experts", 8)
    num_shared = getattr(getattr(config.model, "moe", None), "num_shared_experts", 2)
    cmd = [
        str(venv), "scripts/upcycle_to_moe.py",
        "--config", args.config,
        "--checkpoint", f"checkpoints/{config_stem}/latest",
        "--num-experts", str(num_experts),
        "--num-shared", str(num_shared),
    ]
    subprocess.run(cmd, check=True)


def _stage_train_router(config: Config, args: argparse.Namespace) -> None:
    """Stage 8: Train semantic router."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    data_path = Path("data/router_training_data.jsonl")

    cmd = [str(venv), "scripts/train_router.py"]
    if data_path.exists():
        cli.info("Router data", f"Using existing {data_path}")
        cmd.extend(["--data", str(data_path)])
    else:
        cli.dim("  Generating router training data...")
        cmd.append("--generate-data")
    cmd.extend(["--arch", "mlp"])
    subprocess.run(cmd, check=True)


def _stage_train_reasoning(config: Config, args: argparse.Namespace) -> None:
    """Stage 9: Run GRPO reasoning training."""
    import subprocess

    venv = Path(".venv/Scripts/python")

    # Resolve the latest checkpoint produced by Stage 3 (pretrain)
    ckpt_dir = Path(config.checkpoint.output_dir)
    latest = ckpt_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {latest}. "
            "Run Stage 3 (pretrain) before Stage 9 (train-reasoning)."
        )

    cmd = [
        str(venv), "scripts/train_reasoning.py",
        "--config", "configs/reasoning.yaml",
        "--base-checkpoint", str(latest),
    ]
    subprocess.run(cmd, check=True)


def _stage_evaluate(config: Config, args: argparse.Namespace) -> None:
    """Stage 10: Run full evaluation (smoke + HumanEval + quality report)."""
    import subprocess

    venv = Path(".venv/Scripts/python")
    config_stem = Path(args.config).stem
    ckpt = f"checkpoints/{config_stem}/latest"

    cli.step(1, 3, "Running smoke test")
    subprocess.run([
        str(venv), "scripts/smoke_test.py",
        "--checkpoint", ckpt, "--config", args.config,
    ], check=False)

    cli.step(2, 3, "Running HumanEval")
    subprocess.run([
        str(venv), "scripts/evaluate.py",
        "--checkpoint", ckpt, "--config", args.config,
    ], check=False)

    cli.step(3, 3, "Generating quality report")
    subprocess.run([
        str(venv), "scripts/quality_report.py",
        "--checkpoint", ckpt, "--config", args.config, "--eval",
    ], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Cola-Coder Training Pipeline")
    parser.add_argument("--config", required=True, help="Model config YAML path")
    parser.add_argument("--stages", help="Comma-separated stage numbers (e.g., 1,2,3)")
    parser.add_argument("--start-from", type=int, help="Start from stage N")
    parser.add_argument("--stop-at", type=int, help="Stop after stage N")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-resume training")
    parser.add_argument("--tokenizer", help="Tokenizer path override")
    parser.add_argument(
        "--skip-optional", action="store_true", help="Skip optional stages (4, 7)"
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Determine which stages to run
    if args.stages:
        stage_nums = [int(s.strip()) for s in args.stages.split(",")]
    else:
        start = args.start_from or 1
        stop = args.stop_at or 10
        stage_nums = list(range(start, stop + 1))

    # Skip optional stages if requested
    optional_stages = {4, 7}
    if args.skip_optional:
        stage_nums = [s for s in stage_nums if s not in optional_stages]

    # Validate
    valid = set(STAGES.keys())
    invalid = [s for s in stage_nums if s not in valid]
    if invalid:
        cli.error("Invalid stages", f"{invalid}. Valid: 1-10")
        sys.exit(1)

    # Show pipeline
    cli.header("Cola-Coder Full Pipeline", config.summary())
    cli.rule("Pipeline Stages")

    for num in sorted(STAGES.keys()):
        stage = STAGES[num]
        marker = "→" if num in stage_nums else " "
        optional = " (optional)" if num in optional_stages else ""
        cli.print(f"  {marker} {num:2d}. {stage['name']:<25s} {stage['description']}{optional}")

    cli.rule()

    if args.dry_run:
        cli.dim("DRY RUN — no changes will be made")

    # Run stages
    results: dict[int, bool] = {}
    for num in stage_nums:
        success = run_stage(num, config, args)
        results[num] = success

        if not success and not args.dry_run:
            cli.warn(f"Stage {num} failed. Stopping pipeline.")
            cli.dim("Fix the issue and rerun with --start-from " + str(num))
            break

    # Summary
    cli.rule("Pipeline Summary")
    for num, success in results.items():
        stage = STAGES[num]
        status = "PASS" if success else "FAIL"
        cli.info(f"Stage {num}", f"{status} — {stage['name']}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    cli.done(f"Pipeline: {passed}/{total} stages completed")


if __name__ == "__main__":
    main()
