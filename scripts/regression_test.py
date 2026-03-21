"""Run regression tests on a cola-coder model checkpoint.

Compares generated outputs against known-good baselines and reports any
regressions.  Optionally saves results to a JSON file for later comparison
between checkpoints.

Usage:
    python scripts/regression_test.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml
    python scripts/regression_test.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --save results_v1.json
    python scripts/regression_test.py --compare results_v1.json results_v2.json
"""

import argparse
import json
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def _load_result_json(path: str) -> dict:
    """Load a previously saved RegressionResult as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _result_from_dict(data: dict):
    """Reconstruct a lightweight RegressionResult-like object from a saved dict."""
    from cola_coder.evaluation.regression import RegressionResult

    return RegressionResult(
        total=data["total"],
        passed=data["passed"],
        failed=data["failed"],
        details=data["details"],
    )


def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Run regression tests on a cola-coder model checkpoint."
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
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=storage.tokenizer_path,
        help=f"Path to tokenizer.json (default: {storage.tokenizer_path}).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save regression results to this JSON file for future comparison.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RESULTS_A", "RESULTS_B"),
        help="Compare two saved regression result JSON files instead of running a model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per baseline (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Regression Tests")

    # ---- Compare mode (no model needed) ----
    if args.compare:
        from cola_coder.evaluation.regression import RegressionSuite

        path_a, path_b = args.compare
        if not Path(path_a).exists():
            cli.fatal(f"File not found: {path_a}")
        if not Path(path_b).exists():
            cli.fatal(f"File not found: {path_b}")

        result_a = _result_from_dict(_load_result_json(path_a))
        result_b = _result_from_dict(_load_result_json(path_b))

        suite = RegressionSuite()
        comparison = suite.compare_checkpoints(
            result_a,
            result_b,
            label_a=path_a,
            label_b=path_b,
        )
        cli.print(f"\n{comparison}\n")
        return

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

    # ---- Load regression suite ----
    try:
        from cola_coder.evaluation.regression import RegressionSuite
    except ImportError:
        cli.fatal(
            "Could not import RegressionSuite. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    suite = RegressionSuite()
    cli.success(f"Running {len(suite.BASELINES)} regression baselines")
    cli.info("Max new tokens", args.max_new_tokens)
    cli.info("Temperature", args.temperature)
    print()

    # ---- Run suite ----
    reg_result = suite.run(
        generator=generator,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # ---- Print per-baseline results ----
    for detail in reg_result.details:
        if detail["passed"]:
            cli.print(f"  [green]PASS[/green]  [{detail['category']:<14}]  {detail['description']}")
        else:
            reasons = "; ".join(detail.get("failures", []))
            cli.print(
                f"  [red]FAIL[/red]  [{detail['category']:<14}]  {detail['description']}  <-- {reasons}"
            )

    cli.rule("Results")
    cli.print(f"\n{reg_result.summary()}\n")

    # ---- Save results ----
    if args.save:
        data = {
            "checkpoint": args.checkpoint,
            "total": reg_result.total,
            "passed": reg_result.passed,
            "failed": reg_result.failed,
            "pass_rate": reg_result.pass_rate,
            "details": reg_result.details,
        }
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        cli.success(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
