"""Quick model benchmark — run after training to see how your model performs.

Combines the smoke-test style "show me what the model generates" experience
with a more structured set of test prompts across Python, TypeScript, and async
code. Optionally runs the nano benchmark suite (10 ultra-simple TypeScript
problems with scoring) if it is enabled.

Usage:
    python scripts/benchmark.py                          # auto-detect checkpoint
    python scripts/benchmark.py --checkpoint checkpoints/tiny/step_00017000
    python scripts/benchmark.py --checkpoint path/to/ckpt --max-tokens 256
    python scripts/benchmark.py --temperature 0.0        # greedy (most deterministic)

What you'll see:
    - Each prompt printed, followed by the generated continuation
    - Wall-clock time for each generation
    - Summary table: total time, tokens/sec, average output length
    - Nano benchmark results (syntax, types, test pass rate) if enabled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------
# Five prompts that cover different generation scenarios.  Each is a tuple of
# (label, prompt_text) so we can print a clean name in the summary.
#
# Why these five?
#   1. fibonacci  — classic recursive function; tests basic Python generation
#   2. sort_list  — function signature completion; tests "fill in the blanks"
#   3. Calculator — class / OOP context; tests multi-line class generation
#   4. React App  — JSX / TypeScript; tests the TS-heavy side of the dataset
#   5. fetchData  — async JS; tests async/await patterns
# ---------------------------------------------------------------------------

TEST_PROMPTS: list[tuple[str, str]] = [
    (
        "Python: fibonacci",
        "def fibonacci(n):\n",
    ),
    (
        "Python: sort_list",
        "# Sort a list of integers\ndef sort_list(",
    ),
    (
        "Python: Calculator class",
        "class Calculator:\n    def __init__(self):\n",
    ),
    (
        "TypeScript/React: App component",
        "import React from 'react';\n\nfunction App() {\n",
    ),
    (
        "JavaScript: async fetchData",
        "// Fetch data from API\nasync function fetchData(",
    ),
]


# ---------------------------------------------------------------------------
# Helpers (mirrors run.py's auto-detection strategy)
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk up from this script's location to find the directory with pyproject.toml."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parent


def auto_detect_checkpoint(checkpoints_dir: Path) -> tuple[str, dict] | None:
    """Return (checkpoint_path, metadata) for the highest-step checkpoint, or None."""
    from cola_coder.training.checkpoint import detect_latest_checkpoint

    result = detect_latest_checkpoint(str(checkpoints_dir))
    if result is None:
        return None
    ckpt_path, metadata = result
    resolved = Path(ckpt_path)
    if not resolved.is_absolute():
        resolved = checkpoints_dir.parent / resolved
    return str(resolved), metadata


def build_config_from_metadata(metadata: dict):
    """Reconstruct a Config object from the dict embedded in metadata.json."""
    try:
        from cola_coder.model.config import Config, ModelConfig, TrainingConfig, DataConfig, CheckpointConfig

        raw = metadata.get("config", {})
        return Config(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(),
            checkpoint=CheckpointConfig(),
        )
    except Exception:
        return None


def auto_detect_config(checkpoint_dir: str, project_root: Path) -> str | None:
    """Match checkpoint dir name to configs/<name>.yaml, or return any .yaml found."""
    size_name = Path(checkpoint_dir).parent.name
    yaml_candidate = project_root / "configs" / f"{size_name}.yaml"
    if yaml_candidate.exists():
        return str(yaml_candidate)
    configs_dir = project_root / "configs"
    if configs_dir.exists():
        yamls = sorted(configs_dir.glob("*.yaml"))
        if yamls:
            return str(yamls[0])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from cola_coder.cli import cli

    # ── Argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Quick post-training benchmark for cola-coder models.\n"
            "Auto-detects your latest checkpoint — no flags required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory. Default: auto-detect latest.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Default: auto-detect from checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json. Default: tokenizer.json in project root.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        dest="max_tokens",
        help="Maximum new tokens to generate per prompt (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help=(
            "Sampling temperature (default: 0.3). "
            "Low values give more consistent, repeatable results — good for benchmarking."
        ),
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Benchmark")

    # ── Project root ──────────────────────────────────────────────────────
    project_root = find_project_root()
    checkpoints_dir = project_root / "checkpoints"

    # ── Auto-detect checkpoint ────────────────────────────────────────────
    checkpoint_dir: str
    metadata: dict

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            cli.fatal(
                f"Checkpoint not found: {args.checkpoint}",
                hint="Check the path or omit --checkpoint to auto-detect.",
            )
        checkpoint_dir = str(ckpt_path.resolve())
        meta_path = ckpt_path / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        cli.info("Checkpoint", checkpoint_dir)
    else:
        result = auto_detect_checkpoint(checkpoints_dir)
        if result is None:
            cli.error(
                "No checkpoints found.",
                hint=(
                    "Train a model first:\n"
                    f"  python scripts/train.py --config configs/tiny.yaml\n"
                    f"\nExpected structure: {checkpoints_dir}/<size>/step_XXXXXXXX/"
                ),
            )
            sys.exit(1)
        checkpoint_dir, metadata = result
        size_name = Path(checkpoint_dir).parent.name
        step = metadata.get("step", "?")
        label = (
            f"Auto-detected: {size_name} step {step:,}"
            if isinstance(step, int)
            else f"Auto-detected: {checkpoint_dir}"
        )
        cli.success(label)

    # ── Config ────────────────────────────────────────────────────────────
    config = None

    if args.config is not None:
        from cola_coder.model.config import Config

        config_path = Path(args.config)
        if not config_path.exists():
            cli.fatal(f"Config file not found: {args.config}")
        config = Config.from_yaml(str(config_path))
        cli.info("Config", str(config_path))
    else:
        # Prefer config embedded in metadata.json (no YAML file required)
        config = build_config_from_metadata(metadata)
        if config is not None:
            cli.info("Config", "loaded from checkpoint metadata.json")
        else:
            yaml_path = auto_detect_config(checkpoint_dir, project_root)
            if yaml_path is None:
                cli.fatal(
                    "Could not auto-detect a config file.",
                    hint="Provide --config configs/<size>.yaml explicitly.",
                )
            from cola_coder.model.config import Config

            config = Config.from_yaml(yaml_path)
            cli.info("Config", yaml_path)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    if args.tokenizer is not None:
        tokenizer_path = str(Path(args.tokenizer).resolve())
    else:
        tokenizer_path = str(project_root / "tokenizer.json")

    if not Path(tokenizer_path).exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint=(
                "Train one first: python scripts/train_tokenizer.py\n"
                "Or provide --tokenizer <path>"
            ),
        )

    # ── Device ────────────────────────────────────────────────────────────
    device = cli.gpu_info()

    # ── Load model ────────────────────────────────────────────────────────
    cli.print("\nLoading model...")

    try:
        import torch
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError as exc:
        cli.fatal(
            f"Import error: {exc}",
            hint="Make sure the package is installed: pip install -e .",
        )

    try:
        model = Transformer(config.model).to(device)
        load_model_only(checkpoint_dir, model, device=device)
        tokenizer = CodeTokenizer(tokenizer_path)
        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except torch.cuda.OutOfMemoryError:
        cli.fatal(
            "CUDA out of memory while loading the model.",
            hint=(
                "Try:\n"
                "  - Close other GPU applications (check nvidia-smi)\n"
                "  - Use a smaller model size (configs/tiny.yaml)\n"
                "  - Reduce --max-tokens"
            ),
        )
    except Exception as exc:
        cli.fatal(f"Failed to load model: {exc}")

    # ── Show model info ───────────────────────────────────────────────────
    ckpt_step = metadata.get("step", 0)
    ckpt_loss = metadata.get("loss", float("nan"))
    size_name = Path(checkpoint_dir).parent.name

    cli.kv_table(
        {
            "Model size": size_name,
            "Parameters": config.model.total_params_human,
            "Checkpoint step": f"{ckpt_step:,}",
            "Loss at checkpoint": f"{ckpt_loss:.4f}",
            "Device": device.upper(),
            "Max tokens / prompt": args.max_tokens,
            "Temperature": args.temperature,
        },
        title="Model Info",
    )

    # ── Run benchmark prompts ─────────────────────────────────────────────
    cli.rule("Benchmark Prompts")

    total_tokens_generated = 0
    prompt_times: list[float] = []
    output_lengths: list[int] = []

    for i, (label, prompt) in enumerate(TEST_PROMPTS, 1):
        cli.print(f"\n[bold cyan][{i}/{len(TEST_PROMPTS)}] {label}[/bold cyan]")
        cli.print(f"[dim]Prompt:[/dim]")
        # Print each line of the prompt with a leading │ so it's visually distinct
        for line in prompt.splitlines():
            cli.print(f"  [dim]│[/dim] {line}")

        try:
            t0 = time.perf_counter()
            output = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=40,
                top_p=0.9,
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            cli.warn(f"Generation failed: {exc}")
            continue

        # Strip prompt from output (generator returns prompt + generated text)
        new_text = output[len(prompt):] if output.startswith(prompt) else output

        # Estimate token count via tokenizer if available, otherwise approximate
        try:
            token_ids = tokenizer.encode(new_text)
            num_tokens = len(token_ids)
        except Exception:
            # Rough approximation: ~4 chars per token
            num_tokens = max(1, len(new_text) // 4)

        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0

        cli.print(f"[dim]Generated ({num_tokens} tokens, {elapsed:.2f}s, {tokens_per_sec:.0f} tok/s):[/dim]")
        # Print the generated text, indented
        for line in new_text.splitlines():
            cli.print(f"  {line}")

        prompt_times.append(elapsed)
        output_lengths.append(num_tokens)
        total_tokens_generated += num_tokens

    # ── Summary ───────────────────────────────────────────────────────────
    cli.rule("Summary")

    if prompt_times:
        total_time = sum(prompt_times)
        avg_time = total_time / len(prompt_times)
        avg_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        overall_tok_per_sec = total_tokens_generated / total_time if total_time > 0 else 0.0

        cli.kv_table(
            {
                "Prompts run": f"{len(prompt_times)}/{len(TEST_PROMPTS)}",
                "Total time": f"{total_time:.2f}s",
                "Average time / prompt": f"{avg_time:.2f}s",
                "Total tokens generated": str(total_tokens_generated),
                "Average output length": f"{avg_length:.0f} tokens",
                "Overall throughput": f"{overall_tok_per_sec:.0f} tok/s",
            },
            title="Benchmark Results",
        )
    else:
        cli.warn("No prompts completed successfully.")

    # ── Optional: nano benchmark ──────────────────────────────────────────
    # The nano benchmark runs 10 ultra-simple TypeScript problems and scores
    # them for syntax validity, type correctness, and test execution.
    # It requires `tsc` and `node` to be on PATH for the type/execution checks.
    try:
        from cola_coder.features.nano_benchmark import is_enabled, NanoBenchmark

        if is_enabled():
            cli.rule("Nano Benchmark")
            cli.dim("Running 10 TypeScript problems (requires tsc + node on PATH)...")
            bench = NanoBenchmark()
            # NanoBenchmark.run() accepts any object with .generate(prompt, ...) method.
            # CodeGenerator matches that interface exactly.
            bench.run(generator)
        else:
            cli.dim("Nano benchmark disabled (set FEATURE_ENABLED=True in nano_benchmark.py to enable).")
    except ImportError:
        cli.dim("Nano benchmark not available (could not import nano_benchmark module).")
    except Exception as exc:
        cli.warn(f"Nano benchmark skipped: {exc}")

    # ── Done ──────────────────────────────────────────────────────────────
    cli.done(
        "Benchmark complete.",
        extras={
            "Checkpoint": checkpoint_dir,
            "Step": f"{ckpt_step:,}",
            "Loss": f"{ckpt_loss:.4f}",
        },
    )


if __name__ == "__main__":
    main()
