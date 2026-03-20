"""Compare two model checkpoints — see training progress.

Loads two checkpoints and shows a side-by-side metrics table (loss, perplexity,
improvement deltas), then runs the same test prompt through both models and
prints the outputs for a qualitative feel of how generation changed.

Usage:
    python scripts/compare_checkpoints.py                         # compare last 2 checkpoints
    python scripts/compare_checkpoints.py --a step_00015000 --b step_00017000
    python scripts/compare_checkpoints.py --a checkpoints/tiny/step_00015000 --b checkpoints/tiny/step_00017000
    python scripts/compare_checkpoints.py --no-generate           # skip model loading, metrics only
    python scripts/compare_checkpoints.py --prompt "class Stack:" # custom test prompt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Make sure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


# ---------------------------------------------------------------------------
# Checkpoint scanning helpers
# ---------------------------------------------------------------------------

def _find_all_checkpoint_dirs(checkpoints_root: Path) -> list[Path]:
    """Return all step_* checkpoint directories, sorted by step number (ascending).

    Searches one level deep: checkpoints/<size>/step_XXXXXXXX
    """
    found: list[Path] = []
    if not checkpoints_root.exists():
        return found

    for size_dir in sorted(checkpoints_root.iterdir()):
        if not size_dir.is_dir():
            continue
        for step_dir in sorted(
            size_dir.glob("step_*"),
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        ):
            if (step_dir / "metadata.json").exists():
                found.append(step_dir)

    return found


def _resolve_checkpoint(name: str, checkpoints_root: Path) -> Path:
    """Resolve a checkpoint name/path to an absolute Path.

    Accepts:
    - An absolute path: /full/path/to/step_00015000
    - A relative path:  checkpoints/tiny/step_00015000
    - A bare name:      step_00015000  (searched under checkpoints_root)
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    # Search under checkpoints root for a matching step dir
    for size_dir in checkpoints_root.iterdir():
        if not size_dir.is_dir():
            continue
        candidate = size_dir / name
        if candidate.exists():
            return candidate

    cli.fatal(
        f"Checkpoint not found: {name!r}",
        hint=f"Searched in {checkpoints_root} and as a direct path.",
    )


def _read_metadata(ckpt_dir: Path) -> dict:
    """Read metadata.json from a checkpoint directory. Returns {} on failure."""
    import json

    meta_path = ckpt_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity: e^loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _fmt_delta(delta: float, percent: float, lower_is_better: bool = True) -> str:
    """Format a numeric delta with a directional indicator."""
    direction = "-" if delta < 0 else "+"
    sign = "↓" if delta < 0 else "↑"
    result = f"{direction}{abs(delta):.4f} ({sign}{abs(percent):.1f}%)"
    return result


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def _print_metrics_table(
    dir_a: Path,
    dir_b: Path,
    meta_a: dict,
    meta_b: dict,
) -> None:
    """Print a side-by-side metrics comparison table."""
    step_a = meta_a.get("step", "?")
    step_b = meta_b.get("step", "?")
    loss_a = float(meta_a.get("loss", float("nan")))
    loss_b = float(meta_b.get("loss", float("nan")))
    ppl_a = _perplexity(loss_a)
    ppl_b = _perplexity(loss_b)

    loss_delta = loss_b - loss_a
    loss_pct = (loss_delta / loss_a * 100) if loss_a != 0 else 0.0
    ppl_delta = ppl_b - ppl_a
    ppl_pct = (ppl_delta / ppl_a * 100) if ppl_a != 0 else 0.0

    cli.kv_table(
        {
            "Checkpoint A": f"{dir_a.name}  ({dir_a.parent.name})",
            "Checkpoint B": f"{dir_b.name}  ({dir_b.parent.name})",
        },
        title="Checkpoints",
    )

    # Individual stats
    cli.kv_table(
        {
            "Step A": f"{step_a:,}" if isinstance(step_a, int) else str(step_a),
            "Step B": f"{step_b:,}" if isinstance(step_b, int) else str(step_b),
            "Loss A": f"{loss_a:.4f}",
            "Loss B": f"{loss_b:.4f}",
            "Loss delta (B − A)": _fmt_delta(loss_delta, loss_pct, lower_is_better=True),
            "Perplexity A": f"{ppl_a:.2f}",
            "Perplexity B": f"{ppl_b:.2f}",
            "Perplexity delta (B − A)": _fmt_delta(ppl_delta, ppl_pct, lower_is_better=True),
        },
        title="Metrics",
    )

    # Verdict
    if loss_b < loss_a:
        cli.success(f"Training improved: loss dropped by {abs(loss_delta):.4f} ({abs(loss_pct):.1f}%)")
    elif loss_b > loss_a:
        cli.warn(f"Loss increased by {loss_delta:.4f} ({loss_pct:.1f}%) — check for overfitting or LR issues")
    else:
        cli.info("Verdict", "Loss unchanged between checkpoints")


def _print_generation_comparison(
    dir_a: Path,
    dir_b: Path,
    meta_a: dict,
    meta_b: dict,
    prompt: str,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
) -> None:
    """Load both models and run the test prompt through each, showing outputs side-by-side."""
    try:
        from cola_coder.model.config import ModelConfig
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError as e:
        cli.warn(f"Cannot import model components: {e}")
        cli.warn("Skipping generation comparison.")
        return

    tokenizer_path_p = Path(tokenizer_path)
    if not tokenizer_path_p.exists():
        cli.warn(f"Tokenizer not found at {tokenizer_path!r} — skipping generation.")
        return

    def _load_generator(ckpt_dir: Path, meta: dict) -> CodeGenerator | None:
        """Load a model from a checkpoint directory, using config embedded in metadata."""
        config_raw = meta.get("config", {})
        model_cfg_raw = config_raw.get("model", {})

        if not model_cfg_raw:
            cli.warn(f"No model config in metadata for {ckpt_dir.name} — cannot load model.")
            return None

        try:
            # ModelConfig uses only the fields it knows about
            valid_fields = ModelConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in model_cfg_raw.items() if k in valid_fields}
            model_cfg = ModelConfig(**filtered)

            tokenizer = CodeTokenizer(tokenizer_path)
            model = Transformer(model_cfg).to(device)
            load_model_only(str(ckpt_dir), model, device=device)
            return CodeGenerator(model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            cli.warn(f"Failed to load model from {ckpt_dir.name}: {e}")
            return None

    cli.rule("Generation Comparison")
    cli.info("Test prompt", repr(prompt))
    cli.info("Max new tokens", max_new_tokens)

    cli.print(f"\n  Loading checkpoint A ({dir_a.name})...")
    gen_a = _load_generator(dir_a, meta_a)
    if gen_a:
        output_a = gen_a.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_k=1)
    else:
        output_a = "(failed to load)"

    cli.print(f"  Loading checkpoint B ({dir_b.name})...")
    gen_b = _load_generator(dir_b, meta_b)
    if gen_b:
        output_b = gen_b.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_k=1)
    else:
        output_b = "(failed to load)"

    # Print results side-by-side (stacked, since terminals are narrow)
    cli.rule(f"Output — Checkpoint A ({dir_a.name})")
    cli.print(output_a)

    cli.rule(f"Output — Checkpoint B ({dir_b.name})")
    cli.print(output_b)

    cli.rule()


def _print_weight_diff(dir_a: Path, dir_b: Path) -> None:
    """Use CheckpointComparison to show top changed layers."""
    try:
        from cola_coder.features.checkpoint_comparison import is_enabled, CheckpointComparison

        if not is_enabled():
            return

        cli.rule("Weight Change Analysis")
        cli.print("  Computing weight diffs (loading both state dicts onto CPU)...")

        comp = CheckpointComparison(dir_a, dir_b)
        summary = comp.summary()

        cli.kv_table(
            {
                "Shared layers": str(summary["shared_layers"]),
                "Mean L2 distance": f"{summary['mean_l2_distance']:.4f}",
                "Max L2 distance": f"{summary['max_l2_distance']:.4f}",
                "Mean cosine similarity": f"{summary['mean_cosine_similarity']:.6f}",
                "Fraction of layers changed": f"{summary['fraction_changed']:.1%}",
            },
            title="Weight Diffs (A → B)",
        )

        top_changes = comp.biggest_changes(top_k=5)
        if top_changes:
            cli.rule("Top 5 Most-Changed Layers")
            for i, entry in enumerate(top_changes, 1):
                cli.print(
                    f"  [bold cyan]{i}.[/bold cyan] {entry['name']}"
                    f"  L2={entry['l2_distance']:.4f}"
                    f"  cos={entry['cosine_similarity']:.5f}"
                    f"  ({entry['numel']:,} params)"
                )

        cli.rule()

    except Exception as e:
        # The feature is optional — don't crash if it fails
        cli.warn(f"Weight diff analysis skipped: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two cola-coder checkpoints side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--a",
        metavar="CHECKPOINT_A",
        default=None,
        help="First checkpoint (older). Name like step_00015000, or a full path. "
             "Defaults to second-most-recent checkpoint.",
    )
    parser.add_argument(
        "--b",
        metavar="CHECKPOINT_B",
        default=None,
        help="Second checkpoint (newer). Name like step_00017000, or a full path. "
             "Defaults to most-recent checkpoint.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        metavar="DIR",
        default="checkpoints",
        help="Base checkpoints directory (default: checkpoints).",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer.json",
        help="Path to tokenizer.json (default: tokenizer.json).",
    )
    parser.add_argument(
        "--prompt",
        default="def fibonacci(n):\n",
        help='Test prompt for generation comparison (default: "def fibonacci(n):\\n").',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate per checkpoint (default: 128).",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip model loading and generation — show metrics table only (fast, CPU-only).",
    )
    parser.add_argument(
        "--no-weight-diff",
        action="store_true",
        help="Skip weight diff analysis (saves time if you only care about loss metrics).",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Checkpoint Comparison")

    # ---- Resolve checkpoint root ----
    checkpoints_root = Path(args.checkpoints_dir)
    if not checkpoints_root.exists():
        cli.fatal(
            f"Checkpoints directory not found: {checkpoints_root}",
            hint="Run training first, or pass --checkpoints-dir with the correct path.",
        )

    # ---- Auto-detect if not specified ----
    if args.a is None or args.b is None:
        all_ckpts = _find_all_checkpoint_dirs(checkpoints_root)
        if len(all_ckpts) < 2:
            cli.fatal(
                "Need at least 2 checkpoints for comparison.",
                hint=f"Found {len(all_ckpts)} checkpoint(s) in {checkpoints_root}. "
                     "Train more steps, or pass --a and --b explicitly.",
            )
        dir_a = all_ckpts[-2] if args.a is None else _resolve_checkpoint(args.a, checkpoints_root)
        dir_b = all_ckpts[-1] if args.b is None else _resolve_checkpoint(args.b, checkpoints_root)
        cli.info("Auto-detected A", dir_a.name)
        cli.info("Auto-detected B", dir_b.name)
    else:
        dir_a = _resolve_checkpoint(args.a, checkpoints_root)
        dir_b = _resolve_checkpoint(args.b, checkpoints_root)

    # ---- Read metadata ----
    meta_a = _read_metadata(dir_a)
    meta_b = _read_metadata(dir_b)

    if not meta_a:
        cli.warn(f"No metadata.json found in {dir_a} — some metrics may be missing.")
    if not meta_b:
        cli.warn(f"No metadata.json found in {dir_b} — some metrics may be missing.")

    # ---- Metrics table ----
    cli.rule("Training Metrics")
    _print_metrics_table(dir_a, dir_b, meta_a, meta_b)

    # ---- Weight diff (optional) ----
    if not args.no_weight_diff:
        _print_weight_diff(dir_a, dir_b)

    # ---- Generation comparison (optional) ----
    if not args.no_generate:
        device = cli.gpu_info()
        _print_generation_comparison(
            dir_a=dir_a,
            dir_b=dir_b,
            meta_a=meta_a,
            meta_b=meta_b,
            prompt=args.prompt,
            tokenizer_path=args.tokenizer,
            max_new_tokens=args.max_tokens,
            device=device,
        )
    else:
        cli.dim("Generation skipped (--no-generate).")

    cli.done("Comparison complete.")


if __name__ == "__main__":
    main()
