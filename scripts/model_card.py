"""Generate a model card for a trained checkpoint.

A model card is a markdown file that documents your trained model — what it is,
how it was trained, how to use it, and what its known limitations are.  Think
of it as a README that travels with the model weights.

Usage:
    python scripts/model_card.py                          # auto-detect checkpoint
    python scripts/model_card.py --checkpoint checkpoints/tiny/step_00017000
    python scripts/model_card.py --output MODEL_CARD.md   # custom output path
    python scripts/model_card.py --checkpoint path/to/ckpt --output path/to/output.md

Output:
    MODEL_CARD.md in the project root (or the path you provide via --output).

The generated card includes:
    - Model name, size, and version
    - Architecture details (from config embedded in metadata.json)
    - Training details: step, loss, tokens seen, data source, hardware
    - How to run the model (copy-pasteable run.py command)
    - Known limitations of small code-generation models
    - License
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Helpers (same pattern as run.py / benchmark.py)
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk up from this script's directory to find the directory with pyproject.toml."""
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


def _format_params(n: int) -> str:
    """Format a parameter count as a human-readable string (50M, 125M, 1.3B …)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n) if n else "unknown"


def _count_params(config_dict: dict) -> int:
    """Estimate total parameters from a ModelConfig dict.

    Uses the same fields that ModelConfig exposes:
        vocab_size, d_model, n_heads, n_kv_heads, n_layers, d_ffn, max_seq_len
    Falls back to 0 if the config is missing those keys.
    """
    try:
        d = config_dict.get("model", config_dict)  # handle wrapped or flat dicts
        vocab = d.get("vocab_size", 0)
        dm = d.get("d_model", 0)
        nh = d.get("n_heads", 0)
        n_kv = d.get("n_kv_heads", nh)
        nl = d.get("n_layers", 0)
        dff = d.get("d_ffn", dm * 4)

        if not (vocab and dm and nl):
            return 0

        head_dim = dm // nh if nh else 0

        # Embeddings
        emb = vocab * dm

        # Per-layer: attention (Q, K, V, O projections) + feed-forward (3 matrices for SwiGLU)
        attn = dm * dm + 2 * (n_kv * head_dim) * dm + dm * dm
        ffn = 3 * dm * dff  # SwiGLU has gate, up, down projections
        layer = attn + ffn + 2 * dm  # +2*dm for RMSNorm weights

        return emb + nl * layer + dm  # final RMSNorm
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Fallback card generator (used when ModelCardGenerator feature is disabled)
# ---------------------------------------------------------------------------

def _build_fallback_card(
    *,
    checkpoint_dir: str,
    metadata: dict,
    manifest: dict,
    project_root: Path,
) -> str:
    """Build a model card as a markdown string without using ModelCardGenerator.

    This is the fallback path: it constructs the card directly from the data
    we have on hand (metadata.json, training_manifest.yaml).
    """
    raw_cfg = metadata.get("config", {})
    model_cfg = raw_cfg.get("model", raw_cfg)
    training_cfg = raw_cfg.get("training", {})

    step = metadata.get("step", "unknown")
    loss = metadata.get("loss", None)
    size_name = Path(checkpoint_dir).parent.name
    model_name = f"Cola-Coder {size_name.capitalize()}"

    # Parameter count — prefer config if available
    n_params = _count_params(raw_cfg)
    if n_params == 0 and model_cfg:
        # Try direct lookup (ModelConfig stores total_params as a property, not in
        # the serialised dict; we estimate from the architecture dimensions instead)
        pass
    params_str = _format_params(n_params) if n_params else "unknown"

    # Architecture fields
    d_model = model_cfg.get("d_model", "?")
    n_layers = model_cfg.get("n_layers", "?")
    n_heads = model_cfg.get("n_heads", "?")
    n_kv_heads = model_cfg.get("n_kv_heads", n_heads)
    vocab_size = model_cfg.get("vocab_size", "?")
    max_seq_len = model_cfg.get("max_seq_len", "?")
    d_ffn = model_cfg.get("d_ffn", "?")

    # Training details — prefer manifest, fall back to metadata
    prog = manifest.get("progress", {})
    tokens_seen = prog.get("tokens_seen") or training_cfg.get("tokens_seen", "unknown")
    epochs_completed = prog.get("epochs_completed", "unknown")
    best_loss = prog.get("best_loss") or (f"{loss:.4f}" if loss is not None else "unknown")
    best_step = prog.get("best_step") or step

    # Hardware
    hw = manifest.get("hardware", {})
    gpu_name = hw.get("gpu", "unknown")
    vram_gb = hw.get("vram_gb", "?")
    hardware_str = f"{gpu_name} ({vram_gb} GB VRAM)" if gpu_name != "unknown" else "unknown"

    # Data source
    dataset_src = manifest.get("source", {}).get("dataset", "bigcode/starcoderdata")
    languages = manifest.get("source", {}).get("languages", ["Python", "TypeScript", "JavaScript"])
    lang_str = ", ".join(languages) if languages else "Python, TypeScript, JavaScript"

    # Training hyperparams
    lr = training_cfg.get("learning_rate", "?")
    batch_size = training_cfg.get("batch_size", "?")
    grad_accum = training_cfg.get("gradient_accumulation", "?")
    effective_batch = (
        batch_size * grad_accum
        if isinstance(batch_size, int) and isinstance(grad_accum, int)
        else "?"
    )
    precision = training_cfg.get("precision", "bf16")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    card = f"""\
# {model_name}

**Version:** {size_name}
**License:** Apache 2.0
**Generated:** {now_utc}

## Overview

{model_name} is a decoder-only transformer trained from scratch on open-source
code.  It was built as part of the [Cola-Coder](https://github.com/cola-coder)
learning project — a from-scratch code generation transformer that mirrors the
LLaMA / Mistral architecture.

## Model Information

| Field | Value |
|-------|-------|
| Architecture | Decoder-only transformer (RoPE, GQA, SwiGLU, RMSNorm) |
| Parameters | {params_str} |
| Languages | {lang_str} |
| Vocabulary size | {vocab_size} |
| Context length | {max_seq_len} tokens |
| License | Apache 2.0 |

## Architecture Details

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension (`d_model`) | {d_model} |
| Feed-forward dimension (`d_ffn`) | {d_ffn} |
| Attention heads | {n_heads} |
| KV heads (GQA) | {n_kv_heads} |
| Layers | {n_layers} |
| Positional encoding | RoPE |
| Activation | SwiGLU |
| Normalization | RMSNorm (pre-norm) |

## Training Details

| Field | Value |
|-------|-------|
| Dataset | {dataset_src} |
| Languages | {lang_str} |
| Training steps | {step:,} |
| Best step | {best_step:,} |
| Best loss | {best_loss} |
| Tokens seen | {tokens_seen:,} |
| Epochs completed | {epochs_completed} |
| Learning rate | {lr} |
| Batch size | {batch_size} |
| Gradient accumulation | {grad_accum} |
| Effective batch | {effective_batch} |
| Precision | {precision} |
| Hardware | {hardware_str} |
| Checkpoint | `{checkpoint_dir}` |

## How to Use

Make sure you have the Cola-Coder package installed and a trained tokenizer:

```bash
# Install the package
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

# Interactive code generation REPL (auto-detects this checkpoint)
python scripts/run.py

# Or point explicitly at this checkpoint:
python scripts/run.py --checkpoint {checkpoint_dir}

# Non-interactive generation:
python scripts/generate.py \\
    --checkpoint {checkpoint_dir} \\
    --config configs/{size_name}.yaml \\
    --temperature 0.3 \\
    --max-tokens 256
```

## Quick Benchmark

```bash
python scripts/benchmark.py --checkpoint {checkpoint_dir}
```

## Limitations

- **Small model, narrow data** — Cola-Coder {size_name.capitalize()} has {params_str} parameters and
  was trained on a limited token budget.  It will struggle with complex
  multi-file reasoning, long-range dependencies, and tasks that require
  broad world knowledge.

- **No instruction tuning** — The model is a base language model trained on
  raw code files.  It is not fine-tuned on instruction-following data, so
  it will not reliably respond to natural-language requests like "write me a
  function that …".  Feed it code prefixes, not English prompts.

- **TypeScript / Python bias** — The training data skews toward Python,
  TypeScript, and JavaScript.  Performance on other languages (Go, Rust,
  Java, etc.) will be lower.

- **Not production-ready** — This model is a learning artefact.  It has not
  been evaluated for safety, bias, or correctness on real-world tasks.  Do
  not deploy it in production without your own evaluation.

- **Context window** — The model supports up to {max_seq_len} tokens of context.
  Very long files or conversations will be truncated.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

*This model card was generated automatically by `scripts/model_card.py`.*
"""
    return card


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from cola_coder.cli import cli

    # ── Argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Generate a MODEL_CARD.md for a trained cola-coder checkpoint.\n"
            "Auto-detects the latest checkpoint — no flags required."
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
        "--output",
        type=str,
        default=None,
        help="Output path for MODEL_CARD.md. Default: MODEL_CARD.md in project root.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Model Card Generator")

    # ── Project root ──────────────────────────────────────────────────────
    storage = get_storage_config()
    project_root = find_project_root()
    checkpoints_dir = Path(storage.checkpoints_dir)

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

    # ── Display what we found ─────────────────────────────────────────────
    ckpt_step = metadata.get("step", 0)
    ckpt_loss = metadata.get("loss", float("nan"))
    size_name = Path(checkpoint_dir).parent.name

    cli.kv_table(
        {
            "Model size": size_name,
            "Step": f"{ckpt_step:,}",
            "Loss": f"{ckpt_loss:.4f}",
            "Checkpoint": checkpoint_dir,
        },
        title="Checkpoint Info",
    )

    # ── Read training_manifest.yaml if it exists ──────────────────────────
    # The manifest lives next to the step_XXXXXXXX directory (i.e., in
    # checkpoints/<size>/training_manifest.yaml).
    manifest: dict = {}
    manifest_path = Path(checkpoint_dir).parent / "training_manifest.yaml"
    if manifest_path.exists():
        try:
            from cola_coder.manifest import read_manifest

            manifest = read_manifest(str(manifest_path))
            cli.info("Manifest", str(manifest_path))
        except Exception as exc:
            cli.warn(f"Could not read training manifest: {exc}")
    else:
        cli.dim("No training_manifest.yaml found — training details will be limited.")

    # ── Determine output path ─────────────────────────────────────────────
    output_path = Path(args.output) if args.output else project_root / "MODEL_CARD.md"

    # ── Try the ModelCardGenerator feature first ──────────────────────────
    card_content: str | None = None

    try:
        from cola_coder.features.model_card_generator import (
            is_enabled,
            ModelCardGenerator,
            ModelInfo,
            TrainingInfo,
        )

        if is_enabled():
            cli.dim("Using ModelCardGenerator feature...")

            raw_cfg = metadata.get("config", {})
            training_cfg = raw_cfg.get("training", {})

            n_params = _count_params(raw_cfg)

            model_info = ModelInfo(
                name=f"Cola-Coder {size_name.capitalize()}",
                version=size_name,
                architecture=(
                    "Decoder-only transformer (RoPE, GQA, SwiGLU, RMSNorm) — "
                    "same family as LLaMA 3 / Mistral"
                ),
                parameters=n_params,
                languages=manifest.get("source", {}).get(
                    "languages", ["Python", "TypeScript", "JavaScript"]
                ),
                license="Apache 2.0",
            )

            # Build TrainingInfo from manifest + metadata
            prog = manifest.get("progress", {})
            hw = manifest.get("hardware", {})
            gpu_name = hw.get("gpu", "unknown")
            vram_gb = hw.get("vram_gb", "?")
            hardware_str = (
                f"{gpu_name} ({vram_gb} GB VRAM)" if gpu_name != "unknown" else "unknown"
            )
            dataset_str = manifest.get("source", {}).get(
                "dataset", "bigcode/starcoderdata"
            )
            epochs = prog.get("epochs_completed", 0)
            if isinstance(epochs, float):
                epochs = int(epochs) or 1

            training_info = TrainingInfo(
                dataset=dataset_str,
                epochs=epochs,
                learning_rate=float(training_cfg.get("learning_rate", 0.0)),
                batch_size=int(training_cfg.get("batch_size", 0)),
                hardware=hardware_str,
                training_time=f"{ckpt_step:,} steps",
            )

            generator_obj = ModelCardGenerator(model_info, training_info)

            # Add metrics if we have them
            if ckpt_loss and not (isinstance(ckpt_loss, float) and ckpt_loss != ckpt_loss):
                generator_obj.add_metric("Training loss (final)", round(ckpt_loss, 4))
                best_loss = prog.get("best_loss")
                if best_loss:
                    generator_obj.add_metric("Best training loss", round(best_loss, 4))

            # Add a usage example using run.py
            generator_obj.add_example(
                prompt="def fibonacci(n):\n",
                output=(
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fibonacci(n - 1) + fibonacci(n - 2)"
                ),
            )

            # Add limitations
            generator_obj.add_limitation(
                f"Small model ({_format_params(n_params)} parameters) with limited training "
                "budget — struggles with complex multi-file reasoning."
            )
            generator_obj.add_limitation(
                "Base language model — not instruction-tuned.  Feed code prefixes, not "
                "natural-language requests."
            )
            generator_obj.add_limitation(
                "Biased toward Python, TypeScript, and JavaScript.  Other languages will "
                "see lower quality output."
            )
            generator_obj.add_limitation(
                "Not evaluated for safety or correctness on real-world tasks.  "
                "Do not deploy in production without your own evaluation."
            )

            card_content = generator_obj.generate()
        else:
            cli.dim(
                "ModelCardGenerator feature is disabled "
                "(set FEATURE_ENABLED=True in model_card_generator.py to enable). "
                "Using built-in template instead."
            )
    except ImportError:
        cli.dim("ModelCardGenerator not available — using built-in template.")
    except Exception as exc:
        cli.warn(f"ModelCardGenerator failed ({exc}) — falling back to built-in template.")

    # ── Fallback: build card from template ────────────────────────────────
    if card_content is None:
        card_content = _build_fallback_card(
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
            manifest=manifest,
            project_root=project_root,
        )

    # ── Write output ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_content, encoding="utf-8")

    cli.done(
        "Model card saved.",
        extras={
            "Output": str(output_path),
            "Checkpoint": checkpoint_dir,
            "Step": f"{ckpt_step:,}",
            "Loss": f"{ckpt_loss:.4f}",
            "Size": f"{len(card_content):,} bytes",
        },
    )


if __name__ == "__main__":
    main()
