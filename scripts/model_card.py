"""Generate a model card for a trained checkpoint.

A model card is a markdown file that documents your trained model — what it is,
how it was trained, how to use it, and what its known limitations are.  Think
of it as a README that travels with the model weights.

Usage:
    python scripts/model_card.py                          # auto-detect checkpoint
    python scripts/model_card.py --checkpoint checkpoints/tiny/step_00017000
    python scripts/model_card.py --output MODEL_CARD.md   # custom output path
    python scripts/model_card.py --checkpoint path/to/ckpt --output path/to/output.md
    python scripts/model_card.py --benchmarks eval_results.json  # add benchmark section

Output:
    MODEL_CARD.md in the project root (or the path you provide via --output).

The generated card includes:
    - Model name, size, and version
    - Architecture details (from config embedded in metadata.json)
    - Training details: step, loss, tokens seen, data source, hardware
    - Benchmark results section (if --benchmarks provided or eval_results.json found)
    - Training data summary
    - Hardware requirements section
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
# Language-specific prompts used to generate usage examples
# ---------------------------------------------------------------------------

LANGUAGE_PROMPTS: dict[str, list[tuple[str, str]]] = {
    "typescript": [
        ("TypeScript", "function fibonacci(n: number): number {\n"),
        ("TypeScript", "const fetchUser = async (id: string): Promise<User> => {\n"),
        ("TypeScript", "interface Config {\n"),
    ],
    "typescript-react": [
        ("TypeScript React", "import React from 'react';\n\nconst Button: React.FC<{ label: string }> = ("),
        ("TypeScript React", "export default function HomePage() {\n  const [data, setData] = useState"),
    ],
    "javascript": [
        ("JavaScript", "function debounce(fn, delay) {\n"),
        ("JavaScript", "class EventEmitter {\n"),
    ],
    "javascript-react": [
        ("JavaScript React", "import React, { useState } from 'react';\n\nexport default function App("),
    ],
    "python": [
        ("Python", "def fibonacci(n: int) -> int:\n"),
        ("Python", "class LinkedList:\n"),
        ("Python", "async def fetch_data(url: str) -> dict:\n"),
    ],
}


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
    benchmarks_path: Path | None = None,
    languages_arg: str = "typescript,python",
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

    # Data source — training languages come from manifest; display languages
    # come from the --languages flag (used for examples/overview).
    dataset_src = manifest.get("source", {}).get("dataset", "bigcode/starcoderdata")
    training_languages = manifest.get("source", {}).get(
        "languages", ["Python", "TypeScript", "JavaScript"]
    )
    training_lang_str = ", ".join(training_languages) if training_languages else "Python, TypeScript, JavaScript"
    # Derive display-friendly names from the --languages flag
    _req_langs_fb = [
        lang.strip() for lang in (languages_arg or "typescript,python").split(",") if lang.strip()
    ]
    _display_langs_fb = [
        LANGUAGE_PROMPTS.get(lang, [(lang.title(), "")])[0][0] for lang in _req_langs_fb
    ] if _req_langs_fb else training_languages
    lang_str = ", ".join(_display_langs_fb)

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

    # Formatted tokens / loss for the description
    tokens_str_fb = (
        f"{tokens_seen / 1e9:.2f}B"
        if isinstance(tokens_seen, int) and tokens_seen >= 1e9
        else f"{tokens_seen / 1e6:.0f}M"
        if isinstance(tokens_seen, int)
        else "unknown"
    )
    loss_str_fb = f"{loss:.4f}" if isinstance(loss, float) else "unknown"

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    card = f"""\
# {model_name}

**Version:** {size_name}
**License:** Apache 2.0
**Generated:** {now_utc}

## Overview

**{model_name}** is a {params_str}-parameter decoder-only transformer trained
from scratch on open-source code ({tokens_str_fb} tokens, {step:,} steps,
final loss {loss_str_fb}).

The architecture mirrors LLaMA 3 / Mistral: rotary positional embeddings (RoPE),
grouped-query attention (GQA, {n_heads}Q / {n_kv_heads}KV heads), SwiGLU
feed-forward networks, and pre-norm RMSNorm throughout.  It was built as part of
the [Cola-Coder](https://github.com/cola-coder) project — a from-scratch code
generation transformer targeting {lang_str}.

The model is a **base language model** (code completion style).  Feed it a code
prefix and it continues the code.  It is not instruction-tuned.

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
| Training languages | {training_lang_str} |
| Training steps | {step:,} |
| Best step | {best_step:,} |
| Best loss | {best_loss} |
| Tokens seen | {tokens_str_fb} |
| Epochs completed | {epochs_completed} |
| Learning rate | {lr} |
| Batch size | {batch_size} |
| Gradient accumulation | {grad_accum} |
| Effective batch | {effective_batch} |
| Precision | {precision} |
| Hardware | {hardware_str} |
| Checkpoint | `{checkpoint_dir}` |

{_build_benchmark_section(benchmarks_path)}{_build_data_summary_section(manifest, training_cfg)}{_build_hardware_section(manifest, training_cfg)}## How to Use

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

- **Small model, limited budget** — {model_name} ({params_str} parameters,
  {tokens_str_fb} tokens seen) will struggle with complex multi-file reasoning,
  long-range dependencies, and tasks that require broad world knowledge.

- **No instruction tuning** — This is a base language model trained on raw code
  files.  It is not fine-tuned on instruction-following data, so it will not
  reliably respond to natural-language requests.  Feed it code prefixes, not
  English prompts.

- **Training language bias** — The training data is weighted toward
  {training_lang_str}.  Performance on other languages (Go, Rust, Java, etc.)
  will be lower.

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

def _build_benchmark_section(benchmarks_path: Path | None) -> str:
    """Build a markdown Benchmark Results section from an eval_results.json file.

    Returns an empty string if no benchmark data is available.
    """
    if benchmarks_path is None or not benchmarks_path.exists():
        return ""

    try:
        data = json.loads(benchmarks_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    results = data.get("results", [])
    if not results:
        return ""

    lines = ["## Benchmark Results\n"]
    lines.append("| Benchmark | Status | Duration |")
    lines.append("|-----------|--------|----------|")
    for r in results:
        name = r.get("name", "?")
        if r.get("skipped"):
            status = "Skipped"
        elif r.get("passed"):
            status = "PASS"
        else:
            status = "FAIL"
        dur = r.get("duration_sec", 0)
        dur_str = f"{dur:.1f}s" if dur else "—"
        lines.append(f"| {name} | {status} | {dur_str} |")

    summary = (
        f"\n*Ran {data.get('n_passed', 0)} / "
        f"{data.get('n_passed', 0) + data.get('n_failed', 0)} benchmarks passing.  "
        f"Total evaluation time: {data.get('total_sec', 0):.1f}s.*"
    )
    lines.append(summary)
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_data_summary_section(manifest: dict, training_cfg: dict) -> str:
    """Build a markdown Training Data Summary section."""
    prog = manifest.get("progress", {})
    source = manifest.get("source", {})
    dataset = source.get("dataset", "bigcode/starcoderdata")
    languages = source.get("languages", ["Python", "TypeScript", "JavaScript"])
    tokens_seen = prog.get("tokens_seen") or training_cfg.get("tokens_seen", "unknown")
    filter_mode = source.get("filter_mode", "conservative")

    if isinstance(tokens_seen, int):
        tokens_str = f"{tokens_seen / 1e9:.2f}B" if tokens_seen >= 1e9 else f"{tokens_seen / 1e6:.0f}M"
    else:
        tokens_str = str(tokens_seen)

    lang_str = ", ".join(languages) if languages else "Python, TypeScript, JavaScript"

    return f"""\
## Training Data Summary

| Field | Value |
|-------|-------|
| Dataset | [{dataset}](https://huggingface.co/datasets/{dataset}) |
| Languages | {lang_str} |
| Tokens seen | {tokens_str} |
| Quality filter | {filter_mode} (conservative ≈ 48% rejection, strict ≈ 65%) |
| Quality scoring | Continuous 0.0–1.0 weights; high-quality code trains harder |
| Data source | Open-source code repositories (GitHub via StarCoder pipeline) |

> **Note**: The dataset is gated — requires HuggingFace account with accepted
> [terms of service](https://huggingface.co/datasets/bigcode/starcoderdata) and
> `HF_TOKEN` environment variable.

"""


def _build_hardware_section(manifest: dict, training_cfg: dict) -> str:
    """Build a markdown Hardware Requirements section."""
    hw = manifest.get("hardware", {})
    gpu_name = hw.get("gpu", "NVIDIA RTX 4080")
    vram_gb = hw.get("vram_gb", 16)
    precision = training_cfg.get("precision", "bf16")
    batch_size = training_cfg.get("batch_size", "?")
    grad_accum = training_cfg.get("gradient_accumulation", "?")
    max_seq_len = training_cfg.get("max_seq_len", 2048)

    # Inference VRAM estimate: model weights + KV cache
    # Very rough: 2 bytes/param for bf16; KV cache scales with seq_len
    if isinstance(vram_gb, (int, float)):
        inference_vram = max(2, int(vram_gb / 4))
        inference_vram_str = f"~{inference_vram} GB (bf16 weights + KV cache)"
    else:
        inference_vram_str = "depends on model size"

    return f"""\
## Hardware Requirements

### Training

| Field | Value |
|-------|-------|
| Training GPU | {gpu_name} ({vram_gb} GB VRAM) |
| Precision | {precision} |
| Batch size | {batch_size} |
| Gradient accumulation | {grad_accum} |
| Sequence length | {max_seq_len} tokens |

> **Minimum for training**: GPU with at least 6 GB VRAM for the tiny (50M) config.
> Gradient checkpointing is required for medium (350M) on 16 GB VRAM.

### Inference

| Field | Value |
|-------|-------|
| Minimum VRAM | {inference_vram_str} |
| Supported precisions | bf16 (RTX 4080+), fp16 (RTX 3080) |
| CPU inference | Supported but slow (no KV-cache optimisation) |

"""


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
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Path to eval_results.json from run_eval_suite.py (adds benchmark section).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="typescript,python",
        help=(
            "Comma-separated languages for example generation "
            "(e.g. typescript,typescript-react,python). "
            "Supported: typescript, typescript-react, javascript, javascript-react, python."
        ),
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

            # Resolve target languages: --languages flag takes priority, then
            # manifest source languages, then a sensible default.
            _req_langs = [
                lang.strip() for lang in (args.languages or "").split(",") if lang.strip()
            ]
            _manifest_langs = manifest.get("source", {}).get(
                "languages", ["Python", "TypeScript", "JavaScript"]
            )
            _display_langs = (
                [LANGUAGE_PROMPTS.get(lang, [(lang.title(), "")])[0][0] for lang in _req_langs]
                if _req_langs
                else _manifest_langs
            )

            # Build a concise but informative architecture description that
            # includes the key design choices and training statistics.
            _model_cfg_for_desc = raw_cfg.get("model", raw_cfg)
            _d_model = _model_cfg_for_desc.get("d_model", "?")
            _n_layers = _model_cfg_for_desc.get("n_layers", "?")
            _n_heads = _model_cfg_for_desc.get("n_heads", "?")
            _n_kv = _model_cfg_for_desc.get("n_kv_heads", _n_heads)
            _seq_len = _model_cfg_for_desc.get("max_seq_len", "?")
            _prog_for_desc = manifest.get("progress", {})
            _tokens_seen = _prog_for_desc.get("tokens_seen") or metadata.get("tokens_seen")
            _tokens_str = (
                f"{_tokens_seen / 1e9:.2f}B" if isinstance(_tokens_seen, int) and _tokens_seen >= 1e9
                else f"{_tokens_seen / 1e6:.0f}M" if isinstance(_tokens_seen, int)
                else "unknown"
            )
            _loss_str = (
                f"{metadata['loss']:.4f}" if isinstance(metadata.get("loss"), float) else "unknown"
            )
            _arch_desc = (
                f"Decoder-only transformer trained from scratch on open-source code. "
                f"Architecture mirrors LLaMA 3 / Mistral: RoPE positional encoding, "
                f"grouped-query attention (GQA, {_n_heads}Q / {_n_kv}KV heads), "
                f"SwiGLU feed-forward, and pre-norm RMSNorm. "
                f"{_n_layers} layers, d_model={_d_model}, {_seq_len}-token context window. "
                f"Trained for {ckpt_step:,} steps ({_tokens_str} tokens seen); "
                f"final training loss {_loss_str}."
            )

            model_info = ModelInfo(
                name=f"Cola-Coder {size_name.capitalize()}",
                version=size_name,
                architecture=_arch_desc,
                parameters=n_params,
                languages=_display_langs,
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

            # Try to load the model for live example generation
            live_generator = None
            if args.checkpoint is not None:
                try:
                    from cola_coder.inference.generator import CodeGenerator
                    from cola_coder.model.config import ModelConfig
                    from cola_coder.model.transformer import Transformer
                    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

                    _tok_path = project_root / "tokenizer.json"
                    _cfg_dict = metadata.get("config", {})
                    _model_cfg_dict = _cfg_dict.get("model", _cfg_dict)
                    if _tok_path.exists() and _model_cfg_dict:
                        import torch

                        _device = "cuda" if torch.cuda.is_available() else "cpu"
                        _tok = CodeTokenizer(str(_tok_path))
                        _model_cfg = ModelConfig(**_model_cfg_dict)
                        _model = Transformer(_model_cfg)
                        _model.load_checkpoint(checkpoint_dir)
                        _model = _model.to(_device)
                        live_generator = CodeGenerator(_model, _tok, device=_device)
                        cli.dim("Model loaded — generating live examples.")
                except Exception as _gen_exc:
                    cli.dim(f"Live generation unavailable ({_gen_exc}) — using fallback text.")

            # Generate examples for each requested language (max 2 per language)
            _fallback_output = "# [Generation requires model loaded with --checkpoint]"
            _langs = [
                lang.strip()
                for lang in (args.languages or "typescript,python").split(",")
                if lang.strip()
            ]
            for _lang in _langs:
                _prompts = LANGUAGE_PROMPTS.get(_lang, [])
                for _lang_label, _prompt in _prompts[:2]:
                    if live_generator is not None:
                        try:
                            _output = live_generator.generate(
                                _prompt, max_new_tokens=128, temperature=0.3
                            )
                        except Exception:
                            _output = _fallback_output
                    else:
                        _output = _fallback_output
                    generator_obj.add_example(
                        prompt=_prompt,
                        output=_output,
                        language=_lang_label,
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

    # ── Resolve benchmarks file ───────────────────────────────────────────
    benchmarks_path: Path | None = None
    if args.benchmarks:
        benchmarks_path = Path(args.benchmarks)
    else:
        # Auto-detect eval_results.json in project root
        auto_bench = project_root / "eval_results.json"
        if auto_bench.exists():
            benchmarks_path = auto_bench
            cli.dim(f"Auto-detected benchmark results: {auto_bench}")

    # ── Fallback: build card from template ────────────────────────────────
    if card_content is None:
        card_content = _build_fallback_card(
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
            manifest=manifest,
            project_root=project_root,
            benchmarks_path=benchmarks_path,
            languages_arg=args.languages,
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
