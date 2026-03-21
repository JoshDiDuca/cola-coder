"""Interactive CLI for GGUF export and quantization.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml

    # Or non-interactively:
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action gguf-f16
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action gguf-q8
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action gguf-q4
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action ollama
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action quantize
    python scripts/export_model.py --checkpoint checkpoints/tiny/latest --config configs/tiny.yaml --action benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve project root so imports work when called directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402
from cola_coder.model.config import Config  # noqa: E402
from cola_coder.model.transformer import Transformer  # noqa: E402
from cola_coder.export.gguf_export import GGUFExporter, GGUF_PACKAGE_AVAILABLE  # noqa: E402
from cola_coder.export.ollama_export import OllamaExporter  # noqa: E402
from cola_coder.export.quantize import ModelQuantizer  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _default_output_dir(checkpoint_path: str) -> Path:
    """Derive an exports/ directory next to the checkpoint."""
    ckpt = Path(checkpoint_path)
    # checkpoints/tiny/latest → checkpoints/tiny/exports/
    if ckpt.name == "latest":
        return ckpt.parent / "exports"
    return ckpt.parent / "exports"


def _load_config(config_path: str) -> Config:
    return Config.from_yaml(config_path)


def _load_model(config: Config, checkpoint_path: str) -> Transformer:
    """Load the Transformer from a checkpoint (weights only, no optimizer)."""
    from cola_coder.training.checkpoint import load_model_only

    model = Transformer(config.model)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    load_model_only(checkpoint_path, model, device=device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────

def action_gguf(
    checkpoint: str,
    config: Config,
    quantization: str,
    output_dir: Path,
) -> None:
    cli.info("GGUF export", f"quantization={quantization}")
    if GGUF_PACKAGE_AVAILABLE:
        cli.info("Backend", "gguf Python package")
    else:
        cli.warn("gguf package not installed — using built-in GGUF writer")
        cli.info("Tip", "pip install gguf  for full K-quant support")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"cola-coder-{quantization}.gguf")

    exporter = GGUFExporter(config.model)
    result = exporter.export(checkpoint, output_path, quantization=quantization)

    if result.success:
        cli.success(
            f"Exported to {result.output_path}  "
            f"({result.file_size_mb:.1f} MB, {result.num_tensors} tensors)"
        )
    else:
        cli.error("Export failed", hint=result.error)
        sys.exit(1)


def action_ollama(
    checkpoint: str,
    config: Config,
    output_dir: Path,
) -> None:
    """Export F16 GGUF then generate Modelfile."""
    # First ensure a GGUF exists
    gguf_path = output_dir / "cola-coder-f16.gguf"
    if not gguf_path.exists():
        cli.info("No F16 GGUF found", "exporting now...")
        action_gguf(checkpoint, config, "f16", output_dir)

    cli.info("Creating Ollama Modelfile", "")
    exporter = OllamaExporter()
    modelfile_path = exporter.create_modelfile(
        str(gguf_path),
        str(output_dir),
        model_name="cola-coder",
    )
    cli.success(f"Modelfile written to: {modelfile_path}")
    cli.info("Next step", f"ollama create cola-coder -f {modelfile_path}")


def action_quantize(
    checkpoint: str,
    config: Config,
    output_dir: Path,
) -> None:
    cli.info("Loading model for quantization", "")
    model = _load_model(config, checkpoint)

    quantizer = ModelQuantizer(model)
    cli.info("Running dynamic INT8 quantization", "")
    q_model, result = quantizer.quantize_dynamic()

    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    q_path = str(output_dir / "cola-coder-int8.pt")
    torch.save(q_model.state_dict(), q_path)

    cli.success(
        f"Quantized model saved to {q_path}  "
        f"({result.original_size_mb:.1f} MB → {result.quantized_size_mb:.1f} MB, "
        f"{result.compression_ratio:.1f}× compression)"
    )


def action_benchmark(
    checkpoint: str,
    config: Config,
) -> None:
    cli.info("Loading original model", "")
    model = _load_model(config, checkpoint)

    quantizer = ModelQuantizer(model)
    cli.info("Quantizing for benchmark", "")
    q_model, quant_result = quantizer.quantize_dynamic()

    cli.info("Running benchmark", "comparing original vs INT8 on CPU")
    prompts = [
        "function hello() {",
        "const x: number =",
        "interface User {",
    ]
    stats = quantizer.benchmark(model, q_model, prompts)

    cli.info("Original size", f"{stats['original_size_mb']:.1f} MB")
    cli.info("Quantized size", f"{stats['quantized_size_mb']:.1f} MB")
    cli.info("Compression", f"{stats['compression_ratio']:.2f}×")
    cli.info("Original latency", f"{stats['original_ms']:.1f} ms")
    cli.info("Quantized latency", f"{stats['quantized_ms']:.1f} ms")
    cli.info("Speedup", f"{stats['speedup']:.2f}×")
    cli.info("Logit cosine similarity", f"{stats['logit_cosine_sim']:.4f}  (1.0 = identical)")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ──────────────────────────────────────────────────────────────────────────────

_MENU_CHOICES = [
    ("1", "Export to GGUF (F16)"),
    ("2", "Export to GGUF (Q8_0)"),
    ("3", "Export to GGUF (Q4_K_M)"),
    ("4", "Create Ollama Modelfile"),
    ("5", "Quantize for fast CPU inference (INT8)"),
    ("6", "Benchmark original vs quantized"),
    ("q", "Quit"),
]

_ACTION_MAP = {
    "gguf-f16": "1",
    "gguf-q8": "2",
    "gguf-q4": "3",
    "ollama": "4",
    "quantize": "5",
    "benchmark": "6",
}


def _show_menu() -> str:
    print("\nCola-Coder Export & Quantization")
    print("=" * 40)
    for key, label in _MENU_CHOICES:
        print(f"  [{key}] {label}")
    print()
    choice = input("Select option: ").strip().lower()
    return choice


def main():
    parser = argparse.ArgumentParser(
        description="Export and quantize cola-coder models."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory or 'latest' file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/tiny.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for exported files. Defaults to <checkpoint_parent>/exports/.",
    )
    parser.add_argument(
        "--action",
        choices=list(_ACTION_MAP.keys()),
        default=None,
        help=(
            "Non-interactive action: gguf-f16, gguf-q8, gguf-q4, ollama, "
            "quantize, benchmark. If omitted, shows interactive menu."
        ),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = (
        Path(args.output_dir) if args.output_dir else _default_output_dir(args.checkpoint)
    )

    # Map --action to menu choice
    if args.action:
        choice = _ACTION_MAP[args.action]
    else:
        choice = _show_menu()

    if choice == "1":
        action_gguf(args.checkpoint, config, "f16", output_dir)
    elif choice == "2":
        action_gguf(args.checkpoint, config, "q8_0", output_dir)
    elif choice == "3":
        action_gguf(args.checkpoint, config, "q4_k_m", output_dir)
    elif choice == "4":
        action_ollama(args.checkpoint, config, output_dir)
    elif choice == "5":
        action_quantize(args.checkpoint, config, output_dir)
    elif choice == "6":
        action_benchmark(args.checkpoint, config)
    elif choice == "q":
        cli.info("Bye", "")
    else:
        cli.error(f"Unknown choice: {choice}")
        sys.exit(1)


if __name__ == "__main__":
    main()
