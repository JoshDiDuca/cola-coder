"""Multi-source data collection and mixing.

Reads configs/data_sources.yaml and downloads code, text, and math data
from HuggingFace. Tokenizes each source into separate .npy files, then
combines them with weighted mixing per Qwen2.5-Coder ratios (70/20/10).

Usage:
    .venv/Scripts/python scripts/collect_data.py --config configs/small.yaml
    .venv/Scripts/python scripts/collect_data.py --config configs/4080_max.yaml --sources code,text
    .venv/Scripts/python scripts/collect_data.py --config configs/tiny.yaml --max-samples 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cola_coder.cli import cli
from cola_coder.data.combine import DatasetCombiner, DatasetInput
from cola_coder.data.download import stream_code_data
from cola_coder.data.preprocess import tokenize_and_chunk
from cola_coder.model.config import Config
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer


# ── Generic HF text streaming ────────────────────────────────────────────

def stream_hf_text(
    dataset_name: str,
    *,
    split: str = "train",
    text_field: str = "text",
    min_length: int = 50,
    max_length: int = 50_000,
    max_samples: int | None = None,
):
    """Stream text from a HuggingFace dataset (non-code, no language dirs).

    Works for FineWeb-Edu, OpenWebMath, C4, OpenWebText2, etc.
    """
    from datasets import load_dataset

    cli.dim(f"  Streaming from {dataset_name} (field: {text_field})...")
    ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    count = 0
    for sample in ds:
        content = sample.get(text_field, "")
        if not content:
            continue
        if len(content) < min_length or len(content) > max_length:
            continue
        yield content
        count += 1
        if count % 10_000 == 0:
            cli.dim(f"    {count:,} samples streamed...")
        if max_samples is not None and count >= max_samples:
            cli.dim(f"  Reached sample limit: {max_samples:,}")
            return

    cli.dim(f"  Total: {count:,} text samples yielded from {dataset_name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source data collection (code + text + math)",
    )
    parser.add_argument("--config", required=True, help="Model config YAML path")
    parser.add_argument(
        "--data-sources", default="configs/data_sources.yaml",
        help="Data sources config (default: configs/data_sources.yaml)",
    )
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated sources to collect: code,text,math (default: all enabled)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per source (for testing)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Output directory for .npy files",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument("--no-combine", action="store_true", help="Skip combining step")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────
    config = Config.from_yaml(args.config)

    ds_path = Path(args.data_sources)
    if not ds_path.exists():
        cli.error("Data sources config not found", str(ds_path))
        sys.exit(1)

    with open(ds_path, encoding="utf-8") as f:
        ds_config = yaml.safe_load(f)

    sources_config = ds_config.get("sources", {})
    requested = set(args.sources.split(",")) if args.sources else None

    # ── Load tokenizer ────────────────────────────────────────────────
    tok_path = args.tokenizer or "tokenizer.json"
    if not Path(tok_path).exists():
        cli.error("Tokenizer not found", tok_path)
        cli.dim("  Run: .venv/Scripts/python scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = CodeTokenizer(tok_path)
    seq_len = config.model.max_seq_len
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cli.header("Multi-Source Data Collection", f"Config: {args.config}")

    collected: list[DatasetInput] = []

    # ── Code ──────────────────────────────────────────────────────────
    code_cfg = sources_config.get("code", {})
    if code_cfg.get("enabled", True) and (requested is None or "code" in requested):
        dataset = code_cfg.get("dataset", "bigcode/starcoderdata")
        languages = code_cfg.get("languages", ["python", "typescript", "javascript"])
        weight = code_cfg.get("weight", 0.7)

        cli.step(1, 3, f"Collecting code from {dataset}")
        cli.info("Languages", ", ".join(languages))

        code_iter = stream_code_data(
            dataset, languages=languages, max_samples=args.max_samples,
        )
        output_path = tokenize_and_chunk(
            code_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="code_data",
        )
        collected.append(DatasetInput(path=output_path, weight=weight, name="code"))
        cli.success(f"Code data saved: {output_path}")

    # ── Text ──────────────────────────────────────────────────────────
    text_cfg = sources_config.get("text", {})
    if text_cfg.get("enabled", True) and (requested is None or "text" in requested):
        dataset = text_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
        weight = text_cfg.get("weight", 0.2)
        min_len = text_cfg.get("min_length", 100)
        max_len = text_cfg.get("max_length", 50_000)

        cli.step(2, 3, f"Collecting text from {dataset}")

        text_iter = stream_hf_text(
            dataset, min_length=min_len, max_length=max_len,
            max_samples=args.max_samples,
        )
        output_path = tokenize_and_chunk(
            text_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="text_data",
        )
        collected.append(DatasetInput(path=output_path, weight=weight, name="text"))
        cli.success(f"Text data saved: {output_path}")

    # ── Math ──────────────────────────────────────────────────────────
    math_cfg = sources_config.get("math", {})
    if math_cfg.get("enabled", True) and (requested is None or "math" in requested):
        dataset = math_cfg.get("dataset", "open-web-math/open-web-math")
        weight = math_cfg.get("weight", 0.1)
        min_len = math_cfg.get("min_length", 50)
        max_len = math_cfg.get("max_length", 30_000)

        cli.step(3, 3, f"Collecting math from {dataset}")

        math_iter = stream_hf_text(
            dataset, min_length=min_len, max_length=max_len,
            max_samples=args.max_samples,
        )
        output_path = tokenize_and_chunk(
            math_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="math_data",
        )
        collected.append(DatasetInput(path=output_path, weight=weight, name="math"))
        cli.success(f"Math data saved: {output_path}")

    # ── Combine ───────────────────────────────────────────────────────
    if len(collected) > 1 and not args.no_combine:
        cli.header("Combining Datasets", "Weighted interleaving per data_sources.yaml")

        for ds_input in collected:
            cli.info(ds_input.name, f"weight={ds_input.weight:.0%}, path={ds_input.path}")

        combiner = DatasetCombiner()
        combined_path = str(Path(output_dir) / "mixed_train_data.npy")
        result = combiner.combine(
            collected, strategy="interleave", output_path=combined_path,
        )
        cli.success(f"Combined dataset: {result.output_path}")
        cli.info("Total chunks", f"{result.total_chunks:,}")
    elif len(collected) == 1:
        cli.info("Single source", "No combining needed")
    else:
        cli.warn("No data collected. Check data_sources.yaml settings.")

    cli.done("Data collection complete.")


if __name__ == "__main__":
    main()
