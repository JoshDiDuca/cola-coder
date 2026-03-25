"""Train a BPE tokenizer on code data.

Downloads a sample of code data from HuggingFace and trains a Byte Pair Encoding
tokenizer suitable for code generation models.

Usage:
    python scripts/train_tokenizer.py --vocab-size 32768 --num-samples 10000
    python scripts/train_tokenizer.py --languages python,typescript --output my_tokenizer.json
    python scripts/train_tokenizer.py --config configs/model.yaml
    python scripts/train_tokenizer.py --config configs/model.yaml --data-sources configs/data_sources.yaml
"""

import argparse
from pathlib import Path
from typing import Iterator

import yaml

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def _stream_hf_text_sample(
    dataset_name: str,
    text_field: str,
    max_samples: int | None,
) -> Iterator[str]:
    """Stream text strings from a HuggingFace dataset.

    Downloads rows one at a time via the HF streaming API and yields the
    text field from each row.  Stops after *max_samples* rows or when the
    dataset is exhausted.

    Args:
        dataset_name: HuggingFace dataset identifier, e.g. "HuggingFaceFW/fineweb-edu".
        text_field: Name of the column that holds the text, e.g. "text".
        max_samples: Maximum number of rows to yield.  None = stream entire dataset.

    Yields:
        One text string per dataset row.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        cli.fatal(
            "Could not import `datasets`",
            hint="pip install datasets",
        )

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    count = 0
    for row in dataset:
        if max_samples is not None and count >= max_samples:
            break
        text = row.get(text_field, "")
        if isinstance(text, str) and text.strip():
            yield text
            count += 1


def main() -> None:
    storage = get_storage_config()
    storage.apply_hf_cache()

    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on code data from HuggingFace."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Target vocabulary size (default: 32768).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the trained tokenizer (default: tokenizer.json).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of code files to download for training (default: 10000).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,typescript,javascript",
        help="Comma-separated list of languages to download (default: python,typescript,javascript).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Model config YAML. When given, resolves output path via DatasetResolver.",
    )
    parser.add_argument(
        "--tok-samples",
        type=int,
        default=100_000,
        dest="tok_samples",
        help="Max samples per source during tokenizer training (default: 100000). "
             "Code reads from local cache; text/math streams from HuggingFace. "
             "Set 0 for unlimited code (may OOM on large caches).",
    )
    parser.add_argument(
        "--data-sources",
        type=str,
        default="configs/data_sources.yaml",
        dest="data_sources",
        help="Data sources config (default: configs/data_sources.yaml).",
    )
    args = parser.parse_args()

    # Resolve output path: DatasetResolver (when --config given) or legacy storage path
    if args.output is None:
        if args.config is not None:
            from cola_coder.data.dataset_resolver import DatasetResolver

            args.output = str(DatasetResolver.get_tokenizer_path(args.data_sources, config_path=args.config))
        else:
            args.output = storage.tokenizer_path

    languages = [lang.strip() for lang in args.languages.split(",")]

    cli.header("Cola-Coder", "Tokenizer Training")

    # ------------------------------------------------------------------
    # Multi-source training path (--config given)
    # ------------------------------------------------------------------
    if args.config is not None:
        from cola_coder.data.dataset_resolver import DatasetResolver
        from cola_coder.tokenizer.train_tokenizer import train_from_iterator

        # Load data_sources.yaml
        try:
            with open(args.data_sources) as f:
                ds_config: dict = yaml.safe_load(f) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
            cli.fatal(f"Could not load data sources config: {exc}")

        # Load model config languages — override data_sources.yaml code languages
        config_languages: list[str] | None = None
        try:
            with open(args.config) as f:
                model_cfg: dict = yaml.safe_load(f) or {}
            cl = model_cfg.get("data", {}).get("languages")
            if isinstance(cl, list) and cl:
                config_languages = [str(lang) for lang in cl]
        except Exception:
            pass

        sources_cfg: dict = ds_config.get("sources", {})

        # Count total enabled sources upfront
        total_sources = sum(
            1 for cfg in sources_cfg.values()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        )

        sources_used: list[str] = []
        iterators: list[Iterator[str]] = []
        step_num = 1
        tok_cap: int | None = args.tok_samples if args.tok_samples > 0 else None

        for source_name, cfg in sources_cfg.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                continue

            sources_used.append(source_name)

            if source_name == "code":
                # Model config languages take precedence over data_sources.yaml
                if config_languages is not None:
                    source_languages: list[str] = config_languages
                elif isinstance(cfg.get("languages"), list):
                    source_languages = [str(lang) for lang in cfg["languages"]]
                else:
                    source_languages = languages

                cli.step(
                    step_num,
                    total_sources,
                    f"Streaming {str(tok_cap) if tok_cap else 'ALL'} cached {'+'.join(source_languages)} files",
                )
                from cola_coder.data.download import stream_code_data
                iterators.append(
                    stream_code_data(
                        cfg.get("dataset", "bigcode/starcoderdata"),
                        languages=source_languages,
                        max_samples=tok_cap,
                    )
                )

            else:
                hf_dataset: str = str(cfg.get("dataset", ""))
                if not hf_dataset:
                    cli.fatal(f"Source '{source_name}' is missing 'dataset' key in {args.data_sources}")

                cap_str = str(tok_cap) if tok_cap is not None else "unlimited"
                cli.step(
                    step_num,
                    total_sources,
                    f"Streaming {cap_str} samples from {hf_dataset}",
                )
                iterators.append(
                    _stream_hf_text_sample(hf_dataset, "text", tok_cap)
                )

            step_num += 1

        if not iterators:
            cli.fatal(
                "No enabled data sources found",
                hint=f"Check {args.data_sources} — at least one source must be enabled.",
            )

        cli.info("Sources", ", ".join(sources_used))
        cli.info("Output", args.output)

        # Chain all iterators into one
        def _combined_iterator(iters: list[Iterator[str]]) -> Iterator[str]:
            for it in iters:
                yield from it

        combined: Iterator[str] = _combined_iterator(iterators)

        cli.step(step_num, total_sources + 1, f"Training BPE tokenizer with vocab size {args.vocab_size}")

        try:
            tokenizer = train_from_iterator(
                iterator=combined,
                vocab_size=args.vocab_size,
                output_path=args.output,
            )
        except Exception as exc:
            cli.fatal(f"Error training tokenizer: {exc}")

        # Write tokenizer_meta.json
        DatasetResolver.save_tokenizer_meta(
            tokenizer_path=Path(args.output),
            vocab_size=tokenizer.get_vocab_size(),
            sources=sources_used,
            num_samples=0,  # code source is unbounded; not tracked
        )

    # ------------------------------------------------------------------
    # Legacy path (no --config): code-only training via files
    # ------------------------------------------------------------------
    else:
        cli.step(1, 2, f"Downloading {args.num_samples} code samples")
        cli.info("Languages", ", ".join(languages))

        try:
            from cola_coder.data.download import download_sample_data
        except ImportError:
            cli.fatal(
                "Could not import cola_coder",
                hint="Make sure the package is installed: pip install -e .",
            )

        try:
            file_paths = download_sample_data(
                output_dir=str(Path(storage.data_dir) / "raw"),
                languages=languages,
                num_samples=args.num_samples,
            )
        except Exception as exc:
            cli.fatal(f"Error downloading data: {exc}")

        if not file_paths:
            cli.fatal(
                "No files were downloaded",
                hint="Check your network connection and dataset access.",
            )

        cli.success(f"Downloaded {len(file_paths)} files")

        cli.step(2, 2, f"Training BPE tokenizer with vocab size {args.vocab_size}")

        try:
            from cola_coder.tokenizer.train_tokenizer import train_from_files
        except ImportError:
            cli.fatal("Could not import tokenizer training module")

        try:
            tokenizer = train_from_files(
                file_paths=file_paths,
                vocab_size=args.vocab_size,
                output_path=args.output,
            )
        except Exception as exc:
            cli.fatal(f"Error training tokenizer: {exc}")

    # ---- Summary ----
    test_code = "def hello_world():\n    print('Hello, world!')"
    encoded = tokenizer.encode(test_code)
    decoded = tokenizer.decode(encoded.ids)

    cli.dim(f"Quick test: {test_code!r}")
    cli.dim(f"  Tokens: {len(encoded.ids)}  Decoded: {decoded!r}")

    cli.done("Tokenizer training complete!", extras={
        "Vocabulary size": str(tokenizer.get_vocab_size()),
        "Saved to": str(Path(args.output).resolve()),
    })


if __name__ == "__main__":
    main()
