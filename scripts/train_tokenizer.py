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
    max_samples: int,
) -> Iterator[str]:
    """Stream text strings from a HuggingFace dataset.

    Downloads rows one at a time via the HF streaming API and yields the
    text field from each row.  Stops after *max_samples* rows or when the
    dataset is exhausted.

    Args:
        dataset_name: HuggingFace dataset identifier, e.g. "HuggingFaceFW/fineweb-edu".
        text_field: Name of the column that holds the text, e.g. "text".
        max_samples: Maximum number of rows to yield.

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
        if count >= max_samples:
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

            args.output = str(DatasetResolver.get_tokenizer_path(args.data_sources))
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

        try:
            from cola_coder.data.download import download_sample_data
        except ImportError:
            cli.fatal(
                "Could not import cola_coder",
                hint="Make sure the package is installed: pip install -e .",
            )

        # Load data_sources.yaml
        try:
            with open(args.data_sources) as f:
                ds_config: dict = yaml.safe_load(f) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
            cli.fatal(f"Could not load data sources config: {exc}")

        sources_cfg: dict = ds_config.get("sources", {})

        # Build per-source sample counts (proportional to weight)
        total_weight = sum(
            float(cfg.get("weight", 0))
            for cfg in sources_cfg.values()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        )

        sources_used: list[str] = []
        iterators: list[Iterator[str]] = []
        total_samples_used = 0

        for source_name, cfg in sources_cfg.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", False):
                continue

            weight = float(cfg.get("weight", 0))
            proportional = int(args.num_samples * (weight / total_weight)) if total_weight > 0 else 0
            if proportional <= 0:
                continue

            total_samples_used += proportional
            sources_used.append(source_name)

            if source_name == "code":
                cli.step(
                    len(sources_used),
                    None,
                    f"Collecting {proportional} code samples (weight={weight})",
                )
                source_languages: list[str] = cfg.get("languages", languages)
                if isinstance(source_languages, list):
                    source_languages = [str(lang) for lang in source_languages]
                else:
                    source_languages = languages

                file_paths = download_sample_data(
                    output_dir=str(Path(storage.data_dir) / "raw"),
                    languages=source_languages,
                    num_samples=proportional,
                )

                def _file_iter(paths: list[str]) -> Iterator[str]:
                    for fp in paths:
                        try:
                            with open(fp, encoding="utf-8", errors="replace") as fh:
                                yield fh.read()
                        except OSError:
                            continue

                iterators.append(_file_iter(file_paths))

            else:
                dataset_name: str = str(cfg.get("dataset", ""))
                if not dataset_name:
                    cli.fatal(f"Source '{source_name}' is missing 'dataset' key in {args.data_sources}")

                text_field = "text"
                cli.step(
                    len(sources_used),
                    None,
                    f"Streaming {proportional} samples from {dataset_name} (weight={weight})",
                )
                iterators.append(
                    _stream_hf_text_sample(dataset_name, text_field, proportional)
                )

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

        cli.step(len(sources_used) + 1, None, f"Training BPE tokenizer with vocab size {args.vocab_size}")

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
            num_samples=total_samples_used,
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
