"""Train a BPE tokenizer on code data.

Downloads a sample of code data from HuggingFace and trains a Byte Pair Encoding
tokenizer suitable for code generation models.

Usage:
    python scripts/train_tokenizer.py --vocab-size 32768 --num-samples 10000
    python scripts/train_tokenizer.py --languages python,typescript --output my_tokenizer.json
"""

import argparse
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main():
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
        default=storage.tokenizer_path,
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
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]

    cli.header("Cola-Coder", "Tokenizer Training")

    # ---- Step 1: Download sample data ----
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
    except Exception as e:
        cli.fatal(f"Error downloading data: {e}")

    if not file_paths:
        cli.fatal(
            "No files were downloaded",
            hint="Check your network connection and dataset access.",
        )

    cli.success(f"Downloaded {len(file_paths)} files")

    # ---- Step 2: Train the tokenizer ----
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
    except Exception as e:
        cli.fatal(f"Error training tokenizer: {e}")

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
