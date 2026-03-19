"""Train a BPE tokenizer on code data.

Downloads a sample of code data from HuggingFace and trains a Byte Pair Encoding
tokenizer suitable for code generation models.

Usage:
    python scripts/train_tokenizer.py --vocab-size 32768 --num-samples 10000
    python scripts/train_tokenizer.py --languages python,typescript --output my_tokenizer.json
"""

import argparse
import sys
from pathlib import Path


def main():
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
        default="tokenizer.json",
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

    # ---- Step 1: Download sample data ----
    print(f"Step 1: Downloading {args.num_samples} code samples...")
    print(f"  Languages: {', '.join(languages)}")

    try:
        from cola_coder.data.download import download_sample_data
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        file_paths = download_sample_data(
            output_dir="./data/raw",
            languages=languages,
            num_samples=args.num_samples,
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

    if not file_paths:
        print("Error: No files were downloaded. Check your network connection and dataset access.")
        sys.exit(1)

    print(f"  Downloaded {len(file_paths)} files.\n")

    # ---- Step 2: Train the tokenizer ----
    print(f"Step 2: Training BPE tokenizer with vocab size {args.vocab_size}...")

    try:
        from cola_coder.tokenizer.train_tokenizer import train_from_files
    except ImportError:
        print("Error: Could not import tokenizer training module.")
        sys.exit(1)

    try:
        tokenizer = train_from_files(
            file_paths=file_paths,
            vocab_size=args.vocab_size,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        sys.exit(1)

    # ---- Summary ----
    print(f"\nTokenizer training complete!")
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"  Saved to: {Path(args.output).resolve()}")

    # Quick test
    test_code = "def hello_world():\n    print('Hello, world!')"
    encoded = tokenizer.encode(test_code)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n  Quick test:")
    print(f"    Input:    {test_code!r}")
    print(f"    Tokens:   {len(encoded.ids)} tokens")
    print(f"    Decoded:  {decoded!r}")


if __name__ == "__main__":
    main()
