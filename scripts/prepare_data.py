"""Download and preprocess training data.

Streams code data from HuggingFace, tokenizes it with a trained BPE tokenizer,
and saves chunked token arrays ready for training.

Usage:
    python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json
    python scripts/prepare_data.py --tokenizer tokenizer.json --max-tokens 1000000
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess code data for training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (used to read data/language settings).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to trained tokenizer.json file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to process (default: no limit).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data (default: ./data/processed).",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable quality filtering (use raw data as-is).",
    )
    filter_group.add_argument(
        "--filter-strict",
        action="store_true",
        help="Use strict quality filtering (rejects mediocre code, not just bad code). "
             "Keeps only clearly high-quality files. Typical rejection rate: 30-50%%.",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found: {tokenizer_path}")
        print("  Train a tokenizer first with: python scripts/train_tokenizer.py")
        sys.exit(1)

    # ---- Load config if provided ----
    dataset_name = "bigcode/starcoderdata"
    languages = ["python", "typescript", "javascript"]
    max_seq_len = 2048

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        print(f"Loading config from {config_path}...")
        try:
            from cola_coder.model.config import Config
            config = Config.from_yaml(str(config_path))
            dataset_name = config.data.dataset
            languages = config.data.languages
            max_seq_len = config.model.max_seq_len
            print(f"  Dataset: {dataset_name}")
            print(f"  Languages: {', '.join(languages)}")
            print(f"  Chunk size: {max_seq_len}")
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    # ---- Step 1: Load tokenizer ----
    print(f"\nStep 1: Loading tokenizer from {tokenizer_path}...")

    try:
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        tokenizer = CodeTokenizer(str(tokenizer_path))
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # ---- Step 2: Stream and preprocess data ----
    print(f"\nStep 2: Streaming data from {dataset_name}...")
    if args.max_tokens:
        print(f"  Token limit: {args.max_tokens:,}")

    try:
        from cola_coder.data.download import stream_code_data
        from cola_coder.data.preprocess import tokenize_and_chunk
        from cola_coder.data.quality_filter import filtered_stream, FilterStats, FilterMode
    except ImportError:
        print("Error: Could not import data modules.")
        sys.exit(1)

    try:
        data_stream = stream_code_data(
            dataset_name=dataset_name,
            languages=languages,
        )

        # Apply quality filtering
        if args.no_filter:
            print("Quality filtering: DISABLED")
        elif args.filter_strict:
            print("Quality filtering: STRICT (only keeping high-quality code)")
            print("  Expected rejection rate: 30-50%")
            stats = FilterStats()
            data_stream = filtered_stream(data_stream, mode=FilterMode.STRICT, stats=stats)
        else:
            print("Quality filtering: CONSERVATIVE (use --filter-strict for stricter filtering)")
            stats = FilterStats()
            data_stream = filtered_stream(data_stream, mode=FilterMode.CONSERVATIVE, stats=stats)

        output_file = tokenize_and_chunk(
            text_iterator=data_stream,
            tokenizer=tokenizer,
            chunk_size=max_seq_len,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial data may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    # ---- Summary ----
    print(f"\nData preprocessing complete!")
    print(f"  Output: {Path(output_file).resolve()}")
    print(f"  Ready for training with: python scripts/train.py --data {output_file}")


if __name__ == "__main__":
    main()
