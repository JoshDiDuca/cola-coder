"""Generate SFT instruction-tuning data from source code files.

Reads .py/.ts/.js source code, extracts functions and classes, and
generates instruction-response pairs in ChatML format suitable for
supervised fine-tuning with ``scripts/train_sft.py``.

Usage:
    python scripts/generate_sft_data.py --source ./my-code --output data/sft_train.jsonl
    python scripts/generate_sft_data.py --source ./src --output sft.jsonl --num-samples 500
    python scripts/generate_sft_data.py --source app.py --output sft.jsonl --quality-threshold 0.7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cola_coder.cli import cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT instruction-tuning data from source code."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source code directory or a single file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSONL file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="Minimum quality score 0-1 (default: 0.6).",
    )

    args = parser.parse_args()

    cli.header("Cola-Coder", "SFT Data Generator")

    # ---- Validate inputs ----
    source_path = Path(args.source)
    if not source_path.exists():
        cli.fatal(
            f"Source path not found: {args.source}",
            hint="Check the path and try again.",
        )

    # Determine if source is a directory or file
    source_dir = str(source_path) if source_path.is_dir() else None
    source_file = str(source_path) if source_path.is_file() else None

    cli.info("Source", args.source)
    cli.info("Output", args.output)
    cli.info("Target samples", args.num_samples)
    cli.info("Quality threshold", args.quality_threshold)

    # ---- Generate ----
    cli.step(1, 2, "Generating instruction pairs")

    from cola_coder.data.sources.instruction_gen import (
        CodeToInstructionGenerator,
    )

    generator = CodeToInstructionGenerator(
        source_dir=source_dir,
        source_file=source_file,
    )

    start = time.time()
    examples = generator.generate(
        num_samples=args.num_samples,
        quality_threshold=args.quality_threshold,
    )
    elapsed = time.time() - start

    cli.success(f"Generated {len(examples)} examples in {elapsed:.1f}s")

    if not examples:
        cli.warn(
            "No examples generated. Check that your source path "
            "contains .py/.ts/.js files with functions or classes."
        )
        return

    # ---- Stats ----
    # Count pair types by inspecting the user message
    write_count = sum(
        1 for ex in examples
        if not ex["messages"][1]["content"].startswith(("Explain", "Read",
                                                        "Describe", "What does",
                                                        "The following", "Fix",
                                                        "There is", "Debug"))
    )
    explain_count = sum(
        1 for ex in examples
        if ex["messages"][1]["content"].startswith(("Explain", "Read",
                                                    "Describe", "What does"))
    )
    fix_count = sum(
        1 for ex in examples
        if ex["messages"][1]["content"].startswith(("The following", "Fix",
                                                    "There is", "Debug"))
    )

    cli.kv_table({
        "Total examples": str(len(examples)),
        "Write/Implement pairs": str(write_count),
        "Explain pairs": str(explain_count),
        "Fix-the-bug pairs": str(fix_count),
        "Time": f"{elapsed:.1f}s",
    }, title="Generation Summary")

    # Show a sample
    sample = examples[0]
    cli.print()
    cli.print("[bold]Sample example:[/bold]")
    user_msg = sample["messages"][1]["content"]
    asst_msg = sample["messages"][2]["content"]
    cli.print(f"  [cyan]User:[/cyan] {user_msg[:120]}...")
    cli.print(f"  [cyan]Assistant:[/cyan] {asst_msg[:120]}...")

    # ---- Save ----
    cli.step(2, 2, "Saving to JSONL")
    CodeToInstructionGenerator.save_jsonl(examples, args.output)

    output_size = Path(args.output).stat().st_size
    cli.done("SFT data generated", extras={
        "File": args.output,
        "Examples": str(len(examples)),
        "Size": f"{output_size / 1024:.1f} KB",
    })


if __name__ == "__main__":
    main()
