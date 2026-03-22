"""Convert scraped markdown documentation into tokenized .npy format for training.

Scans data/docs/ for .md files organized as {framework}/{version}/*.md,
wraps each document in context tokens, tokenizes, chunks, and saves as a
uint16 numpy array compatible with CodeDataset and create_dataloader().

Usage:
    python scripts/prepare_docs_data.py --tokenizer tokenizer.json --seq-len 2048
    python scripts/prepare_docs_data.py --docs-dir data/docs/ --output data/processed/docs_data.npy
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Make cola_coder importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cola_coder.cli import cli


def _format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _parse_doc_header(content: str) -> tuple[str, str]:
    """Extract framework@version label from the first header comment line.

    Expected format (first line of file):
        // Framework: react@18.2.0

    Returns:
        (framework_version, remaining_content) — e.g. ("react@18.2.0", rest_of_file)
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return ("unknown@0.0.0", content)

    first = lines[0].strip()
    if first.startswith("// Framework:"):
        label = first[len("// Framework:") :].strip()
        return (label, "".join(lines[1:]))

    return ("unknown@0.0.0", content)


def _build_doc_text(framework_version: str, page_title: str, body: str) -> str:
    """Wrap a documentation page in context tokens.

    Format:
        <|doc|>react@18.2.0 - useState<|/doc|>
        {body}
        <|eos|>
    """
    header = f"<|doc|>{framework_version} - {page_title}<|/doc|>"
    return f"{header}\n{body.strip()}\n<|eos|>"


def _scan_docs(docs_dir: Path) -> list[dict]:
    """Scan docs_dir for .md files and return file metadata records."""
    records = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        try:
            rel = md_file.relative_to(docs_dir)
        except ValueError:
            rel = md_file

        # Derive page title from the filename stem (sans extension)
        page_title = md_file.stem.replace("-", " ").replace("_", " ")

        records.append({
            "path": md_file,
            "rel": str(rel),
            "page_title": page_title,
        })

    return records


def _tokenize_docs(
    records: list[dict],
    tokenizer,
    seq_len: int,
) -> tuple[np.ndarray, dict]:
    """Tokenize all doc records and chunk into fixed-length arrays.

    Returns:
        (chunks_array, stats) where chunks_array has shape [N, seq_len] uint16.
    """
    all_tokens: list[int] = []
    total_files = 0
    total_chars = 0

    cli.dim(f"Processing {len(records)} markdown files...")

    for record in records:
        try:
            raw = record["path"].read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            cli.warn(f"Could not read {record['path']}: {exc}")
            continue

        framework_version, body = _parse_doc_header(raw)
        doc_text = _build_doc_text(framework_version, record["page_title"], body)

        ids = tokenizer.encode(doc_text, add_bos=True, add_eos=False)
        all_tokens.extend(ids)
        total_files += 1
        total_chars += len(raw)

    if not all_tokens:
        return np.zeros((0, seq_len), dtype=np.uint16), {
            "total_files": 0,
            "total_tokens": 0,
            "num_chunks": 0,
        }

    # Chunk into seq_len blocks (discard the trailing partial chunk)
    total_tokens = len(all_tokens)
    num_complete = total_tokens // seq_len
    trimmed = all_tokens[: num_complete * seq_len]

    arr = np.array(trimmed, dtype=np.uint16).reshape(num_complete, seq_len)

    stats = {
        "total_files": total_files,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "num_chunks": num_complete,
    }
    return arr, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scraped markdown docs into tokenized .npy training data."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to trained tokenizer.json file.",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="data/docs",
        help="Root directory containing {framework}/{version}/*.md files "
             "(default: data/docs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/docs_data.npy",
        help="Output .npy file path (default: data/processed/docs_data.npy).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for chunking (default: 2048). Override with --config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If provided, seq_len is read from model.max_seq_len.",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Docs Data Preparation")

    # ---- Resolve seq_len from config if provided ----
    seq_len = args.seq_len
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            cli.fatal(f"Config not found: {config_path}")
        try:
            from cola_coder.model.config import Config

            config = Config.from_yaml(str(config_path))
            seq_len = config.model.max_seq_len
            cli.info("Config", str(config_path))
        except Exception as exc:
            cli.fatal(f"Error loading config: {exc}")

    cli.info("Sequence length", seq_len)

    # ---- Validate inputs ----
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint="Train a tokenizer first: python scripts/train_tokenizer.py",
        )

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        cli.fatal(
            f"Docs directory not found: {docs_dir}",
            hint="Create it or run a docs scraper first.",
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load tokenizer ----
    cli.step(1, 3, "Loading tokenizer")
    cli.dim(f"Source: {tokenizer_path}")

    try:
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Is the package installed?",
            hint="Try: pip install -e .",
        )

    try:
        tokenizer = CodeTokenizer(str(tokenizer_path))
        cli.info("Vocabulary size", tokenizer.vocab_size)
    except Exception as exc:
        cli.fatal(f"Error loading tokenizer: {exc}")

    # ---- Step 2: Scan docs ----
    cli.step(2, 3, "Scanning docs directory")
    cli.dim(f"Source: {docs_dir}")

    records = _scan_docs(docs_dir)

    if not records:
        cli.fatal(
            f"No .md files found under {docs_dir}",
            hint="Ensure docs are structured as {framework}/{version}/*.md",
        )

    # Show breakdown by framework/version directory
    frameworks: dict[str, int] = {}
    for r in records:
        parts = Path(r["rel"]).parts
        key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        frameworks[key] = frameworks.get(key, 0) + 1

    cli.info("Total files", len(records))
    for fw, count in sorted(frameworks.items()):
        cli.dim(f"  {fw}: {count} pages")

    # ---- Step 3: Tokenize and chunk ----
    cli.step(3, 3, "Tokenizing and chunking")
    start = time.time()

    try:
        chunks, stats = _tokenize_docs(records, tokenizer, seq_len)
    except KeyboardInterrupt:
        cli.warn("Interrupted.")
        sys.exit(1)
    except Exception as exc:
        cli.fatal(f"Tokenization failed: {exc}")

    elapsed = time.time() - start

    if stats["num_chunks"] == 0:
        cli.warn("No complete chunks produced. The docs may be too short for the seq_len.")
        cli.warn("Consider using --seq-len with a smaller value.")
        sys.exit(1)

    # ---- Save output ----
    np.save(str(output_path), chunks)

    file_size = output_path.stat().st_size
    throughput = stats["total_tokens"] / elapsed if elapsed > 0 else 0

    cli.info("Shape", f"{chunks.shape}  (chunks × seq_len)")
    cli.info("Dtype", "uint16")
    cli.info("File size", _format_size(file_size))

    # ---- Write manifest ----
    manifest_path = output_path.with_suffix("").with_suffix(".manifest.yaml")
    try:
        from cola_coder.manifest import write_data_manifest

        write_data_manifest(
            str(manifest_path),
            output_file=output_path.name,
            output_size_bytes=file_size,
            num_chunks=stats["num_chunks"],
            chunk_size=seq_len,
            total_tokens=stats["total_tokens"],
            dtype="uint16",
            total_files=stats["total_files"],
            dataset="local/docs",
            languages=sorted(frameworks.keys()),
            filter_mode="none",
            tokenizer_path=str(tokenizer_path),
            vocab_size=tokenizer.vocab_size,
            wall_time_seconds=elapsed,
            throughput_tokens_per_sec=throughput,
        )
        cli.dim(f"Manifest: {manifest_path}")
    except Exception as exc:
        cli.warn(f"Could not write manifest: {exc}")

    cli.done(
        "Docs data preparation complete!",
        extras={
            "Output": str(output_path.resolve()),
            "Chunks": f"{stats['num_chunks']:,}",
            "Tokens": f"{stats['total_tokens']:,}",
            "Files": f"{stats['total_files']:,}",
            "Created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Next step": "python scripts/combine_datasets.py",
        },
    )


if __name__ == "__main__":
    main()
