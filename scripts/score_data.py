"""Score data quality and generate .weights.npy for weighted training.

Usage:
    python scripts/score_data.py --data code_data.npy --tokenizer tokenizer.json
    python scripts/score_data.py --data code_data.npy --scorers tsc,eslint --tokenizer tokenizer.json
    python scripts/score_data.py --jsonl github_scraped.jsonl
    python scripts/score_data.py --data code_data.npy --tokenizer tokenizer.json --curriculum easy_to_hard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.data.scorers.registry import build_composite_scorer, list_available_scorers


def main() -> None:
    parser = argparse.ArgumentParser(description="Score data quality for training.")
    parser.add_argument("--data", type=str, default=None, help="Path to .npy data file")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL file (GitHub scraped)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json (required for .npy)")
    parser.add_argument("--scorers", type=str, default=None, help="Comma-separated scorer names (default: all enabled)")
    parser.add_argument("--config", type=str, default="configs/scoring.yaml", help="Scoring config path")
    parser.add_argument("--curriculum", type=str, default=None, help="Curriculum strategy: easy_to_hard, hard_to_easy, staged, random")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to score (for testing)")
    args = parser.parse_args()

    if args.data is None and args.jsonl is None:
        cli.error("Specify --data (for .npy) or --jsonl (for JSONL)")
        sys.exit(1)

    cli.header("Cola-Coder", "Data Quality Scoring")

    # Show available scorers
    available = list_available_scorers(args.config)
    cli.info("Available scorers", "")
    for s in available:
        status = "+" if s["available"] else "-"
        enabled = "enabled" if s["enabled"] else "disabled"
        cli.dim(f"  {status} {s['name']} (weight={s['weight']}, {enabled})")

    # Build composite scorer
    scorer_names = args.scorers.split(",") if args.scorers else None
    composite = build_composite_scorer(args.config, scorer_names)

    if args.jsonl:
        _score_jsonl(args, composite)
    elif args.data:
        _score_npy(args, composite)

    cli.success("Scoring complete!")


def _score_jsonl(args, composite) -> None:
    """Score a JSONL file (GitHub scraped data)."""
    cli.step(1, 2, f"Scoring JSONL: {args.jsonl}")

    input_path = Path(args.jsonl)
    scores_path = input_path.with_suffix(".scores.jsonl")
    weights_list: list[float] = []

    with open(input_path, encoding="utf-8") as fin, \
         open(scores_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.max_samples and i >= args.max_samples:
                break
            try:
                entry = json.loads(line)
                code = entry.get("content", "")
                if not code:
                    continue
                metadata = {k: v for k, v in entry.items() if k != "content"}
            except json.JSONDecodeError:
                continue

            result = composite.score(code, metadata)
            weights_list.append(result.weight)

            score_entry = {
                "index": i,
                "overall": round(result.overall, 4),
                "weight": round(result.weight, 2),
            }
            for name, sr in result.per_scorer.items():
                score_entry[name] = round(sr.score, 4)
            fout.write(json.dumps(score_entry) + "\n")

            if (i + 1) % 1000 == 0:
                cli.dim(f"  Scored {i + 1} samples...")

    # Save weights
    weights = np.array(weights_list, dtype=np.float32)
    weights_path = input_path.with_suffix(".weights.npy")
    np.save(str(weights_path), weights)

    _print_distribution(weights)
    cli.info("Weights saved", str(weights_path))
    cli.info("Details saved", str(scores_path))


def _score_npy(args, composite) -> None:
    """Score a .npy file (tokenized data — requires tokenizer to decode)."""
    if args.tokenizer is None:
        cli.error("--tokenizer required when scoring .npy files")
        sys.exit(1)

    cli.step(1, 3 if args.curriculum else 2, f"Loading tokenizer: {args.tokenizer}")
    try:
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        tokenizer = CodeTokenizer(args.tokenizer)
    except Exception as e:
        cli.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    data_path = Path(args.data)
    data = np.load(str(data_path), mmap_mode="r")
    n_samples = args.max_samples or len(data)
    n_samples = min(n_samples, len(data))

    # Warn about subprocess-based scorers on large datasets
    subprocess_scorers = {"tsc", "eslint"}
    active_scorers = {s.name for s, _ in composite._scorers} if hasattr(composite, "_scorers") else set()
    slow_scorers = active_scorers & subprocess_scorers
    if slow_scorers and n_samples > 10000:
        cli.warn(
            f"  Subprocess scorers ({', '.join(slow_scorers)}) are active on {n_samples:,} samples."
        )
        cli.warn(
            f"  This will be very slow (~0.5-2 sec/sample). Consider:"
        )
        cli.dim(f"    --max-samples 5000  (score a subset)")
        cli.dim(f"    --scorers heuristic  (pure Python, ~10k/sec)")
        cli.dim(f"    --scorers heuristic,stars  (fast scorers only)")

    cli.step(2, 3 if args.curriculum else 2, f"Scoring {n_samples:,} chunks from {data_path.name}")
    cli.dim(f"  Active scorers: {', '.join(active_scorers) if active_scorers else 'none'}")
    cli.dim(f"  Output: {data_path.with_suffix('.scores.jsonl').name} + {data_path.with_suffix('.weights.npy').name}")

    weights_list: list[float] = []
    scores_path = data_path.with_suffix(".scores.jsonl")
    start_time = time.time()
    last_log_time = start_time
    errors = 0

    # Adaptive log interval: log more frequently for slow scorers
    log_interval = 10 if slow_scorers else 500
    flush_interval = max(log_interval, 100)

    with open(scores_path, "w", encoding="utf-8") as fout:
        for i in range(n_samples):
            try:
                # Decode chunk back to text
                tokens = data[i].tolist()
                text = tokenizer.decode(tokens)

                result = composite.score(text)
                weights_list.append(result.weight)

                score_entry = {
                    "index": i,
                    "overall": round(result.overall, 4),
                    "weight": round(result.weight, 2),
                }
                for name, sr in result.per_scorer.items():
                    score_entry[name] = round(sr.score, 4)
                fout.write(json.dumps(score_entry) + "\n")

            except Exception as e:
                errors += 1
                weights_list.append(1.0)  # Neutral weight on error
                if errors <= 5:
                    cli.warn(f"  Error scoring chunk {i}: {e}")
                elif errors == 6:
                    cli.warn(f"  Suppressing further error messages...")

            # Progress logging
            now = time.time()
            if (i + 1) % log_interval == 0 or (now - last_log_time) >= 10:
                elapsed = now - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = n_samples - i - 1
                eta = remaining / rate if rate > 0 else 0
                pct = (i + 1) / n_samples * 100
                cli.dim(
                    f"  [{pct:5.1f}%] {i + 1:,}/{n_samples:,} "
                    f"({rate:.1f}/sec, ETA: {_format_eta(eta)}, "
                    f"errors: {errors})"
                )
                last_log_time = now

            # Flush periodically to see partial results
            if (i + 1) % flush_interval == 0:
                fout.flush()

    elapsed = time.time() - start_time
    cli.success(
        f"  Scored {n_samples:,} chunks in {_format_eta(elapsed)} "
        f"({n_samples / elapsed:.1f}/sec, {errors} errors)"
    )

    weights = np.array(weights_list, dtype=np.float32)
    weights_path = data_path.with_suffix(".weights.npy")
    np.save(str(weights_path), weights)

    _print_distribution(weights)
    cli.info("Weights saved", str(weights_path))
    cli.info("Details saved", str(scores_path))

    # Optional curriculum ordering
    if args.curriculum:
        from cola_coder.data.scorers.curriculum import CurriculumOrderer, CurriculumStrategy
        cli.step(3, 3, f"Applying curriculum: {args.curriculum}")
        strategy = CurriculumStrategy(args.curriculum)
        orderer = CurriculumOrderer(strategy=strategy)
        schedule = orderer.reorder(data_path, weights_path)
        cli.info("Curriculum", f"{schedule.strategy} — {len(schedule.phases)} phases")


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def _print_distribution(weights: np.ndarray) -> None:
    """Print score distribution summary."""
    n = len(weights)
    if n == 0:
        return

    tiers = {
        "excellent (2.0x)": np.sum(weights >= 2.0),
        "good (1.5x)": np.sum((weights >= 1.5) & (weights < 2.0)),
        "average (1.0x)": np.sum((weights >= 1.0) & (weights < 1.5)),
        "poor (0.3x)": np.sum((weights > 0.0) & (weights < 1.0)),
        "reject (0.0x)": np.sum(weights <= 0.0),
    }
    cli.info("Distribution", "")
    for tier, count in tiers.items():
        pct = count / n * 100
        cli.dim(f"  {tier}: {int(count):,} ({pct:.1f}%)")
    cli.dim(f"  Mean weight: {np.mean(weights):.3f}")


if __name__ == "__main__":
    main()
