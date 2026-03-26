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
from cola_coder.data.scorers.protocol import CompositeScorer
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
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for subprocess scorers on large datasets")
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

    # Build composite scorer (this also logs sandbox status)
    scorer_names = args.scorers.split(",") if args.scorers else None
    composite = build_composite_scorer(args.config, scorer_names)

    # Show sandbox status banner
    _show_sandbox_banner(composite)

    if args.jsonl:
        _score_jsonl(args, composite)
    elif args.data:
        _score_npy(args, composite)

    # Show sandbox execution summary
    _show_sandbox_summary(composite)

    cli.success("Scoring complete!")


def _get_sandbox_runner(composite: CompositeScorer) -> "SandboxedRunner | None":
    """Extract the SandboxedRunner from any subprocess-based scorer in the composite."""
    from cola_coder.data.scorers.sandbox import SandboxedRunner

    for scorer, _ in composite._scorers:
        # TscScorer has _runner, EslintScorer has _runner
        runner = getattr(scorer, "_runner", None)
        if isinstance(runner, SandboxedRunner):
            return runner
        # TscScorer wraps TscRunner which has _runner
        tsc = getattr(scorer, "_tsc", None)
        if tsc is not None:
            inner = getattr(tsc, "_runner", None)
            if isinstance(inner, SandboxedRunner):
                return inner
    return None


def _show_sandbox_banner(composite: CompositeScorer) -> None:
    """Show sandbox/Docker status at the top of scoring output."""
    runner = _get_sandbox_runner(composite)
    if runner is None:
        cli.dim("  Sandbox: N/A (no subprocess scorers active)")
        return

    if runner.use_docker:
        cli.info(
            "Sandbox",
            f"Docker CONNECTED — image={runner.docker_image}, "
            f"network=none, memory={runner.memory_mb}MB",
        )
    elif runner._docker_requested:
        cli.warn(
            "  Sandbox: Docker was REQUESTED but NOT AVAILABLE — "
            "using native isolation (temp dir + timeout)",
        )
    else:
        cli.info(
            "Sandbox",
            f"Native mode — temp dir isolation, timeout={runner.timeout}s "
            "(set security.mode=docker for container isolation)",
        )


def _show_sandbox_summary(composite: CompositeScorer) -> None:
    """Show sandbox execution statistics at end of scoring."""
    runner = _get_sandbox_runner(composite)
    if runner is None:
        return

    summary = runner.get_run_summary()
    total = summary["total_runs"]
    if total == 0:
        return

    cli.info("Sandbox summary", "")
    if runner.use_docker:
        cli.dim(
            f"  Docker: {summary['docker_runs']:,}/{total:,} executions "
            f"(all sandboxed in {runner.docker_image})"
        )
    else:
        cli.dim(f"  Native: {summary['native_runs']:,}/{total:,} executions (temp dir isolation)")

    if summary["errors"] > 0:
        cli.warn(f"  Errors: {summary['errors']:,} tool execution failures")


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
    """Score a .npy file with tiered approach for efficiency."""
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

    # Classify scorers into fast (pure Python) and slow (subprocess)
    fast_scorers: list[tuple[object, float]] = []
    slow_scorers: list[tuple[object, float]] = []
    for scorer, weight in composite._scorers:
        if scorer.name in ("tsc", "eslint", "llm_judge"):
            slow_scorers.append((scorer, weight))
        else:
            fast_scorers.append((scorer, weight))

    has_slow = len(slow_scorers) > 0

    if has_slow and n_samples > 10000:
        cli.info("Strategy", "Sample-and-distill (large dataset with subprocess scorers)")
        cli.dim(f"  Fast scorers (all {n_samples:,} samples): {[s.name for s, _ in fast_scorers]}")
        cli.dim(f"  Slow scorers (sampled): {[s.name for s, _ in slow_scorers]}")
        _score_tiered(args, data, tokenizer, fast_scorers, slow_scorers, n_samples)
    else:
        cli.info("Strategy", f"Direct scoring ({n_samples:,} samples)")
        _score_direct(args, data, tokenizer, composite, n_samples)

    # Optional curriculum ordering
    if args.curriculum:
        from cola_coder.data.scorers.curriculum import CurriculumOrderer, CurriculumStrategy
        cli.step(3, 3, f"Applying curriculum: {args.curriculum}")
        strategy = CurriculumStrategy(args.curriculum)
        orderer = CurriculumOrderer(strategy=strategy)
        weights_path = Path(args.data).with_suffix(".weights.npy")
        schedule = orderer.reorder(data_path, weights_path)
        cli.info("Curriculum", f"{schedule.strategy} — {len(schedule.phases)} phases")


def _score_tiered(args, data, tokenizer, fast_scorers, slow_scorers, n_samples: int) -> None:
    """Three-tier scoring for large datasets.

    1. Score ALL samples with fast scorers (heuristic, stars, classifier)
    2. Score a SAMPLE with slow scorers (tsc, eslint)
    3. Train a classifier from the slow scorer sample
    4. Apply classifier to ALL samples
    5. Combine fast scores + classifier scores into final weights
    """
    data_path = Path(args.data)
    start_time = time.time()

    # --- Tier 1: Fast scorers on ALL data ---
    cli.step(1, 4, f"Fast scoring all {n_samples:,} samples")
    fast_composite = CompositeScorer(fast_scorers)
    fast_weights = np.ones(n_samples, dtype=np.float32)
    fast_score_sum: float = 0.0

    for i in range(n_samples):
        tokens = data[i].tolist()
        text = tokenizer.decode(tokens)
        result = fast_composite.score(text)
        fast_weights[i] = result.overall  # Store raw overall, not tier weight
        fast_score_sum += result.overall

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            avg = fast_score_sum / (i + 1)
            cli.dim(
                f"  [{(i+1)/n_samples*100:5.1f}%] {i+1:,}/{n_samples:,} "
                f"({rate:.0f}/sec, ETA: {_format_eta(eta)}) avg={avg:.3f}"
            )

    fast_avg = fast_score_sum / n_samples if n_samples > 0 else 0.0
    cli.success(
        f"  Fast scoring done: {n_samples:,} samples in "
        f"{_format_eta(time.time() - start_time)} (avg={fast_avg:.3f})"
    )

    # --- Tier 2: Slow scorers on SAMPLE ---
    sample_size = min(args.sample_size or 10000, n_samples)
    cli.step(2, 4, f"Subprocess scoring {sample_size:,} sampled chunks")

    # Stratified sample: pick across the quality spectrum
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    sample_indices.sort()

    slow_composite = CompositeScorer(slow_scorers)
    annotations_path = data_path.with_suffix(".slow_annotations.jsonl")

    with open(annotations_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(sample_indices), 50):
            batch_end = min(batch_start + 50, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]

            # Decode batch
            batch_items: list[tuple[str, dict[str, object] | None]] = []
            for idx in batch_indices:
                tokens = data[idx].tolist()
                text = tokenizer.decode(tokens)
                batch_items.append((text, None))

            # Score batch (uses score_batch for tsc/eslint batch optimization)
            batch_results = slow_composite.score_batch(batch_items)

            for j, result in enumerate(batch_results):
                entry = {
                    "code_prefix": batch_items[j][0][:500],
                    "score": int(round(result.overall * 5)),  # 0-5 scale for classifier
                    "overall": round(result.overall, 4),
                }
                f.write(json.dumps(entry) + "\n")

            done = min(batch_end, len(sample_indices))
            if done % 200 == 0 or done == len(sample_indices):
                cli.dim(f"  Sampled {done:,}/{sample_size:,}")

    cli.success(f"  Subprocess scoring done: {sample_size:,} samples")

    # --- Tier 3: Train classifier and bulk score ---
    cli.step(3, 4, "Training quality classifier from sample")

    try:
        from cola_coder.data.scorers.classifier import QualityClassifierTrainer, QualityClassifier

        model_dir = str(data_path.parent / "quality_classifier")
        trainer = QualityClassifierTrainer()
        metrics = trainer.train(str(annotations_path), model_dir)
        cli.dim(f"  Accuracy: {metrics.accuracy:.3f}, MAE: {metrics.mean_absolute_error:.3f}")

        # Bulk predict on ALL samples
        classifier = QualityClassifier(model_dir)
        cli.step(4, 4, f"Classifier scoring all {n_samples:,} samples")

        classifier_weights = np.ones(n_samples, dtype=np.float32)
        batch_size = 500
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            texts: list[str] = []
            for i in range(batch_start, batch_end):
                tokens = data[i].tolist()
                texts.append(tokenizer.decode(tokens))

            preds = classifier.predict_batch(texts)
            for j, pred in enumerate(preds):
                classifier_weights[batch_start + j] = pred

            if (batch_end) % 5000 == 0:
                cli.dim(f"  [{batch_end/n_samples*100:5.1f}%] {batch_end:,}/{n_samples:,}")

        # Combine: weighted average of fast scores + classifier scores
        # Fast scorers get their configured weights, classifier gets the slow scorers' total weight
        fast_total = sum(w for _, w in fast_scorers) if fast_scorers else 0
        slow_total = sum(w for _, w in slow_scorers) if slow_scorers else 0
        total = fast_total + slow_total

        if total > 0:
            final_scores = (fast_weights * (fast_total / total)) + (classifier_weights * (slow_total / total))
        else:
            final_scores = fast_weights

    except ImportError:
        cli.warn("  scikit-learn not installed — using fast scores only")
        final_scores = fast_weights
    except Exception as e:
        cli.warn(f"  Classifier training failed: {e} — using fast scores only")
        final_scores = fast_weights

    # Convert scores to tier weights using the same mapping as CompositeScorer
    final_weights = np.ones(n_samples, dtype=np.float32)
    for i in range(n_samples):
        s = float(final_scores[i])
        if s >= 0.8:
            final_weights[i] = 2.0
        elif s >= 0.6:
            final_weights[i] = 1.5
        elif s >= 0.4:
            final_weights[i] = 1.0
        elif s >= 0.2:
            final_weights[i] = 0.3
        else:
            final_weights[i] = 0.0

    # Save
    weights_path = data_path.with_suffix(".weights.npy")
    np.save(str(weights_path), final_weights)

    _print_distribution(final_weights)
    cli.info("Weights saved", str(weights_path))

    elapsed = time.time() - start_time
    cli.success(f"Total scoring time: {_format_eta(elapsed)}")


def _score_direct(args, data, tokenizer, composite, n_samples: int) -> None:
    """Direct scoring for small datasets or fast-only scorers."""
    data_path = Path(args.data)
    active_scorers = {s.name for s, _ in composite._scorers} if hasattr(composite, "_scorers") else set()
    subprocess_scorers = {"tsc", "eslint"}
    slow_scorers = active_scorers & subprocess_scorers

    cli.step(2, 3 if args.curriculum else 2, f"Scoring {n_samples:,} chunks from {data_path.name}")
    cli.dim(f"  Active scorers: {', '.join(active_scorers) if active_scorers else 'none'}")
    cli.dim(f"  Output: {data_path.with_suffix('.scores.jsonl').name} + {data_path.with_suffix('.weights.npy').name}")

    weights_list: list[float] = []
    score_sum: float = 0.0  # Running sum for average
    scores_path = data_path.with_suffix(".scores.jsonl")
    start_time = time.time()
    last_log_time = start_time
    errors = 0

    # Per-scorer running sums for breakdown
    scorer_sums: dict[str, float] = {}
    scorer_counts: dict[str, int] = {}

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
                score_sum += result.overall

                # Track per-scorer averages
                for name, sr in result.per_scorer.items():
                    scorer_sums[name] = scorer_sums.get(name, 0.0) + sr.score
                    scorer_counts[name] = scorer_counts.get(name, 0) + 1

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
                scored_count = i + 1
                avg = score_sum / scored_count if scored_count > 0 else 0.0

                # Build per-scorer avg string
                parts: list[str] = []
                for sn in sorted(scorer_sums.keys()):
                    sc = scorer_counts.get(sn, 1)
                    parts.append(f"{sn}={scorer_sums[sn] / sc:.2f}")
                scorer_str = ", ".join(parts)

                cli.dim(
                    f"  [{pct:5.1f}%] {scored_count:,}/{n_samples:,} "
                    f"({rate:.1f}/sec, ETA: {_format_eta(eta)}) "
                    f"avg={avg:.3f} [{scorer_str}]"
                )
                last_log_time = now

            # Flush periodically to see partial results
            if (i + 1) % flush_interval == 0:
                fout.flush()

    elapsed = time.time() - start_time
    final_avg = score_sum / n_samples if n_samples > 0 else 0.0
    cli.success(
        f"  Scored {n_samples:,} chunks in {_format_eta(elapsed)} "
        f"({n_samples / elapsed:.1f}/sec, avg={final_avg:.3f}, {errors} errors)"
    )

    weights = np.array(weights_list, dtype=np.float32)
    weights_path = data_path.with_suffix(".weights.npy")
    np.save(str(weights_path), weights)

    _print_distribution(weights)
    cli.info("Weights saved", str(weights_path))
    cli.info("Details saved", str(scores_path))


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
