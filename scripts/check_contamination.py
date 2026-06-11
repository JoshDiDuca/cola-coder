"""Benchmark decontamination — detect eval problems leaking into training data.

Contaminated benchmarks inflate pass@k: if an eval problem (or its solution)
appears in the training corpus, the model may have memorised it. This wires the
existing DataLeakageDetector (MinHash + exact containment) into a usable check.

Default metric is `containment` (|eval∩train| / |eval|), which catches a short
eval problem EMBEDDED in a larger training file — the common contamination case
that plain Jaccard similarity misses.

Usage:
    # Check the 62 built-in problems against an SFT/instruction corpus
    python scripts/check_contamination.py --eval all --train-jsonl data/sft/instructions.jsonl

    # Check against a sample of tokenized training chunks
    python scripts/check_contamination.py --eval all \\
        --train-npy data/processed/train_data.npy --tokenizer tokenizer.json

Exit code is 1 when contamination is found (so it can gate a pipeline/CI step).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


def _load_eval_docs(args) -> list[str]:
    """Build eval documents from the chosen problem set.

    Each eval document is a SINGLE contamination unit checked independently for
    embedding in the training corpus: the problem STATEMENT (prompt) the model
    sees, and — as a separate document — its canonical SOLUTION when present.
    The hidden test grader (test_code) is excluded, and prompt/solution are kept
    separate so a leak of EITHER is caught at full containment (concatenating
    them would dilute the signal when only one leaked — the common case).
    """
    from cola_coder.evaluation.problem_loader import ProblemSet

    ps = ProblemSet()
    if args.eval == "jsonl":
        if not args.eval_jsonl:
            cli.fatal("--eval-jsonl PATH is required when --eval jsonl")
        ps.add_from_jsonl(args.eval_jsonl)
    elif args.eval == "typescript":
        ps.add_typescript()
    else:
        ps.add_builtin(extended=args.eval in ("extended", "all"))

    docs: list[str] = []
    for p in ps._problems:
        docs.append(p.prompt)
        sol = getattr(p, "canonical_solution", "")
        if sol and sol.strip():
            docs.append(sol)
    return docs


def _load_train_docs(args) -> list[str]:
    """Load the training corpus as text documents (JSONL fields or decoded .npy)."""
    if args.train_jsonl:
        docs: list[str] = []
        with open(args.train_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Accept the common text-bearing fields used across the project.
                text = (
                    obj.get("content") or obj.get("text") or obj.get("code")
                    or obj.get("output") or obj.get("completion")
                    or obj.get("instruction") or ""
                )
                if text:
                    docs.append(text)
                if args.max_train and len(docs) >= args.max_train:
                    break
        return docs

    # Tokenized .npy: decode a sample of chunks back to text.
    import numpy as np

    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

    tok = CodeTokenizer(args.tokenizer)
    data = np.load(args.train_npy, mmap_mode="r")
    n_total = len(data)
    n = min(n_total, args.max_train) if args.max_train else n_total
    # Evenly spaced sample so we cover the whole file, not just the head.
    idxs = np.linspace(0, n_total - 1, n).astype(int) if n_total else []
    return [tok.decode(data[int(i)].tolist()) for i in idxs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect eval/training data contamination (benchmark decontamination).",
    )
    parser.add_argument(
        "--eval", default="all",
        choices=["builtin", "extended", "all", "typescript", "jsonl"],
        help="Evaluation problem set to check (default: all 62 built-in).",
    )
    parser.add_argument("--eval-jsonl", default=None, help="Custom eval problems JSONL.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-jsonl", default=None, help="Training corpus JSONL.")
    src.add_argument("--train-npy", default=None, help="Tokenized training .npy.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer (required for --train-npy).")

    parser.add_argument(
        "--metric", default="containment", choices=["containment", "jaccard"],
        help="containment (default) catches embedded problems; jaccard finds "
             "near-duplicate same-size docs.",
    )
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Similarity/containment threshold to flag (default: 0.8).")
    parser.add_argument("--shingle-size", type=int, default=5,
                        help="Character n-gram size (default: 5).")
    parser.add_argument("--max-train", type=int, default=20000,
                        help="Cap training docs sampled (default: 20000; 0 = all).")
    parser.add_argument("--max-matches", type=int, default=25,
                        help="Max contamination matches to print (default: 25).")
    args = parser.parse_args()

    if args.train_npy and not args.tokenizer:
        cli.fatal("--tokenizer is required with --train-npy")

    cli.header("Cola-Coder", "Benchmark Decontamination")

    eval_docs = _load_eval_docs(args)
    cli.info("Eval problems", len(eval_docs))

    cli.info("Loading training corpus", args.train_jsonl or args.train_npy)
    train_docs = _load_train_docs(args)
    cli.info("Training docs", f"{len(train_docs):,}")

    if not eval_docs or not train_docs:
        cli.warn("Nothing to compare (empty eval or training set).")
        sys.exit(0)

    from cola_coder.features.data_leakage_detector import DataLeakageDetector

    detector = DataLeakageDetector(
        similarity_threshold=args.threshold, shingle_size=args.shingle_size
    )
    cli.step(1, 2, "Indexing training corpus")
    detector.index_train(train_docs)
    cli.step(2, 2, f"Checking {len(eval_docs)} eval docs ({args.metric})")
    report = detector.check_eval(eval_docs, metric=args.metric)

    cli.rule("Results")
    cli.kv_table({
        "Metric": args.metric,
        "Threshold": f"{args.threshold:.2f}",
        "Eval docs": str(report.num_eval_docs),
        "Train docs": str(report.num_train_docs),
        "Contaminated": str(report.num_contaminated),
        "Contamination rate": f"{report.contamination_rate:.1%}",
    }, title="Contamination Report")

    if report.has_leakage():
        cli.warn(f"{report.num_contaminated} eval problem(s) overlap the training data:")
        for m in report.matches[:args.max_matches]:
            cli.dim(f"  {m.summary()}")
            cli.dim(f"    eval:  {m.eval_preview.strip()[:80]!r}")
            cli.dim(f"    train: {m.train_preview.strip()[:80]!r}")
        cli.error(
            "Contamination detected — pass@k on these problems is unreliable. "
            "Remove the overlapping training data or exclude these problems."
        )
        sys.exit(1)

    cli.success("No contamination detected above the threshold.")


if __name__ == "__main__":
    main()
