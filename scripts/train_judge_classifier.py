"""Train quality classifier from LLM annotations.

Usage:
    python scripts/train_judge_classifier.py annotate --provider ollama --model codellama --data data.jsonl
    python scripts/train_judge_classifier.py annotate --provider claude --model claude-sonnet-4-6 --data data.jsonl
    python scripts/train_judge_classifier.py train --annotations data/annotations.jsonl
    python scripts/train_judge_classifier.py evaluate --model-dir models/quality_classifier --annotations data/annotations.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge classifier pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # annotate
    ann = sub.add_parser("annotate", help="Annotate code samples with LLM")
    ann.add_argument("--provider", default="ollama", choices=["ollama", "claude"])
    ann.add_argument("--model", default="codellama")
    ann.add_argument("--base-url", default="http://localhost:11434")
    ann.add_argument("--num-samples", type=int, default=10000)
    ann.add_argument("--data", type=str, required=True, help="Path to .npy or .jsonl data")
    ann.add_argument("--tokenizer", type=str, default=None, help="Tokenizer (required for .npy)")
    ann.add_argument("--output", default="data/annotations.jsonl")

    # train
    tr = sub.add_parser("train", help="Train classifier from annotations")
    tr.add_argument("--annotations", required=True, help="Path to annotations.jsonl")
    tr.add_argument("--output-dir", default="models/quality_classifier")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate trained classifier")
    ev.add_argument("--model-dir", required=True)
    ev.add_argument("--annotations", required=True, help="Path to test annotations")

    args = parser.parse_args()

    if args.command == "annotate":
        _annotate(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)


def _annotate(args) -> None:
    from cola_coder.data.scorers.llm_judge import LlmJudge

    cli.header("Cola-Coder", "LLM-as-Judge Annotation")
    cli.info("Provider", f"{args.provider} ({args.model})")
    cli.info("Samples", str(args.num_samples))

    judge = LlmJudge(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    # Load data samples
    data_path = Path(args.data)
    codes: list[str] = []

    if data_path.suffix == ".jsonl":
        import json
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.num_samples:
                    break
                try:
                    entry = json.loads(line)
                    code = entry.get("content", "")
                    if code:
                        codes.append(code)
                except json.JSONDecodeError:
                    continue
    elif data_path.suffix == ".npy":
        if args.tokenizer is None:
            cli.error("--tokenizer required for .npy data")
            sys.exit(1)
        import numpy as np
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
        tokenizer = CodeTokenizer(args.tokenizer)
        data = np.load(str(data_path), mmap_mode="r")
        n = min(args.num_samples, len(data))
        for i in range(n):
            text = tokenizer.decode(data[i].tolist())
            if text.strip():
                codes.append(text)

    cli.info("Loaded", f"{len(codes)} samples")
    result_path = judge.annotate_batch(codes, output_path=args.output)
    cli.success(f"Annotations saved to {result_path}")


def _train(args) -> None:
    from cola_coder.data.scorers.classifier import QualityClassifierTrainer

    cli.header("Cola-Coder", "Quality Classifier Training")
    trainer = QualityClassifierTrainer()
    metrics = trainer.train(args.annotations, args.output_dir)
    cli.success(f"Model saved to {args.output_dir}")
    cli.info("Accuracy", f"{metrics.accuracy:.4f}")
    cli.info("MAE", f"{metrics.mean_absolute_error:.4f}")
    cli.info("Training samples", str(metrics.num_train))
    cli.info("Test samples", str(metrics.num_test))


def _evaluate(args) -> None:
    from cola_coder.data.scorers.classifier import QualityClassifierTrainer

    cli.header("Cola-Coder", "Classifier Evaluation")
    trainer = QualityClassifierTrainer()
    metrics = trainer.evaluate(args.model_dir, args.annotations)
    cli.info("Accuracy", f"{metrics.accuracy:.4f}")
    cli.info("MAE", f"{metrics.mean_absolute_error:.4f}")


if __name__ == "__main__":
    main()
