#!/usr/bin/env python3
"""Train a FineWeb-Edu style quality classifier for code.

Workflow:
    1. Sample N code files from an existing .npy dataset
    2. Score them with an LLM (Claude API) or load existing labels
    3. Fine-tune a small classifier (DistilBERT/CodeBERTa) on those labels
    4. Evaluate on held-out test set
    5. Save the trained model

Usage:
    # Step 1: Generate annotations (needs ANTHROPIC_API_KEY)
    python scripts/train_quality_classifier.py annotate \
        --data data/processed/train_data.npy \
        --tokenizer tokenizer.json \
        --num-samples 10000 \
        --output data/quality_labels.jsonl

    # Step 2: Train classifier (needs transformers + torch)
    python scripts/train_quality_classifier.py train \
        --labels data/quality_labels.jsonl \
        --output models/quality_classifier \
        --epochs 5 --lr 2e-5

    # Step 3: Evaluate
    python scripts/train_quality_classifier.py evaluate \
        --model models/quality_classifier \
        --labels data/quality_labels.jsonl

    # Quick test with heuristic scorer (no dependencies)
    python scripts/train_quality_classifier.py demo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def cmd_demo(args: argparse.Namespace) -> None:
    """Demo the heuristic scorer on some sample code."""
    from cola_coder.data.filters.quality_classifier import HeuristicQualityScorer

    scorer = HeuristicQualityScorer()

    samples = {
        "excellent_python": '''\
"""Calculator module with basic arithmetic operations."""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


class Calculator:
    """A simple calculator that tracks operation history.

    Usage:
        calc = Calculator()
        result = calc.add(2, 3)  # 5
        print(calc.history)      # [('+', 2, 3, 5)]
    """

    def __init__(self) -> None:
        self.history: list[tuple[str, Number, Number, Number]] = []

    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers and record in history."""
        result = a + b
        self.history.append(("+", a, b, result))
        return result

    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a and record in history."""
        result = a - b
        self.history.append(("-", a, b, result))
        return result

    def clear_history(self) -> None:
        """Clear the operation history."""
        self.history.clear()
''',
        "poor_python": '''\
x=1
y=2
z=x+y
a=z*2
b=a-1
print(b)
if b>5:
    print("big")
else:
    print("small")
for i in range(10):
    for j in range(10):
        for k in range(10):
            print(i,j,k)
''',
        "minified_js": (
            'function a(b,c){return b+c}function d(e,f){return e*f}'
            'var g=a(1,2);var h=d(3,4);console.log(g,h);'
        ),
        "good_typescript": '''\
import { useState, useCallback } from 'react';

interface TodoItem {
    id: string;
    text: string;
    completed: boolean;
}

/**
 * Custom hook for managing a todo list.
 *
 * @returns Todo list state and mutation functions
 */
export function useTodos() {
    const [todos, setTodos] = useState<TodoItem[]>([]);

    const addTodo = useCallback((text: string) => {
        setTodos(prev => [
            ...prev,
            { id: crypto.randomUUID(), text, completed: false },
        ]);
    }, []);

    const toggleTodo = useCallback((id: string) => {
        setTodos(prev =>
            prev.map(todo =>
                todo.id === id ? { ...todo, completed: !todo.completed } : todo
            )
        );
    }, []);

    const removeTodo = useCallback((id: string) => {
        setTodos(prev => prev.filter(todo => todo.id !== id));
    }, []);

    return { todos, addTodo, toggleTodo, removeTodo };
}
''',
        "empty": "",
        "trivial": "x = 1\n",
    }

    print("=== Heuristic Quality Scorer Demo ===\n")
    for name, code in samples.items():
        score = scorer.score(code)
        label = (
            "garbage" if score < 0.2
            else "poor" if score < 0.4
            else "average" if score < 0.6
            else "good" if score < 0.8
            else "excellent"
        )
        print(f"  {name:25s} -> {score:.3f} ({label})")

    print("\n=== Filter Demo ===\n")
    from cola_coder.data.filters.quality_classifier import QualityClassifierFilter
    from cola_coder.data.pipeline import DataRecord

    filt = QualityClassifierFilter(threshold=0.4, mode="heuristic")
    for name, code in samples.items():
        record = DataRecord(content=code, metadata={"language": "python"})
        keep, reason = filt.check(record)
        status = "KEEP" if keep else f"REJECT ({reason})"
        print(f"  {name:25s} -> {status}")


def cmd_annotate(args: argparse.Namespace) -> None:
    """Score code samples with an LLM to create training labels."""
    from cola_coder.data.filters.quality_classifier import QualityAnnotator

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set ANTHROPIC_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {data_path}...")

    # Load tokenizer to decode tokens back to text
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
    except Exception as e:
        print(f"ERROR: Could not load tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    data = np.load(str(data_path), mmap_mode="r")
    total_tokens = len(data)
    print(f"Dataset has {total_tokens:,} tokens")

    # Sample random chunks and decode them back to code
    # Each chunk is seq_len tokens (default 2048)
    seq_len = args.seq_len
    num_chunks = total_tokens // seq_len
    sample_indices = random.sample(range(num_chunks), min(args.num_samples, num_chunks))

    annotator = QualityAnnotator(api_key=api_key, model=args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Annotating {len(sample_indices)} samples...")
    annotated = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk_idx in enumerate(sample_indices):
            start = chunk_idx * seq_len
            end = start + seq_len
            tokens = data[start:end].tolist()

            # Decode tokens back to text
            code = tokenizer.decode(tokens, skip_special_tokens=True)

            if not code.strip():
                continue

            try:
                scores = annotator.annotate_batch([code], language=args.language)
                score = scores[0]
            except Exception as e:
                print(f"  Warning: annotation failed for chunk {chunk_idx}: {e}")
                continue

            record = {
                "code": code,
                "score": score,
                "language": args.language,
                "chunk_idx": chunk_idx,
            }
            f.write(json.dumps(record) + "\n")
            annotated += 1

            if (i + 1) % 100 == 0:
                print(f"  Annotated {i + 1}/{len(sample_indices)} samples...")

    print(f"\nDone! Annotated {annotated} samples -> {output_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train the quality classifier on labeled data."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print(
            "ERROR: Training requires transformers and torch.\n"
            "Install with: pip install transformers torch",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load labeled data
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading labels from {labels_path}...")
    records = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} labeled samples")

    # Train/val/test split (80/10/10)
    random.shuffle(records)
    n = len(records)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_data = records[:train_end]
    val_data = records[train_end:val_end]
    test_data = records[val_end:]

    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Load pretrained model
    base_model = args.base_model
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,  # Regression: single score output
        problem_type="regression",
    )

    # Create datasets
    class CodeQualityDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            record = self.data[idx]
            encoding = self.tokenizer(
                record["code"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(float(record["score"]), dtype=torch.float),
            }

    train_dataset = CodeQualityDataset(train_data, tokenizer)
    val_dataset = CodeQualityDataset(val_data, tokenizer)

    # Training arguments
    output_dir = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")

    # Quick evaluation on test set
    print("\nEvaluating on test set...")
    test_dataset = CodeQualityDataset(test_data, tokenizer)
    results = trainer.evaluate(test_dataset)
    print(f"Test loss: {results['eval_loss']:.4f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained classifier against labeled data."""
    try:
        from cola_coder.data.filters.quality_classifier import CodeQualityClassifier
    except ImportError:
        print("ERROR: Could not import quality classifier", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    classifier = CodeQualityClassifier(model_path=str(model_path))

    # Load labels
    labels_path = Path(args.labels)
    records = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Evaluating on {len(records)} samples...")

    # Score all samples
    codes = [r["code"] for r in records]
    true_scores = [r["score"] for r in records]
    pred_scores = classifier.score_batch(codes)

    # Convert predicted 0-1 back to 1-5 for comparison
    pred_labels = [s * 4 + 1 for s in pred_scores]

    # Calculate metrics
    n = len(true_scores)
    mae = sum(abs(t - p) for t, p in zip(true_scores, pred_labels)) / n
    mse = sum((t - p) ** 2 for t, p in zip(true_scores, pred_labels)) / n

    # Pearson correlation
    mean_true = sum(true_scores) / n
    mean_pred = sum(pred_labels) / n
    cov = sum((t - mean_true) * (p - mean_pred) for t, p in zip(true_scores, pred_labels)) / n
    std_true = (sum((t - mean_true) ** 2 for t in true_scores) / n) ** 0.5
    std_pred = (sum((p - mean_pred) ** 2 for p in pred_labels) / n) ** 0.5
    correlation = cov / max(std_true * std_pred, 1e-8)

    print("\nResults:")
    print(f"  MAE:         {mae:.3f}")
    print(f"  RMSE:        {mse ** 0.5:.3f}")
    print(f"  Correlation: {correlation:.3f}")

    # Score distribution
    print("\nScore distribution (predicted, 1-5 scale):")
    for bucket in range(1, 6):
        count = sum(1 for s in pred_labels if bucket - 0.5 <= s < bucket + 0.5)
        pct = count / n * 100
        bar = "#" * int(pct / 2)
        print(f"  {bucket}: {count:5d} ({pct:5.1f}%) {bar}")


def main() -> None:
    from cola_coder.model.config import get_storage_config

    storage = get_storage_config()
    storage.apply_hf_cache()

    parser = argparse.ArgumentParser(
        description="Train a FineWeb-Edu style quality classifier for code"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Demo command
    sub.add_parser("demo", help="Demo the heuristic scorer")

    # Annotate command
    ann = sub.add_parser("annotate", help="Score samples with LLM")
    ann.add_argument("--data", required=True, help="Path to .npy token data")
    ann.add_argument("--tokenizer", default=storage.tokenizer_path, help="Path to tokenizer.json")
    ann.add_argument("--num-samples", type=int, default=10000, help="Number of samples")
    ann.add_argument("--seq-len", type=int, default=2048, help="Tokens per sample")
    ann.add_argument("--output", default=str(Path(storage.data_dir) / "quality_labels.jsonl"),
                     help="Output path")
    ann.add_argument("--api-key", help="Anthropic API key")
    ann.add_argument("--model", default="claude-3-haiku-20240307", help="LLM model name")
    ann.add_argument("--language", default="python", help="Code language")

    # Train command
    trn = sub.add_parser("train", help="Train classifier on labels")
    trn.add_argument("--labels", required=True, help="Path to quality_labels.jsonl")
    trn.add_argument("--output", default=str(Path(storage.checkpoints_dir) / "quality_classifier"),
                     help="Output dir")
    trn.add_argument("--base-model", default="distilbert-base-uncased", help="Base model")
    trn.add_argument("--epochs", type=int, default=5, help="Training epochs")
    trn.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    trn.add_argument("--batch-size", type=int, default=16, help="Batch size")

    # Evaluate command
    evl = sub.add_parser("evaluate", help="Evaluate trained classifier")
    evl.add_argument("--model", required=True, help="Path to trained model")
    evl.add_argument("--labels", required=True, help="Path to quality_labels.jsonl")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
