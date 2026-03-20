# Feature 21: Router Evaluation Suite

**Status:** Optional | **CLI Flag:** `--eval-router` | **Complexity:** Medium

---

## Overview

A comprehensive evaluation harness for the domain router. Measures accuracy, per-domain precision/recall/F1, and confusion matrix on a labeled test set. Compares three routing strategies: heuristic-only (Feature 24), learned router (Feature 16), and oracle (ground truth). Generates a Rich CLI report with tables, confusion matrix, and per-strategy comparison charts. Also evaluates on deliberately ambiguous examples to stress-test robustness.

---

## Motivation

Without systematic evaluation it is impossible to know:
- Whether the learned router actually outperforms the simple heuristic
- Which domains are confused with each other
- What the optimal confidence threshold is for each domain
- Whether router quality degrades on out-of-domain prompts

A dedicated evaluation suite enables data-driven iteration on the router architecture, training data quality, and routing thresholds.

---

## Architecture / Design

### Evaluation Inputs

1. **Labeled test set:** JSONL file with `{text, domain}` pairs, held out from the training set (Feature 17)
2. **Ambiguous examples:** Manually curated set of edge cases (React code with Prisma imports, etc.)
3. **Three routing strategies:**
   - `heuristic`: Feature 24 rules engine (no ML)
   - `learned`: RouterModel (Feature 16) with calibrated temperature
   - `oracle`: Always uses the ground truth label (upper bound)

### Metrics

| Metric                | Description                               |
|-----------------------|-------------------------------------------|
| Overall Accuracy      | Correct / Total predictions               |
| Per-domain Precision  | TP / (TP + FP) for each domain            |
| Per-domain Recall     | TP / (TP + FN) for each domain            |
| Per-domain F1         | Harmonic mean of precision and recall     |
| Macro F1              | Unweighted average of per-domain F1       |
| Weighted F1           | Domain-frequency weighted average F1      |
| Confusion Matrix      | N×N counts of (predicted, actual) pairs   |
| Fallback Rate         | % of predictions below confidence threshold |
| Mean Confidence       | Average router softmax confidence         |

---

## Implementation Steps

### Step 1: Evaluation Dataset Loader

```python
# cola_coder/eval/router_eval_dataset.py
import json
from dataclasses import dataclass
from typing import Iterator

@dataclass
class EvalSample:
    text: str
    true_domain: str
    source: str = ""
    is_ambiguous: bool = False


def load_eval_dataset(
    jsonl_path: str,
    ambiguous_path: str = None,
) -> tuple[list[EvalSample], list[EvalSample]]:
    """Load standard and ambiguous eval sets."""
    standard = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            standard.append(EvalSample(
                text=record["text"],
                true_domain=record["domain"],
                source=record.get("source", ""),
            ))

    ambiguous = []
    if ambiguous_path:
        with open(ambiguous_path) as f:
            for line in f:
                record = json.loads(line)
                ambiguous.append(EvalSample(
                    text=record["text"],
                    true_domain=record["domain"],
                    source=record.get("source", ""),
                    is_ambiguous=True,
                ))

    return standard, ambiguous
```

### Step 2: RouterEvaluator Class

```python
# cola_coder/eval/router_evaluator.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable
from collections import defaultdict

DOMAIN_NAMES = ["react", "nextjs", "graphql", "prisma", "zod", "testing", "general_ts"]

@dataclass
class DomainMetrics:
    domain: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of true examples for this domain

@dataclass
class EvalReport:
    strategy_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_domain: list[DomainMetrics]
    confusion_matrix: np.ndarray
    fallback_rate: float
    mean_confidence: float
    total_samples: int


class RouterEvaluator:
    def __init__(self, domain_names: list[str] = None):
        self.domains = domain_names or DOMAIN_NAMES
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}
        self.n = len(self.domains)

    def evaluate(
        self,
        samples: list,
        predict_fn: Callable[[str], tuple[str, float]],
        strategy_name: str,
    ) -> EvalReport:
        """
        predict_fn: (text) → (predicted_domain, confidence)
        Returns EvalReport with all metrics.
        """
        true_labels = []
        pred_labels = []
        confidences = []
        fallbacks = 0

        for sample in samples:
            pred_domain, confidence = predict_fn(sample.text)
            true_labels.append(self.domain_to_idx.get(sample.true_domain, -1))
            pred_labels.append(self.domain_to_idx.get(pred_domain, -1))
            confidences.append(confidence)
            if pred_domain == "general_ts" and sample.true_domain != "general_ts":
                fallbacks += 1

        true_arr = np.array(true_labels)
        pred_arr = np.array(pred_labels)

        # Accuracy
        accuracy = (true_arr == pred_arr).mean()

        # Confusion matrix
        cm = np.zeros((self.n, self.n), dtype=int)
        for t, p in zip(true_arr, pred_arr):
            if 0 <= t < self.n and 0 <= p < self.n:
                cm[t, p] += 1

        # Per-domain metrics
        per_domain = []
        for i, domain in enumerate(self.domains):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_domain.append(DomainMetrics(domain, precision, recall, f1, int(support)))

        macro_f1 = np.mean([m.f1 for m in per_domain])
        total = sum(m.support for m in per_domain)
        weighted_f1 = sum(m.f1 * m.support for m in per_domain) / total if total > 0 else 0.0

        return EvalReport(
            strategy_name=strategy_name,
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            per_domain=per_domain,
            confusion_matrix=cm,
            fallback_rate=fallbacks / len(samples) if samples else 0.0,
            mean_confidence=np.mean(confidences),
            total_samples=len(samples),
        )
```

### Step 3: Rich Report Generator

```python
# cola_coder/eval/report.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

console = Console()

def print_eval_report(report: EvalReport):
    """Print a formatted evaluation report to the terminal."""
    # Summary panel
    console.print(Panel(
        f"[bold]Strategy:[/bold] {report.strategy_name}  "
        f"[bold]Accuracy:[/bold] {report.accuracy:.3f}  "
        f"[bold]Macro F1:[/bold] {report.macro_f1:.3f}  "
        f"[bold]Weighted F1:[/bold] {report.weighted_f1:.3f}\n"
        f"[bold]Fallback Rate:[/bold] {report.fallback_rate:.1%}  "
        f"[bold]Mean Confidence:[/bold] {report.mean_confidence:.3f}  "
        f"[bold]Samples:[/bold] {report.total_samples}",
        title=f"[bold cyan]Router Eval: {report.strategy_name}[/bold cyan]"
    ))

    # Per-domain table
    table = Table(title="Per-Domain Metrics", show_lines=True)
    table.add_column("Domain", style="cyan", width=14)
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")
    table.add_column("Quality", width=12)

    for m in sorted(report.per_domain, key=lambda x: -x.f1):
        bar_len = int(m.f1 * 10)
        bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (10 - bar_len)
        table.add_row(
            m.domain,
            f"{m.precision:.3f}",
            f"{m.recall:.3f}",
            f"{m.f1:.3f}",
            str(m.support),
            bar,
        )
    console.print(table)
    print_confusion_matrix(report.confusion_matrix, report.per_domain)


def print_confusion_matrix(cm: np.ndarray, per_domain: list):
    """Print ASCII confusion matrix with domain labels."""
    domains = [m.domain[:6] for m in per_domain]
    console.print("\n[bold]Confusion Matrix (rows=true, cols=predicted):[/bold]")
    header = "       " + "  ".join(f"{d:>6}" for d in domains)
    console.print(header)
    for i, row_domain in enumerate(domains):
        row_str = f"{row_domain:>6} "
        for j, val in enumerate(cm[i]):
            if i == j:
                row_str += f"  [green]{val:4d}[/green]"
            elif val > 0:
                row_str += f"  [red]{val:4d}[/red]"
            else:
                row_str += f"  {val:4d}"
        console.print(row_str)


def print_strategy_comparison(reports: list[EvalReport]):
    """Compare multiple strategies side by side."""
    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Accuracy")
    table.add_column("Macro F1")
    table.add_column("Weighted F1")
    table.add_column("Fallback Rate")
    table.add_column("Mean Conf")
    table.add_column("vs Oracle", justify="right")

    oracle = next((r for r in reports if r.strategy_name == "oracle"), None)
    for r in reports:
        gap = (f"[red]-{(oracle.accuracy - r.accuracy):.3f}[/red]"
               if oracle and r.strategy_name != "oracle" else "-")
        table.add_row(
            r.strategy_name,
            f"{r.accuracy:.3f}",
            f"{r.macro_f1:.3f}",
            f"{r.weighted_f1:.3f}",
            f"{r.fallback_rate:.1%}",
            f"{r.mean_confidence:.3f}",
            gap,
        )
    console.print(table)
```

### Step 4: Ambiguous Example Generator

```python
# cola_coder/eval/ambiguous_examples.py

AMBIGUOUS_EXAMPLES = [
    # Next.js + React overlap
    {
        "text": "import React from 'react';\nimport { GetServerSideProps } from 'next';\n"
                "export const getServerSideProps: GetServerSideProps = async () => ({ props: {} });\n"
                "const Page: React.FC = () => <div>Hello</div>;",
        "domain": "nextjs",  # Next.js wins due to getServerSideProps
    },
    # Prisma + Zod overlap (form validation + DB)
    {
        "text": "import { z } from 'zod';\nimport { PrismaClient } from '@prisma/client';\n"
                "const schema = z.object({ name: z.string() });\n"
                "const prisma = new PrismaClient();",
        "domain": "prisma",  # Primary domain: DB access
    },
    # Testing + React (RTL)
    {
        "text": "import { render, screen } from '@testing-library/react';\n"
                "import { Button } from './Button';\n"
                "describe('Button', () => { it('renders', () => { render(<Button />); }); });",
        "domain": "testing",  # Testing wins: describe/it/render from testing library
    },
    # GraphQL + Apollo (looks like general TS)
    {
        "text": "const GET_USER = `\n  query GetUser($id: ID!) {\n    user(id: $id) { name email }\n  }\n`;",
        "domain": "graphql",
    },
]

def save_ambiguous_examples(path: str):
    import json
    with open(path, "w") as f:
        for ex in AMBIGUOUS_EXAMPLES:
            f.write(json.dumps(ex) + "\n")
```

### Step 5: CLI Command

```python
@app.command()
def eval_router(
    test_data: str = typer.Argument(..., help="Path to test JSONL"),
    router_checkpoint: str = typer.Option(None, "--router-ckpt"),
    ambiguous_data: str = typer.Option(None, "--ambiguous"),
    compare_heuristic: bool = typer.Option(True, "--compare-heuristic"),
    output_json: str = typer.Option(None, "--output-json"),
):
    """Evaluate router accuracy and compare strategies."""
    from cola_coder.eval import RouterEvaluator, load_eval_dataset, print_strategy_comparison

    standard, ambiguous = load_eval_dataset(test_data, ambiguous_data)
    evaluator = RouterEvaluator()
    reports = []

    if router_checkpoint:
        router = load_router(router_checkpoint)
        def learned_predict(text):
            ids = tokenizer.encode(text, max_length=128, return_tensors="pt")
            domain_id, probs = router.predict(ids)
            return DOMAIN_NAMES[domain_id], probs[0, domain_id].item()
        reports.append(evaluator.evaluate(standard, learned_predict, "learned"))

    if compare_heuristic:
        from cola_coder.data.domain_classifier import classify_domain
        def heuristic_predict(text):
            result = classify_domain(text)
            if result:
                domain, conf = result
            else:
                domain, conf = "general_ts", 0.5
            return domain, conf
        reports.append(evaluator.evaluate(standard, heuristic_predict, "heuristic"))

    # Oracle (always correct)
    def oracle_predict(sample_text):
        # Cheat: look up true label — used via closure in full implementation
        return "oracle_domain", 1.0
    # reports.append(evaluator.evaluate(standard, oracle_predict, "oracle"))

    print_strategy_comparison(reports)
    for r in reports:
        print_eval_report(r)

    if ambiguous:
        console.print("\n[bold yellow]Ambiguous Examples:[/bold yellow]")
        for r_name, predict_fn in [("learned", learned_predict), ("heuristic", heuristic_predict)]:
            correct = sum(
                1 for s in ambiguous
                if predict_fn(s.text)[0] == s.true_domain
            )
            console.print(f"  {r_name}: {correct}/{len(ambiguous)} ambiguous correct")
```

---

## Key Files to Modify

- `cola_coder/eval/__init__.py` — new package
- `cola_coder/eval/router_evaluator.py` — evaluation metrics
- `cola_coder/eval/router_eval_dataset.py` — dataset loading
- `cola_coder/eval/report.py` — Rich report generation
- `cola_coder/eval/ambiguous_examples.py` — curated edge cases
- `cola_coder/cli.py` — `eval-router` command
- `data/eval/router_test.jsonl` — held-out test data
- `data/eval/router_ambiguous.jsonl` — ambiguous examples

---

## Testing Strategy

```python
def test_evaluator_perfect_score():
    evaluator = RouterEvaluator()
    samples = [EvalSample("code", "react") for _ in range(10)]
    report = evaluator.evaluate(samples, lambda _: ("react", 0.9), "test")
    assert report.accuracy == 1.0
    assert abs(report.macro_f1 - 1.0) < 0.01  # Only 1 domain present, F1=1.0

def test_evaluator_confusion_matrix_shape():
    evaluator = RouterEvaluator()
    samples = [EvalSample("code", d) for d in ["react", "prisma", "zod"]]
    report = evaluator.evaluate(samples, lambda _: ("general_ts", 0.3), "test")
    assert report.confusion_matrix.shape == (7, 7)

def test_evaluator_fallback_rate():
    evaluator = RouterEvaluator()
    samples = [EvalSample("code", "react") for _ in range(10)]
    # Predict general_ts for all → fallback rate = 100%
    report = evaluator.evaluate(samples, lambda _: ("general_ts", 0.3), "test")
    assert report.fallback_rate == 1.0
```

---

## Performance Considerations

- **Batch evaluation:** For learned router, batch all test samples through the model in one pass rather than one-by-one. Use DataLoader with batch_size=256.
- **Cached tokenization:** Tokenize all samples once before evaluation loop starts.
- **Large test sets:** For 100k+ samples, stream results and compute confusion matrix incrementally rather than storing all predictions in memory.
- **Parallelism:** Heuristic evaluation is CPU-bound and embarrassingly parallel; use `multiprocessing.Pool`.

---

## Dependencies

- Feature 16 (RouterModel) — for learned strategy
- Feature 17 (training data generator) — source of test set
- Feature 24 (domain detection heuristic) — for heuristic strategy comparison
- `rich` — for formatted tables and panels
- `numpy` — for confusion matrix computation

---

## Estimated Complexity

| Task                        | Effort  |
|-----------------------------|---------|
| EvalSample + dataset loader | 1h      |
| RouterEvaluator metrics     | 3h      |
| Rich report generation      | 2h      |
| Ambiguous examples          | 1h      |
| CLI command + integration   | 1.5h    |
| Tests                       | 1.5h    |
| **Total**                   | **~10h** |

Overall complexity: **Medium** (mostly metrics math and visualization)

---

## 2026 Best Practices

- **Macro vs Weighted F1:** Report both. Macro F1 reveals per-domain failures; weighted F1 reflects overall user experience. For production decisions, use weighted F1.
- **Bootstrap confidence intervals:** For small test sets (<1000 samples), report 95% CI on accuracy/F1 using bootstrap resampling to avoid overfitting to a particular split.
- **Stratified sampling:** Ensure the test set has proportional representation of all domains. If General TS dominates, accuracy metric is inflated.
- **Human baseline:** If feasible, have one human label 100 samples to establish a human-performance ceiling. Router accuracy relative to human is more meaningful than absolute number.
- **Regression testing:** Save eval reports as JSON and track metrics over time. Alert if macro F1 drops more than 2% between router versions.
- **Cohen's Kappa:** Report inter-rater agreement statistic (agreement beyond chance) as a complement to raw accuracy.
