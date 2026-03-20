"""Router Evaluation Suite: measure domain router accuracy and quality.

Evaluates whether the router model sends prompts to the right specialist.
Computes overall accuracy, per-domain precision/recall/F1, and a confusion
matrix from a set of (predicted_domain, actual_domain, confidence) triples.

Self-contained — no ML dependencies required to run the core metrics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

KNOWN_DOMAINS: list[str] = [
    "react",
    "nextjs",
    "graphql",
    "prisma",
    "zod",
    "testing",
    "python",
    "general_ts",
]


# ---------------------------------------------------------------------------
# EvalDataset type
# ---------------------------------------------------------------------------

class EvalSample(NamedTuple):
    prompt: str
    expected_domain: str


# EvalDataset is simply a list of EvalSample tuples.
EvalDataset = list[EvalSample]


def create_test_dataset() -> EvalDataset:
    """Return a small built-in labeled dataset for smoke-testing the router."""
    return [
        EvalSample(
            prompt="import React, { useState } from 'react';\nconst Counter = () => {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(count + 1)}>{count}</button>;\n};",
            expected_domain="react",
        ),
        EvalSample(
            prompt="import { GetServerSideProps } from 'next';\nexport const getServerSideProps: GetServerSideProps = async (ctx) => ({ props: {} });\nexport default function Page() { return <div>Hello</div>; }",
            expected_domain="nextjs",
        ),
        EvalSample(
            prompt="const GET_USER = `\n  query GetUser($id: ID!) {\n    user(id: $id) { name email }\n  }\n`;",
            expected_domain="graphql",
        ),
        EvalSample(
            prompt="import { PrismaClient } from '@prisma/client';\nconst prisma = new PrismaClient();\nconst user = await prisma.user.findUnique({ where: { id: 1 } });",
            expected_domain="prisma",
        ),
        EvalSample(
            prompt="import { z } from 'zod';\nconst UserSchema = z.object({ name: z.string(), age: z.number().positive() });\ntype User = z.infer<typeof UserSchema>;",
            expected_domain="zod",
        ),
        EvalSample(
            prompt="import { render, screen } from '@testing-library/react';\ndescribe('Button', () => {\n  it('renders label', () => {\n    render(<button>Click me</button>);\n    expect(screen.getByText('Click me')).toBeInTheDocument();\n  });\n});",
            expected_domain="testing",
        ),
        EvalSample(
            prompt="def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
            expected_domain="python",
        ),
        EvalSample(
            prompt="function add(a: number, b: number): number {\n  return a + b;\n}",
            expected_domain="general_ts",
        ),
        # A few ambiguous / cross-domain examples with canonical labels
        EvalSample(
            prompt="import React from 'react';\nimport { GetServerSideProps } from 'next';\nexport const getServerSideProps: GetServerSideProps = async () => ({ props: {} });\nconst Page: React.FC = () => <div />;",
            expected_domain="nextjs",
        ),
        EvalSample(
            prompt="import { z } from 'zod';\nimport { PrismaClient } from '@prisma/client';\nconst schema = z.object({ name: z.string() });\nconst prisma = new PrismaClient();",
            expected_domain="prisma",
        ),
    ]


# ---------------------------------------------------------------------------
# RouterEvaluator
# ---------------------------------------------------------------------------

class RouterEvaluator:
    """Accumulate routing decisions and compute evaluation metrics.

    Usage::

        evaluator = RouterEvaluator()
        for pred, actual, conf in routing_log:
            evaluator.add_result(pred, actual, conf)
        print(evaluator.summary())
    """

    def __init__(self) -> None:
        # Each entry: (predicted_domain, actual_domain, confidence)
        self._results: list[tuple[str, str, float]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_result(self, predicted_domain: str, actual_domain: str, confidence: float) -> None:
        """Record one routing decision."""
        self._results.append((predicted_domain, actual_domain, float(confidence)))

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def accuracy(self) -> float:
        """Return fraction of routing decisions that were correct."""
        if not self._results:
            return 0.0
        correct = sum(1 for pred, actual, _ in self._results if pred == actual)
        return correct / len(self._results)

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """Return a nested dict confusion_matrix[actual][predicted] = count.

        Only domains that appear in the recorded results are included as
        rows/columns, keeping the structure compact.
        """
        # Collect all domain labels seen (preserve insertion order via dict)
        seen: dict[str, None] = {}
        for pred, actual, _ in self._results:
            seen[actual] = None
            seen[pred] = None
        domains = list(seen.keys())

        cm: dict[str, dict[str, int]] = {d: {d2: 0 for d2 in domains} for d in domains}
        for pred, actual, _ in self._results:
            if actual in cm and pred in cm[actual]:
                cm[actual][pred] += 1
        return cm

    def per_domain_metrics(self) -> dict[str, dict[str, float]]:
        """Return precision, recall, F1, and support per domain.

        Returns a dict keyed by domain name::

            {
                "react": {"precision": 0.9, "recall": 1.0, "f1": 0.947, "support": 10},
                ...
            }
        """
        if not self._results:
            return {}

        # Collect labels
        seen: dict[str, None] = {}
        for pred, actual, _ in self._results:
            seen[actual] = None
            seen[pred] = None
        domains = list(seen.keys())

        # tp, fp, fn per domain
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)
        support: dict[str, int] = defaultdict(int)

        for pred, actual, _ in self._results:
            support[actual] += 1
            if pred == actual:
                tp[actual] += 1
            else:
                fp[pred] += 1
                fn[actual] += 1

        metrics: dict[str, dict[str, float]] = {}
        for domain in domains:
            t = tp[domain]
            p_denom = t + fp[domain]
            r_denom = t + fn[domain]
            precision = t / p_denom if p_denom > 0 else 0.0
            recall = t / r_denom if r_denom > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            metrics[domain] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": float(support[domain]),
            }
        return metrics

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def macro_f1(self) -> float:
        """Unweighted average F1 across all domains that appear as actual labels."""
        dm = self.per_domain_metrics()
        if not dm:
            return 0.0
        # Only average over domains that have at least one actual example
        actual_domains = {actual for _, actual, _ in self._results}
        scores = [dm[d]["f1"] for d in actual_domains if d in dm]
        return sum(scores) / len(scores) if scores else 0.0

    def weighted_f1(self) -> float:
        """Support-weighted average F1."""
        dm = self.per_domain_metrics()
        if not dm:
            return 0.0
        actual_domains = {actual for _, actual, _ in self._results}
        total = sum(dm[d]["support"] for d in actual_domains if d in dm)
        if total == 0:
            return 0.0
        return sum(dm[d]["f1"] * dm[d]["support"] for d in actual_domains if d in dm) / total

    def mean_confidence(self) -> float:
        """Average confidence score across all recorded results."""
        if not self._results:
            return 0.0
        return sum(c for _, _, c in self._results) / len(self._results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a full summary dict with all computed metrics."""
        return {
            "total_samples": len(self._results),
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "weighted_f1": self.weighted_f1(),
            "mean_confidence": self.mean_confidence(),
            "per_domain": self.per_domain_metrics(),
            "confusion_matrix": self.confusion_matrix(),
        }
