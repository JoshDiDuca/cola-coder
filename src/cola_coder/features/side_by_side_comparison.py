"""
Side-by-side comparison of two model outputs for evaluation.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ComparisonResult:
    model_a_name: str
    model_b_name: str
    prompt: str
    output_a: str
    output_b: str
    similarity: float
    differences: List[str] = field(default_factory=list)


class SideBySideComparator:
    """Compare two model outputs side-by-side."""

    def compare(
        self,
        prompt: str,
        output_a: str,
        output_b: str,
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonResult:
        """Compare two outputs and return a ComparisonResult."""
        sim = self.compute_similarity(output_a, output_b)
        diffs = self.find_differences(output_a, output_b)
        return ComparisonResult(
            model_a_name=name_a,
            model_b_name=name_b,
            prompt=prompt,
            output_a=output_a,
            output_b=output_b,
            similarity=sim,
            differences=diffs,
        )

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute a 0-1 similarity score between two texts using SequenceMatcher."""
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()

    def find_differences(self, text_a: str, text_b: str) -> List[str]:
        """Return a list of human-readable difference descriptions."""
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        diffs: List[str] = []
        for group in difflib.SequenceMatcher(None, lines_a, lines_b).get_grouped_opcodes(n=3):
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    continue
                if tag == "replace":
                    removed = "".join(lines_a[i1:i2]).rstrip("\n")
                    added = "".join(lines_b[j1:j2]).rstrip("\n")
                    diffs.append(f"Changed lines {i1+1}-{i2}: {removed!r} -> {added!r}")
                elif tag == "delete":
                    removed = "".join(lines_a[i1:i2]).rstrip("\n")
                    diffs.append(f"Deleted lines {i1+1}-{i2}: {removed!r}")
                elif tag == "insert":
                    added = "".join(lines_b[j1:j2]).rstrip("\n")
                    diffs.append(f"Inserted after line {i1}: {added!r}")
        return diffs

    def format_comparison(self, result: ComparisonResult) -> str:
        """Format a ComparisonResult as a readable text block."""
        sep = "=" * 60
        half_sep = "-" * 60
        lines = [
            sep,
            f"PROMPT: {result.prompt}",
            sep,
            f"[{result.model_a_name}]",
            half_sep,
            result.output_a,
            "",
            f"[{result.model_b_name}]",
            half_sep,
            result.output_b,
            "",
            half_sep,
            f"Similarity: {result.similarity:.4f}",
            f"Differences ({len(result.differences)}):",
        ]
        for diff in result.differences:
            lines.append(f"  - {diff}")
        lines.append(sep)
        return "\n".join(lines)

    def batch_compare(
        self,
        prompts: List[str],
        outputs_a: List[str],
        outputs_b: List[str],
        name_a: str = "A",
        name_b: str = "B",
    ) -> List[ComparisonResult]:
        """Compare multiple prompt/output pairs in batch."""
        results = []
        for prompt, out_a, out_b in zip(prompts, outputs_a, outputs_b):
            results.append(self.compare(prompt, out_a, out_b, name_a=name_a, name_b=name_b))
        return results

    def summary(self, results: List[ComparisonResult]) -> dict:
        """Return aggregate statistics over a list of ComparisonResults."""
        if not results:
            return {
                "count": 0,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "total_differences": 0,
                "avg_differences": 0.0,
            }
        similarities = [r.similarity for r in results]
        diff_counts = [len(r.differences) for r in results]
        return {
            "count": len(results),
            "avg_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
            "total_differences": sum(diff_counts),
            "avg_differences": sum(diff_counts) / len(diff_counts),
        }
