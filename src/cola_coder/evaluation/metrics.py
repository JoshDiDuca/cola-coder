"""Evaluation metrics for code generation.

The primary metric is pass@k: generate k code samples for each problem,
and report what fraction of problems have at least one correct solution
among the k samples.

pass@1: "Does the model get it right on the first try?"
pass@10: "If the model gets 10 attempts, does at least one work?"

Higher k is easier to achieve — even a weak model might get lucky with
enough attempts. pass@1 is the most meaningful for practical use.

The unbiased estimator for pass@k (from the original paper):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples, C = combinations.
This avoids bias from naively computing "fraction with at least one correct."
"""

from dataclasses import dataclass


@dataclass
class ProblemResult:
    """Results for a single coding problem."""
    task_id: str
    num_samples: int  # Total number of generated solutions (n)
    num_correct: int  # How many passed all tests (c)

    @property
    def pass_rate(self) -> float:
        """Simple pass rate (fraction of correct solutions)."""
        if self.num_samples == 0:
            return 0.0
        return self.num_correct / self.num_samples


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased pass@k estimator.

    This is the proper statistical way to compute pass@k,
    avoiding bias from naive estimation.

    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: The k in pass@k.

    Returns:
        Estimated probability of getting at least one correct in k samples.
    """
    if n - c < k:
        return 1.0

    # Use logarithms to avoid overflow with large combinations
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # = 1 - prod((n-c-i) / (n-i) for i in range(k))
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def compute_pass_at_k(
    results: list[ProblemResult],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Compute pass@k for multiple k values across all problems.

    Args:
        results: List of per-problem results.
        k_values: Which k values to compute (e.g., [1, 5, 10]).

    Returns:
        Dictionary mapping "pass@k" to the score (0.0 to 1.0).
    """
    metrics = {}

    for k in k_values:
        scores = []
        for r in results:
            if r.num_samples >= k:
                scores.append(pass_at_k(r.num_samples, r.num_correct, k))
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)
        else:
            metrics[f"pass@{k}"] = 0.0

    return metrics


def format_results(
    results: list[ProblemResult],
    k_values: list[int] = [1, 5, 10],
) -> str:
    """Format evaluation results as a readable table.

    Args:
        results: List of per-problem results.
        k_values: Which k values to report.

    Returns:
        Formatted string with results table.
    """
    metrics = compute_pass_at_k(results, k_values)

    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        "",
    ]

    # Overall metrics
    for key, value in metrics.items():
        lines.append(f"  {key}: {value:.1%}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"{'Problem':<30} {'Correct':>10} {'Total':>10} {'Rate':>10}")
    lines.append("-" * 60)

    # Per-problem breakdown
    for r in results:
        lines.append(
            f"  {r.task_id:<28} {r.num_correct:>10} {r.num_samples:>10} "
            f"{r.pass_rate:>9.1%}"
        )

    lines.append("-" * 60)
    lines.append(f"  {'TOTAL':<28} "
                 f"{sum(r.num_correct for r in results):>10} "
                 f"{sum(r.num_samples for r in results):>10}")
    lines.append("=" * 60)

    return "\n".join(lines)
