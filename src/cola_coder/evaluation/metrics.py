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

import logging
import math
import random
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    """Results for a single coding problem."""
    task_id: str
    num_samples: int  # Total number of generated solutions (n)
    num_correct: int  # How many passed all tests (c)
    # How many passed all tests AND were secure (no dangerous patterns) — the
    # "secure-correct" count for secure-pass@k (CWEval, EVAL-024). None means
    # security was not assessed for this problem (back-compat default).
    num_secure_correct: int | None = None

    def __post_init__(self) -> None:
        # secure-correct is a SUBSET of correct: a sample can't be secure-correct
        # without being correct. Catch a miswired harness early rather than
        # silently reporting secure-pass@k > pass@k.
        if (
            self.num_secure_correct is not None
            and self.num_secure_correct > self.num_correct
        ):
            raise ValueError(
                f"num_secure_correct ({self.num_secure_correct}) > num_correct "
                f"({self.num_correct}) for {self.task_id}: secure-correct must be "
                "a subset of correct."
            )

    @property
    def pass_rate(self) -> float:
        """Simple pass rate (fraction of correct solutions)."""
        if self.num_samples == 0:
            return 0.0
        return self.num_correct / self.num_samples

    @property
    def secure_pass_rate(self) -> float:
        """Fraction of samples that passed tests AND were secure.

        0.0 when there are no samples or security was not assessed.
        """
        if self.num_samples == 0 or self.num_secure_correct is None:
            return 0.0
        return self.num_secure_correct / self.num_samples


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
) -> dict[str, float | None]:
    """Compute pass@k for multiple k values across all problems.

    Only problems with at least k samples contribute to pass@k — the unbiased
    estimator is undefined for n < k (and `pass_at_k` would return a spurious
    1.0 there). Two consequences are surfaced rather than hidden:

    - If NO problem has n >= k, the metric is ``None`` ("not estimable with this
      many samples"), NOT ``0.0`` — reporting 0.0 falsely reads as total failure
      when the real cause is too few samples. Callers/formatters must handle None.
    - If SOME problems are excluded (mixed sample counts), a warning is logged
      because the average is then over an easier subset.

    Args:
        results: List of per-problem results.
        k_values: Which k values to compute (e.g., [1, 5, 10]).

    Returns:
        Dictionary mapping "pass@k" to the score (0.0-1.0), or None when the
        metric cannot be estimated (no problem had >= k samples).
    """
    metrics: dict[str, float | None] = {}

    for k in k_values:
        scores = [
            pass_at_k(r.num_samples, r.num_correct, k)
            for r in results
            if r.num_samples >= k
        ]
        excluded = len(results) - len(scores)

        if not scores:
            metrics[f"pass@{k}"] = None
            logger.warning(
                "pass@%d not estimable: no problem has >= %d samples "
                "(generate at least %d samples per problem to report it).",
                k, k, k,
            )
        else:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)
            if excluded:
                logger.warning(
                    "pass@%d averages %d/%d problems — %d excluded for having "
                    "< %d samples; the score is biased toward the easier subset.",
                    k, len(scores), len(results), excluded, k,
                )

    return metrics


def compute_secure_pass_at_k(
    results: list[ProblemResult],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float | None]:
    """Compute secure-pass@k: a sample counts only if it passed tests AND is secure.

    The 2026 secure-codegen standard (CWEval / CodeGuard+): functional correctness
    alone over-credits a model that writes working-but-insecure code. secure-pass@k
    reuses the unbiased ``pass_at_k`` estimator with ``c = num_secure_correct`` (the
    samples that both pass and are clean), so it is directly comparable to pass@k —
    the gap pass@k − secure-pass@k is the share of "correct but insecure" solutions.

    Only problems where security was assessed (``num_secure_correct is not None``)
    AND that have >= k samples contribute. Mirrors ``compute_pass_at_k``: returns
    ``None`` (not 0.0) for a k that can't be estimated, and warns when problems are
    excluded so a partial/biased average isn't read as complete.
    """
    metrics: dict[str, float | None] = {}
    assessed = [r for r in results if r.num_secure_correct is not None]
    if results and not assessed:
        logger.warning(
            "secure-pass@k not computed: no problem has a security assessment "
            "(num_secure_correct). Run the eval with the security scanner enabled."
        )
        return {f"secure-pass@{k}": None for k in k_values}
    if len(assessed) < len(results):
        logger.warning(
            "secure-pass@k assessed on %d/%d problems — %d lack a security "
            "assessment and are excluded.",
            len(assessed), len(results), len(results) - len(assessed),
        )

    for k in k_values:
        scores = [
            pass_at_k(r.num_samples, r.num_secure_correct, k)
            for r in assessed
            if r.num_samples >= k
        ]
        if not scores:
            metrics[f"secure-pass@{k}"] = None
            logger.warning(
                "secure-pass@%d not estimable: no assessed problem has >= %d "
                "samples.", k, k,
            )
        else:
            metrics[f"secure-pass@{k}"] = sum(scores) / len(scores)

    return metrics


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. ``q`` in [0, 1]."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def pass_at_k_stderr(results: list[ProblemResult], k: int) -> float | None:
    """Standard error of pass@k, treating each PROBLEM as the random unit.

    pass@k is the mean of per-problem unbiased estimates, so the standard error of
    that mean is ``stdev(per-problem scores) / sqrt(num_problems)``. This is the
    cheap CLT-style error bar; for a small problem set (cola-coder's HumanEval is
    62 problems) prefer ``bootstrap_pass_at_k`` whose interval makes no normal
    assumption (CLT is unreliable below a few hundred problems).

    Only problems with ``num_samples >= k`` contribute (same exclusion as
    ``compute_pass_at_k``). Returns ``None`` when fewer than 2 such problems exist
    (a single point has no estimable spread).
    """
    scores = [
        pass_at_k(r.num_samples, r.num_correct, k)
        for r in results
        if r.num_samples >= k
    ]
    if len(scores) < 2:
        return None
    return statistics.stdev(scores) / math.sqrt(len(scores))


def bootstrap_pass_at_k(
    results: list[ProblemResult],
    k: int,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float] | None:
    """Bootstrap confidence interval for pass@k by resampling PROBLEMS.

    Resample the eligible problems (those with ``num_samples >= k``) with
    replacement ``n_boot`` times, recompute the aggregate pass@k each draw, and
    take the percentile interval. This captures per-problem difficulty variance
    without assuming normality — the right tool on a ~62-problem set where the
    CLT is shaky (arXiv:2503.01747).

    Returns ``(point, lo, hi)`` where ``point`` is the usual aggregate pass@k and
    ``[lo, hi]`` is the ``ci`` percentile interval. Deterministic for a given
    ``seed``. Returns ``None`` when no problem has ``>= k`` samples (mirrors
    ``compute_pass_at_k``'s not-estimable semantics). With a single eligible
    problem the interval collapses to the point estimate.
    """
    scores = [
        pass_at_k(r.num_samples, r.num_correct, k)
        for r in results
        if r.num_samples >= k
    ]
    if not scores:
        return None
    m = len(scores)
    point = sum(scores) / m
    if m == 1:
        return (point, point, point)

    rng = random.Random(seed)
    boot = [
        sum(scores[rng.randrange(m)] for _ in range(m)) / m
        for _ in range(n_boot)
    ]
    boot.sort()
    alpha = (1.0 - ci) / 2.0
    return (point, _percentile(boot, alpha), _percentile(boot, 1.0 - alpha))


def paired_bootstrap_delta(
    results_a: list[ProblemResult],
    results_b: list[ProblemResult],
    k: int,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float] | None:
    """Paired bootstrap of the pass@k difference (B − A) on a shared problem set.

    Matches problems by ``task_id`` and bootstraps the per-problem differences.
    Pairing cancels per-problem difficulty, giving a far tighter interval than
    comparing two independent CIs — the correct way to ask "did B really beat A?"
    (arXiv:2411.00640). A returned interval that EXCLUDES 0 is a credible change;
    one that spans 0 is within noise.

    Only problems present in both sets with ``num_samples >= k`` in each are used.
    Returns ``(mean_delta, lo, hi)`` (positive ⇒ B better), or ``None`` when no
    paired problem qualifies. Deterministic for a given ``seed``.
    """
    by_b = {r.task_id: r for r in results_b}
    deltas = [
        pass_at_k(by_b[a.task_id].num_samples, by_b[a.task_id].num_correct, k)
        - pass_at_k(a.num_samples, a.num_correct, k)
        for a in results_a
        if a.task_id in by_b
        and a.num_samples >= k
        and by_b[a.task_id].num_samples >= k
    ]
    if not deltas:
        return None
    m = len(deltas)
    mean_delta = sum(deltas) / m
    if m == 1:
        return (mean_delta, mean_delta, mean_delta)

    rng = random.Random(seed)
    boot = [
        sum(deltas[rng.randrange(m)] for _ in range(m)) / m
        for _ in range(n_boot)
    ]
    boot.sort()
    alpha = (1.0 - ci) / 2.0
    return (mean_delta, _percentile(boot, alpha), _percentile(boot, 1.0 - alpha))


def format_results(
    results: list[ProblemResult],
    k_values: list[int] = [1, 5, 10],
    bootstrap: bool = True,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> str:
    """Format evaluation results as a readable table.

    Args:
        results: List of per-problem results.
        k_values: Which k values to report.
        bootstrap: Append a bootstrap confidence interval to each pass@k line.
        n_boot: Bootstrap resamples (only used when ``bootstrap``).
        ci: Confidence level for the interval (e.g. 0.95).
        seed: RNG seed for a reproducible interval.

    Returns:
        Formatted string with results table.
    """
    metrics = compute_pass_at_k(results, k_values)

    intervals: dict[str, tuple[float, float]] = {}
    if bootstrap:
        for k in k_values:
            boot = bootstrap_pass_at_k(results, k, n_boot=n_boot, ci=ci, seed=seed)
            if boot is not None:
                intervals[f"pass@{k}"] = (boot[1], boot[2])

    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        "",
    ]

    # Overall metrics
    for key, value in metrics.items():
        if value is None:
            lines.append(f"  {key}: n/a (need more samples)")
        elif key in intervals:
            lo, hi = intervals[key]
            lines.append(f"  {key}: {value:.1%}  [{ci:.0%} CI {lo:.1%}–{hi:.1%}]")
        else:
            lines.append(f"  {key}: {value:.1%}")

    # Secure-pass@k — only shown when at least one problem was security-assessed.
    if any(r.num_secure_correct is not None for r in results):
        for key, value in compute_secure_pass_at_k(results, k_values).items():
            if value is None:
                lines.append(f"  {key}: n/a (need more samples)")
            else:
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
