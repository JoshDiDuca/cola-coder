"""Contamination-trust-stratified pass@k (EVAL-036).

A bare pass@k number is *unfalsifiable* once the benchmark leaks into training:
a model that memorised three answers and reasoned about none can post the same
score as one that reasoned about all three. The 2026 contamination-aware eval
standard (LiveCodeBench's "evaluation over time"; OpenAI's verbatim-reproduction
audit; SWE-ReBench's decontaminated-variant comparison) responds by reporting how
much of a score *survives decontamination* — i.e. the pass@k on the subset of
problems the model could NOT have memorised, alongside the contaminated subset.

This module is the analysis half of that method, decoupled from any model load:

1. ``score_problem_contamination`` assigns each benchmark problem a CONTINUOUS
   recoverability score (max containment of its prompt OR canonical solution in
   the training corpus) by REUSING ``DataLeakageDetector``'s exact shingle
   containment — not a re-implementation. The detector itself only emits a binary
   ``has_leakage`` verdict at a single threshold; a continuous per-problem score is
   what trust stratification needs.
2. ``contamination_tier`` buckets that score into clean / suspect / contaminated.
3. ``stratified_pass_at_k`` joins the per-problem tiers with the eval's own
   ``ProblemResult`` records and reports pass@k per tier PLUS the
   clean-minus-contaminated gap — the headline "trust delta". A large positive
   gap (high on memorised problems, low on novel ones) is the memorisation
   signature; a gap near zero means the headline score is trustworthy.

Everything here is pure text/statistics over already-collected results — no GPU,
no checkpoint, no sandbox — so it runs and tests in milliseconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cola_coder.evaluation.humaneval import CodingProblem
from cola_coder.evaluation.metrics import ProblemResult, compute_pass_at_k
from cola_coder.features.data_leakage_detector import (
    DataLeakageDetector,
    _containment,
    _shingles,
)

# Trust tiers, ordered most-trustworthy → least. "clean" problems are the ones a
# decontaminated score should rely on; "contaminated" ones inflate the headline.
TIERS: tuple[str, str, str] = ("clean", "suspect", "contaminated")

# Default containment bands (fraction of the eval doc's shingles found verbatim in
# SOME training doc). 0.80 is the established contamination flag (arXiv:2502.14425,
# matching DataLeakageDetector's default similarity_threshold); 0.50 marks the grey
# zone where a substantial fragment leaked but the whole problem may not have.
DEFAULT_SUSPECT_THRESHOLD: float = 0.50
DEFAULT_CONTAMINATED_THRESHOLD: float = 0.80


def contamination_tier(
    score: float,
    suspect_threshold: float = DEFAULT_SUSPECT_THRESHOLD,
    contaminated_threshold: float = DEFAULT_CONTAMINATED_THRESHOLD,
) -> str:
    """Bucket a continuous contamination score into a trust tier.

    Args:
        score: Max containment of the problem in the training corpus, 0.0–1.0.
        suspect_threshold: At/above this (but below contaminated) → "suspect".
        contaminated_threshold: At/above this → "contaminated".

    Returns:
        One of :data:`TIERS`.

    Raises:
        ValueError: If the thresholds are not ordered ``0 <= suspect <=
            contaminated <= 1``.
    """
    if not 0.0 <= suspect_threshold <= contaminated_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= suspect <= contaminated <= 1, got "
            f"suspect={suspect_threshold}, contaminated={contaminated_threshold}"
        )
    if score >= contaminated_threshold:
        return "contaminated"
    if score >= suspect_threshold:
        return "suspect"
    return "clean"


@dataclass
class ProblemContamination:
    """Per-problem contamination diagnosis."""

    task_id: str
    """Identifier matching the problem's :class:`ProblemResult.task_id`."""
    score: float
    """Max containment of the problem's prompt OR solution in training, 0.0–1.0."""
    tier: str
    """One of :data:`TIERS`, derived from ``score``."""
    matched_unit: str
    """Which unit leaked most: "prompt", "solution", or "none" (score 0)."""

    def summary(self) -> str:
        return f"{self.task_id}: {self.tier} (containment={self.score:.2f}, via {self.matched_unit})"


def score_problem_contamination(
    problems: Sequence[CodingProblem],
    train_docs: Sequence[str],
    shingle_size: int = 5,
    suspect_threshold: float = DEFAULT_SUSPECT_THRESHOLD,
    contaminated_threshold: float = DEFAULT_CONTAMINATED_THRESHOLD,
) -> list[ProblemContamination]:
    """Score how recoverable each benchmark problem is from the training corpus.

    For every problem, the prompt and the canonical solution are each treated as a
    separate contamination unit (a leak of EITHER inflates pass@k) and scored by
    its maximum exact-shingle containment over all training docs. The higher of the
    two is the problem's contamination score; the unit that produced it is recorded.

    Reuses :class:`DataLeakageDetector`'s ``_shingles`` / ``_containment`` so the
    containment definition stays identical to the binary decontamination path —
    this is the *continuous* companion to that detector, not a fork of it.

    Args:
        problems: Benchmark problems (need ``task_id``/``prompt``; ``canonical_solution``
            is used when present).
        train_docs: Training-corpus texts to check containment against.
        shingle_size: Character n-gram size; must match the detector's setting for
            scores to be comparable to its binary verdict (default 5).
        suspect_threshold: Forwarded to :func:`contamination_tier`.
        contaminated_threshold: Forwarded to :func:`contamination_tier`.

    Returns:
        One :class:`ProblemContamination` per input problem, in input order.

    Raises:
        ValueError: If ``shingle_size`` < 1.
    """
    if shingle_size < 1:
        raise ValueError(f"shingle_size must be >= 1, got {shingle_size}")

    train_shingle_sets = [_shingles(doc, shingle_size) for doc in train_docs]

    diagnoses: list[ProblemContamination] = []
    for problem in problems:
        units: dict[str, str] = {"prompt": problem.prompt}
        solution = getattr(problem, "canonical_solution", "")
        if solution and solution.strip():
            units["solution"] = solution

        best_score = 0.0
        best_unit = "none"
        for unit_name, text in units.items():
            unit_shingles = _shingles(text, shingle_size)
            if not unit_shingles:
                continue
            unit_best = 0.0
            for train_shingles in train_shingle_sets:
                containment = _containment(unit_shingles, train_shingles)
                if containment > unit_best:
                    unit_best = containment
                    if unit_best >= 1.0:
                        break
            if unit_best > best_score:
                best_score = unit_best
                best_unit = unit_name

        diagnoses.append(
            ProblemContamination(
                task_id=problem.task_id,
                score=best_score,
                tier=contamination_tier(
                    best_score, suspect_threshold, contaminated_threshold
                ),
                matched_unit=best_unit,
            )
        )
    return diagnoses


@dataclass
class TierPassAtK:
    """pass@k for one contamination tier."""

    tier: str
    num_problems: int
    """Problems in this tier that had a matching :class:`ProblemResult`."""
    pass_at_k: dict[int, float]
    """k → pass@k over this tier's problems (empty dict when no problems)."""


@dataclass
class StratifiedPassAtKReport:
    """Contamination-trust-stratified pass@k with a clean-vs-contaminated gap."""

    k_values: tuple[int, ...]
    overall: dict[int, float]
    """k → pass@k over ALL matched problems (the headline, contaminated number)."""
    by_tier: dict[str, TierPassAtK]
    """tier name → its :class:`TierPassAtK` (always one entry per :data:`TIERS`)."""
    trust_delta: dict[int, float | None]
    """k → clean pass@k − contaminated pass@k. ``None`` when either tier is empty.

    A large positive value is the memorisation signature (the model does much
    better on problems it could have memorised); ~0 means the headline is trusted.
    """
    unmatched_task_ids: list[str] = field(default_factory=list)
    """Contamination diagnoses with no corresponding ``ProblemResult`` (skipped)."""

    def trusted_pass_at_k(self, k: int) -> float | None:
        """Return the decontaminated (clean-subset) pass@k for ``k``.

        This is the number to quote as the contamination-resistant score. Returns
        ``None`` when no clean problems were available at this ``k``.
        """
        clean = self.by_tier["clean"]
        return clean.pass_at_k.get(k)

    def summary(self) -> str:
        parts = [f"Stratified pass@k over tiers {TIERS}:"]
        for k in self.k_values:
            overall = self.overall.get(k)
            clean = self.trusted_pass_at_k(k)
            delta = self.trust_delta.get(k)
            overall_s = "n/a" if overall is None else f"{overall:.3f}"
            clean_s = "n/a" if clean is None else f"{clean:.3f}"
            delta_s = "n/a" if delta is None else f"{delta:+.3f}"
            parts.append(
                f"  pass@{k}: headline={overall_s} clean={clean_s} trust_delta={delta_s}"
            )
        return "\n".join(parts)


def stratified_pass_at_k(
    results: Sequence[ProblemResult],
    contamination: Sequence[ProblemContamination],
    k_values: Sequence[int] = (1, 5),
) -> StratifiedPassAtKReport:
    """Join eval results with contamination tiers and report pass@k per tier.

    Args:
        results: Per-problem eval results (from the eval harness).
        contamination: Per-problem contamination diagnoses (from
            :func:`score_problem_contamination`), matched to ``results`` by
            ``task_id``.
        k_values: The ``k`` values to compute pass@k at.

    Returns:
        A :class:`StratifiedPassAtKReport` whose ``trusted_pass_at_k`` gives the
        decontaminated score and whose ``trust_delta`` exposes the memorisation gap.

    Raises:
        ValueError: If ``k_values`` is empty or contains a non-positive ``k``.
    """
    ks = tuple(k_values)
    if not ks:
        raise ValueError("k_values must contain at least one k")
    if any(k <= 0 for k in ks):
        raise ValueError(f"all k_values must be positive, got {ks}")

    results_by_id = {r.task_id: r for r in results}

    tier_results: dict[str, list[ProblemResult]] = {t: [] for t in TIERS}
    unmatched: list[str] = []
    for diag in contamination:
        result = results_by_id.get(diag.task_id)
        if result is None:
            unmatched.append(diag.task_id)
            continue
        tier_results[diag.tier].append(result)

    def _estimable(pak: dict[str, float | None]) -> dict[int, float]:
        # Keep only k whose pass@k is estimable (not None — too-few-samples).
        out: dict[int, float] = {}
        for k in ks:
            value = pak.get(f"pass@{k}")
            if value is not None:
                out[k] = value
        return out

    by_tier: dict[str, TierPassAtK] = {}
    for tier, tier_res in tier_results.items():
        pak = compute_pass_at_k(tier_res, list(ks)) if tier_res else {}
        by_tier[tier] = TierPassAtK(
            tier=tier,
            num_problems=len(tier_res),
            pass_at_k=_estimable(pak),
        )

    matched_results = [results_by_id[d.task_id] for d in contamination
                       if d.task_id in results_by_id]
    overall = _estimable(
        compute_pass_at_k(matched_results, list(ks)) if matched_results else {}
    )

    clean_pak = by_tier["clean"].pass_at_k
    contaminated_pak = by_tier["contaminated"].pass_at_k
    trust_delta: dict[int, float | None] = {}
    for k in ks:
        if k in clean_pak and k in contaminated_pak:
            trust_delta[k] = clean_pak[k] - contaminated_pak[k]
        else:
            trust_delta[k] = None

    return StratifiedPassAtKReport(
        k_values=ks,
        overall=overall,
        by_tier=by_tier,
        trust_delta=trust_delta,
        unmatched_task_ids=unmatched,
    )


def build_contamination_detector(
    train_docs: Sequence[str],
    shingle_size: int = 5,
    threshold: float = DEFAULT_CONTAMINATED_THRESHOLD,
) -> DataLeakageDetector:
    """Construct and index a :class:`DataLeakageDetector` for the binary path.

    Convenience for callers that want BOTH the continuous per-problem scores (via
    :func:`score_problem_contamination`) and the detector's binary match list over
    the same corpus, without indexing twice in two different ways. Keeps the
    detector the single source of the containment definition.

    Args:
        train_docs: Training-corpus texts to index.
        shingle_size: Character n-gram size (default 5).
        threshold: Containment threshold for the binary verdict (default 0.80).

    Returns:
        A :class:`DataLeakageDetector` already indexed on ``train_docs``.
    """
    detector = DataLeakageDetector(
        similarity_threshold=threshold, shingle_size=shingle_size
    )
    detector.index_train(train_docs)
    return detector
