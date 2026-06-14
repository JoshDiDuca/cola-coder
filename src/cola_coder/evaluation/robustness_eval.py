"""Verifier-graded functional robustness evaluation (EVAL-030).

A model can ace pass@1 yet be *fragile* — solving a problem when its docstring
reads one way and failing the moment the same spec is reworded. This module
re-uses the existing sandbox verifier (``runner.evaluate_solution``) to measure
that fragility directly.

For each problem we verify the model's solution on the CLEAN docstring and on
every semantically-preserving perturbation (see ``perturbations.py``), then report:

    - robust_pass@1   — fraction solved under the WORST perturbation (per-problem
                        min over variants, averaged across problems). This is the
                        headline: it answers "how often is the model right no
                        matter how we phrase the spec?"
    - consistency_rate — fraction of problems whose pass/fail verdict is INVARIANT
                        across every variant (all pass or all fail).
    - fragile_task_ids — problems solved on the clean docstring but failing on at
                        least one mere rewording. The actionable list.

``generate_fn(prompt: str) -> str`` is injected so tests can pass a deterministic
stub with no model, and the CLI can pass a real generator. The function body is
extracted and graded exactly as ``evaluate.py`` does (DRY: same
``extract_function`` + ``evaluate_solution``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .difficulty_profile import TIERS
from .metrics import ProblemResult, bootstrap_pass_at_k
from .perturbations import ALL_KINDS, PerturbedProblem, perturb_docstring
from .runner import evaluate_solution, extract_function

GenerateFn = Callable[[str], str]

# Bucket for problems whose difficulty tier was not supplied / is unrecognized.
UNKNOWN_TIER = "unknown"


@dataclass
class RobustnessReport:
    """Aggregate robustness metrics plus a per-problem breakdown.

    ``robust_pass_at_1_ci`` is ``(point, lo, hi)`` from the bootstrap when requested,
    else ``None``. ``per_problem`` carries one dict per problem with its clean verdict,
    the worst-case (robust) verdict, the per-variant verdicts, and a fragile flag.
    """

    robust_pass_at_1: float
    consistency_rate: float
    fragile_task_ids: list[str]
    per_problem: list[dict] = field(default_factory=list)
    robust_pass_at_1_ci: tuple[float, float, float] | None = None
    num_problems: int = 0
    # EVAL-031: per-difficulty-tier breakdown. Empty unless a tier mapping was
    # supplied. Each value is a dict: {"n", "robust_pass_at_1", "consistency_rate",
    # "ci": (point, lo, hi) | None}. Keys are difficulty tier labels (see
    # ``difficulty_profile.TIERS``) plus ``UNKNOWN_TIER`` for unmapped problems.
    by_tier: dict[str, dict] = field(default_factory=dict)


def _grade(
    generate_fn: GenerateFn,
    variant: PerturbedProblem,
    max_new_tokens: int,
) -> bool:
    """Generate on this variant's prompt and return whether it passes the tests.

    The verifier is shared (``evaluate_solution``); a harness exception counts as a
    fail for that variant (a crash is not a pass) but never aborts the run.
    """
    problem = variant.problem
    try:
        generated = generate_fn(problem.prompt)
        function_code = extract_function(generated, problem.entry_point)
        passed, _ = evaluate_solution(problem, function_code)
        return bool(passed)
    except Exception:  # noqa: BLE001 — a harness crash is a fail, not a pass
        return False


def evaluate_robustness(
    generate_fn: GenerateFn,
    problem_set,
    kinds: list[str] | None = None,
    seed: int = 42,
    max_new_tokens: int = 256,
    compute_ci: bool = False,
    n_boot: int = 10_000,
    ci: float = 0.95,
    difficulty_tiers: dict[str, str] | None = None,
) -> RobustnessReport:
    """Measure functional robustness under semantically-preserving rewordings.

    Args:
        generate_fn: ``prompt -> generated_text`` (stub in tests, real model in CLI).
        problem_set: Iterable of ``CodingProblem`` (``ProblemSet`` qualifies).
        kinds: Perturbation kinds to apply (defaults to all).
        seed: Seed forwarded to ``perturb_docstring`` for reproducibility.
        max_new_tokens: Generation budget (forwarded to the verifier path only via
            extraction; ``generate_fn`` owns generation, so this is advisory).
        compute_ci: When True, attach a bootstrap CI on robust_pass@1 (overall, and
            per tier when ``difficulty_tiers`` is supplied).
        n_boot, ci: Bootstrap parameters (only used when ``compute_ci``).
        difficulty_tiers: Optional ``task_id -> tier`` mapping (EVAL-031). When
            supplied, the report's ``by_tier`` is populated with per-tier
            robust_pass@1, consistency, n, and (when ``compute_ci``) a bootstrap CI.
            Tiers come from the caller (e.g. EVAL-026 verifier-effort tiers); this
            keeps the function decoupled from best-of-N data and hermetically
            testable. A ``task_id`` missing from the mapping (or with an
            unrecognized tier) falls into the ``UNKNOWN_TIER`` bucket.

    Returns:
        A populated ``RobustnessReport``.
    """
    if kinds is None:
        kinds = list(ALL_KINDS)

    per_problem: list[dict] = []
    fragile: list[str] = []
    robust_scores: list[float] = []
    consistent = 0

    for problem in problem_set:
        variants = perturb_docstring(problem, kinds=kinds, seed=seed)
        verdicts: dict[str, bool] = {}
        for variant in variants:
            verdicts[variant.perturbation] = _grade(generate_fn, variant, max_new_tokens)

        clean_pass = verdicts.get("clean", False)
        all_verdicts = list(verdicts.values())
        worst_pass = all(all_verdicts)  # robust@1 = solved under EVERY variant
        is_consistent = len(set(all_verdicts)) == 1
        # Fragile = clean docstring passed but at least one mere rewording failed.
        is_fragile = clean_pass and not all(all_verdicts)

        robust_scores.append(1.0 if worst_pass else 0.0)
        if is_consistent:
            consistent += 1
        if is_fragile:
            fragile.append(problem.task_id)

        tier = _resolve_tier(problem.task_id, difficulty_tiers)
        per_problem.append(
            {
                "task_id": problem.task_id,
                "clean_pass": clean_pass,
                "robust_pass": worst_pass,
                "consistent": is_consistent,
                "fragile": is_fragile,
                "num_variants": len(variants),
                "verdicts": verdicts,
                "tier": tier,
            }
        )

    n = len(robust_scores)
    robust_pass_at_1 = (sum(robust_scores) / n) if n else 0.0
    consistency_rate = (consistent / n) if n else 0.0

    robust_ci: tuple[float, float, float] | None = None
    if compute_ci and n:
        robust_ci = _bootstrap_robust_ci(per_problem, n_boot=n_boot, ci=ci, seed=seed)

    by_tier: dict[str, dict] = {}
    if difficulty_tiers is not None:
        by_tier = _stratify_by_tier(
            per_problem, compute_ci=compute_ci, n_boot=n_boot, ci=ci, seed=seed
        )

    return RobustnessReport(
        robust_pass_at_1=robust_pass_at_1,
        consistency_rate=consistency_rate,
        fragile_task_ids=fragile,
        per_problem=per_problem,
        robust_pass_at_1_ci=robust_ci,
        num_problems=n,
        by_tier=by_tier,
    )


def _resolve_tier(task_id: str, difficulty_tiers: dict[str, str] | None) -> str:
    """Map a task_id to its difficulty tier, defaulting to ``UNKNOWN_TIER``.

    A missing mapping, an absent task_id, or a tier label not in
    ``difficulty_profile.TIERS`` all resolve to ``UNKNOWN_TIER`` (never crashes).
    """
    if not difficulty_tiers:
        return UNKNOWN_TIER
    tier = difficulty_tiers.get(task_id)
    return tier if tier in TIERS else UNKNOWN_TIER


def _bootstrap_robust_ci(
    rows: list[dict],
    n_boot: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float] | None:
    """Bootstrap CI on robust_pass@1 for a set of per-problem rows.

    Reuses the existing bootstrap (DRY): model each problem as a single-sample
    ``ProblemResult`` whose "correct" count is its robust verdict, so pass@1 of that
    set IS robust_pass@1 and its bootstrap CI is the robust CI.
    """
    results = [
        ProblemResult(
            task_id=row["task_id"],
            num_samples=1,
            num_correct=1 if row["robust_pass"] else 0,
        )
        for row in rows
    ]
    return bootstrap_pass_at_k(results, k=1, n_boot=n_boot, ci=ci, seed=seed)


def _stratify_by_tier(
    per_problem: list[dict],
    compute_ci: bool,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, dict]:
    """Group per-problem rows by difficulty tier and aggregate per-tier metrics.

    Tiers are emitted in ``difficulty_profile.TIERS`` order, with ``UNKNOWN_TIER``
    last; only tiers that actually have problems appear. Each entry carries ``n``,
    ``robust_pass_at_1``, ``consistency_rate`` and (when ``compute_ci``) a bootstrap
    ``ci`` of ``(point, lo, hi)`` on robust_pass@1 (``None`` otherwise).
    """
    grouped: dict[str, list[dict]] = {}
    for row in per_problem:
        grouped.setdefault(row["tier"], []).append(row)

    ordered = [t for t in (*TIERS, UNKNOWN_TIER) if t in grouped]
    out: dict[str, dict] = {}
    for tier in ordered:
        rows = grouped[tier]
        n = len(rows)
        tier_ci = (
            _bootstrap_robust_ci(rows, n_boot=n_boot, ci=ci, seed=seed)
            if compute_ci
            else None
        )
        out[tier] = {
            "n": n,
            "robust_pass_at_1": sum(1 for r in rows if r["robust_pass"]) / n,
            "consistency_rate": sum(1 for r in rows if r["consistent"]) / n,
            "ci": tier_ci,
        }
    return out
