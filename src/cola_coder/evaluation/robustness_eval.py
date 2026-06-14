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

from .metrics import ProblemResult, bootstrap_pass_at_k
from .perturbations import ALL_KINDS, PerturbedProblem, perturb_docstring
from .runner import evaluate_solution, extract_function

GenerateFn = Callable[[str], str]


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
) -> RobustnessReport:
    """Measure functional robustness under semantically-preserving rewordings.

    Args:
        generate_fn: ``prompt -> generated_text`` (stub in tests, real model in CLI).
        problem_set: Iterable of ``CodingProblem`` (``ProblemSet`` qualifies).
        kinds: Perturbation kinds to apply (defaults to all).
        seed: Seed forwarded to ``perturb_docstring`` for reproducibility.
        max_new_tokens: Generation budget (forwarded to the verifier path only via
            extraction; ``generate_fn`` owns generation, so this is advisory).
        compute_ci: When True, attach a bootstrap CI on robust_pass@1.
        n_boot, ci: Bootstrap parameters (only used when ``compute_ci``).

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

        per_problem.append(
            {
                "task_id": problem.task_id,
                "clean_pass": clean_pass,
                "robust_pass": worst_pass,
                "consistent": is_consistent,
                "fragile": is_fragile,
                "num_variants": len(variants),
                "verdicts": verdicts,
            }
        )

    n = len(robust_scores)
    robust_pass_at_1 = (sum(robust_scores) / n) if n else 0.0
    consistency_rate = (consistent / n) if n else 0.0

    robust_ci: tuple[float, float, float] | None = None
    if compute_ci and n:
        # Reuse the existing bootstrap (DRY): model each problem as a single-sample
        # ProblemResult whose "correct" count is its robust verdict, then pass@1 of
        # that set IS robust_pass@1, and its bootstrap CI is the robust CI.
        results = [
            ProblemResult(
                task_id=p["task_id"],
                num_samples=1,
                num_correct=1 if p["robust_pass"] else 0,
            )
            for p in per_problem
        ]
        robust_ci = bootstrap_pass_at_k(results, k=1, n_boot=n_boot, ci=ci, seed=seed)

    return RobustnessReport(
        robust_pass_at_1=robust_pass_at_1,
        consistency_rate=consistency_rate,
        fragile_task_ids=fragile,
        per_problem=per_problem,
        robust_pass_at_1_ci=robust_ci,
        num_problems=n,
    )
