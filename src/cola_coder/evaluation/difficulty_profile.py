"""Verifier-effort difficulty profiling (EVAL-026).

Adaptive best-of-N already records how much compute each prompt needed (candidates
generated + escalation rounds) before the sandbox verifier was satisfied. That
"verifier effort" is a free, OBJECTIVE, MODEL-RELATIVE difficulty label — no human
annotation, and it tracks what THIS model actually finds hard (unlike static
heuristics). 2026 curriculum-RL (E2H, arXiv:2506.06632; Self-Evolving Curriculum,
arXiv:2505.14970) needs exactly such difficulty metrics.

Two uses:
- An eval report stratified by difficulty tier (where is the model weak?).
- Difficulty tiers fed back into the GRPO curriculum (which IDEA-020's per-difficulty
  entropy floors already consume) — closing eval → curriculum with measured difficulty.

Pure logic over best-of-N results (duck-typed on .candidates_used / .solved), so it
needs no GPU/sandbox to run or test.
"""

from __future__ import annotations

from typing import Iterable

# Difficulty tiers, easy → hard, plus "unsolved" for prompts the verifier never passed.
TIERS = ("easy", "medium", "hard", "unsolved")


def verifier_effort_tier(candidates_used: int, max_candidates: int, solved: bool) -> str:
    """Classify a prompt's difficulty from how much best-of-N compute it took.

    - Not solved at all → "unsolved" (hardest; the model couldn't do it in budget).
    - Solved within ≤25% of the budget → "easy".
    - Solved within ≤50% → "medium".
    - Solved but needed more than half the budget → "hard".
    """
    if not solved:
        return "unsolved"
    frac = candidates_used / max(1, max_candidates)
    if frac <= 0.25:
        return "easy"
    if frac <= 0.5:
        return "medium"
    return "hard"


def profile_difficulty(results: Iterable, max_candidates: int) -> dict:
    """Aggregate best-of-N results into a difficulty distribution report.

    Args:
        results: iterable of BestOfNResult-like objects (need .candidates_used, .solved).
        max_candidates: the adaptive budget cap used (for tier normalization).

    Returns:
        dict with per-tier counts, total, solve_rate, and mean candidates used.
    """
    results = list(results)
    tiers = {t: 0 for t in TIERS}
    for r in results:
        tiers[verifier_effort_tier(r.candidates_used, max_candidates, r.solved)] += 1
    n = len(results)
    solved = n - tiers["unsolved"]
    return {
        "n": n,
        "tiers": tiers,
        "solve_rate": solved / n if n else 0.0,
        "mean_candidates": (
            sum(r.candidates_used for r in results) / n if n else 0.0
        ),
    }
