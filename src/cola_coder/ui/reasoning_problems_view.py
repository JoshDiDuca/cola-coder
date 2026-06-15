"""Reasoning-problems endpoint helper for the local UI.

Read-only browser over the GRPO/reasoning problem set. Mirrors the reasoning
training pipeline, which builds its problems through
:func:`cola_coder.evaluation.problem_loader.load_problem_set` — the same
``ProblemSet`` of :class:`cola_coder.evaluation.humaneval.CodingProblem`
instances that ``scripts/train_reasoning.py`` trains on (20 original /
62 extended built-ins, plus optional JSONL custom problems).

This view only *enumerates* problems (truncating prompts to a single-line
preview); it never executes test code or runs generation. Robust to import or
load failures: returns an ``{"error": ...}`` dict, never raises.

The model keys are snake_case to match the project's TS⇄Python schema bar.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Map the public ``which`` selector onto load_problem_set's ``source`` arg.
# "all"/"extended" -> 62 built-ins; "builtin" -> original 20; "curriculum"
# -> 62 built-ins sorted easy→medium→hard.
_PROMPT_PREVIEW_CHARS = 160


def _prompt_preview(prompt: str) -> str:
    """Collapse a problem prompt to a single truncated line for display."""
    single_line = " ".join(prompt.split())
    if len(single_line) > _PROMPT_PREVIEW_CHARS:
        return single_line[:_PROMPT_PREVIEW_CHARS].rstrip() + "…"
    return single_line


def reasoning_problems(which: str = "all") -> dict:
    """Enumerate the reasoning/GRPO problem set as a serializable dict.

    Args:
        which: Which set to load. One of ``"all"``/``"extended"`` (62 built-ins),
            ``"builtin"`` (original 20), or ``"curriculum"`` (62 sorted
            easy→medium→hard). Anything else falls back to the extended set.

    Returns:
        A dict matching :class:`cola_coder.ui.schemas.ReasoningProblemSet`, or
        ``{"error": ...}`` on genuine failure (never raises).
    """
    try:
        from cola_coder.evaluation.problem_loader import load_problem_set
    except Exception as exc:  # import-time failure (missing dep, etc.)
        logger.warning("reasoning_problems: import failed: %r", exc)
        return {"error": f"problem loader unavailable: {exc}"}

    selector = which.lower()
    if selector == "builtin":
        source = "builtin"
        curriculum = False
    elif selector == "curriculum":
        source = "extended"
        curriculum = True
    else:  # "all", "extended", or any unknown selector
        source = "extended"
        curriculum = False

    try:
        problem_set = load_problem_set(source=source, curriculum=curriculum)
    except Exception as exc:  # malformed problems, bad source, etc.
        logger.warning("reasoning_problems: load failed for %r: %r", which, exc)
        return {"error": str(exc)}

    problems: list[dict] = []
    difficulties: set[str] = set()
    languages: set[str] = set()

    for problem in problem_set:
        difficulty = problem.difficulty or "unknown"
        language = problem.language or "unknown"
        difficulties.add(difficulty)
        languages.add(language)
        problems.append(
            {
                "id": problem.task_id,
                "difficulty": difficulty,
                "language": language,
                "prompt_preview": _prompt_preview(problem.prompt),
                "has_tests": bool(problem.test_code.strip()),
            }
        )

    return {
        "problems": problems,
        "count": len(problems),
        "difficulties": sorted(difficulties),
        "languages": sorted(languages),
    }
