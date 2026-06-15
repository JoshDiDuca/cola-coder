"""Project-health endpoint helper for the local UI.

Mirrors the CLI ``scripts/project_health.py`` battery, but DELIBERATELY cheap and
non-blocking: a synchronous HTTP handler must never shell out to the full pytest
suite or ``ruff`` (those take tens of seconds to minutes). Instead each dimension
is derived from quick filesystem signals — file existence, glob counts, and a
content grep that the CLI already uses (``FEATURE_ENABLED`` presence).

Dimensions scored (0..1 each), then averaged into an overall score + letter grade:

- ``tests``       — tests/ present and well-populated (count vs. documented ~150).
- ``features``    — fraction of feature modules that declare ``FEATURE_ENABLED``
                    (the exact check ``project_health.py`` runs, sans subprocess).
- ``configs``     — required config YAMLs present (tiny/small/medium/4080_max/large).
- ``checkpoints`` — at least one ``step_*`` checkpoint dir exists on disk.
- ``docs``        — educational guides present under ``docs/``.
- ``scripts``     — scripts/ present and well-populated (count vs. documented ~62).

The heavyweight CLI checks (running pytest / ruff for real PASS/FAIL) are NOT
executed here; ``tests`` and ``scripts`` are presence/count proxies, noted as such
in their ``detail``. Never raises — returns ``{"error": ...}`` on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Documented project sizes from CLAUDE.md — used as denominators for count ratios.
_EXPECTED_TESTS = 150
_EXPECTED_SCRIPTS = 62
_REQUIRED_CONFIGS = ("tiny", "small", "medium", "4080_max", "large")


def _project_root() -> Path:
    """Resolve the repo root (``src/cola_coder/ui/`` → up three levels)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _grade_for(score: float) -> str:
    """Map a 0..1 overall score to a letter grade."""
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


def _score_tests(root: Path) -> tuple[float, str]:
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return 0.0, "tests/ directory missing"
    n = sum(1 for _ in tests_dir.glob("test_*.py"))
    score = min(1.0, n / _EXPECTED_TESTS) if _EXPECTED_TESTS else 0.0
    return score, f"{n} test files (presence proxy; suite not run)"


def _score_features(root: Path) -> tuple[float, str]:
    features_dir = root / "src" / "cola_coder" / "features"
    if not features_dir.is_dir():
        return 0.0, "features/ directory missing"
    skip = {"__init__.py"}
    modules = [p for p in features_dir.glob("*.py") if p.name not in skip]
    if not modules:
        return 0.0, "no feature modules found"
    missing: list[str] = []
    for mod in modules:
        try:
            content = mod.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read feature module %s: %s", mod, exc)
            missing.append(mod.name)
            continue
        if "FEATURE_ENABLED" not in content:
            missing.append(mod.name)
    have = len(modules) - len(missing)
    score = have / len(modules)
    detail = f"{have}/{len(modules)} modules declare FEATURE_ENABLED"
    if missing:
        detail += f" (missing: {', '.join(sorted(missing)[:3])})"
    return score, detail


def _score_configs(root: Path) -> tuple[float, str]:
    configs_dir = root / "configs"
    present = [name for name in _REQUIRED_CONFIGS if (configs_dir / f"{name}.yaml").is_file()]
    score = len(present) / len(_REQUIRED_CONFIGS)
    return score, f"{len(present)}/{len(_REQUIRED_CONFIGS)} required config YAMLs present"


def _score_checkpoints(root: Path) -> tuple[float, str]:
    ckpt_root = root / "checkpoints"
    if not ckpt_root.is_dir():
        return 0.0, "no checkpoints/ directory (no model trained yet)"
    step_dirs = [d for d in ckpt_root.rglob("step_*") if d.is_dir()]
    if not step_dirs:
        return 0.0, "checkpoints/ exists but contains no step_* dirs"
    return 1.0, f"{len(step_dirs)} checkpoint dir(s) on disk"


def _score_docs(root: Path) -> tuple[float, str]:
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        return 0.0, "docs/ directory missing"
    n = sum(1 for _ in docs_dir.rglob("*.md"))
    score = 1.0 if n >= 6 else (n / 6.0)
    return score, f"{n} markdown guide(s) under docs/"


def _score_scripts(root: Path) -> tuple[float, str]:
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return 0.0, "scripts/ directory missing"
    n = sum(1 for p in scripts_dir.glob("*.py") if p.name != "__init__.py")
    score = min(1.0, n / _EXPECTED_SCRIPTS) if _EXPECTED_SCRIPTS else 0.0
    return score, f"{n} CLI scripts (presence proxy; --help not run)"


def project_health() -> dict:
    """Compute a cheap, non-blocking project-health report.

    Returns a ``ProjectHealthReport``-shaped dict, or ``{"error": ...}`` on
    failure. Never raises and never runs a heavy subprocess.
    """
    try:
        root = _project_root()
        scorers: list[tuple[str, tuple[float, str]]] = [
            ("Tests", _score_tests(root)),
            ("Features", _score_features(root)),
            ("Configs", _score_configs(root)),
            ("Checkpoints", _score_checkpoints(root)),
            ("Docs", _score_docs(root)),
            ("Scripts", _score_scripts(root)),
        ]
        dimensions = [
            {"name": name, "score": round(score, 4), "detail": detail}
            for name, (score, detail) in scorers
        ]
        overall = (
            sum(d["score"] for d in dimensions) / len(dimensions) if dimensions else 0.0
        )
        grade = _grade_for(overall)
        n_strong = sum(1 for d in dimensions if d["score"] >= 0.9)
        summary = (
            f"Grade {grade} — {n_strong}/{len(dimensions)} dimensions strong "
            f"({overall * 100:.0f}% overall). Cheap filesystem checks; "
            "run scripts/project_health.py for full pytest/ruff PASS/FAIL."
        )
        return {
            "overall_score": round(overall, 4),
            "grade": grade,
            "dimensions": dimensions,
            "summary": summary,
        }
    except Exception as exc:  # never raise out of a request handler
        logger.exception("project_health failed")
        return {"error": str(exc)}
