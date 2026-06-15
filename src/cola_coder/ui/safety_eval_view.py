"""Safety-eval results endpoint helper for the local UI.

Read-only discovery of *past* safety-evaluation artifacts — the secrets /
dangerous-patterns / PII / license probes that ``scripts/safety_eval.py`` runs
(suites: ``basic``, ``extended``, ``pii``, ``license``, ``injection``). The CLI
currently only prints results to the console, so persisted artifacts may not
exist yet; this viewer never runs an evaluation (that needs the GPU the live
trainer is using). It simply scans the conventional artifact directories for
JSON files shaped like a safety run and returns them newest-first.

A recognised artifact is a JSON object carrying a ``suite`` plus either a list
of per-probe results (``probes``/``results``/``checks``) or aggregate counts
(``total``/``passed``/``failed``). Anything that does not look like a safety run
is skipped. All functions are robust to missing/malformed inputs and never
raise — an empty result (``{"runs": [], "count": 0}``) is returned when nothing
is found.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Conventional directories that may hold safety-eval JSON artifacts.
_SCAN_DIRS: tuple[str, ...] = ("reports", "eval_results", "evaluations", "evals", "safety")

# Known suite names (from cola_coder.evaluation.safety_probes.SUITES).
_KNOWN_SUITES: frozenset[str] = frozenset(
    {"basic", "extended", "pii", "license", "injection", "all"}
)

# Keys under which a list of per-probe results may be stored.
_PROBE_LIST_KEYS: tuple[str, ...] = ("probes", "results", "checks")

# Filename hints — a JSON file whose name mentions safety is preferred, but
# content sniffing is authoritative (a ``suite`` field is required regardless).
_SAFETY_HINTS: tuple[str, ...] = ("safety", "safety_eval", "safety-eval")


def safety_eval_results(root: str = ".") -> dict:
    """Scan ``root`` for persisted safety-eval artifacts and return the model dict.

    Returns a ``SafetyEvalResults``-shaped dict: ``{"runs": [...], "count": N}``
    with runs sorted newest-first by ``mtime``. Unreadable or non-safety JSON
    files are skipped. Returns an empty result when nothing is found. Never
    raises.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return {"runs": [], "count": 0}

    seen: set[str] = set()
    runs: list[dict] = []

    for candidate in _candidate_paths(root_path):
        try:
            resolved = str(candidate.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        run = _parse_run(candidate, root_path)
        if run is not None:
            runs.append(run)

    runs.sort(key=lambda entry: entry["mtime"], reverse=True)
    return {"runs": runs, "count": len(runs)}


# ── Internals ───────────────────────────────────────────────────────────────


def _candidate_paths(root_path: Path) -> list[Path]:
    """Collect ``*.json`` files from the conventional dirs + the top level."""
    candidates: list[Path] = []

    for dirname in _SCAN_DIRS:
        scan_dir = root_path / dirname
        if not scan_dir.is_dir():
            continue
        try:
            for found in scan_dir.rglob("*.json"):
                if found.is_file():
                    candidates.append(found)
        except OSError:
            continue

    # Loose top-level artifacts whose name hints at a safety run.
    try:
        for found in root_path.iterdir():
            if not found.is_file() or found.suffix.lower() != ".json":
                continue
            if any(hint in found.name.lower() for hint in _SAFETY_HINTS):
                candidates.append(found)
    except OSError:
        pass

    return candidates


def _parse_run(path: Path, root_path: Path) -> dict | None:
    """Parse one candidate into a ``SafetyEvalRun`` dict, or ``None`` if not one."""
    try:
        stat = path.stat()
    except OSError:
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        logger.debug("skipping unreadable safety artifact %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        return None

    suite = data.get("suite")
    if not isinstance(suite, str) or suite not in _KNOWN_SUITES:
        return None

    probes = _parse_probes(data, suite)
    has_counts = any(
        isinstance(data.get(key), int) for key in ("total", "passed", "failed")
    )
    # A safety run must carry either explicit per-probe results or aggregate
    # counts; otherwise a bare ``{"suite": "basic"}`` would false-positive.
    if not probes and not has_counts:
        return None

    total = _as_int(data.get("total"), default=len(probes))
    passed = _as_int(
        data.get("passed"), default=sum(1 for p in probes if p["passed"])
    )
    failed = _as_int(data.get("failed"), default=max(total - passed, 0))

    try:
        rel = os.path.relpath(str(path), str(root_path))
    except (OSError, ValueError):
        rel = str(path)

    return {
        "name": path.name,
        "path": rel,
        "checkpoint": _as_str_or_none(data.get("checkpoint")),
        "suite": suite,
        "total": total,
        "passed": passed,
        "failed": failed,
        "mtime": stat.st_mtime,
        "probes": probes,
    }


def _parse_probes(data: dict, run_suite: str) -> list[dict]:
    """Extract the per-probe ``SafetyProbe`` dicts from a run object."""
    raw: object = None
    for key in _PROBE_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            raw = value
            break
    if not isinstance(raw, list):
        return []

    probes: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        probe = _parse_probe(item, run_suite)
        if probe is not None:
            probes.append(probe)
    return probes


def _parse_probe(item: dict, run_suite: str) -> dict | None:
    """Coerce one raw probe object into a ``SafetyProbe`` dict, or ``None``."""
    name = item.get("name")
    if not isinstance(name, str):
        prompt = item.get("prompt")
        name = prompt if isinstance(prompt, str) else None
    if name is None:
        return None

    suite = item.get("suite")
    if not isinstance(suite, str):
        suite = run_suite

    passed = _as_bool(item.get("passed"))
    if passed is None:
        # Fall back to common boolean shapes: an explicit ``ok`` flag, or the
        # presence of recorded issues meaning the probe did NOT pass.
        ok = _as_bool(item.get("ok"))
        if ok is not None:
            passed = ok
        else:
            issues = item.get("issues")
            passed = not issues if isinstance(issues, list) else True

    return {
        "suite": suite,
        "name": name,
        "passed": passed,
        "detail": _as_str_or_none(item.get("detail")),
    }


def _as_int(value: object, *, default: int) -> int:
    """Return ``value`` as int (excluding bool), else ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_bool(value: object) -> bool | None:
    """Return ``value`` when it is a real bool, else ``None``."""
    return value if isinstance(value, bool) else None


def _as_str_or_none(value: object) -> str | None:
    """Return ``value`` when it is a str, else ``None``."""
    return value if isinstance(value, str) else None
