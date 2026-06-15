"""Regression-history endpoint helper for the local cola-coder UI.

Read-only viewer of *past* quality-regression artifacts — the persisted output
of the CLI ``scripts/regression_test.py``. It NEVER runs the regression suite
(that loads a checkpoint onto the GPU the live trainer is using); it only scans
the filesystem for JSON reports that were written earlier, newest-first.

Where the artifacts come from (verified against the real script)
---------------------------------------------------------------
``scripts/regression_test.py --save <path.json>`` is the only thing that
persists results, and the ``<path>`` is user-chosen (no fixed dir). It writes a
JSON dict shaped exactly::

    {"checkpoint": str,
     "total": int, "passed": int, "failed": int, "pass_rate": float,
     "details": [{"description": str, "category": str, "passed": bool,
                  "output": str, "failures": list[str]}, ...]}

(see ``scripts/regression_test.py`` lines 251-264 and
``cola_coder.evaluation.regression.RegressionResult``.)

Mapping to the UI model (snake_case — note the deliberate adaptation)
---------------------------------------------------------------------
The requested ``RegressionMetric`` carries numeric ``value`` / ``baseline`` /
``delta`` fields, but the real artifact's per-baseline ``details`` are *string
pattern* pass/fail checks — they do NOT persist a numeric value or a per-metric
baseline. So each baseline detail becomes one :class:`RegressionMetric` with:

- ``name``     = the baseline ``description`` (prefixed with its ``category``);
- ``value`` / ``baseline`` / ``delta`` = ``None`` (no numeric value exists in
  the artifact — these stay null rather than being fabricated);
- ``regressed`` = ``not passed`` (a failing baseline is a regression).

The genuinely numeric run-level signals (``pass_rate``, ``passed``, ``total``)
are surfaced as the FIRST few ``metrics`` rows so the panel shows real numbers:
``pass_rate`` carries ``value`` = the run's rate (``baseline``/``delta`` null,
``regressed`` = whole-run fail), ``passed``/``total`` carry their counts.

The conventional drop spots are scanned (``regression``, ``reports``,
``regressions``, ``results``, ``eval_results`` and the project root, name-hinted)
plus ``checkpoints/<model>/`` (a natural place to save a report next to its
checkpoint). All functions are best-effort and never raise: a genuinely broken
discovery returns ``{"error": ...}``; finding no artifacts is NOT an error
(``{"runs": [], "count": 0}`` is returned).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Conventional directories a user might point ``--save`` at, relative to root.
_SCAN_DIRS: tuple[str, ...] = (
    ".",
    "regression",
    "regressions",
    "reports",
    "results",
    "eval_results",
)

# Checkpoint roots — a report may be saved alongside the checkpoint it measured.
_CKPT_DIRS: tuple[str, ...] = ("checkpoints",)

# Filename substrings that mark a file as a likely regression artifact. Keeps the
# project-root scan cheap and avoids parsing every unrelated JSON.
_NAME_HINTS: tuple[str, ...] = ("regression", "regress", "results_v")


def _read_json(path: Path) -> object | None:
    """Parse a JSON file, or return ``None`` on any read/decode failure."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _as_float(value: object) -> float | None:
    """Coerce a JSON value to ``float``, or ``None`` (bool excluded)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object, *, default: int) -> int:
    """Return ``value`` as int (excluding bool), else ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_str_or_none(value: object) -> str | None:
    """Return ``value`` when it is a non-empty string, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _is_regression_report(raw: object) -> bool:
    """True if a parsed payload looks like a ``regression_test.py --save`` dict.

    The telltale shape is a dict carrying a ``details`` list together with the
    aggregate ``passed`` and ``total`` counts — exactly what the CLI serializes.
    This filters out unrelated JSON (configs, metadata, benchmark reports, etc.).
    """
    if not isinstance(raw, dict):
        return False
    if not isinstance(raw.get("details"), list):
        return False
    return "passed" in raw and "total" in raw


def _detail_metric(detail: dict) -> dict | None:
    """Map one persisted baseline ``detail`` to a ``RegressionMetric`` dict.

    The per-baseline checks are string-pattern pass/fail (no numeric value in the
    artifact), so ``value``/``baseline``/``delta`` are null and ``regressed`` is
    simply ``not passed``. Returns ``None`` for a malformed entry.
    """
    if not isinstance(detail, dict):
        return None
    description = _as_str_or_none(detail.get("description")) or "baseline"
    category = _as_str_or_none(detail.get("category"))
    name = f"{category}: {description}" if category else description
    passed = bool(detail.get("passed"))
    return {
        "name": name,
        "value": None,
        "baseline": None,
        "delta": None,
        "regressed": not passed,
    }


def _run_level_metrics(raw: dict, total: int, passed: int, run_passed: bool) -> list[dict]:
    """Build the numeric run-level metric rows (``pass_rate``/``passed``/``total``)."""
    pass_rate = _as_float(raw.get("pass_rate"))
    if pass_rate is None and total > 0:
        pass_rate = passed / total
    return [
        {
            "name": "pass_rate",
            "value": pass_rate,
            "baseline": None,
            "delta": None,
            "regressed": not run_passed,
        },
        {
            "name": "passed",
            "value": float(passed),
            "baseline": None,
            "delta": None,
            "regressed": False,
        },
        {
            "name": "total",
            "value": float(total),
            "baseline": None,
            "delta": None,
            "regressed": False,
        },
    ]


def _summarize(raw: dict, path: Path, mtime: float) -> dict:
    """Collapse one regression report into a single ``RegressionRun``-shaped dict."""
    details_raw = raw.get("details")
    details: list[dict] = (
        [d for d in details_raw if isinstance(d, dict)] if isinstance(details_raw, list) else []
    )
    total = _as_int(raw.get("total"), default=len(details))
    passed = _as_int(raw.get("passed"), default=sum(1 for d in details if d.get("passed")))
    # A run "passes" only when every baseline passed (no regressions).
    run_passed = total > 0 and passed == total

    metrics: list[dict] = _run_level_metrics(raw, total, passed, run_passed)
    for detail in details:
        metric = _detail_metric(detail)
        if metric is not None:
            metrics.append(metric)

    return {
        "name": path.name,
        "path": str(path),
        "checkpoint": _as_str_or_none(raw.get("checkpoint")),
        "mtime": mtime,
        "passed": run_passed,
        "metrics": metrics,
    }


def _candidate_files(root_path: Path) -> list[Path]:
    """Find JSON files that may be regression artifacts under ``root``.

    Scans the conventional output dirs (and the project root, name-filtered so
    the root scan stays cheap) plus ``checkpoints/<model>/``. Missing dirs are
    silently ignored. Duplicates (same resolved path) are de-duplicated.
    """
    seen: set[Path] = set()
    files: list[Path] = []

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(path)

    for dirname in _SCAN_DIRS:
        scan_dir = root_path / dirname
        if not scan_dir.is_dir():
            continue
        root_level = dirname == "."
        try:
            entries = [p for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".json"]
        except OSError:
            continue
        for path in entries:
            # At the project root, restrict to name-hinted files to avoid parsing
            # every stray JSON; dedicated dirs scan all JSON (content-sniffed below).
            if root_level and not any(h in path.name.lower() for h in _NAME_HINTS):
                continue
            _add(path)

    for dirname in _CKPT_DIRS:
        ckpt_root = root_path / dirname
        if not ckpt_root.is_dir():
            continue
        try:
            model_dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for model_dir in model_dirs:
            try:
                jsons = [p for p in model_dir.iterdir() if p.is_file() and p.suffix == ".json"]
            except OSError:
                continue
            for path in jsons:
                if any(h in path.name.lower() for h in _NAME_HINTS):
                    _add(path)

    return files


def regression_history(root: str = ".") -> dict:
    """Collect persisted quality-regression reports, newest first.

    Returns a :class:`~cola_coder.ui.schemas.RegressionHistory`-shaped dict::

        {"runs": [RegressionRun, ...], "count": int}

    sorted by modification time (newest first). Reads only past artifacts written
    by ``scripts/regression_test.py --save`` — it never runs the regression suite.
    On any failure returns ``{"error": "..."}`` and never raises. Finding no
    artifacts is NOT an error: ``{"runs": [], "count": 0}``.
    """
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return {"error": f"root not found: {root}"}

        runs: list[dict] = []
        for path in _candidate_files(root_path):
            raw = _read_json(path)
            if not _is_regression_report(raw):
                continue
            assert isinstance(raw, dict)  # narrowed by _is_regression_report
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            runs.append(_summarize(raw, path, mtime))

        # Newest first; path tiebreak for stable ordering.
        runs.sort(key=lambda r: (-r["mtime"], r["path"]))

        return {"runs": runs, "count": len(runs)}
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        logger.warning("regression_history scan failed: %s", exc)
        return {"error": str(exc)}
