"""Repo-scoring endpoint helper for the local cola-coder UI.

Read-only viewer of *past* repo-quality scoring artifacts — the persisted output
of the CLI ``scripts/score_repos.py``. It NEVER scores repos (that may install
deps / run tests / hit GitHub); it only scans the filesystem for JSON reports
that were written earlier and surfaces them ranked by score (best first).

Where the artifacts come from (verified against the real script):

- ``scripts/score_repos.py`` with ``--json`` prints a JSON object to stdout shaped
  as a mapping of *repo path* -> :meth:`RepoScore.to_dict` output::

      {
        "/path/to/repo": {
          "tests_detected": bool,
          "tests_pass": bool,
          "quality_tier": "verified" | "tested" | "detected" | "none",
          "score": float,                       # 0.0-1.0 composite
          "test_result": {                       # null when tests never ran
            "framework": str, "total_tests": int, "passed": int,
            "failed": int, "skipped": int, "error": str | null,
            "coverage": float | null, "duration_seconds": float
          } | null,
          "details": { "reason"?: str, ... }
        },
        ...
      }

  The script writes to stdout, so a user redirects it to a file of their choosing
  (no fixed dir). We scan the conventional drop spots — the project root,
  ``reports/``, ``results/``, ``repo_scores/`` — for JSON whose contents match the
  shape above, and pick the newest matching file.

The real scorer is *test-quality* based: it has no GitHub ``stars`` / ``license``
fields, and the only language-ish signal is the test ``framework`` (jest/pytest/
...). To fit the UI ``RepoScore`` model we map defensively:

- ``repo``     <- the mapping key (the repo path), basename surfaced as the label
- ``score``    <- ``score``
- ``stars``    <- ``details.stars`` if a future/enriched report carries it, else ``None``
- ``language`` <- ``details.language`` else the test ``framework`` else ``None``
- ``license``  <- ``details.license`` if present, else ``None``
- ``reason``   <- ``details.reason`` else the ``quality_tier`` (always informative)

All snake_case (matches the Pydantic model). Repo paths are *untrusted* scraped
metadata — they are passed through as plain strings only (never executed).

Every function is best-effort and never raises: a genuinely broken discovery
returns ``{"error": ...}``; finding no artifacts is NOT an error — an empty,
valid result is returned (``repos: []``, ``count: 0``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Conventional places a user might redirect ``score_repos.py --json`` output to,
# relative to the project root. The root is name-filtered so the scan stays cheap.
_SCAN_DIRS: tuple[str, ...] = (
    ".",
    "reports",
    "results",
    "repo_scores",
)

# Filename substrings that mark a root-level file as a likely repo-score artifact.
_NAME_HINTS: tuple[str, ...] = ("repo", "score")

# Tier keys the real scorer emits — used to validate a payload looks like ours.
_VALID_TIERS: frozenset[str] = frozenset({"verified", "tested", "detected", "none"})

# Cap on the number of repos surfaced (the report can list many repos).
_MAX_REPOS: int = 200


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


def _as_int(value: object) -> int | None:
    """Coerce a JSON value to ``int``, or ``None`` (bool excluded)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_str(value: object) -> str | None:
    """Return ``value`` if it is a non-empty string, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _is_repo_score_report(raw: object) -> bool:
    """True if a parsed payload looks like a ``score_repos.py --json`` report.

    The telltale shape is a non-empty mapping whose values are themselves dicts
    carrying a ``quality_tier`` (one of the four known tiers) and a numeric
    ``score``. This filters out unrelated JSON (configs, benchmark reports, etc.).
    """
    if not isinstance(raw, dict) or not raw:
        return False
    for entry in raw.values():
        if not isinstance(entry, dict):
            return False
        tier = entry.get("quality_tier")
        if tier not in _VALID_TIERS:
            return False
        if _as_float(entry.get("score")) is None:
            return False
    return True


def _summarize_repo(repo_path: str, entry: dict) -> dict:
    """Collapse one repo entry into a single ``RepoScore``-shaped dict.

    ``entry`` is the per-repo ``RepoScore.to_dict()`` mapping. ``repo_path`` is the
    mapping key (the scored repo's path). Field mapping is documented in the module
    docstring — defensive, since the test-quality scorer carries no stars/license.
    """
    details = entry.get("details")
    details_map: dict[str, object] = details if isinstance(details, dict) else {}

    test_result = entry.get("test_result")
    framework: str | None = None
    if isinstance(test_result, dict):
        framework = _as_str(test_result.get("framework"))

    # repo: surface the basename as the label, but keep the full path available.
    name = Path(repo_path).name or repo_path

    language = _as_str(details_map.get("language")) or framework
    reason = _as_str(details_map.get("reason")) or _as_str(entry.get("quality_tier"))

    return {
        "repo": name,
        "score": _as_float(entry.get("score")) or 0.0,
        "stars": _as_int(details_map.get("stars")),
        "language": language,
        "license": _as_str(details_map.get("license")),
        "reason": reason,
    }


def _candidate_files(root_path: Path) -> list[Path]:
    """Find JSON files that may be repo-score artifacts under ``root``.

    Scans the conventional output dirs (and the project root, name-filtered so the
    root scan stays cheap). Missing dirs are silently ignored; duplicate resolved
    paths are de-duplicated.
    """
    seen: set[Path] = set()
    files: list[Path] = []

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
            if root_level and not any(h in path.name.lower() for h in _NAME_HINTS):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)

    return files


def repo_scores(root: str = ".") -> dict:
    """Return the newest persisted repo-scoring report, ranked by score (best first).

    Returns a :class:`~cola_coder.ui.schemas.RepoScoresResult`-shaped dict::

        {"path": str, "repos": [RepoScore, ...], "count": int, "mtime": float}

    ``repos`` is sorted by ``score`` descending and capped at 200. Reads only past
    artifacts — it never scores a repo. On any failure returns ``{"error": "..."}``
    and never raises. Finding no artifacts is NOT an error: a valid empty result is
    returned (``{"path": "", "repos": [], "count": 0, "mtime": 0.0}``).
    """
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return {"error": f"root not found: {root}"}

        # Pick the newest report whose contents match the repo-score shape.
        best_path: Path | None = None
        best_mtime = -1.0
        for path in _candidate_files(root_path):
            raw = _read_json(path)
            if not _is_repo_score_report(raw):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = path

        if best_path is None:
            return {"path": "", "repos": [], "count": 0, "mtime": 0.0}

        raw = _read_json(best_path)
        assert isinstance(raw, dict)  # narrowed by _is_repo_score_report

        repos: list[dict] = [
            _summarize_repo(str(repo_path), entry)
            for repo_path, entry in raw.items()
            if isinstance(entry, dict)
        ]
        # Best score first; repo name as a stable tiebreak.
        repos.sort(key=lambda r: (-r["score"], r["repo"]))
        repos = repos[:_MAX_REPOS]

        return {
            "path": str(best_path),
            "repos": repos,
            "count": len(repos),
            "mtime": max(best_mtime, 0.0),
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        logger.warning("repo_scores scan failed: %s", exc)
        return {"error": str(exc)}
