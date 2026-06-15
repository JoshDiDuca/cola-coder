"""LR-finder-results endpoint helper for the local cola-coder UI.

Read-only viewer of *past* learning-rate range-test artifacts — mirrors the CLI
``scripts/find_lr.py`` (Smith's LR Range Test). It NEVER runs the finder (that
needs the GPU the live trainer is using); it only scans the filesystem for
result files written by an earlier run.

Where the artifacts come from (verified against the real script + module):

- ``scripts/find_lr.py`` runs :class:`cola_coder.training.lr_finder.LRFinder`
  and, today, persists ONLY a PNG plot (``lr_finder_plot.png`` by default via
  ``--save-plot``); the (lr, loss) points and the suggested LR are printed to the
  console and NOT serialized to a structured file. So in the common case there is
  no machine-readable artifact and this endpoint correctly returns an empty list.
- To stay useful if a structured result is ever dropped (a user redirecting the
  summary, or a future ``--json`` flag), we ALSO scan the conventional spots for a
  JSON file shaped like the :class:`~cola_coder.training.lr_finder.LRFinderResult`
  dataclass — i.e. carrying parallel ``lrs``/``losses`` arrays (plus optional
  ``smoothed_losses``, ``suggested_lr``, ``suggested_min_lr``). Such a file is
  summarized into one :class:`LrFinderRun`-shaped dict.

All field names are snake_case to match the Pydantic UI schema 1:1. The
``points`` list is capped to keep the curve payload small. All functions are
best-effort and never raise: a genuinely broken scan returns ``{"error": ...}``;
finding no artifacts is NOT an error (an empty list is returned).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum number of (lr, loss) points kept per run for the curve payload.
_MAX_POINTS: int = 300

# Conventional directories a user might point ``--save-plot``/output at, relative
# to the project root. The PNG sits next to where a JSON result would land.
_SCAN_DIRS: tuple[str, ...] = (
    ".",
    "lr_finder",
    "results",
    "lr_finder_results",
)

# Filename substrings that mark a file as a likely LR-finder artifact. Keeps the
# root scan cheap and avoids parsing every unrelated JSON in the project root.
_NAME_HINTS: tuple[str, ...] = ("lr_finder", "lr-finder", "lrfinder", "find_lr")


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


def _as_str(value: object) -> str | None:
    """Return ``value`` if it is a non-empty string, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _float_list(value: object) -> list[float]:
    """Extract a list of floats from a JSON array, skipping non-numerics."""
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        f = _as_float(item)
        if f is not None:
            out.append(f)
    return out


def _is_lr_finder_report(raw: object) -> bool:
    """True if a parsed payload looks like an LR-finder result.

    The telltale shape is a dict carrying parallel ``lrs`` and ``losses`` arrays —
    exactly the fields of :class:`LRFinderResult`. This filters out unrelated JSON
    (configs, metadata, etc.).
    """
    if not isinstance(raw, dict):
        return False
    return isinstance(raw.get("lrs"), list) and isinstance(raw.get("losses"), list)


def _downsample(points: list[dict], max_points: int) -> list[dict]:
    """Uniform-stride downsample, always keeping the first and last points."""
    count = len(points)
    if count <= max_points:
        return points
    stride = (count + max_points - 1) // max_points
    kept = points[::stride]
    if kept[-1] is not points[-1]:
        if len(kept) >= max_points:
            kept = kept[: max_points - 1]
        kept.append(points[-1])
    return kept


def _summarize(raw: dict, path: Path, mtime: float) -> dict:
    """Collapse one LR-finder result into a single ``LrFinderRun``-shaped dict."""
    lrs = _float_list(raw.get("lrs"))
    losses = _float_list(raw.get("losses"))
    # Pair only where both an lr and a loss exist (truncate to the shorter).
    paired: list[dict] = [
        {"lr": lr, "loss": loss} for lr, loss in zip(lrs, losses) if lr > 0
    ]
    points = _downsample(paired, _MAX_POINTS)

    min_loss = min((p["loss"] for p in paired), default=None)

    return {
        "name": path.name,
        "path": str(path),
        "config": _as_str(raw.get("config")),
        "suggested_lr": _as_float(raw.get("suggested_lr")),
        "min_loss": min_loss,
        "num_points": len(paired),
        "mtime": mtime,
        "points": points,
    }


def _candidate_files(root_path: Path) -> list[Path]:
    """Find JSON files that may be LR-finder artifacts under ``root``.

    Scans the conventional output dirs (and the project root, name-filtered so the
    root scan stays cheap). Missing dirs are silently ignored. Duplicates (same
    resolved path) are de-duplicated.
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
            # every stray JSON; dedicated lr-finder dirs scan all JSON.
            if root_level and not any(h in path.name.lower() for h in _NAME_HINTS):
                continue
            _add(path)

    return files


def lr_finder_results(root: str = ".") -> dict:
    """Collect persisted LR-finder results, newest first.

    Returns a :class:`~cola_coder.ui.schemas.LrFinderResults`-shaped dict::

        {"runs": [LrFinderRun, ...], "count": int}

    sorted by modification time (newest first). Reads only past artifacts — it
    never runs the LR finder. On any failure returns ``{"error": "..."}`` and
    never raises. Finding no artifacts is NOT an error: ``{"runs": [], "count": 0}``
    (the common case today, since ``find_lr.py`` persists only a PNG plot).
    """
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return {"error": f"root not found: {root}"}

        runs: list[dict] = []
        for path in _candidate_files(root_path):
            raw = _read_json(path)
            if not _is_lr_finder_report(raw):
                continue
            assert isinstance(raw, dict)  # narrowed by _is_lr_finder_report
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            runs.append(_summarize(raw, path, mtime))

        # Newest first; path tiebreak for stable ordering.
        runs.sort(key=lambda r: (-r["mtime"], r["path"]))

        return {"runs": runs, "count": len(runs)}
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        logger.warning("lr_finder_results scan failed: %s", exc)
        return {"error": str(exc)}
