"""Auto-eval-over-training history helpers for the local cola-coder UI/dashboard.

Pure library module (no Rich, no CLI) that gathers the auto-evaluation snapshots
recorded over the life of a training run into a single chronological series — a
quality-vs-step view for charting (e.g. pass@1 / pass@5 over steps).

Where the snapshots come from (mirrors ``scripts/training_eval_history.py``):

- ``checkpoints/<model>/step_*/metadata.json`` — each may carry an ``auto_eval``
  state dict whose ``history`` list holds per-step ``EvalSnapshot`` dicts
  (``training.auto_eval.EvalSnapshot.to_dict``: ``step``, ``timestamp``,
  ``pass_at_1``, ``pass_at_5``, ``num_problems``, ``avg_generation_time``,
  ``is_best``).
- ``checkpoints/<model>/auto_eval_history.json`` — a standalone JSON array of
  the same per-step snapshot dicts.

All functions are best-effort and never raise — on a genuinely broken discovery
(e.g. an unreadable root) they return an ``{"error": ...}`` dict; finding no
snapshots is NOT an error (empty series is returned).
"""

from __future__ import annotations

import json
from pathlib import Path

# Directories conventionally holding checkpoints (which carry eval history).
_CKPT_DIRS = ("checkpoints",)

# Snapshot dict fields that are NOT charting metrics (everything else numeric is).
_NON_METRIC_KEYS = frozenset({"step", "timestamp"})


def _read_json(path: Path) -> object | None:
    """Parse a JSON file, or return ``None`` on any read/decode failure."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _to_step(value: object) -> int | None:
    """Coerce a snapshot ``step`` to int, or ``None`` if not a real number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _is_number(value: object) -> bool:
    """True for a real int/float (excludes bool, which is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_metrics(snapshot: dict) -> dict:
    """Pull the numeric, chartable values out of a raw snapshot dict.

    Everything numeric except ``step`` (the x-axis) becomes a metric. ``bool``
    values (e.g. ``is_best``) are included as ints since they are chartable
    series too, but the structural ``timestamp`` string is dropped.
    """
    metrics: dict = {}
    for key, value in snapshot.items():
        if key in _NON_METRIC_KEYS:
            continue
        if isinstance(value, bool):
            metrics[key] = int(value)
        elif _is_number(value):
            metrics[key] = value
    return metrics


def _history_from_payload(raw: object) -> list:
    """Locate the list of snapshot dicts inside a parsed JSON payload.

    Handles the three real shapes:
    - checkpoint ``metadata.json`` — a dict carrying an ``auto_eval`` state dict
      whose ``history`` is the snapshot list;
    - a dict that *is* the ``auto_eval`` state (has ``history`` directly);
    - standalone ``auto_eval_history.json`` — a top-level list of snapshots.

    A plain ``metadata.json`` with no ``auto_eval`` key yields no snapshots (we
    never treat the metadata dict itself as a snapshot).
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        state = raw.get("auto_eval")
        if isinstance(state, dict):
            history = state.get("history")
            return history if isinstance(history, list) else []
        history = raw.get("history")
        return history if isinstance(history, list) else []
    return []


def _snapshot_records(raw: object, path: Path, mtime: float) -> list[dict]:
    """Turn a parsed JSON payload into zero or more snapshot records."""
    rows: list[dict] = []
    candidates = _history_from_payload(raw)

    path_str = str(path)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        metrics = _extract_metrics(item)
        if not metrics:
            continue
        rows.append(
            {
                "step": _to_step(item.get("step")),
                "path": path_str,
                "mtime": mtime,
                "metrics": metrics,
            }
        )
    return rows


def _candidate_files(root_path: Path) -> list[Path]:
    """Find files that may carry auto-eval history under ``root``.

    Scans ``checkpoints/<model>/`` for standalone ``auto_eval_history.json`` and
    every ``step_*/metadata.json``. Missing dirs are silently ignored.
    """
    files: list[Path] = []
    for dirname in _CKPT_DIRS:
        ckpt_root = root_path / dirname
        if not ckpt_root.is_dir():
            continue
        try:
            model_dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for model_dir in model_dirs:
            standalone = model_dir / "auto_eval_history.json"
            if standalone.is_file():
                files.append(standalone)
            try:
                step_dirs = [
                    d
                    for d in model_dir.iterdir()
                    if d.is_dir() and d.name.startswith("step_")
                ]
            except OSError:
                continue
            for step_dir in step_dirs:
                meta = step_dir / "metadata.json"
                if meta.is_file():
                    files.append(meta)
    return files


def eval_history(root: str = ".") -> dict:
    """Collect auto-eval-over-training snapshots into a chronological series.

    Returns::

        {"snapshots": [ {"step": int | None, "path": str, "mtime": float,
                         "metrics": dict} ],   # chronological (by step then mtime)
         "count": int,
         "metric_keys": list[str]}             # union of metric names seen

    Discovers the snapshot file(s) wherever ``training_eval_history.py`` writes
    them: ``checkpoints/<model>/step_*/metadata.json`` (the ``auto_eval.history``
    list) and the standalone ``checkpoints/<model>/auto_eval_history.json`` array.
    Each parsed row is one snapshot whose ``metrics`` holds the numeric eval
    values for that step (``pass_at_1``, ``pass_at_5``, ...).

    On any failure returns ``{"error": "..."}``. Never raises. Returns
    ``{"snapshots": [], "count": 0, "metric_keys": []}`` when nothing is found
    (not an error).
    """
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return {"error": f"root not found: {root}"}

        snapshots: list[dict] = []
        for path in _candidate_files(root_path):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            raw = _read_json(path)
            if raw is None:
                continue
            snapshots.extend(_snapshot_records(raw, path, mtime))

        # Chronological: by step (None last), then mtime, then path for stability.
        snapshots.sort(
            key=lambda s: (
                s["step"] is None,
                s["step"] if s["step"] is not None else 0,
                s["mtime"],
                s["path"],
            )
        )

        metric_keys: list[str] = []
        seen: set[str] = set()
        for snap in snapshots:
            for key in snap["metrics"]:
                if key not in seen:
                    seen.add(key)
                    metric_keys.append(key)

        return {
            "snapshots": snapshots,
            "count": len(snapshots),
            "metric_keys": metric_keys,
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        return {"error": str(exc)}
