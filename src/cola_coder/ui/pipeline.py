"""Pipeline-run browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of named pipeline runs stored as JSON at
``pipeline_runs/{name}.json`` (a 10-stage state machine). All functions are
robust to missing or malformed inputs and never raise on bad data — they return
empty results or an {"error": ...} dict instead.

Run files are user-authored and of varying shape, so every field is inferred
best-effort and falls back to ``None`` when it cannot be determined.
"""

from __future__ import annotations

import json
import os

_COMPLETED = "completed"
_FAILED = "failed"
_RUNNING = "running"


def list_pipeline_runs(runs_dir: str = "pipeline_runs") -> list[dict]:
    """Scan ``runs_dir`` for ``*.json`` pipeline-run files.

    Each entry is a dict with keys: name (file stem), path, mtime,
    num_stages, status (overall, best-effort), completed (count of completed
    stages). A file that cannot be parsed yields
    {"name", "path", "mtime", "error": str} instead.

    Missing ``runs_dir`` yields ``[]``. Results are sorted by name.
    """
    if not os.path.isdir(runs_dir):
        return []

    results: list[dict] = []
    try:
        filenames = os.listdir(runs_dir)
    except OSError:
        return []

    for filename in filenames:
        if not filename.endswith(".json"):
            continue

        path = os.path.join(runs_dir, filename)
        if not os.path.isfile(path):
            continue

        name = filename[: -len(".json")]

        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            mtime = None

        run = read_pipeline_run(path)
        if "error" in run and not _looks_like_run(run):
            results.append(
                {
                    "name": name,
                    "path": path,
                    "mtime": mtime,
                    "error": run["error"],
                }
            )
            continue

        num_stages, completed, status = _summarize_stages(run)
        results.append(
            {
                "name": name,
                "path": path,
                "mtime": mtime,
                "num_stages": num_stages,
                "status": status,
                "completed": completed,
            }
        )

    results.sort(key=lambda entry: entry["name"])
    return results


def read_pipeline_run(path: str) -> dict:
    """Read and parse one pipeline-run JSON file.

    Returns the parsed dict (the full run state) on success, or {"error": str}
    if the file is missing or unparseable. Never raises.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    if not isinstance(data, dict):
        return {"error": f"unexpected run shape: {type(data).__name__}"}

    return data


def _looks_like_run(run: dict) -> bool:
    """Whether a dict carrying an 'error' is still a usable run (vs a parse failure)."""
    return any(key in run for key in ("name", "config_path", "stages"))


def _summarize_stages(run: dict) -> tuple[int | None, int | None, str | None]:
    """Best-effort (num_stages, completed, overall_status) from a run dict.

    Inspects a ``stages`` field if present: either a list of stage dicts (each
    with a ``status`` key) or a dict mapping stage name -> status (or -> dict
    with a ``status`` key). Returns ``(None, None, None)`` when nothing can be
    inferred.
    """
    stages = run.get("stages") if isinstance(run, dict) else None
    statuses = _extract_statuses(stages)
    if statuses is None:
        return None, None, None

    num_stages = len(statuses)
    completed = sum(1 for status in statuses if status == _COMPLETED)
    return num_stages, completed, _overall_status(statuses)


def _extract_statuses(stages: object) -> list[str | None] | None:
    """Pull a flat list of per-stage status strings from a stages field."""
    if isinstance(stages, list):
        statuses: list[str | None] = []
        for stage in stages:
            if isinstance(stage, dict):
                statuses.append(_as_status(stage.get("status")))
            else:
                statuses.append(_as_status(stage))
        return statuses

    if isinstance(stages, dict):
        statuses = []
        for value in stages.values():
            if isinstance(value, dict):
                statuses.append(_as_status(value.get("status")))
            else:
                statuses.append(_as_status(value))
        return statuses

    return None


def _as_status(value: object) -> str | None:
    """Normalize a status value to a lowercase string, or None."""
    if isinstance(value, str):
        return value.strip().lower() or None
    return None


def _overall_status(statuses: list[str | None]) -> str | None:
    """Derive a single overall status from per-stage statuses.

    Precedence: any failed -> failed; any running -> running; all completed ->
    completed; otherwise pending. ``None`` when there are no stages.
    """
    if not statuses:
        return None
    if any(status == _FAILED for status in statuses):
        return _FAILED
    if any(status == _RUNNING for status in statuses):
        return _RUNNING
    if all(status == _COMPLETED for status in statuses):
        return _COMPLETED
    return "pending"
