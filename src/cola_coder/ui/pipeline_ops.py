"""Pipeline-run lifecycle operations for the local UI.

Thin, side-effect-explicit wrappers over :class:`PipelineRunManager` that the
FastAPI layer exposes so the dashboard can CREATE, RESET, OVERRIDE, and DELETE
named pipeline runs (the 10-stage state machine persisted to
``pipeline_runs/{name}.json``).

These are pure STATE operations on the run JSON — they never execute a stage,
load a model, or touch the GPU, so they are always safe to call while the live
trainer is running. Actual stage execution stays behind the trainer-guarded
job runner (``/api/run`` → ``full_pipeline``).

Every function is robust to bad input: a missing run, an invalid name, or a
bad stage number returns an ``{"error": ...}`` dict rather than raising, so the
HTTP layer can surface it as a typed ``ErrorResponse`` instead of a 500.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cola_coder.pipeline.run_manager import (
    ALL_STAGE_NUMS,
    OPTIONAL_STAGES,
    STAGE_DEFS,
    PipelineRun,
    PipelineRunManager,
)

logger = logging.getLogger(__name__)

# Run names become filenames (``pipeline_runs/{name}.json``), so they must be a
# strict whitelist — no path separators, no traversal, no surprises.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_COMPLETED = "completed"
_FAILED = "failed"
_RUNNING = "running"
_SKIPPED = "skipped"


def _overall_status(stages: dict[int, object], completed: int, active: int) -> str:
    """Derive a single overall status from per-stage states.

    ``failed`` if any stage failed, else ``running`` if any is running, else
    ``completed`` once every active (non-skipped) stage is done, else ``pending``.
    """
    statuses = [getattr(s, "status", "pending") for s in stages.values()]
    if _FAILED in statuses:
        return _FAILED
    if _RUNNING in statuses:
        return _RUNNING
    if active > 0 and completed >= active:
        return _COMPLETED
    return "pending"


def _detail(run: PipelineRun) -> dict:
    """Serialise a :class:`PipelineRun` into the UI detail shape (named stages)."""
    stage_rows: list[dict] = []
    completed = 0
    active = 0
    for num in ALL_STAGE_NUMS:
        state = run.stages.get(num)
        status = state.status if state else "pending"
        if status != _SKIPPED:
            active += 1
        if status == _COMPLETED:
            completed += 1
        defn = STAGE_DEFS[num]
        stage_rows.append(
            {
                "num": num,
                "name": str(defn["name"]),
                "description": str(defn["description"]),
                "optional": num in OPTIONAL_STAGES,
                "status": status,
                "artifact": state.artifact if state else "",
                "override": state.override if state else "",
                "error": state.error if state else "",
                "duration_secs": state.duration_secs if state else 0.0,
                "started_at": state.started_at if state else None,
                "completed_at": state.completed_at if state else None,
            }
        )
    return {
        "name": run.name,
        "config_path": run.config_path,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "notes": run.notes,
        "stages": stage_rows,
        "num_stages": len(ALL_STAGE_NUMS),
        "active_stages": active,
        "completed": completed,
        "status": _overall_status(run.stages, completed, active),
    }


def _manager(runs_dir: str) -> PipelineRunManager:
    return PipelineRunManager(Path(runs_dir))


def create_run(
    name: str,
    config_path: str,
    skip_stages: list[int] | None = None,
    runs_dir: str = "pipeline_runs",
) -> dict:
    """Create a new named pipeline run, or return an error dict.

    Validates the *name* (filename-safe whitelist), refuses to clobber an
    existing run, and verifies *config_path* exists before writing state.
    """
    if not _NAME_RE.match(name):
        return {"error": f"invalid run name {name!r}: use letters, digits, '-' and '_' (max 64)"}
    if not Path(config_path).is_file():
        return {"error": f"config not found: {config_path}"}
    mgr = _manager(runs_dir)
    if mgr.exists(name):
        return {"error": f"run {name!r} already exists"}
    bad = [n for n in (skip_stages or []) if n not in STAGE_DEFS]
    if bad:
        return {"error": f"unknown stage number(s): {bad}"}
    run = mgr.create(name, config_path, skip_stages=set(skip_stages) if skip_stages else None)
    logger.info("created pipeline run %r (config=%s)", name, config_path)
    return _detail(run)


def delete_run(name: str, runs_dir: str = "pipeline_runs") -> dict:
    """Delete a named run. Returns ``{ok, name}`` or an error dict."""
    if not _NAME_RE.match(name):
        return {"error": f"invalid run name {name!r}"}
    mgr = _manager(runs_dir)
    if not mgr.exists(name):
        return {"error": f"run {name!r} not found"}
    ok = mgr.delete(name)
    logger.info("deleted pipeline run %r (ok=%s)", name, ok)
    return {"ok": ok, "name": name}


def reset_run(name: str, stage_num: int, runs_dir: str = "pipeline_runs") -> dict:
    """Reset *name* so *stage_num* and everything after it are pending again."""
    if not _NAME_RE.match(name):
        return {"error": f"invalid run name {name!r}"}
    if stage_num not in STAGE_DEFS:
        return {"error": f"unknown stage number: {stage_num}"}
    mgr = _manager(runs_dir)
    if not mgr.exists(name):
        return {"error": f"run {name!r} not found"}
    run = mgr.load(name)
    mgr.reset_to_stage(run, stage_num)
    logger.info("reset pipeline run %r to stage %d", name, stage_num)
    return _detail(run)


def set_override(
    name: str,
    stage_num: int,
    path: str,
    runs_dir: str = "pipeline_runs",
) -> dict:
    """Set the input-artifact override for *stage_num* of *name*."""
    if not _NAME_RE.match(name):
        return {"error": f"invalid run name {name!r}"}
    if stage_num not in STAGE_DEFS:
        return {"error": f"unknown stage number: {stage_num}"}
    mgr = _manager(runs_dir)
    if not mgr.exists(name):
        return {"error": f"run {name!r} not found"}
    run = mgr.load(name)
    mgr.set_override(run, stage_num, path)
    logger.info("set override for run %r stage %d -> %s", name, stage_num, path)
    return _detail(run)


def get_run_detail(name: str, runs_dir: str = "pipeline_runs") -> dict:
    """Return the full per-stage detail for *name*, or an error dict."""
    if not _NAME_RE.match(name):
        return {"error": f"invalid run name {name!r}"}
    mgr = _manager(runs_dir)
    if not mgr.exists(name):
        return {"error": f"run {name!r} not found"}
    return _detail(mgr.load(name))
