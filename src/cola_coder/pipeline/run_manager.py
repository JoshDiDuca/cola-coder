"""Named pipeline run management with state persistence.

Each pipeline run tracks 10 stages (collect → pretrain → align → eval),
persists state to JSON, and supports resume, override, and re-run.

Storage: ``pipeline_runs/{name}.json`` under the project root.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Stage definitions ─────────────────────────────────────────────────────

STAGE_DEFS: dict[int, dict[str, str | bool]] = {
    1: {
        "name": "collect-data",
        "description": "Collect code, text, and math data",
        "optional": False,
    },
    2: {
        "name": "prepare-data",
        "description": "Filter, score, tokenize, and mix data",
        "optional": False,
    },
    3: {
        "name": "pretrain",
        "description": "Base model pretraining",
        "optional": False,
    },
    4: {
        "name": "extend-context",
        "description": "RoPE scaling for longer context",
        "optional": True,
    },
    5: {
        "name": "generate-instructions",
        "description": "Create SFT instruction pairs from code",
        "optional": False,
    },
    6: {
        "name": "instruction-tune",
        "description": "SFT on ChatML instruction data",
        "optional": False,
    },
    7: {
        "name": "upcycle-moe",
        "description": "Convert dense model to MoE + differentiate experts (fine-tune)",
        "optional": True,
    },
    8: {
        "name": "train-router",
        "description": "Train semantic domain router",
        "optional": False,
    },
    9: {
        "name": "train-reasoning",
        "description": "GRPO reasoning with thinking tokens",
        "optional": False,
    },
    10: {
        "name": "evaluate",
        "description": "Full evaluation suite",
        "optional": False,
    },
}

ALL_STAGE_NUMS = sorted(STAGE_DEFS.keys())
OPTIONAL_STAGES = {n for n, d in STAGE_DEFS.items() if d.get("optional")}


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class StageState:
    """Tracks the status and artifacts for a single pipeline stage."""

    status: str = "pending"
    """One of: pending, running, completed, failed, skipped."""

    started_at: str | None = None
    completed_at: str | None = None
    duration_secs: float = 0.0
    error: str = ""
    artifact: str = ""
    """Output path produced by this stage (checkpoint, data file, etc.)."""

    override: str = ""
    """User-specified input override — takes precedence over previous stage artifact."""


@dataclass
class PipelineRun:
    """A named pipeline run with per-stage state tracking."""

    name: str
    config_path: str
    created_at: str = ""
    updated_at: str = ""
    stages: dict[int, StageState] = field(default_factory=dict)
    notes: str = ""


# ── Manager ───────────────────────────────────────────────────────────────

class PipelineRunManager:
    """Create, load, save, and query named pipeline runs.

    Runs are stored as individual JSON files in *runs_dir*.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        runs_dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        config_path: str,
        skip_stages: set[int] | None = None,
    ) -> PipelineRun:
        """Create a new pipeline run with all stages initialised."""
        now = _now_iso()
        stages: dict[int, StageState] = {}
        for num in ALL_STAGE_NUMS:
            if skip_stages and num in skip_stages:
                stages[num] = StageState(status="skipped")
            else:
                stages[num] = StageState()
        run = PipelineRun(
            name=name,
            config_path=config_path,
            created_at=now,
            updated_at=now,
            stages=stages,
        )
        self.save(run)
        return run

    def load(self, name: str) -> PipelineRun:
        """Load a run from disk.  Raises FileNotFoundError if missing."""
        path = self._path_for(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        return _run_from_dict(data)

    def save(self, run: PipelineRun) -> None:
        """Persist the run to disk (atomic write)."""
        run.updated_at = _now_iso()
        path = self._path_for(run.name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_run_to_dict(run), indent=2), encoding="utf-8")
        tmp.replace(path)

    def list_runs(self) -> list[PipelineRun]:
        """Return all saved runs, newest first."""
        runs: list[PipelineRun] = []
        for p in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append(_run_from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return runs

    def delete(self, name: str) -> bool:
        """Delete a run file.  Returns True if it existed."""
        path = self._path_for(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, name: str) -> bool:
        return self._path_for(name).exists()

    # ── Stage queries ─────────────────────────────────────────────────

    def next_pending(self, run: PipelineRun) -> int | None:
        """Return the number of the next non-skipped pending/failed stage, or None."""
        for num in ALL_STAGE_NUMS:
            st = run.stages.get(num)
            if st and st.status in ("pending", "failed", "running"):
                return num
        return None

    def completed_count(self, run: PipelineRun) -> int:
        return sum(
            1 for s in run.stages.values() if s.status in ("completed", "skipped")
        )

    def total_active(self, run: PipelineRun) -> int:
        return sum(1 for s in run.stages.values() if s.status != "skipped")

    def summary_line(self, run: PipelineRun) -> str:
        """One-line summary like 'tiny-v1 — 5/8 done, failed at stage 6'."""
        done = self.completed_count(run)
        total = self.total_active(run)
        failed = [
            n for n, s in run.stages.items() if s.status == "failed"
        ]
        status_part = f"{done}/{total} done"
        if failed:
            names = ", ".join(str(n) for n in failed)
            status_part += f", failed at stage {names}"
        nxt = self.next_pending(run)
        if nxt and not failed:
            status_part += f", next: stage {nxt}"
        return f"{run.name} — {status_part}"

    # ── Stage state transitions ───────────────────────────────────────

    def mark_running(self, run: PipelineRun, stage_num: int) -> None:
        st = run.stages[stage_num]
        st.status = "running"
        st.started_at = _now_iso()
        st.error = ""
        self.save(run)

    def mark_completed(
        self, run: PipelineRun, stage_num: int, artifact: str = "",
        duration: float = 0.0,
    ) -> None:
        st = run.stages[stage_num]
        st.status = "completed"
        st.completed_at = _now_iso()
        st.duration_secs = duration
        st.artifact = artifact
        st.error = ""
        self.save(run)

    def mark_failed(
        self, run: PipelineRun, stage_num: int, error: str = "",
        duration: float = 0.0,
    ) -> None:
        st = run.stages[stage_num]
        st.status = "failed"
        st.completed_at = _now_iso()
        st.duration_secs = duration
        st.error = error
        self.save(run)

    def set_override(self, run: PipelineRun, stage_num: int, path: str) -> None:
        run.stages[stage_num].override = path
        self.save(run)

    def reset_to_stage(self, run: PipelineRun, stage_num: int) -> None:
        """Reset a run so it re-executes from *stage_num* onward.

        Stages before *stage_num* keep their status (completed/skipped/etc).
        Stage *stage_num* and all later stages are reset to pending,
        preserving artifacts from earlier stages so downstream stages
        can still resolve inputs.
        """
        for num in sorted(run.stages):
            if num >= stage_num:
                st = run.stages[num]
                if st.status == "skipped":
                    continue  # Don't un-skip stages the user chose to skip
                st.status = "pending"
                st.started_at = None
                st.completed_at = None
                st.duration_secs = 0.0
                st.error = ""
                # Keep artifact from previous runs — it might still be valid
                # and downstream stages may need it as fallback
        self.save(run)

    # ── Artifact resolution ───────────────────────────────────────────

    def resolve_input(self, run: PipelineRun, stage_num: int) -> str:
        """Resolve the primary input for *stage_num*.

        Priority: override for this stage → artifact from nearest earlier
        completed stage → empty string (auto-detect).
        """
        st = run.stages.get(stage_num)
        if st and st.override:
            return st.override
        # Walk backwards through earlier stages for an artifact
        for prev in range(stage_num - 1, 0, -1):
            prev_st = run.stages.get(prev)
            if prev_st and prev_st.status == "completed" and prev_st.artifact:
                return prev_st.artifact
        return ""

    # ── Internal ──────────────────────────────────────────────────────

    def _path_for(self, name: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
        return self.runs_dir / f"{safe}.json"


# ── Serialisation helpers ─────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_to_dict(run: PipelineRun) -> dict:
    d = asdict(run)
    # Convert int keys to strings for JSON
    d["stages"] = {str(k): v for k, v in d["stages"].items()}
    return d


def _run_from_dict(d: dict) -> PipelineRun:
    stages: dict[int, StageState] = {}
    for k, v in d.get("stages", {}).items():
        stages[int(k)] = StageState(**v)
    return PipelineRun(
        name=d["name"],
        config_path=d["config_path"],
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
        stages=stages,
        notes=d.get("notes", ""),
    )
