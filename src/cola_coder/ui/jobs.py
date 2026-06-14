"""Background job manager for the local UI/dashboard.

Launches existing project scripts as detached background subprocesses, streams
their combined stdout/stderr to a per-job log file, and tracks live status by
polling each ``Popen`` handle. Also provides a safety guard that refuses to
launch a second training run while one is already in progress.
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

try:  # psutil is optional; is_training_running degrades gracefully without it.
    import psutil
except ImportError:  # pragma: no cover - exercised only when psutil is absent
    psutil = None  # type: ignore[assignment]


class JobManager:
    """Manage background jobs launched as detached subprocesses."""

    def __init__(self, log_dir: str = "ui_jobs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # job_id -> {"meta": dict, "proc": Popen, "log_handle": file-or-None}
        self._jobs: dict[str, dict] = {}

    def start(self, name: str, cmd: list[str], cwd: Optional[str] = None) -> dict:
        """Launch ``cmd`` as a detached background subprocess.

        stdout and stderr are merged into a per-job log file under ``log_dir``.
        Returns the job metadata dict (status "running").
        """
        job_id = uuid.uuid4().hex
        log_path = self.log_dir / f"{name}-{job_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

        meta = {
            "id": job_id,
            "name": name,
            "pid": proc.pid,
            "status": "running",
            "cmd": list(cmd),
            "log": str(log_path),
            "started": time.time(),
        }
        self._jobs[job_id] = {"meta": meta, "proc": proc, "log_handle": log_handle}
        return dict(meta)

    def _refresh(self, entry: dict) -> dict:
        """Poll the process and return an up-to-date metadata dict."""
        meta = entry["meta"]
        proc: subprocess.Popen = entry["proc"]
        returncode = proc.poll()
        if returncode is None:
            meta["status"] = "running"
        elif returncode == 0:
            meta["status"] = "done"
        else:
            meta["status"] = "failed"

        if returncode is not None and entry.get("log_handle") is not None:
            try:
                entry["log_handle"].close()
            except OSError:
                pass
            entry["log_handle"] = None

        result = dict(meta)
        result["returncode"] = returncode
        return result

    def list(self) -> list[dict]:
        """Return all jobs with their current (freshly polled) status."""
        return [self._refresh(entry) for entry in self._jobs.values()]

    def get(self, job_id: str) -> Optional[dict]:
        """Return a single job's current status, or None if unknown."""
        entry = self._jobs.get(job_id)
        if entry is None:
            return None
        return self._refresh(entry)

    def stop(self, job_id: str) -> bool:
        """Terminate the job's process if running.

        Returns True if a running process was signalled, else False.
        """
        entry = self._jobs.get(job_id)
        if entry is None:
            return False
        proc: subprocess.Popen = entry["proc"]
        if proc.poll() is not None:
            return False
        proc.terminate()
        return True

    def is_training_running(self) -> bool:
        """Return True if any python process is running a training script.

        Detected via a cmdline containing "train.py". Uses psutil when
        importable; returns False (best-effort) if psutil is missing. Never
        raises.
        """
        if psutil is None:
            return False
        try:
            for proc in psutil.process_iter(["cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if any("train.py" in str(part) for part in cmdline):
                    return True
        except Exception:
            return False
        return False

    def start_training(self, config: str, resume: Optional[str] = None) -> dict:
        """Start a training run, refusing if one is already in progress.

        SAFETY: if ``is_training_running()`` is True, does NOT launch and
        returns ``{"error": "training already running"}``. This implementation
        returns an error dict (it does not raise).
        """
        if self.is_training_running():
            return {"error": "training already running"}

        cmd = [sys.executable, "scripts/train.py", "--config", config]
        if resume is not None:
            cmd += ["--resume", resume]
        else:
            cmd += ["--auto-resume"]
        return self.start("train", cmd)
