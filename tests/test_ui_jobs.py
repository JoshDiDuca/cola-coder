"""Tests for the background JobManager (src/cola_coder/ui/jobs.py).

All tests are LIGHT and SAFE: harmless subprocesses only (short python -c
snippets). They NEVER launch real training.
"""

import sys
import time

from cola_coder.ui.jobs import JobManager


def _poll_until(manager: JobManager, job_id: str, predicate, cap: float = 5.0):
    """Poll a job until predicate(meta) is true or cap seconds elapse."""
    deadline = time.time() + cap
    meta = manager.get(job_id)
    while time.time() < deadline:
        meta = manager.get(job_id)
        if meta is not None and predicate(meta):
            return meta
        time.sleep(0.05)
    return meta


def test_start_and_complete(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    job = mgr.start("sleeper", [sys.executable, "-c", "import time; time.sleep(0.2)"])

    assert job["status"] == "running"
    assert job["name"] == "sleeper"
    assert isinstance(job["pid"], int)
    assert job["cmd"][0] == sys.executable

    listed = mgr.list()
    assert any(j["id"] == job["id"] for j in listed)
    statuses = {j["id"]: j["status"] for j in listed}
    assert statuses[job["id"]] in ("running", "done")

    final = _poll_until(mgr, job["id"], lambda m: m["status"] == "done")
    assert final["status"] == "done"
    assert final["returncode"] == 0


def test_failing_job(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    job = mgr.start("failer", [sys.executable, "-c", "import sys; sys.exit(3)"])

    final = _poll_until(mgr, job["id"], lambda m: m["status"] == "failed")
    assert final["status"] == "failed"
    assert final["returncode"] == 3


def test_get_unknown_returns_none(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    assert mgr.get("does-not-exist") is None


def test_stop_running_job(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    job = mgr.start("longsleep", [sys.executable, "-c", "import time; time.sleep(30)"])

    # Ensure it is actually running before we stop it.
    assert mgr.get(job["id"])["status"] == "running"
    assert mgr.stop(job["id"]) is True

    final = _poll_until(mgr, job["id"], lambda m: m["status"] != "running")
    assert final["status"] != "running"

    # Stopping an already-finished job returns False.
    assert mgr.stop(job["id"]) is False


def test_is_training_running_returns_bool(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    result = mgr.is_training_running()
    assert isinstance(result, bool)


def test_start_training_refuses_when_already_running(tmp_path):
    mgr = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    before = len(mgr.list())

    mgr.is_training_running = lambda: True  # type: ignore[method-assign]

    result = mgr.start_training("configs/small.yaml")
    assert "error" in result
    # No subprocess was spawned.
    assert len(mgr.list()) == before
