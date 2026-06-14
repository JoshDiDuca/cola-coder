"""Tests for the live job-log SSE stream + GPU action tagging.

The stream endpoint follows a job's log file and pushes JobLogChunk frames; the
GPU tags let the UI warn before an action competes with the live trainer for
VRAM. None of this touches the training path.
"""

from __future__ import annotations

import sys
import time

from fastapi.testclient import TestClient

from cola_coder.ui.app import create_app
from cola_coder.ui.jobs import JobManager
from cola_coder.ui.schemas import JobLogChunk


def _client(tmp_path) -> tuple[TestClient, JobManager]:
    jm = JobManager(log_dir=str(tmp_path / "ui_jobs"))
    return TestClient(create_app(job_manager=jm)), jm


def _await_done(jm: JobManager, jid: str, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jm.get(jid)
        if status and status["status"] != "running":
            return
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_stream_unknown_job_404(tmp_path) -> None:
    c, _ = _client(tmp_path)
    assert c.get("/api/jobs/nope/stream").status_code == 404


def test_stream_captures_output_and_terminal_done(tmp_path) -> None:
    c, jm = _client(tmp_path)
    job = jm.start("echo", [sys.executable, "-c", "print('hello-stream'); print('line2')"])
    _await_done(jm, job["id"])

    frames: list[JobLogChunk] = []
    with c.stream("GET", f"/api/jobs/{job['id']}/stream?tail=50") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith("data: "):
                frame = JobLogChunk.model_validate_json(line[len("data: "):])
                frames.append(frame)
                if frame.done:
                    break

    assert frames, "expected at least one SSE frame"
    assert frames[-1].done is True
    joined = "".join(f.text for f in frames)
    assert "hello-stream" in joined and "line2" in joined


def test_actions_carry_gpu_flag(tmp_path) -> None:
    c, _ = _client(tmp_path)
    r = c.get("/api/actions")
    assert r.status_code == 200, r.text  # response_model would 500 if gpu field were unmodeled
    by_key = {a["key"]: a for a in r.json()}
    # GPU-heavy eval/benchmark actions are flagged; pure CPU utilities are not.
    assert by_key["evaluate"]["gpu"] is True
    assert by_key["benchmark"]["gpu"] is True
    assert by_key["env_check"]["gpu"] is False
    assert by_key["vram_estimate"]["gpu"] is False
    # trainer actions stay trainer-flagged (guarded separately).
    assert by_key["train_sft"]["trainer"] is True
