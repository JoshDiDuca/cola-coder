"""UI server-push (SSE) + snapshot tests.

Light + safe: temp JobManager + temp data/ckpt roots (same pattern as
tests/test_ui_app.py), no GPU, no training. Reads exactly one SSE event then
closes so nothing hangs on the 1s server-side tick loop.
"""

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cola_coder.ui import create_app  # noqa: E402
from cola_coder.ui.jobs import JobManager  # noqa: E402


@pytest.fixture
def client(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    np.save(str(data / "train.npy"), np.zeros((5, 8), dtype=np.uint16))
    jm = JobManager(log_dir=str(tmp_path / "jobs"))
    app = create_app(job_manager=jm, project_root=tmp_path, data_root=str(data),
                     ckpt_root=str(tmp_path / "checkpoints"))
    return TestClient(app), jm, data


def test_status_endpoint_unchanged(client):
    c, _, _ = client
    r = c.get("/api/status")
    assert r.status_code == 200
    assert set(r.json()) == {"training", "system", "checkpoints"}


def test_events_stream_pushes_full_snapshot(client):
    c, _, _ = client
    with c.stream("GET", "/api/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        payload = None
        for line in r.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                break  # one event is enough — don't block on the tick loop
    assert payload is not None
    assert set(payload) == {"training", "system", "checkpoints", "jobs"}
    assert isinstance(payload["jobs"], list)
