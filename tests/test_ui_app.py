"""UI FastAPI app: endpoint wiring over the status/jobs/datasets modules.

Light + safe: injects a temp JobManager + temp data_root, runs only harmless
subprocess jobs, never touches training or the GPU.
"""

import sys

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
    np.save(str(data / "train.weights.npy"), np.array([0.1, 0.5, 0.9, 1.0, 0.3], dtype=np.float32))
    jm = JobManager(log_dir=str(tmp_path / "jobs"))
    app = create_app(job_manager=jm, project_root=tmp_path, data_root=str(data),
                     ckpt_root=str(tmp_path / "checkpoints"))
    return TestClient(app), jm, data


def test_status_endpoint_shape(client):
    c, _, _ = client
    r = c.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"training", "system", "checkpoints"}
    assert "alive" in body["training"]
    assert "gpu_util_pct" in body["system"]


def test_datasets_and_scores(client):
    c, _, data = client
    rows = c.get("/api/datasets").json()
    names = {d["name"] for d in rows}
    assert "train.npy" in names
    assert "train.weights.npy" not in names  # weights sidecar excluded from listing
    npy = next(d for d in rows if d["name"] == "train.npy")
    assert npy["has_weights"] is True and npy["num_samples"] == 5
    scores = c.get("/api/datasets/scores", params={"path": str(data / "train.weights.npy")}).json()
    assert scores["n"] == 5 and len(scores["histogram"]) == 10


def test_actions_listed_and_runnable(client):
    c, jm, _ = client
    actions = c.get("/api/actions").json()
    assert any(a["key"] == "smoke_test" for a in actions)
    # Run a harmless action by overriding the cmd through the action's args is not
    # allowed (scripts are fixed); instead drive the JobManager directly to prove
    # the /api/jobs plumbing, then assert /api/run validates the action key.
    bad = c.post("/api/run", json={"action": "nope"})
    assert bad.status_code == 400


def test_run_starts_job_and_log(client):
    c, jm, _ = client
    job = jm.start("probe", [sys.executable, "-c", "print('hello-ui')"])
    # job appears via the API
    listed = c.get("/api/jobs").json()
    assert any(j["id"] == job["id"] for j in listed)
    # log endpoint returns text (may be empty until flushed) without error
    log = c.get(f"/api/jobs/{job['id']}/log").json()
    assert "log" in log
    assert c.get("/api/jobs/does-not-exist/log").status_code == 404


def test_train_start_refuses_second_trainer(client, monkeypatch):
    c, jm, _ = client
    monkeypatch.setattr(jm, "is_training_running", lambda: True)
    r = c.post("/api/train/start", json={"config": "configs/small.yaml"})
    assert r.status_code == 409
    assert "error" in r.json()
