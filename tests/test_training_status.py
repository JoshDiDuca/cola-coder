"""TOOL-018: training_status.py must not mishandle step 0.

`training_status.py` is a CPU-only, read-only observer the user relies on to
know an unattended run is healthy. Very early in a run the only checkpoint on
disk is ``step_00000000`` and the best loss so far is whatever was recorded at
step 0.  The original code used truthiness checks (``if step``, ``if best_step``,
``max_steps and step``) which treat the *valid* value 0 as "missing", so the
readout reported "Latest step: ?", "Progress: ?" and dropped the "(step 0)"
annotation on the best loss — misreporting a healthy run as having no progress.

These tests capture the kv-table that ``_describe_size`` builds and lock the
step-0 behaviour.
"""

import importlib.util
import json
from pathlib import Path

import yaml

_SCRIPT = Path(__file__).parent.parent / "scripts" / "training_status.py"


def _load():
    spec = importlib.util.spec_from_file_location("training_status_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_size_dir(tmp_path: Path, *, step: int, best_step: int, max_steps: int = 20000) -> Path:
    """Create a checkpoints/<size>/ dir with one step_* checkpoint + manifest."""
    size_dir = tmp_path / "tiny"
    step_dir = size_dir / f"step_{step:08d}"
    step_dir.mkdir(parents=True)
    (step_dir / "metadata.json").write_text(
        json.dumps(
            {"step": step, "loss": 10.4, "config": {"training": {"max_steps": max_steps}}}
        )
    )
    manifest = {
        "progress": {
            "current_step": step,
            "total_steps": max_steps,
            "tokens_seen": 1000,
            "epochs_completed": 0.0,
            "current_loss": 10.4,
            "best_loss": 10.4,
            "best_step": best_step,
            "loss_history": {f"step_{step}": 10.4},
        }
    }
    (size_dir / "training_manifest.yaml").write_text(yaml.safe_dump(manifest))
    return size_dir


def _capture_table(monkeypatch, module, size_dir: Path) -> dict:
    """Run _describe_size and capture the dict passed to cli.kv_table."""
    captured: dict = {}

    def fake_kv_table(items, title=""):
        captured.update(items)

    monkeypatch.setattr(module.cli, "kv_table", fake_kv_table)
    # Silence the other cli outputs the function emits.
    monkeypatch.setattr(module.cli, "print", lambda *a, **k: None)
    monkeypatch.setattr(module.cli, "dim", lambda *a, **k: None)

    module._describe_size(size_dir, show_curve=False)
    return captured


class TestStepZeroIsValid:
    def test_latest_step_zero_is_reported_not_unknown(self, tmp_path, monkeypatch):
        m = _load()
        size_dir = _make_size_dir(tmp_path, step=0, best_step=0)
        table = _capture_table(monkeypatch, m, size_dir)
        # Regression: was "?" because `if step` treats 0 as missing.
        assert table["Latest step"] == "0"

    def test_progress_zero_pct_at_step_zero(self, tmp_path, monkeypatch):
        m = _load()
        size_dir = _make_size_dir(tmp_path, step=0, best_step=0, max_steps=20000)
        table = _capture_table(monkeypatch, m, size_dir)
        # Regression: was "?" because `max_steps and step` is False at step 0.
        assert table["Progress"].startswith("0.0%")
        assert "0 / 20,000 steps" in table["Progress"]

    def test_best_loss_keeps_step_zero_annotation(self, tmp_path, monkeypatch):
        m = _load()
        size_dir = _make_size_dir(tmp_path, step=0, best_step=0)
        table = _capture_table(monkeypatch, m, size_dir)
        # Regression: "(step 0)" was dropped because `if best_step` is False.
        assert table["Best loss"] == "10.4000  (step 0)"


class TestNonZeroStepUnaffected:
    def test_normal_step_still_renders(self, tmp_path, monkeypatch):
        m = _load()
        size_dir = _make_size_dir(tmp_path, step=5000, best_step=4000, max_steps=20000)
        table = _capture_table(monkeypatch, m, size_dir)
        assert table["Latest step"] == "5,000"
        assert "25.0%" in table["Progress"]
        assert table["Best loss"] == "10.4000  (step 4,000)"
