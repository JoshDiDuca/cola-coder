"""BUG: checkpoint save crashed on a transient Windows rename lock (WinError 5).
_atomic_replace_dir retries + copy-fallback so an unattended run isn't killed."""

from pathlib import Path


from cola_coder.training.checkpoint import _atomic_replace_dir


def _mktmp(tmp_path, name, content="x"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.safetensors").write_text(content)
    return d


def test_succeeds_normally(tmp_path):
    tmp = _mktmp(tmp_path, ".tmp_step", "weights")
    final = tmp_path / "step"
    _atomic_replace_dir(tmp, final)
    assert final.exists() and not tmp.exists()
    assert (final / "model.safetensors").read_text() == "weights"


def test_replaces_existing_final(tmp_path):
    (tmp_path / "step").mkdir()
    (tmp_path / "step" / "old.txt").write_text("old")
    tmp = _mktmp(tmp_path, ".tmp_step", "new")
    _atomic_replace_dir(tmp, tmp_path / "step")
    final = tmp_path / "step"
    assert (final / "model.safetensors").read_text() == "new"
    assert not (final / "old.txt").exists()  # fully replaced


def test_retries_then_succeeds(tmp_path, monkeypatch):
    tmp = _mktmp(tmp_path, ".tmp_step", "w")
    final = tmp_path / "step"
    calls = {"n": 0}
    real_rename = Path.rename

    def flaky_rename(self, target):
        calls["n"] += 1
        if calls["n"] < 3:           # fail the first 2 attempts
            raise PermissionError(5, "Access is denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # don't actually wait
    _atomic_replace_dir(tmp, final)
    assert final.exists() and calls["n"] == 3


def test_copy_fallback_when_rename_always_fails(tmp_path, monkeypatch):
    tmp = _mktmp(tmp_path, ".tmp_step", "fallback")
    final = tmp_path / "step"
    monkeypatch.setattr(Path, "rename",
                        lambda self, target: (_ for _ in ()).throw(PermissionError(5, "denied")))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _atomic_replace_dir(tmp, final, retries=2)
    # rename never worked, but the copy fallback produced the final dir
    assert final.exists()
    assert (final / "model.safetensors").read_text() == "fallback"
