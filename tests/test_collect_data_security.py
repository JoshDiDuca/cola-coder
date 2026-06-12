"""Security tests for collect_data in-stream malware scanning (SEC-001/SEC-002).

- Disabling in-stream scanning must NEVER be silent (loud SECURITY warning).
- The post-download 'warn' path must fail CLOSED in non-interactive runs
  (cli.confirm default=False), not silently ingest malware.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
_SCRIPT = ROOT / "scripts" / "collect_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("collect_data", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInStreamScanGating:
    def test_enabled_returns_scanning_wrapper_that_passes_clean(self):
        mod = _load()
        stats: dict = {}
        cfg = {"malware_scan": {"enabled": True, "in_stream": True}}
        wrapped = mod._maybe_scan_stream(iter(["hello world\n"]), "code", cfg, stats)
        # It's a different object (the scanning generator), and clean text passes.
        out = list(wrapped)
        assert out == ["hello world\n"]
        assert stats.get("clean") == 1

    def test_disabled_master_switch_warns_loudly(self, monkeypatch):
        mod = _load()
        warnings: list[str] = []
        monkeypatch.setattr(mod.cli, "warn", lambda m, *a, **k: warnings.append(m))
        src = iter(["x"])
        cfg = {"malware_scan": {"enabled": False}}
        result = mod._maybe_scan_stream(src, "text", cfg, {})
        assert result is src  # unscanned, same iterator
        assert any("SECURITY" in w and "DISABLED" in w for w in warnings)

    def test_in_stream_false_warns_loudly(self, monkeypatch):
        mod = _load()
        warnings: list[str] = []
        monkeypatch.setattr(mod.cli, "warn", lambda m, *a, **k: warnings.append(m))
        src = iter(["x"])
        cfg = {"malware_scan": {"enabled": True, "in_stream": False}}
        result = mod._maybe_scan_stream(src, "math", cfg, {})
        assert result is src
        assert any("in-stream" in w.lower() and "DISABLED" in w for w in warnings)

    def test_missing_config_defaults_to_scanning(self):
        # Empty config (e.g. scoring.yaml absent) must DEFAULT to scanning, not
        # silently skip — verified by getting a scanning wrapper, not the source.
        mod = _load()
        src = iter(["clean code\n"])
        wrapped = mod._maybe_scan_stream(src, "code", {}, {})
        assert wrapped is not src


class TestFailClosedAndConfig:
    def test_warn_path_fails_closed(self):
        # The on_threat='warn' branch must pass default=False to cli.confirm so
        # non-interactive runs abort rather than continue with threats.
        text = _SCRIPT.read_text(encoding="utf-8")
        assert "Continue despite threats?" in text
        assert "default=False" in text

    def test_scoring_yaml_safe_defaults(self):
        cfg = (ROOT / "configs" / "scoring.yaml").read_text(encoding="utf-8")
        # Safe defaults present and documented
        assert "enabled: true" in cfg
        assert "in_stream: true" in cfg
        assert 'on_threat: "quarantine"' in cfg
        assert "SECURITY" in cfg


class TestQuarantineDestNoCollision:
    """SEC-008: quarantining two threats with the SAME basename must not
    overwrite — the old `quarantine_dir / src.name` silently lost evidence."""

    def test_same_basename_different_dirs_no_collision(self, tmp_path):
        mod = _load()
        q = tmp_path / "quarantine"
        a = mod._quarantine_dest(q, Path("/repo/pkgA/__init__.py"))
        b = mod._quarantine_dest(q, Path("/repo/pkgB/__init__.py"))
        assert a != b  # distinct destinations despite identical basename
        assert a.name.endswith("___init__.py") and b.name.endswith("___init__.py")

    def test_same_path_is_idempotent(self, tmp_path):
        mod = _load()
        q = tmp_path / "quarantine"
        src = Path("/repo/x/index.js")
        assert mod._quarantine_dest(q, src) == mod._quarantine_dest(q, src)

    def test_dest_is_under_quarantine_dir_and_keeps_basename(self, tmp_path):
        mod = _load()
        q = tmp_path / "quarantine"
        dst = mod._quarantine_dest(q, Path("/a/b/evil.exe"))
        assert dst.parent == q
        assert dst.name.endswith("_evil.exe")

    def test_no_real_overwrite_when_quarantining_collisions(self, tmp_path):
        # End-to-end: two distinct files named the same both survive in quarantine.
        mod = _load()
        q = tmp_path / "quarantine"
        q.mkdir()
        (tmp_path / "p1").mkdir()
        (tmp_path / "p2").mkdir()
        f1 = tmp_path / "p1" / "index.js"
        f1.write_text("AAA")
        f2 = tmp_path / "p2" / "index.js"
        f2.write_text("BBB")
        f1.rename(mod._quarantine_dest(q, f1))
        f2.rename(mod._quarantine_dest(q, f2))
        contents = sorted(p.read_text() for p in q.iterdir())
        assert contents == ["AAA", "BBB"]  # neither was overwritten
