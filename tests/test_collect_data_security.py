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
