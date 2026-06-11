"""DATA-017: QualityFilterPlugin must pass languages to filter_code.

The plugin fell back to record.metadata.get("languages") (PLURAL) when no
config languages were set, but sources emit the canonical SINGULAR "language"
key (DATA-007/008) — so the fallback was always None and the language-aware
quality checks never engaged. filter_code expects a list[str].
"""

import cola_coder.data.quality_filter as qf
from cola_coder.data.pipeline import DataRecord
from cola_coder.data.filters.quality import QualityFilterPlugin


def _spy(monkeypatch):
    captured = {}

    def fake(content, mode, languages=None):
        captured["languages"] = languages
        captured["mode"] = mode
        return True, ""

    monkeypatch.setattr(qf, "filter_code", fake)
    return captured


class TestLanguageForwarding:
    def test_config_languages_take_precedence(self, monkeypatch):
        captured = _spy(monkeypatch)
        plugin = QualityFilterPlugin()
        plugin.setup({"languages": ["python", "go"]})
        plugin.check(DataRecord(content="x=1", metadata={"language": "typescript"}))
        assert captured["languages"] == ["python", "go"]

    def test_falls_back_to_record_language_singular(self, monkeypatch):
        # No config languages → use the record's singular "language", as a list.
        captured = _spy(monkeypatch)
        plugin = QualityFilterPlugin()
        plugin.check(DataRecord(content="x=1", metadata={"language": "typescript"}))
        assert captured["languages"] == ["typescript"]

    def test_none_when_no_language_anywhere(self, monkeypatch):
        captured = _spy(monkeypatch)
        plugin = QualityFilterPlugin()
        plugin.check(DataRecord(content="x=1", metadata={}))
        assert captured["languages"] is None

    def test_plural_languages_key_no_longer_required(self, monkeypatch):
        # A record with only the (wrong) plural key still falls through to None
        # rather than silently picking it up — the singular key is canonical.
        captured = _spy(monkeypatch)
        plugin = QualityFilterPlugin()
        plugin.check(DataRecord(content="x=1", metadata={"languages": ["rust"]}))
        assert captured["languages"] is None


class TestMode:
    def test_mode_forwarded(self, monkeypatch):
        captured = _spy(monkeypatch)
        plugin = QualityFilterPlugin()
        plugin.setup({"mode": "strict"})
        plugin.check(DataRecord(content="x=1", metadata={}))
        assert captured["mode"] == qf.FilterMode.STRICT
