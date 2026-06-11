"""Data source metadata correctness (DATA-007 / DATA-008).

Language-aware scorers/filters infer language from metadata["file_path"] and
metadata["language"] (see scorers/language_detect.py, and github source which
emits both). The local + software_heritage sources used to emit only "path"
(wrong key) and the HuggingFace source emitted no language — so those sources'
files were NOT language-detected by extension and fell back to weaker content
heuristics. These tests lock the canonical keys.
"""

from cola_coder.data.scorers.language_detect import (
    detect_extension,
    is_typescript,
)
from cola_coder.data.sources.local import LocalFileSource


class TestLocalSourceMetadata:
    def test_emits_canonical_file_path_key(self, tmp_path):
        f = tmp_path / "widget.ts"
        # Content WITHOUT TypeScript heuristic markers — so detection must come
        # from the file extension via metadata, not the content.
        f.write_text("export const answer = 42;\n", encoding="utf-8")
        records = list(LocalFileSource([str(tmp_path)], extensions=[".ts"]).stream())
        assert len(records) == 1
        meta = records[0].metadata
        assert meta["file_path"] == str(f)
        assert meta["path"] == str(f)  # legacy key kept for compatibility

    def test_typescript_detected_via_file_path(self, tmp_path):
        f = tmp_path / "plain.ts"
        f.write_text("export const answer = 42;\n", encoding="utf-8")
        rec = next(iter(LocalFileSource([str(tmp_path)], extensions=[".ts"]).stream()))
        # Content alone does NOT look like TS (0 heuristic markers)...
        assert is_typescript(rec.content, None) is False
        # ...but with the file_path metadata it's correctly detected as TS.
        assert is_typescript(rec.content, rec.metadata) is True
        assert detect_extension(rec.metadata) == ".ts"


class TestHuggingFaceSourceMetadata:
    def _patch_stream(self, monkeypatch, items):
        import cola_coder.data.download as dl

        monkeypatch.setattr(dl, "stream_code_data", lambda **kw: iter(items))

    def test_single_language_tags_language(self, monkeypatch):
        from cola_coder.data.sources.huggingface import HuggingFaceSource

        self._patch_stream(monkeypatch, ["const x = 1;\n"])
        src = HuggingFaceSource(dataset="d", languages=["typescript"])
        rec = next(iter(src.stream()))
        assert rec.metadata["language"] == "typescript"
        # And language detection uses it (content has no TS markers).
        assert is_typescript(rec.content, rec.metadata) is True

    def test_multi_language_omits_language(self, monkeypatch):
        from cola_coder.data.sources.huggingface import HuggingFaceSource

        self._patch_stream(monkeypatch, ["x = 1\n"])
        src = HuggingFaceSource(dataset="d", languages=["python", "typescript"])
        rec = next(iter(src.stream()))
        # Per-record language isn't knowable for a multi-language source.
        assert "language" not in rec.metadata
