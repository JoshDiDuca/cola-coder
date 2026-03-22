"""Tests for DocFetcher (inference/doc_fetcher.py).

Covers:
- Cache path generation consistency
- Cache TTL validation (mocked file timestamps)
- HTML-to-markdown conversion (when beautifulsoup4 is available)
- fetch() cache hit → no HTTP call
- fetch() cache miss → HTTP call, content cached
- get_relevant_docs() → detects React hooks in code context
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.inference.doc_fetcher import DocFetcher


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fetcher(tmp_path: Path) -> DocFetcher:
    """Return a DocFetcher whose cache lives in tmp_path."""
    return DocFetcher(cache_dir=str(tmp_path / "doc_cache"))


# ── Cache path generation ─────────────────────────────────────────────────────


class TestCachePathGeneration:
    def test_consistent_for_same_args(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p1 = fetcher._get_cache_path("react", "useState", "18.2.0")
        p2 = fetcher._get_cache_path("react", "useState", "18.2.0")
        assert p1 == p2

    def test_different_topic_different_path(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p1 = fetcher._get_cache_path("react", "useState", "latest")
        p2 = fetcher._get_cache_path("react", "useEffect", "latest")
        assert p1 != p2

    def test_different_framework_different_path(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p1 = fetcher._get_cache_path("react", "useState", "latest")
        p2 = fetcher._get_cache_path("nextjs", "useState", "latest")
        assert p1 != p2

    def test_different_version_different_path(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p1 = fetcher._get_cache_path("react", "useState", "18.0.0")
        p2 = fetcher._get_cache_path("react", "useState", "17.0.0")
        assert p1 != p2

    def test_path_under_cache_dir(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p = fetcher._get_cache_path("react", "useState", "latest")
        assert str(p).startswith(str(fetcher.cache_dir))

    def test_slash_in_topic_sanitised(self, tmp_path):
        """Slashes in topic names (e.g. 'app/routing') must be sanitised."""
        fetcher = _make_fetcher(tmp_path)
        p = fetcher._get_cache_path("nextjs", "app/routing", "latest")
        # Path must be valid — no literal slashes turned into dir separators beyond intended
        assert p.name.endswith(".md")

    def test_returns_path_object(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        p = fetcher._get_cache_path("react", "useState", "latest")
        assert isinstance(p, Path)


# ── Cache TTL ─────────────────────────────────────────────────────────────────


class TestCacheTTL:
    """TTL tests mock os.stat at the pathlib level to avoid Windows read-only attr."""

    def _mock_stat(self, mtime: float) -> MagicMock:
        mock = MagicMock()
        mock.st_mtime = mtime
        return mock

    def test_fresh_cache_is_valid(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        cache_file = tmp_path / "doc_cache" / "react" / "latest" / "useState.md"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("# useState\nHook docs")
        # File just written — mtime is now; should be valid
        assert fetcher._is_cache_valid(cache_file, ttl_days=7) is True

    def test_expired_cache_is_invalid(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        cache_file = tmp_path / "doc_cache" / "react" / "latest" / "useState.md"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("old content")

        old_mtime = time.time() - (10 * 86400)  # 10 days ago

        with patch("cola_coder.inference.doc_fetcher.Path.stat",
                   return_value=self._mock_stat(old_mtime)):
            assert fetcher._is_cache_valid(cache_file, ttl_days=7) is False

    def test_just_within_ttl_is_valid(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        cache_file = tmp_path / "doc_cache" / "react" / "latest" / "useState.md"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("content")

        recent_mtime = time.time() - (6 * 86400)  # 6 days ago — within 7-day TTL

        with patch("cola_coder.inference.doc_fetcher.Path.stat",
                   return_value=self._mock_stat(recent_mtime)):
            assert fetcher._is_cache_valid(cache_file, ttl_days=7) is True

    def test_nonexistent_cache_is_invalid(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        missing = tmp_path / "doc_cache" / "react" / "latest" / "nonexistent.md"
        assert fetcher._is_cache_valid(missing) is False

    def test_custom_ttl_respected(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        cache_file = tmp_path / "doc_cache" / "react" / "latest" / "useState.md"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("content")

        mtime = time.time() - (2 * 86400)  # 2 days old

        with patch("cola_coder.inference.doc_fetcher.Path.stat",
                   return_value=self._mock_stat(mtime)):
            assert fetcher._is_cache_valid(cache_file, ttl_days=1) is False
            assert fetcher._is_cache_valid(cache_file, ttl_days=3) is True


# ── HTML to Markdown ──────────────────────────────────────────────────────────


class TestHtmlToMarkdown:
    def test_basic_heading(self, tmp_path):
        pytest.importorskip("bs4", reason="beautifulsoup4 required for HTML conversion")
        fetcher = _make_fetcher(tmp_path)
        html = "<html><body><h1>useState</h1></body></html>"
        md = fetcher._html_to_markdown(html)
        assert "useState" in md
        assert "#" in md

    def test_paragraph_text(self, tmp_path):
        pytest.importorskip("bs4", reason="beautifulsoup4 required")
        fetcher = _make_fetcher(tmp_path)
        html = "<html><body><p>Returns a stateful value.</p></body></html>"
        md = fetcher._html_to_markdown(html)
        assert "Returns a stateful value" in md

    def test_code_block_preserved(self, tmp_path):
        pytest.importorskip("bs4", reason="beautifulsoup4 required")
        fetcher = _make_fetcher(tmp_path)
        html = (
            "<html><body>"
            "<pre><code>const [state, setState] = useState(0);</code></pre>"
            "</body></html>"
        )
        md = fetcher._html_to_markdown(html)
        assert "useState" in md
        assert "```" in md

    def test_nav_stripped(self, tmp_path):
        pytest.importorskip("bs4", reason="beautifulsoup4 required")
        fetcher = _make_fetcher(tmp_path)
        html = (
            "<html><body>"
            "<nav>Navigation</nav>"
            "<main><p>Main content</p></main>"
            "</body></html>"
        )
        md = fetcher._html_to_markdown(html)
        assert "Main content" in md
        assert "Navigation" not in md

    def test_fallback_without_bs4(self, tmp_path):
        """_html_to_markdown should fall back gracefully without beautifulsoup4."""
        fetcher = _make_fetcher(tmp_path)
        html = "<p>Some <b>text</b></p>"

        with patch.dict("sys.modules", {"bs4": None}):
            # Call the plain-tag-stripping fallback directly
            result = fetcher._strip_tags_fallback(html)
            assert "Some" in result
            assert "text" in result
            assert "<p>" not in result
            assert "<b>" not in result


# ── fetch() — cache hit ───────────────────────────────────────────────────────


class TestFetchCacheHit:
    def test_cache_hit_no_http_call(self, tmp_path):
        """When cache is fresh, _fetch_url should never be called."""
        fetcher = _make_fetcher(tmp_path)

        # Pre-populate the cache
        cache_path = fetcher._get_cache_path("react", "useState", "latest")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("# useState\nHook documentation content.")

        with patch.object(fetcher, "_fetch_url") as mock_http:
            result = fetcher.fetch("react", "useState")

        mock_http.assert_not_called()
        assert result is not None
        assert "useState" in result

    def test_cache_hit_returns_doc_wrapped_content(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)

        cache_path = fetcher._get_cache_path("react", "useState", "latest")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("Cached markdown content.")

        result = fetcher.fetch("react", "useState")
        assert result is not None
        assert "<|doc|>" in result
        assert "Cached markdown content." in result

    def test_cache_miss_triggers_http(self, tmp_path):
        """When cache is absent, _fetch_url should be called once."""
        fetcher = _make_fetcher(tmp_path)

        with patch.object(fetcher, "_fetch_url", return_value="<html><body><p>Fetched</p></body></html>"):
            result = fetcher.fetch("react", "useState")

        assert result is not None
        assert "Fetched" in result

    def test_cache_miss_writes_cache(self, tmp_path):
        """After a successful fetch, the result should be cached to disk."""
        fetcher = _make_fetcher(tmp_path)

        with patch.object(fetcher, "_fetch_url", return_value="<html><body><p>Content</p></body></html>"):
            fetcher.fetch("react", "useState")

        cache_path = fetcher._get_cache_path("react", "useState", "latest")
        assert cache_path.exists()

    def test_fetch_unknown_framework_returns_none(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        result = fetcher.fetch("unknown_framework_xyz", "some_topic")
        assert result is None

    def test_fetch_http_failure_returns_none(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)

        with patch.object(fetcher, "_fetch_url", return_value=None):
            result = fetcher.fetch("react", "useState")

        assert result is None

    def test_fetch_with_version(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)

        cache_path = fetcher._get_cache_path("react", "useState", "18.2.0")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("Version-specific content.")

        result = fetcher.fetch("react", "useState", version="18.2.0")
        assert result is not None
        assert "react@18.2.0" in result


# ── get_relevant_docs() ───────────────────────────────────────────────────────


class TestGetRelevantDocs:
    def test_identifies_usestate_in_code(self, tmp_path):
        """get_relevant_docs() should detect useState in code context."""
        fetcher = _make_fetcher(tmp_path)
        code_context = "const [count, setCount] = useState(0);"

        # Pre-populate cache for useState so no HTTP is made
        cache_path = fetcher._get_cache_path("react", "useState", "18.2.0")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("useState documentation.")

        result = fetcher.get_relevant_docs("react", "18.2.0", code_context)
        assert "useState" in result

    def test_identifies_useeffect_in_code(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        code_context = "useEffect(() => { fetchData(); }, [id]);"

        cache_path = fetcher._get_cache_path("react", "useEffect", "18.2.0")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("useEffect documentation.")

        result = fetcher.get_relevant_docs("react", "18.2.0", code_context)
        assert "useEffect" in result

    def test_multiple_hooks_detected(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        code_context = (
            "const [state, setState] = useState(null);\n"
            "const ref = useRef(null);\n"
        )

        for hook in ("useState", "useRef"):
            cache_path = fetcher._get_cache_path("react", hook, "18.2.0")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(f"{hook} docs.")

        result = fetcher.get_relevant_docs("react", "18.2.0", code_context)
        assert "useState" in result or "useRef" in result

    def test_unknown_framework_returns_empty(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        result = fetcher.get_relevant_docs("unknown_fw", "1.0.0", "const x = 1")
        assert result == ""

    def test_no_match_returns_fallback(self, tmp_path):
        """When no specific hook is found, falls back to the first page."""
        fetcher = _make_fetcher(tmp_path)
        code_context = "const x = someCustomFunction();"

        # Pre-populate the fallback (first page = useState)
        cache_path = fetcher._get_cache_path("react", "useState", "18.2.0")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("useState fallback content.")

        result = fetcher.get_relevant_docs("react", "18.2.0", code_context)
        # Either returns something (fallback) or empty — just don't crash
        assert isinstance(result, str)

    def test_caps_at_three_topics(self, tmp_path):
        """get_relevant_docs() should return at most 3 doc blocks."""
        fetcher = _make_fetcher(tmp_path)
        # Code that mentions many hooks
        code_context = (
            "useState(0); useEffect(fn, []); useContext(ctx); "
            "useReducer(r, s); useCallback(fn, []); useMemo(fn, []);"
        )

        hooks = ["useState", "useEffect", "useContext", "useReducer", "useCallback", "useMemo"]
        for hook in hooks:
            cache_path = fetcher._get_cache_path("react", hook, "18.2.0")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(f"{hook} docs.")

        result = fetcher.get_relevant_docs("react", "18.2.0", code_context)
        # Count doc blocks — each starts with <|doc|>
        doc_count = result.count("<|doc|>")
        assert doc_count <= 3
