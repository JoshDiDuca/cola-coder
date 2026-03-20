"""Tests for the GitHub scraper module.

Tests cover:
- RepoFilter.to_github_query() produces correct query strings
- RepoProcessor.IGNORE_PATTERNS correctly filters paths
- MetadataCache stores and retrieves with TTL
- GitHubClient initialization
- RepoProcessor file extraction and quality checks

All tests use mocks/fixtures — no real API calls.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


from cola_coder.data.sources.github import (
    DataRecord,
    GitHubClient,
    MetadataCache,
    RepoFilter,
    RepoProcessor,
    FILTER_PRESETS,
)


# ---------------------------------------------------------------------------
# RepoFilter tests
# ---------------------------------------------------------------------------


class TestRepoFilter:
    """Test RepoFilter.to_github_query() produces correct query strings."""

    def test_empty_filter(self):
        """An empty filter with defaults should still produce archived:false fork:false."""
        f = RepoFilter()
        query = f.to_github_query()
        assert "archived:false" in query
        assert "fork:false" in query

    def test_min_stars_only(self):
        f = RepoFilter(min_stars=100, not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "stars:>=100" in query

    def test_max_stars_only(self):
        f = RepoFilter(max_stars=500, not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "stars:<=500" in query

    def test_star_range(self):
        f = RepoFilter(min_stars=100, max_stars=500, not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "stars:100..500" in query

    def test_min_forks_only(self):
        f = RepoFilter(min_forks=10, not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "forks:>=10" in query

    def test_fork_range(self):
        f = RepoFilter(min_forks=5, max_forks=100, not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "forks:5..100" in query

    def test_primary_language(self):
        f = RepoFilter(primary_language="TypeScript", not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "language:TypeScript" in query

    def test_licenses(self):
        f = RepoFilter(licenses=["mit", "apache-2.0"], not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "license:mit" in query
        assert "license:apache-2.0" in query

    def test_pushed_after(self):
        f = RepoFilter(pushed_after="2023-01-01", not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "pushed:>=2023-01-01" in query

    def test_created_after(self):
        f = RepoFilter(created_after="2022-06-15", not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "created:>=2022-06-15" in query

    def test_not_archived(self):
        f = RepoFilter(not_archived=True, is_fork=None)
        assert "archived:false" in f.to_github_query()

    def test_archived_not_filtered(self):
        f = RepoFilter(not_archived=False, is_fork=None)
        assert "archived:" not in f.to_github_query()

    def test_fork_false(self):
        f = RepoFilter(is_fork=False, not_archived=False)
        assert "fork:false" in f.to_github_query()

    def test_fork_true(self):
        f = RepoFilter(is_fork=True, not_archived=False)
        assert "fork:true" in f.to_github_query()

    def test_fork_none(self):
        """is_fork=None means no fork filter."""
        f = RepoFilter(is_fork=None, not_archived=False)
        assert "fork:" not in f.to_github_query()

    def test_topics(self):
        f = RepoFilter(topics_include=["react", "nextjs"], not_archived=False, is_fork=None)
        query = f.to_github_query()
        assert "topic:react" in query
        assert "topic:nextjs" in query

    def test_max_size(self):
        f = RepoFilter(max_repo_size_kb=100000, not_archived=False, is_fork=None)
        assert "size:<=100000" in f.to_github_query()

    def test_complex_filter(self):
        """A realistic combined filter produces a valid query."""
        f = RepoFilter(
            min_stars=500,
            primary_language="TypeScript",
            licenses=["mit"],
            pushed_after="2023-01-01",
            not_archived=True,
            is_fork=False,
            max_repo_size_kb=500000,
        )
        query = f.to_github_query()
        assert "language:TypeScript" in query
        assert "stars:>=500" in query
        assert "license:mit" in query
        assert "pushed:>=2023-01-01" in query
        assert "archived:false" in query
        assert "fork:false" in query
        assert "size:<=500000" in query


class TestFilterPresets:
    """Test that all presets produce valid non-empty queries."""

    def test_all_presets_produce_queries(self):
        for name, preset in FILTER_PRESETS.items():
            query = preset.to_github_query()
            assert query, f"Preset '{name}' produced empty query"
            assert len(query) > 10, f"Preset '{name}' query too short: {query}"

    def test_typescript_elite_has_expected_parts(self):
        query = FILTER_PRESETS["typescript_elite"].to_github_query()
        assert "language:TypeScript" in query
        assert "stars:>=500" in query

    def test_popular_any_no_language(self):
        """popular_any should not filter by language."""
        query = FILTER_PRESETS["popular_any"].to_github_query()
        assert "language:" not in query
        assert "stars:>=1000" in query


# ---------------------------------------------------------------------------
# RepoProcessor tests
# ---------------------------------------------------------------------------


class TestRepoProcessorIgnorePatterns:
    """Test that IGNORE_PATTERNS correctly filters paths."""

    def setup_method(self):
        self.processor = RepoProcessor()

    def test_node_modules_ignored(self):
        path = Path("node_modules/express/index.js")
        assert self.processor._should_ignore(path)

    def test_dist_ignored(self):
        path = Path("dist/bundle.js")
        assert self.processor._should_ignore(path)

    def test_git_ignored(self):
        path = Path(".git/objects/abc123")
        assert self.processor._should_ignore(path)

    def test_pycache_ignored(self):
        path = Path("src/__pycache__/module.cpython-310.pyc")
        assert self.processor._should_ignore(path)

    def test_venv_ignored(self):
        path = Path(".venv/lib/python3.10/site-packages/requests/__init__.py")
        assert self.processor._should_ignore(path)

    def test_normal_src_not_ignored(self):
        path = Path("src/app/main.ts")
        assert not self.processor._should_ignore(path)

    def test_nested_src_not_ignored(self):
        path = Path("packages/core/src/index.ts")
        assert not self.processor._should_ignore(path)

    def test_egg_info_ignored(self):
        path = Path("my_package.egg-info/PKG-INFO")
        assert self.processor._should_ignore(path)

    def test_target_ignored(self):
        path = Path("target/debug/build/something")
        assert self.processor._should_ignore(path)


class TestRepoProcessorSecretDetection:
    """Test that secret files are correctly identified."""

    def setup_method(self):
        self.processor = RepoProcessor()

    def test_env_file(self):
        assert self.processor._is_secret_file(Path(".env"))

    def test_env_local(self):
        assert self.processor._is_secret_file(Path(".env.local"))

    def test_credentials_json(self):
        assert self.processor._is_secret_file(Path("credentials.json"))

    def test_normal_file_not_secret(self):
        assert not self.processor._is_secret_file(Path("main.ts"))

    def test_config_file_not_secret(self):
        assert not self.processor._is_secret_file(Path("config.json"))


class TestRepoProcessorLanguageFilter:
    """Test language-based file filtering."""

    def test_typescript_only(self):
        proc = RepoProcessor(languages=["TypeScript"])
        assert ".ts" in proc.allowed_extensions
        assert ".tsx" in proc.allowed_extensions
        assert ".py" not in proc.allowed_extensions

    def test_python_only(self):
        proc = RepoProcessor(languages=["Python"])
        assert ".py" in proc.allowed_extensions
        assert ".ts" not in proc.allowed_extensions

    def test_all_languages(self):
        proc = RepoProcessor(languages=None)
        assert ".ts" in proc.allowed_extensions
        assert ".py" in proc.allowed_extensions
        assert ".go" in proc.allowed_extensions


class TestRepoProcessorExtractFiles:
    """Test file extraction from a mock repo directory."""

    def test_extracts_valid_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some source files
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "main.ts").write_text("console.log('hello world');", encoding="utf-8")
            (src_dir / "utils.py").write_text("def hello():\n    return 'world'", encoding="utf-8")

            proc = RepoProcessor()
            records = list(proc.extract_files(tmpdir, repo_name="test/repo"))

            assert len(records) == 2
            languages = {r.language for r in records}
            assert "TypeScript" in languages
            assert "Python" in languages

    def test_skips_node_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm_dir = Path(tmpdir) / "node_modules" / "express"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.js").write_text("module.exports = {};", encoding="utf-8")

            # Also create a valid file
            (Path(tmpdir) / "app.js").write_text("const x = require('express');", encoding="utf-8")

            proc = RepoProcessor()
            records = list(proc.extract_files(tmpdir))

            assert len(records) == 1
            assert records[0].file_path == "app.js"

    def test_skips_secret_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".env").write_text("SECRET_KEY=abc123", encoding="utf-8")
            (Path(tmpdir) / "app.py").write_text("import os\nprint('hello')", encoding="utf-8")

            proc = RepoProcessor()
            records = list(proc.extract_files(tmpdir))

            assert len(records) == 1
            assert records[0].file_path == "app.py"

    def test_skips_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty.py").write_text("", encoding="utf-8")
            (Path(tmpdir) / "tiny.py").write_text("x=1", encoding="utf-8")  # < 10 chars
            (Path(tmpdir) / "valid.py").write_text("def hello():\n    return 42", encoding="utf-8")

            proc = RepoProcessor()
            records = list(proc.extract_files(tmpdir))

            assert len(records) == 1
            assert records[0].file_path == "valid.py"

    def test_respects_language_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.ts").write_text("console.log('hello world');", encoding="utf-8")
            (Path(tmpdir) / "main.py").write_text("def hello():\n    return 'world'", encoding="utf-8")

            proc = RepoProcessor(languages=["TypeScript"])
            records = list(proc.extract_files(tmpdir))

            assert len(records) == 1
            assert records[0].language == "TypeScript"

    def test_record_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.ts").write_text(
                "export function greet(): string { return 'hi'; }",
                encoding="utf-8",
            )

            proc = RepoProcessor()
            records = list(proc.extract_files(
                tmpdir,
                repo_name="owner/repo",
                repo_stars=1000,
                repo_url="https://github.com/owner/repo",
                repo_license="MIT",
            ))

            assert len(records) == 1
            r = records[0]
            assert r.repo_name == "owner/repo"
            assert r.repo_stars == 1000
            assert r.repo_url == "https://github.com/owner/repo"
            assert r.license == "MIT"
            assert r.language == "TypeScript"


class TestRepoProcessorQualityChecks:
    """Test license detection, test detection, CI detection, strict TS."""

    def test_check_license_mit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "LICENSE").write_text(
                "MIT License\n\nCopyright (c) 2024\n\n"
                "Permission is hereby granted, free of charge...",
                encoding="utf-8",
            )
            assert RepoProcessor.check_license(tmpdir) == "MIT"

    def test_check_license_apache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "LICENSE").write_text(
                "Apache License\nVersion 2.0, January 2004\n"
                "http://www.apache.org/licenses/",
                encoding="utf-8",
            )
            assert RepoProcessor.check_license(tmpdir) == "Apache-2.0"

    def test_check_license_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert RepoProcessor.check_license(tmpdir) is None

    def test_check_has_tests_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "tests").mkdir()
            assert RepoProcessor.check_has_tests(tmpdir) is True

    def test_check_has_tests_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "app.test.ts").write_text("test('foo', () => {})", encoding="utf-8")
            assert RepoProcessor.check_has_tests(tmpdir) is True

    def test_check_no_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "app.ts").write_text("const x = 1;", encoding="utf-8")
            assert RepoProcessor.check_has_tests(tmpdir) is False

    def test_check_has_ci_github_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = Path(tmpdir) / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "ci.yml").write_text("on: push", encoding="utf-8")
            assert RepoProcessor.check_has_ci(tmpdir) is True

    def test_check_has_ci_travis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".travis.yml").write_text("language: node_js", encoding="utf-8")
            assert RepoProcessor.check_has_ci(tmpdir) is True

    def test_check_no_ci(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert RepoProcessor.check_has_ci(tmpdir) is False

    def test_tsconfig_strict_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tsconfig = {"compilerOptions": {"strict": True, "target": "ES2020"}}
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )
            assert RepoProcessor.check_tsconfig_strict(tmpdir) is True

    def test_tsconfig_strict_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tsconfig = {"compilerOptions": {"target": "ES2020"}}
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )
            assert RepoProcessor.check_tsconfig_strict(tmpdir) is False

    def test_tsconfig_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert RepoProcessor.check_tsconfig_strict(tmpdir) is False

    def test_tsconfig_with_comments(self):
        """tsconfig.json often has comments — parser should handle them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """{
                // Enable strict type checking
                "compilerOptions": {
                    "strict": true,
                    "target": "ES2020"
                }
            }"""
            (Path(tmpdir) / "tsconfig.json").write_text(content, encoding="utf-8")
            assert RepoProcessor.check_tsconfig_strict(tmpdir) is True


# ---------------------------------------------------------------------------
# MetadataCache tests
# ---------------------------------------------------------------------------


class TestMetadataCache:
    """Test cache storage, retrieval, and TTL expiry."""

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MetadataCache(cache_dir=tmpdir, ttl_days=7)
            data = {"stars": 1000, "name": "test/repo"}

            cache.set("test-key", data)
            result = cache.get("test-key")

            assert result is not None
            assert result["stars"] == 1000
            assert result["name"] == "test/repo"

    def test_get_missing_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MetadataCache(cache_dir=tmpdir)
            assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MetadataCache(cache_dir=tmpdir, ttl_days=0)  # 0 day TTL = expire immediately
            cache.set("test-key", {"value": 42})

            # Manually backdate the cached_at timestamp
            for f in Path(tmpdir).glob("*.json"):
                raw = json.loads(f.read_text())
                old_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
                raw["_cached_at"] = old_time
                f.write_text(json.dumps(raw))

            result = cache.get("test-key")
            assert result is None

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MetadataCache(cache_dir=tmpdir)
            cache.set("key1", {"a": 1})
            cache.set("key2", {"b": 2})

            count = cache.clear()
            assert count == 2
            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MetadataCache(cache_dir=tmpdir)

            cache.set("key", {"version": 1})
            cache.set("key", {"version": 2})

            result = cache.get("key")
            assert result["version"] == 2


# ---------------------------------------------------------------------------
# GitHubClient tests
# ---------------------------------------------------------------------------


class TestGitHubClientInit:
    """Test GitHubClient initialization and configuration."""

    def test_init_with_token(self):
        client = GitHubClient(token="ghp_test123")
        assert client.token == "ghp_test123"
        assert "token ghp_test123" in client.session.headers.get("Authorization", "")

    def test_init_with_env_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_env_token"}):
            client = GitHubClient()
            assert client.token == "ghp_env_token"

    def test_init_without_token(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove GITHUB_TOKEN if it exists
            env = os.environ.copy()
            env.pop("GITHUB_TOKEN", None)
            with patch.dict(os.environ, env, clear=True):
                client = GitHubClient()
                assert client.token is None
                assert "Authorization" not in client.session.headers

    def test_headers_set(self):
        client = GitHubClient(token="ghp_test")
        assert "application/vnd.github.v3+json" in client.session.headers["Accept"]
        assert "cola-coder" in client.session.headers["User-Agent"]

    @patch("cola_coder.data.sources.github.requests.Session")
    def test_search_repos_calls_api(self, mock_session_class):
        """Test that search_repos makes the right API call (mocked)."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "X-RateLimit-Remaining": "100",
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
        }
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [{"full_name": "test/repo", "stargazers_count": 500}],
        }
        mock_session.request.return_value = mock_response

        client = GitHubClient.__new__(GitHubClient)
        client.token = None
        client.session = mock_session
        client._rate_remaining = None
        client._rate_reset = None

        f = RepoFilter(min_stars=100, primary_language="TypeScript")
        repos = client.search_repos(f, max_results=10)

        assert len(repos) == 1
        assert repos[0]["full_name"] == "test/repo"

        # Verify the API was called
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "GET"
        assert "search/repositories" in call_args[0][1]


# ---------------------------------------------------------------------------
# DataRecord tests
# ---------------------------------------------------------------------------


class TestDataRecord:
    """Test DataRecord dataclass."""

    def test_creation(self):
        record = DataRecord(
            content="console.log('hello');",
            file_path="src/main.ts",
            language="TypeScript",
            repo_name="test/repo",
            repo_stars=100,
        )
        assert record.content == "console.log('hello');"
        assert record.language == "TypeScript"
        assert record.repo_stars == 100

    def test_defaults(self):
        record = DataRecord(
            content="x = 1",
            file_path="main.py",
            language="Python",
            repo_name="",
        )
        assert record.repo_stars == 0
        assert record.repo_url == ""
        assert record.license == ""
        assert record.file_size == 0
