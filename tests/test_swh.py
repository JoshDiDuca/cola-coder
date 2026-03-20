"""Tests for the Software Heritage data source.

Tests SWHClient initialization, rate limit handling, and the full
origin → visit → snapshot → directory → content flow using mocked
HTTP responses. No real API calls are made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch


from cola_coder.data.sources.software_heritage import (
    SoftwareHeritageSource,
    SWHClient,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable mock data matching SWH API shapes
# ---------------------------------------------------------------------------

MOCK_ORIGIN_URL = "https://github.com/example/repo"

MOCK_VISITS = [
    {
        "visit": 3,
        "status": "full",
        "snapshot": "aaa111",
        "date": "2024-06-01T00:00:00Z",
    },
    {
        "visit": 2,
        "status": "partial",
        "snapshot": "bbb222",
        "date": "2024-01-01T00:00:00Z",
    },
]

MOCK_SNAPSHOT = {
    "id": "aaa111",
    "branches": {
        "HEAD": {"target_type": "alias", "target": "refs/heads/main"},
        "refs/heads/main": {"target_type": "revision", "target": "rev001"},
        "refs/heads/develop": {"target_type": "revision", "target": "rev002"},
    },
}

MOCK_ROOT_DIR = [
    {"name": "src", "type": "dir", "target": "dir001"},
    {"name": "README.md", "type": "file", "target": "cnt001", "length": 100},
    {"name": "node_modules", "type": "dir", "target": "dir_skip"},
]

MOCK_SRC_DIR = [
    {"name": "main.py", "type": "file", "target": "cnt002", "length": 200},
    {"name": "utils.ts", "type": "file", "target": "cnt003", "length": 150},
    {"name": "image.png", "type": "file", "target": "cnt004", "length": 50},
]

MOCK_FILE_CONTENTS = {
    "cnt001": "# Example Repo\n\nThis is a readme file with enough content.",
    "cnt002": 'def hello():\n    print("Hello from Python!")\n\nhello()\n',
    "cnt003": 'export function greet(): string {\n  return "hello";\n}\n',
}


# ---------------------------------------------------------------------------
# SWHClient tests
# ---------------------------------------------------------------------------


class TestSWHClientInit:
    """Test SWHClient initialization and configuration."""

    def test_init_with_token(self):
        client = SWHClient(token="test-token-123")
        assert client.token == "test-token-123"
        assert client.session.headers["Authorization"] == "Bearer test-token-123"

    def test_init_without_token(self):
        with patch.dict("os.environ", {}, clear=True):
            client = SWHClient(token=None)
            assert client.token is None
            assert "Authorization" not in client.session.headers

    def test_init_from_env(self):
        with patch.dict("os.environ", {"SWH_API_TOKEN": "env-token-456"}):
            client = SWHClient()
            assert client.token == "env-token-456"
            assert client.session.headers["Authorization"] == "Bearer env-token-456"

    def test_init_custom_timeout(self):
        client = SWHClient(token="t", timeout=60)
        assert client.timeout == 60

    def test_default_headers(self):
        client = SWHClient(token="t")
        assert "User-Agent" in client.session.headers
        assert client.session.headers["Accept"] == "application/json"


class TestSWHClientRateLimit:
    """Test rate limit header parsing and wait behavior."""

    def test_update_rate_limit_from_headers(self):
        client = SWHClient(token="t")
        mock_resp = MagicMock()
        mock_resp.headers = {
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Reset": "1700000000",
        }

        client._update_rate_limit(mock_resp)
        assert client._rate_remaining == 42
        assert client._rate_reset == 1700000000.0

    def test_update_rate_limit_missing_headers(self):
        client = SWHClient(token="t")
        mock_resp = MagicMock()
        mock_resp.headers = {}

        client._update_rate_limit(mock_resp)
        # Should remain None (not crash)
        assert client._rate_remaining is None
        assert client._rate_reset is None

    def test_update_rate_limit_invalid_values(self):
        client = SWHClient(token="t")
        mock_resp = MagicMock()
        mock_resp.headers = {
            "X-RateLimit-Remaining": "not-a-number",
            "X-RateLimit-Reset": "also-bad",
        }

        # Should not raise
        client._update_rate_limit(mock_resp)
        assert client._rate_remaining is None

    @patch("time.sleep")
    def test_wait_for_rate_limit_sleeps_when_low(self, mock_sleep):
        client = SWHClient(token="t")
        client._rate_remaining = 2
        client._rate_reset = time.time() + 5.0

        client._wait_for_rate_limit()
        mock_sleep.assert_called_once()
        # Should sleep for approximately reset_time - now + 1
        slept = mock_sleep.call_args[0][0]
        assert 4.0 < slept < 8.0

    @patch("time.sleep")
    def test_wait_for_rate_limit_no_sleep_when_plenty(self, mock_sleep):
        client = SWHClient(token="t")
        client._rate_remaining = 100
        client._rate_reset = time.time() + 5.0

        client._wait_for_rate_limit()
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_handles_429_with_retry_after(self, mock_sleep):
        """On 429, should sleep for Retry-After seconds and retry."""
        client = SWHClient(token="t")

        # First response: 429
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "3", "X-RateLimit-Remaining": "0"}

        # Second response: 200 OK
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.headers = {"X-RateLimit-Remaining": "100"}
        resp_200.json.return_value = {"ok": True}

        client.session.get = MagicMock(side_effect=[resp_429, resp_200])

        result = client._rate_limited_get("https://example.com/test")
        assert result == {"ok": True}
        assert client.session.get.call_count == 2

    @patch("time.sleep")
    def test_handles_server_error_with_retry(self, mock_sleep):
        """On 500, should retry with exponential backoff."""
        client = SWHClient(token="t")

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.headers = {}
        resp_200.json.return_value = {"ok": True}

        client.session.get = MagicMock(side_effect=[resp_500, resp_200])

        result = client._rate_limited_get("https://example.com/test")
        assert result == {"ok": True}
        assert client.session.get.call_count == 2


# ---------------------------------------------------------------------------
# SoftwareHeritageSource tests
# ---------------------------------------------------------------------------


class TestSoftwareHeritageSourceAvailability:
    """Test is_available() and import handling."""

    def test_is_available_returns_bool(self):
        result = SoftwareHeritageSource.is_available()
        # In test environment, requests is installed
        assert result is True

    def test_name_format(self):
        source = SoftwareHeritageSource(
            origins=["https://github.com/a/b", "https://github.com/c/d"]
        )
        assert source.name() == "software_heritage(2 origins)"

    def test_estimate_size_returns_max_files(self):
        source = SoftwareHeritageSource(origins=["x"], max_files=50)
        assert source.estimate_size() == 50

    def test_estimate_size_returns_none_when_unlimited(self):
        source = SoftwareHeritageSource(origins=["x"])
        assert source.estimate_size() is None


class TestSoftwareHeritageSourceStream:
    """Test the full streaming flow with mocked API responses."""

    def _make_source(self, **kwargs) -> SoftwareHeritageSource:
        defaults = {
            "origins": [MOCK_ORIGIN_URL],
            "content_types": [".py", ".ts"],
            "token": "test-token",
            "max_files": 100,
        }
        defaults.update(kwargs)
        return SoftwareHeritageSource(**defaults)

    def _mock_client(self) -> MagicMock:
        """Create a mock SWHClient with standard responses."""
        client = MagicMock(spec=SWHClient)
        client.get_visits.return_value = MOCK_VISITS
        client.get_snapshot.return_value = MOCK_SNAPSHOT
        client.get_revision_directory.return_value = MOCK_ROOT_DIR
        client.get_directory.return_value = MOCK_SRC_DIR
        client.get_content_raw.side_effect = lambda sha1: MOCK_FILE_CONTENTS.get(sha1, "")
        return client

    def test_stream_yields_matching_files(self):
        source = self._make_source()
        mock_client = self._mock_client()
        source._client = mock_client

        records = list(source.stream())

        # Should yield: main.py, utils.ts (from src/)
        # Should skip: README.md (.md not in content_types), image.png (.png not in content_types)
        # Should skip: node_modules directory entirely
        assert len(records) == 2

        paths = [r.metadata["path"] for r in records]
        assert "src/main.py" in paths
        assert "src/utils.ts" in paths

        # Verify record contents
        py_record = next(r for r in records if r.metadata["path"] == "src/main.py")
        assert "def hello" in py_record.content
        assert py_record.metadata["source"] == "software_heritage"
        assert py_record.metadata["origin"] == MOCK_ORIGIN_URL
        assert py_record.metadata["sha1"] == "cnt002"

    def test_stream_no_origins_yields_nothing(self):
        source = self._make_source(origins=[])
        records = list(source.stream())
        assert records == []

    def test_stream_respects_max_files(self):
        source = self._make_source(max_files=1)
        mock_client = self._mock_client()
        source._client = mock_client

        records = list(source.stream())
        assert len(records) == 1

    def test_stream_no_content_types_includes_all(self):
        """When content_types is None, all text files are included."""
        source = self._make_source(content_types=None)
        mock_client = self._mock_client()

        # Override: make all files have valid content
        def get_content(sha1):
            return MOCK_FILE_CONTENTS.get(sha1, "Some valid content that is long enough to pass")

        mock_client.get_content_raw.side_effect = get_content
        source._client = mock_client

        records = list(source.stream())
        # Should get README.md + main.py + utils.ts + image.png (all pass length check)
        paths = [r.metadata["path"] for r in records]
        assert "README.md" in paths
        assert "src/main.py" in paths
        assert "src/utils.ts" in paths

    def test_stream_handles_no_visits(self):
        source = self._make_source()
        mock_client = self._mock_client()
        mock_client.get_visits.return_value = []
        source._client = mock_client

        records = list(source.stream())
        assert records == []

    def test_stream_handles_api_error(self):
        """API errors on one origin should not stop processing others."""
        source = self._make_source(
            origins=[MOCK_ORIGIN_URL, "https://github.com/other/repo"]
        )
        mock_client = self._mock_client()

        # First origin works, second raises
        call_count = 0

        def visits_side_effect(url):
            nonlocal call_count
            call_count += 1
            if url == "https://github.com/other/repo":
                raise ConnectionError("Network error")
            return MOCK_VISITS

        mock_client.get_visits.side_effect = visits_side_effect
        source._client = mock_client

        # Should still yield records from the first origin
        records = list(source.stream())
        assert len(records) > 0


class TestBranchSelection:
    """Test _pick_branch_revision logic."""

    def test_picks_head_alias(self):
        source = SoftwareHeritageSource(origins=["x"])
        branches = {
            "HEAD": {"target_type": "alias", "target": "refs/heads/main"},
            "refs/heads/main": {"target_type": "revision", "target": "rev-main"},
        }
        assert source._pick_branch_revision(branches) == "rev-main"

    def test_picks_head_direct_revision(self):
        source = SoftwareHeritageSource(origins=["x"])
        branches = {
            "HEAD": {"target_type": "revision", "target": "rev-direct"},
        }
        assert source._pick_branch_revision(branches) == "rev-direct"

    def test_falls_back_to_main(self):
        source = SoftwareHeritageSource(origins=["x"])
        branches = {
            "refs/heads/main": {"target_type": "revision", "target": "rev-main"},
            "refs/heads/feature": {"target_type": "revision", "target": "rev-feat"},
        }
        assert source._pick_branch_revision(branches) == "rev-main"

    def test_falls_back_to_master(self):
        source = SoftwareHeritageSource(origins=["x"])
        branches = {
            "refs/heads/master": {"target_type": "revision", "target": "rev-master"},
        }
        assert source._pick_branch_revision(branches) == "rev-master"

    def test_falls_back_to_any_revision(self):
        source = SoftwareHeritageSource(origins=["x"])
        branches = {
            "refs/heads/develop": {"target_type": "revision", "target": "rev-dev"},
        }
        assert source._pick_branch_revision(branches) == "rev-dev"

    def test_returns_none_for_empty_branches(self):
        source = SoftwareHeritageSource(origins=["x"])
        assert source._pick_branch_revision({}) is None


class TestRegistryIntegration:
    """Test that the source registers itself with the pipeline registry."""

    def test_registered_in_source_registry(self):
        from cola_coder.data.registry import get_source

        cls = get_source("software_heritage")
        assert cls is SoftwareHeritageSource
