"""Software Heritage data source.

Software Heritage (archive.softwareheritage.org) is the universal archive
of software source code. It provides:
- Deduplicated storage (same file content = same hash, regardless of repo)
- Rich metadata (origins, visits, snapshots)
- Permissive access via API and bulk exports

Access methods (in order of practicality):
1. SWH API: REST API, 1200 req/hr unauthenticated, 12000 with token
2. SWH Dataset: Bulk compressed exports on S3 (huge, terabytes)
3. SWH Graph: Relationship graph as compressed files

For a small project, the API is the right approach.
For large-scale, use The Stack v2 on HuggingFace (already SWH-derived).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator

from cola_coder.data.pipeline import DataRecord, DataSource

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SWH API client
# ---------------------------------------------------------------------------


class SWHClient:
    """HTTP client for the Software Heritage REST API.

    Handles authentication, rate limiting, and retries. The SWH API uses
    ``X-RateLimit-*`` response headers (similar to GitHub) and returns
    HTTP 429 when the limit is exceeded.

    Think of this like an Axios instance with interceptors for auth and
    rate-limit back-off.
    """

    BASE_URL = "https://archive.softwareheritage.org/api/1"
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0  # seconds — exponential backoff multiplier

    def __init__(self, token: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        if not _HAS_REQUESTS:
            raise ImportError(
                "The 'requests' library is required for SoftwareHeritageSource. "
                "Install it with: pip install requests"
            )

        self.token = token or os.environ.get("SWH_API_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "cola-coder/0.1"
        self.session.headers["Accept"] = "application/json"

        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("SWH client initialized with API token")
        else:
            logger.warning(
                "No SWH_API_TOKEN set — rate limit is 1200 requests/hour. "
                "Get a token at https://archive.softwareheritage.org for 12000/hr."
            )

        # Rate limit tracking (populated from response headers)
        self._rate_remaining: int | None = None
        self._rate_reset: float | None = None

    # -- Public API methods --------------------------------------------------

    def get_origin(self, url: str) -> dict[str, Any]:
        """Look up an origin (repository) by its URL.

        Args:
            url: The origin URL, e.g. ``https://github.com/pallets/flask``.

        Returns:
            Origin metadata dict with ``url`` and ``origin_visits_url`` keys.
        """
        return self._rate_limited_get(f"{self.BASE_URL}/origin/{url}/get/")

    def get_visits(self, origin_url: str) -> list[dict[str, Any]]:
        """List visits for an origin, most recent first.

        A visit is a point-in-time crawl of the origin. We want the latest
        visit that has status ``full`` (meaning the crawl completed).

        Args:
            origin_url: The origin URL as stored in SWH.

        Returns:
            List of visit dicts (newest first).
        """
        # SWH paginates, but for our purposes the first page is enough —
        # we only need the latest successful visit.
        return self._rate_limited_get(
            f"{self.BASE_URL}/origin/{origin_url}/visits/",
            params={"per_page": 20},
        )

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Get branches/tags from a snapshot.

        Args:
            snapshot_id: Hex SHA-1 of the snapshot.

        Returns:
            Snapshot dict with a ``branches`` mapping.
        """
        return self._rate_limited_get(f"{self.BASE_URL}/snapshot/{snapshot_id}/")

    def get_revision_directory(self, revision_id: str) -> list[dict[str, Any]]:
        """List the root directory of a revision (commit).

        Args:
            revision_id: Hex SHA-1 of the revision.

        Returns:
            List of directory entry dicts (each has ``name``, ``type``,
            ``target`` keys).
        """
        return self._rate_limited_get(
            f"{self.BASE_URL}/revision/{revision_id}/directory/"
        )

    def get_directory(self, dir_id: str) -> list[dict[str, Any]]:
        """List entries in a directory.

        Args:
            dir_id: Hex SHA-1 of the directory.

        Returns:
            List of directory entry dicts.
        """
        return self._rate_limited_get(f"{self.BASE_URL}/directory/{dir_id}/")

    def get_content_raw(self, content_sha1: str) -> str:
        """Download the raw content of a file by its SHA-1 hash.

        Args:
            content_sha1: Hex SHA-1 of the content blob.

        Returns:
            The file content as a UTF-8 string.

        Raises:
            UnicodeDecodeError: If the content is binary.
        """
        url = f"{self.BASE_URL}/content/sha1:{content_sha1}/raw/"
        resp = self._rate_limited_get_response(url)
        resp.raise_for_status()
        return resp.text

    # -- Internal HTTP helpers -----------------------------------------------

    def _rate_limited_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any:
        """GET with rate limit handling. Returns parsed JSON."""
        resp = self._rate_limited_get_response(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _rate_limited_get_response(
        self, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response:
        """GET with rate limit handling. Returns the raw Response object.

        Respects ``X-RateLimit-Remaining`` and ``X-RateLimit-Reset`` headers.
        On HTTP 429, sleeps for the ``Retry-After`` duration and retries.
        """
        for attempt in range(self.MAX_RETRIES):
            # Pre-request: sleep if we know we're near the limit
            self._wait_for_rate_limit()

            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "SWH request failed (%s), retrying in %.1fs...", exc, wait
                    )
                    time.sleep(wait)
                    continue
                raise

            # Update rate limit state from response headers
            self._update_rate_limit(resp)

            # Handle 429 Too Many Requests
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else self.BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "SWH rate limit hit (429), sleeping %.1fs...", wait
                )
                time.sleep(wait)
                continue

            # Handle server errors with retry
            if resp.status_code >= 500:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "SWH server error %d, retrying in %.1fs...",
                        resp.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue

            return resp

        # Exhausted retries — return last response so caller can raise
        return resp  # type: ignore[possibly-undefined]

    def _update_rate_limit(self, resp: requests.Response) -> None:
        """Parse rate limit headers from a response."""
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")

        if remaining is not None:
            try:
                self._rate_remaining = int(remaining)
            except ValueError:
                pass

        if reset is not None:
            try:
                self._rate_reset = float(reset)
            except ValueError:
                pass

    def _wait_for_rate_limit(self) -> None:
        """Sleep if we're close to exhausting the rate limit."""
        if self._rate_remaining is not None and self._rate_remaining < 5:
            if self._rate_reset is not None:
                wait = max(0.0, self._rate_reset - time.time()) + 1
                if wait > 0:
                    logger.info(
                        "SWH rate limit low (%d remaining), waiting %.0fs...",
                        self._rate_remaining,
                        wait,
                    )
                    time.sleep(wait)


# ---------------------------------------------------------------------------
# SoftwareHeritageSource — DataSource plugin
# ---------------------------------------------------------------------------


class SoftwareHeritageSource(DataSource):
    """Stream code from the Software Heritage archive.

    Uses the SWH REST API to search for and retrieve code content.
    This source is most useful for:

    - Getting specific files/repos that aren't in StarCoderData
    - Accessing historical versions of code
    - Dedup-guaranteed data (SWH deduplicates by content hash)

    Rate limits:
    - Unauthenticated: 1200 requests/hour
    - Authenticated: 12000 requests/hour (get token at archive.softwareheritage.org)

    Example::

        source = SoftwareHeritageSource(
            origins=["https://github.com/pallets/flask"],
            content_types=[".py"],
            max_files=100,
        )
        for record in source.stream():
            print(record.metadata["path"], len(record.content))
    """

    def __init__(
        self,
        origins: list[str] | None = None,
        content_types: list[str] | None = None,
        token: str | None = None,
        max_files: int | None = None,
        timeout: int = SWHClient.DEFAULT_TIMEOUT,
    ):
        """Initialize the Software Heritage data source.

        Args:
            origins: List of origin URLs to fetch from (e.g. GitHub repo URLs).
                     At least one origin is required.
            content_types: File extensions to include (e.g. ``[".py", ".ts"]``).
                           If ``None``, all text files are included.
            token: SWH API token. Falls back to ``SWH_API_TOKEN`` env var.
            max_files: Stop after yielding this many files (across all origins).
            timeout: HTTP request timeout in seconds (default 30).
        """
        self._origins = origins or []
        # Normalize to canonical ".ext" lowercase so the filter matches what
        # os.path.splitext yields. Without this, content_types=[".PY"] or
        # ["py"] (both plausible) matched NOTHING — a silent empty stream.
        self._content_types = (
            {"." + ct.strip().lstrip(".").lower() for ct in content_types if ct.strip()}
            if content_types else None
        )
        self._token = token
        self._max_files = max_files
        self._timeout = timeout
        self._client: SWHClient | None = None

    def name(self) -> str:
        n_origins = len(self._origins)
        return f"software_heritage({n_origins} origins)"

    @staticmethod
    def is_available() -> bool:
        """Check whether the requests library is installed."""
        return _HAS_REQUESTS

    def _get_client(self) -> SWHClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = SWHClient(token=self._token, timeout=self._timeout)
        return self._client

    def stream(self) -> Iterator[DataRecord]:
        """Stream code files from the configured SWH origins.

        For each origin URL:
        1. Look up the origin and get the latest successful visit
        2. Get the snapshot (branches) from that visit
        3. Pick the HEAD/main/master branch
        4. Walk the directory tree recursively
        5. Download and yield each matching file

        Yields:
            ``DataRecord`` objects with content and metadata.
        """
        if not self._origins:
            logger.warning("SoftwareHeritageSource: no origins configured, nothing to stream")
            return

        if not _HAS_REQUESTS:
            raise ImportError(
                "The 'requests' library is required for SoftwareHeritageSource. "
                "Install it with: pip install requests"
            )

        client = self._get_client()
        files_yielded = 0

        for origin_url in self._origins:
            if self._max_files is not None and files_yielded >= self._max_files:
                break

            logger.info("SWH: processing origin %s", origin_url)

            try:
                for record in self._stream_origin(client, origin_url):
                    files_yielded += 1
                    yield record
                    if self._max_files is not None and files_yielded >= self._max_files:
                        break
            except Exception:
                logger.exception("SWH: failed to process origin %s", origin_url)
                continue

    def _stream_origin(
        self, client: SWHClient, origin_url: str
    ) -> Iterator[DataRecord]:
        """Stream all matching files from a single origin."""
        # 1. Get latest successful visit
        visits = client.get_visits(origin_url)
        if not visits:
            logger.warning("SWH: no visits found for %s", origin_url)
            return

        # Find most recent visit with a snapshot
        visit = None
        for v in visits:
            if v.get("snapshot") and v.get("status") == "full":
                visit = v
                break

        if visit is None:
            # Fall back to any visit with a snapshot
            for v in visits:
                if v.get("snapshot"):
                    visit = v
                    break

        if visit is None:
            logger.warning("SWH: no usable visit for %s", origin_url)
            return

        snapshot_id = visit["snapshot"]
        logger.info("SWH: using visit %s, snapshot %s", visit.get("visit"), snapshot_id)

        # 2. Get snapshot branches
        snapshot = client.get_snapshot(snapshot_id)
        branches = snapshot.get("branches", {})

        if not branches:
            logger.warning("SWH: snapshot %s has no branches", snapshot_id)
            return

        # 3. Pick the best branch (HEAD > main > master > first available)
        revision_id = self._pick_branch_revision(branches)
        if revision_id is None:
            logger.warning("SWH: could not find a usable branch in snapshot %s", snapshot_id)
            return

        # 4. Get root directory from the revision
        try:
            entries = client.get_revision_directory(revision_id)
        except Exception:
            logger.exception("SWH: failed to get directory for revision %s", revision_id)
            return

        # 5. Recursively walk and yield files
        yield from self._walk_directory(client, entries, origin_url, path_prefix="")

    def _pick_branch_revision(self, branches: dict[str, Any]) -> str | None:
        """Pick the best branch and return its revision ID.

        Priority: HEAD (if alias resolved) > refs/heads/main > refs/heads/master
        > first branch with a revision target.
        """
        # Try HEAD first
        head = branches.get("HEAD")
        if head:
            if head.get("target_type") == "revision":
                return head["target"]
            # HEAD might be an alias — resolve it
            if head.get("target_type") == "alias" and head.get("target") in branches:
                resolved = branches[head["target"]]
                if resolved and resolved.get("target_type") == "revision":
                    return resolved["target"]

        # Try common default branch names
        for name in ("refs/heads/main", "refs/heads/master"):
            branch = branches.get(name)
            if branch and branch.get("target_type") == "revision":
                return branch["target"]

        # Fall back to any branch with a revision target
        for _name, branch in branches.items():
            if branch and branch.get("target_type") == "revision":
                return branch["target"]

        return None

    def _walk_directory(
        self,
        client: SWHClient,
        entries: list[dict[str, Any]],
        origin_url: str,
        path_prefix: str,
    ) -> Iterator[DataRecord]:
        """Recursively walk a directory tree and yield matching files."""
        for entry in entries:
            entry_name = entry.get("name", "")
            entry_type = entry.get("type", "")
            target = entry.get("target", "")

            full_path = f"{path_prefix}/{entry_name}" if path_prefix else entry_name

            # Skip common junk directories
            if entry_type == "dir" and entry_name in (
                "node_modules", ".git", "dist", "build", "__pycache__",
                ".venv", "venv", "vendor", ".next", "coverage",
            ):
                continue

            if entry_type == "dir" and target:
                # Recurse into subdirectory
                try:
                    sub_entries = client.get_directory(target)
                    yield from self._walk_directory(
                        client, sub_entries, origin_url, path_prefix=full_path
                    )
                except Exception:
                    logger.warning("SWH: failed to read directory %s", full_path)
                    continue

            elif entry_type == "file" and target:
                # Check extension filter
                if self._content_types is not None:
                    ext = os.path.splitext(entry_name)[1].lower()
                    if ext not in self._content_types:
                        continue

                # Check file size (skip large files)
                length = entry.get("length", 0)
                if length > 256 * 1024:  # 256 KB max
                    continue
                if length == 0:
                    continue

                # Download content
                try:
                    content = client.get_content_raw(target)
                except UnicodeDecodeError:
                    # Binary file
                    continue
                except Exception:
                    logger.warning("SWH: failed to fetch content for %s", full_path)
                    continue

                if not content or len(content.strip()) < 10:
                    continue

                yield DataRecord(
                    content=content,
                    metadata={
                        "source": "software_heritage",
                        "origin": origin_url,
                        # Canonical key for language detection (alongside "path").
                        "file_path": full_path,
                        "path": full_path,
                        "sha1": target,
                        "extension": os.path.splitext(entry_name)[1],
                    },
                )

    def estimate_size(self) -> int | None:
        return self._max_files


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

try:
    from cola_coder.data.registry import register_source

    register_source("software_heritage")(SoftwareHeritageSource)
except ImportError:  # pragma: no cover
    pass
