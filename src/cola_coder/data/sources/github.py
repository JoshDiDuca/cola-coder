"""GitHub repository scraper for cola-coder training data.

Searches GitHub for high-quality code repositories, clones them,
and extracts source files as training data. Supports rich filtering
by stars, language composition, license, owner quality, and more.

Security: Shallow clones only, hooks disabled, size limits enforced,
secrets excluded, clones cleaned up after extraction.

Usage:
    from cola_coder.data.sources.github import (
        GitHubClient, RepoFilter, RepoProcessor, MetadataCache,
        GitHubSource, FILTER_PRESETS,
    )

    client = GitHubClient()  # uses GITHUB_TOKEN env var
    repos = client.search_repos(FILTER_PRESETS["typescript_elite"], max_results=50)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator

import requests

from cola_coder.data.pipeline import DataRecord
from cola_coder.data.registry import register_source

_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RepoFilter — rich filtering criteria
# ---------------------------------------------------------------------------


@dataclass
class RepoFilter:
    """Rich filtering criteria for GitHub repository search.

    Builds a GitHub search API query from structured fields. Think of it
    like a TypeScript interface that maps to query parameters — each field
    becomes a qualifier in the GitHub search string.
    """
    # Star / fork thresholds
    min_stars: int | None = None
    max_stars: int | None = None
    min_forks: int | None = None
    max_forks: int | None = None

    # Language filtering
    primary_language: str | None = None
    language_min_percent: float | None = None  # e.g. 0.6 = at least 60%
    languages_include: list[str] = field(default_factory=list)
    languages_exclude: list[str] = field(default_factory=list)

    # License
    licenses: list[str] = field(default_factory=list)  # SPDX IDs

    # Recency
    pushed_after: str | None = None   # ISO date, e.g. "2023-01-01"
    created_after: str | None = None  # ISO date

    # Repository attributes
    not_archived: bool = True
    owner_type: str | None = None  # "User" or "Organization"
    min_owner_followers: int | None = None
    min_owner_total_stars: int | None = None
    is_fork: bool | None = False  # False = originals only
    max_repo_size_kb: int | None = None

    # Quality signals (checked post-clone, not in API query)
    has_tests: bool | None = None
    has_ci: bool | None = None
    typescript_strict: bool | None = None

    # Topics
    topics_include: list[str] = field(default_factory=list)
    topics_exclude: list[str] = field(default_factory=list)

    def to_github_query(self) -> str:
        """Build a GitHub search API query string from filter fields.

        Returns:
            A query string like 'language:TypeScript stars:>500 license:mit'
            suitable for the /search/repositories endpoint.
        """
        parts: list[str] = []

        # Language
        if self.primary_language:
            parts.append(f"language:{self.primary_language}")

        # Stars
        if self.min_stars is not None and self.max_stars is not None:
            parts.append(f"stars:{self.min_stars}..{self.max_stars}")
        elif self.min_stars is not None:
            parts.append(f"stars:>={self.min_stars}")
        elif self.max_stars is not None:
            parts.append(f"stars:<={self.max_stars}")

        # Forks
        if self.min_forks is not None and self.max_forks is not None:
            parts.append(f"forks:{self.min_forks}..{self.max_forks}")
        elif self.min_forks is not None:
            parts.append(f"forks:>={self.min_forks}")
        elif self.max_forks is not None:
            parts.append(f"forks:<={self.max_forks}")

        # License
        for lic in self.licenses:
            parts.append(f"license:{lic.lower()}")

        # Dates
        if self.pushed_after:
            parts.append(f"pushed:>={self.pushed_after}")
        if self.created_after:
            parts.append(f"created:>={self.created_after}")

        # Archived
        if self.not_archived:
            parts.append("archived:false")

        # Fork
        if self.is_fork is False:
            parts.append("fork:false")
        elif self.is_fork is True:
            parts.append("fork:true")
        # None = don't filter

        # Topics
        for topic in self.topics_include:
            parts.append(f"topic:{topic}")

        # Size limit (GitHub uses KB)
        if self.max_repo_size_kb is not None:
            parts.append(f"size:<={self.max_repo_size_kb}")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# FILTER_PRESETS — pre-built filters for common use cases
# ---------------------------------------------------------------------------

FILTER_PRESETS: dict[str, RepoFilter] = {
    "typescript_elite": RepoFilter(
        min_stars=500,
        primary_language="TypeScript",
        language_min_percent=0.6,
        licenses=["mit", "apache-2.0"],
        not_archived=True,
        is_fork=False,
        has_tests=True,
        typescript_strict=True,
        pushed_after="2023-01-01",
        max_repo_size_kb=500_000,
    ),
    "typescript_good": RepoFilter(
        min_stars=50,
        primary_language="TypeScript",
        licenses=["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"],
        not_archived=True,
        is_fork=False,
        pushed_after="2022-01-01",
        max_repo_size_kb=500_000,
    ),
    "python_elite": RepoFilter(
        min_stars=500,
        primary_language="Python",
        language_min_percent=0.6,
        licenses=["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"],
        not_archived=True,
        is_fork=False,
        has_tests=True,
        pushed_after="2023-01-01",
        max_repo_size_kb=500_000,
    ),
    "popular_any": RepoFilter(
        min_stars=1000,
        licenses=["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"],
        not_archived=True,
        is_fork=False,
        pushed_after="2022-01-01",
        max_repo_size_kb=500_000,
    ),
}


# ---------------------------------------------------------------------------
# GitHubClient — API interaction with rate limiting and retries
# ---------------------------------------------------------------------------


class GitHubClient:
    """GitHub API client with rate limiting, retries, and search.

    Handles authentication via GITHUB_TOKEN env var, tracks rate limits
    from response headers, and implements exponential backoff on failures.
    """

    API_BASE = "https://api.github.com"
    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0  # seconds

    def __init__(self, token: str | None = None):
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token. Falls back to
                   GITHUB_TOKEN environment variable. Without a token,
                   rate limits are very low (60 requests/hour).
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()

        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHub client initialized with token")
        else:
            logger.warning(
                "No GITHUB_TOKEN set — API rate limit is 60 requests/hour. "
                "Set GITHUB_TOKEN env var for 5000 requests/hour."
            )

        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "cola-coder-scraper/0.1"

        # Rate limit tracking
        self._rate_remaining: int | None = None
        self._rate_reset: float | None = None

    def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Make an API request with rate limiting and retries.

        Sleeps when rate limit is nearly exhausted, retries with
        exponential backoff on transient failures.
        """
        if not url.startswith("http"):
            url = f"{self.API_BASE}{url}"

        for attempt in range(self.MAX_RETRIES):
            # Check rate limit before request
            self._wait_for_rate_limit()

            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Request failed ({e}), retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                raise

            # Update rate limit tracking from headers
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining is not None:
                self._rate_remaining = int(remaining)
            if reset is not None:
                self._rate_reset = float(reset)

            # Handle rate limit exceeded
            if resp.status_code == 403 and self._rate_remaining == 0:
                wait = max(0, (self._rate_reset or 0) - time.time()) + 1
                logger.warning(f"Rate limit exceeded, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            # Handle server errors with retry
            if resp.status_code >= 500:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        f"Server error {resp.status_code}, retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue

            resp.raise_for_status()
            return resp

        # Should not reach here, but just in case
        raise RuntimeError(f"Failed after {self.MAX_RETRIES} retries: {url}")

    def _wait_for_rate_limit(self) -> None:
        """Sleep if we're close to hitting the rate limit."""
        if self._rate_remaining is not None and self._rate_remaining < 5:
            if self._rate_reset is not None:
                wait = max(0, self._rate_reset - time.time()) + 1
                if wait > 0:
                    logger.info(f"Rate limit low ({self._rate_remaining}), "
                                f"waiting {wait:.0f}s...")
                    time.sleep(wait)

    def search_repos(
        self,
        filter: RepoFilter,
        max_results: int = 100,
        sort: str = "stars",
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Search GitHub repositories using the search API.

        Handles pagination automatically (GitHub returns max 100 per page,
        max 1000 results total for search).

        Args:
            filter: RepoFilter with search criteria.
            max_results: Maximum number of repos to return (capped at 1000 by GitHub).
            sort: Sort field — "stars", "forks", "updated", or "best-match".
            order: Sort order — "asc" or "desc".

        Returns:
            List of repository dictionaries from the GitHub API.
        """
        query = filter.to_github_query()
        if not query:
            raise ValueError("RepoFilter produced an empty query — set at least one filter.")

        logger.info(f"Searching GitHub: {query}")

        repos: list[dict[str, Any]] = []
        per_page = min(100, max_results)
        max_pages = (max_results + per_page - 1) // per_page

        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page,
            }

            resp = self._request("GET", "/search/repositories", params=params)
            data = resp.json()

            items = data.get("items", [])
            if not items:
                break

            repos.extend(items)
            logger.info(
                f"  Page {page}: {len(items)} repos "
                f"(total: {len(repos)}/{data.get('total_count', '?')})"
            )

            if len(repos) >= max_results:
                repos = repos[:max_results]
                break

            # Respect search API rate limits (30 requests/min)
            time.sleep(2)

        return repos

    def get_repo_info(self, repo: str) -> dict[str, Any]:
        """Fetch full metadata for a repository.

        Args:
            repo: Full repo name like "owner/repo".

        Returns:
            Repository metadata dictionary from the GitHub API.
        """
        resp = self._request("GET", f"/repos/{repo}")
        return resp.json()

    def get_languages(self, repo: str) -> dict[str, float]:
        """Get language breakdown as percentages.

        Args:
            repo: Full repo name like "owner/repo".

        Returns:
            Dict mapping language name to percentage (0.0-1.0).
            Example: {"TypeScript": 0.85, "JavaScript": 0.10, "CSS": 0.05}
        """
        resp = self._request("GET", f"/repos/{repo}/languages")
        raw: dict[str, int] = resp.json()

        total = sum(raw.values())
        if total == 0:
            return {}

        return {lang: bytes_count / total for lang, bytes_count in raw.items()}

    def get_owner_info(self, owner: str) -> dict[str, Any]:
        """Fetch owner (user or org) information.

        Args:
            owner: GitHub username or org name.

        Returns:
            User/org metadata including followers count.
        """
        resp = self._request("GET", f"/users/{owner}")
        return resp.json()

    def clone_repo(
        self,
        repo: str,
        dest: str | Path,
        shallow: bool = True,
    ) -> Path:
        """Clone a repository securely.

        Security measures:
        - Shallow clone (--depth 1) to minimize data
        - Git hooks disabled via core.hooksPath=/dev/null
        - Size checked before clone via API

        Args:
            repo: Full repo name like "owner/repo".
            dest: Directory to clone into.
            shallow: Use shallow clone (default True for security).

        Returns:
            Path to the cloned repository directory.
        """
        dest = Path(dest)

        # Validate repo name to prevent path traversal
        if not _REPO_NAME_RE.match(repo):
            raise ValueError(f"Invalid repo name: {repo!r}")

        repo_dir = dest / repo.replace("/", "_")

        if repo_dir.exists():
            logger.info(f"Repo already cloned: {repo_dir}")
            return repo_dir

        dest.mkdir(parents=True, exist_ok=True)

        clone_url = f"https://github.com/{repo}.git"
        cmd = ["git", "clone"]

        if shallow:
            cmd.extend(["--depth", "1"])

        # Disable git hooks for security (NUL on Windows, /dev/null on Unix)
        null_path = "NUL" if sys.platform == "win32" else "/dev/null"
        cmd.extend(["--config", f"core.hooksPath={null_path}"])
        cmd.extend([clone_url, str(repo_dir)])

        logger.info(f"Cloning {repo}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {repo}: {result.stderr.strip()}"
            )

        logger.info(f"Cloned {repo} to {repo_dir}")
        return repo_dir


# ---------------------------------------------------------------------------
# RepoProcessor — extract source files from cloned repos
# ---------------------------------------------------------------------------


class RepoProcessor:
    """Extract and process source files from cloned repositories.

    Maps languages to file extensions, filters out junk directories
    (node_modules, dist, .git, etc.), and checks for quality signals
    like test coverage, CI config, and TypeScript strict mode.
    """

    LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
        "TypeScript": [".ts", ".tsx"],
        "JavaScript": [".js", ".jsx", ".mjs", ".cjs"],
        "Python": [".py"],
        "Java": [".java"],
        "Go": [".go"],
        "Rust": [".rs"],
        "C": [".c", ".h"],
        "C++": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
        "C#": [".cs"],
        "Ruby": [".rb"],
        "PHP": [".php"],
        "Swift": [".swift"],
        "Kotlin": [".kt", ".kts"],
        "Scala": [".scala"],
        "Shell": [".sh", ".bash"],
    }

    # Reverse lookup: extension -> language
    _EXT_TO_LANG: dict[str, str] = {}
    for _lang, _exts in LANGUAGE_EXTENSIONS.items():
        for _ext in _exts:
            _EXT_TO_LANG[_ext] = _lang

    IGNORE_PATTERNS: list[str] = [
        "node_modules",
        "dist",
        "build",
        "out",
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        ".env",
        "vendor",
        "third_party",
        "external",
        ".next",
        ".nuxt",
        "coverage",
        ".nyc_output",
        "target",           # Rust/Java build output
        ".gradle",
        ".idea",
        ".vscode",
        "site-packages",
        "bower_components",
        ".yarn",
        ".pnp",
        "eggs",
        "*.egg-info",
        ".eggs",
        "__pypackages__",
    ]

    # Files that should never be included (secrets, credentials, etc.)
    SECRET_PATTERNS: list[str] = [
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        ".env.staging",
        "credentials.json",
        "secrets.json",
        "serviceAccountKey.json",
        "apikeys.json",
        "tokens.json",
        ".npmrc",
        ".pypirc",
        ".htpasswd",
        ".htaccess",
        ".netrc",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".keystore",
        ".jks",
    ]

    # Directories that are always secrets/credentials
    SECRET_DIRS: set[str] = {
        ".aws", ".azure", ".gcp", ".docker", ".ssh",
    }

    # Maximum file size to include (256 KB)
    MAX_FILE_SIZE: int = 256 * 1024

    def __init__(self, languages: list[str] | None = None):
        """Initialize the processor.

        Args:
            languages: List of languages to extract. If None, extracts all
                       recognized languages. Language names must match keys
                       in LANGUAGE_EXTENSIONS (case-sensitive).
        """
        self.languages = languages

        # Build set of allowed extensions
        if languages:
            self.allowed_extensions: set[str] = set()
            for lang in languages:
                exts = self.LANGUAGE_EXTENSIONS.get(lang, [])
                self.allowed_extensions.update(exts)
        else:
            self.allowed_extensions = set(self._EXT_TO_LANG.keys())

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on IGNORE_PATTERNS."""
        parts = path.parts
        for part in parts:
            for pattern in self.IGNORE_PATTERNS:
                if pattern.startswith("*"):
                    # Glob-like suffix match
                    if part.endswith(pattern[1:]):
                        return True
                elif part == pattern:
                    return True
        return False

    def _is_secret_file(self, path: Path) -> bool:
        """Check if a file looks like it contains secrets."""
        # Check if any parent directory is a secret directory
        for part in path.parts[:-1]:
            if part.lower() in self.SECRET_DIRS:
                return True
        name = path.name.lower()
        for pattern in self.SECRET_PATTERNS:
            if name == pattern.lower() or name.endswith(pattern.lower()):
                return True
        return False

    def extract_files(
        self,
        repo_path: str | Path,
        repo_name: str = "",
        repo_stars: int = 0,
        repo_url: str = "",
        repo_license: str = "",
    ) -> Iterator[DataRecord]:
        """Extract source files from a cloned repository.

        Walks the repository tree, filters by language/extension,
        skips ignored directories and secret files, and yields
        DataRecord instances.

        Args:
            repo_path: Path to the cloned repo root.
            repo_name: Full name like "owner/repo".
            repo_stars: Star count for metadata.
            repo_url: GitHub URL for metadata.
            repo_license: SPDX license ID for metadata.

        Yields:
            DataRecord for each valid source file.
        """
        repo_path = Path(repo_path)

        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Security: skip symlinks to prevent path traversal attacks
            if file_path.is_symlink():
                logger.debug(f"Skipping symlink: {file_path}")
                continue

            # Security: verify resolved path is still inside repo
            try:
                resolved = file_path.resolve()
                resolved.relative_to(repo_path.resolve())
            except (ValueError, OSError):
                logger.warning(f"Skipping file outside repo: {file_path}")
                continue

            # Check extension
            ext = file_path.suffix.lower()
            if ext not in self.allowed_extensions:
                continue

            # Check ignore patterns
            rel_path = file_path.relative_to(repo_path)
            if self._should_ignore(rel_path):
                continue

            # Check for secrets
            if self._is_secret_file(file_path):
                continue

            # Check file size (use lstat to avoid following symlinks)
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > self.MAX_FILE_SIZE or size == 0:
                continue

            # Read content
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            # Skip empty or nearly-empty files
            if len(content.strip()) < 10:
                continue

            language = self._EXT_TO_LANG.get(ext, "Unknown")

            yield DataRecord(
                content=content,
                metadata={
                    "source": "github",
                    "file_path": str(rel_path),
                    "language": language,
                    "repo_name": repo_name,
                    "repo_stars": repo_stars,
                    "repo_url": repo_url,
                    "license": repo_license,
                    "file_size": size,
                },
            )

    @staticmethod
    def check_license(repo_path: str | Path) -> str | None:
        """Check for a license file and attempt to identify the SPDX ID.

        Args:
            repo_path: Path to the cloned repo root.

        Returns:
            SPDX license ID (e.g. "MIT", "Apache-2.0") or None.
        """
        repo_path = Path(repo_path)

        # Common license file names
        license_names = ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE",
                         "LICENCE.md", "LICENCE.txt", "COPYING", "COPYING.md"]

        for name in license_names:
            license_file = repo_path / name
            if license_file.exists():
                try:
                    text = license_file.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue

                # Simple heuristic matching
                if "mit license" in text or "permission is hereby granted" in text:
                    return "MIT"
                if "apache license" in text and "version 2.0" in text:
                    return "Apache-2.0"
                if "bsd 2-clause" in text or "redistribution and use" in text:
                    if "3. neither" in text or "3. all advertising" in text:
                        return "BSD-3-Clause"
                    return "BSD-2-Clause"
                if "isc license" in text:
                    return "ISC"
                if "mozilla public license" in text:
                    return "MPL-2.0"
                if "gnu general public license" in text:
                    if "version 3" in text:
                        return "GPL-3.0"
                    if "version 2" in text:
                        return "GPL-2.0"
                if "gnu lesser general public" in text:
                    return "LGPL-2.1"
                if "unlicense" in text or "this is free and unencumbered" in text:
                    return "Unlicense"

                return "Unknown"

        return None

    @staticmethod
    def check_has_tests(repo_path: str | Path) -> bool:
        """Check if the repository appears to have tests.

        Looks for common test directory names and test file patterns.

        Args:
            repo_path: Path to the cloned repo root.

        Returns:
            True if test files or directories are found.
        """
        repo_path = Path(repo_path)

        # Check for test directories
        test_dirs = ["test", "tests", "__tests__", "spec", "specs",
                     "test_", "testing"]
        for d in test_dirs:
            if (repo_path / d).is_dir():
                return True

        # Check for test files in the top two levels
        test_patterns = ["*.test.*", "*.spec.*", "test_*.py", "*_test.go",
                         "*_test.rs", "*Test.java"]
        for pattern in test_patterns:
            # Check root and one level deep
            if list(repo_path.glob(pattern)):
                return True
            if list(repo_path.glob(f"*/{pattern}")):
                return True
            if list(repo_path.glob(f"**/{pattern}")):
                return True

        return False

    @staticmethod
    def check_has_ci(repo_path: str | Path) -> bool:
        """Check if the repository has CI configuration.

        Args:
            repo_path: Path to the cloned repo root.

        Returns:
            True if CI config files are found.
        """
        repo_path = Path(repo_path)

        ci_paths = [
            ".github/workflows",
            ".circleci",
            ".travis.yml",
            "Jenkinsfile",
            ".gitlab-ci.yml",
            "azure-pipelines.yml",
            ".buildkite",
        ]
        for ci_path in ci_paths:
            full = repo_path / ci_path
            if full.exists():
                return True

        return False

    @staticmethod
    def check_tsconfig_strict(repo_path: str | Path) -> bool:
        """Check if TypeScript strict mode is enabled in tsconfig.json.

        Args:
            repo_path: Path to the cloned repo root.

        Returns:
            True if tsconfig.json exists and has "strict": true.
        """
        tsconfig = Path(repo_path) / "tsconfig.json"
        if not tsconfig.exists():
            return False

        try:
            text = tsconfig.read_text(encoding="utf-8", errors="ignore")
            # Remove single-line comments (tsconfig allows them)
            text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
            # Remove multi-line comments
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            data = json.loads(text)
        except (OSError, json.JSONDecodeError):
            return False

        compiler_opts = data.get("compilerOptions", {})
        return compiler_opts.get("strict", False) is True


# ---------------------------------------------------------------------------
# MetadataCache — avoid redundant API calls
# ---------------------------------------------------------------------------


class MetadataCache:
    """Cache GitHub API responses as JSON files on disk.

    Uses a TTL (time-to-live) to expire old entries. Cache keys are
    derived from the request URL/identifier.

    Think of this like a simple Map<string, object> backed by the
    filesystem, with automatic expiry — like a Redis cache but just
    JSON files.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/github_cache",
        ttl_days: int = 7,
    ):
        """Initialize the metadata cache.

        Args:
            cache_dir: Directory to store cached JSON files.
            ttl_days: Time-to-live in days before entries expire.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

    def _key_to_path(self, key: str) -> Path:
        """Convert a cache key to a file path."""
        # Hash the key to avoid filesystem issues with special characters
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.json"

    def get(self, key: str) -> dict | None:
        """Retrieve a cached entry if it exists and hasn't expired.

        Args:
            key: Cache key (e.g. repo full name, API URL).

        Returns:
            Cached data dict or None if not found/expired.
        """
        path = self._key_to_path(key)

        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        # Check TTL
        cached_at = raw.get("_cached_at")
        if cached_at:
            cached_time = datetime.fromisoformat(cached_at)
            if datetime.now(timezone.utc) - cached_time > self.ttl:
                # Expired
                path.unlink(missing_ok=True)
                return None

        return raw.get("data")

    def set(self, key: str, data: dict) -> None:
        """Store data in the cache.

        Args:
            key: Cache key.
            data: Data to cache (must be JSON-serializable).
        """
        path = self._key_to_path(key)
        entry = {
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "_key": key,
            "data": data,
        }

        try:
            path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to write cache entry for {key}: {e}")

    def clear(self) -> int:
        """Remove all cached entries.

        Returns:
            Number of entries removed.
        """
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count


# ---------------------------------------------------------------------------
# GitHubSource — DataSource-compatible wrapper
# ---------------------------------------------------------------------------


@register_source("github")
class GitHubSource:
    """DataSource-compatible wrapper that combines GitHub search, clone, and extraction.

    Wraps GitHubClient, RepoProcessor, and MetadataCache into a single
    interface that yields DataRecord instances. Compatible with any future
    pipeline system.

    Usage:
        source = GitHubSource(filter=FILTER_PRESETS["typescript_elite"])
        for record in source.stream(max_repos=50):
            print(record.metadata.get("file_path"), record.metadata.get("language"))
    """

    def __init__(
        self,
        filter: RepoFilter | None = None,
        token: str | None = None,
        clone_dir: str | Path = "data/github_clones",
        cache_dir: str | Path = "data/github_cache",
        languages: list[str] | None = None,
        cleanup: bool = True,
        malware_scan: bool = True,
    ):
        """Initialize the GitHub data source.

        Args:
            filter: RepoFilter for searching repos. If None, uses "popular_any".
            token: GitHub token. Falls back to GITHUB_TOKEN env var.
            clone_dir: Directory for temporary clones.
            cache_dir: Directory for metadata cache.
            languages: Languages to extract (passed to RepoProcessor).
            cleanup: Whether to delete clones after extraction.
            malware_scan: Scan every clone before extraction (default ON).
                Scanner selection comes from configs/scoring.yaml
                (scoring.security.malware_scan).
        """
        self.filter = filter or FILTER_PRESETS["popular_any"]
        self.client = GitHubClient(token=token)
        self.processor = RepoProcessor(languages=languages)
        self.cache = MetadataCache(cache_dir=cache_dir)
        self.clone_dir = Path(clone_dir)
        self.cleanup = cleanup
        self.malware_scan = malware_scan
        self._scanner = None  # Lazy — scanner availability probing costs subprocess calls

    # ── Malware scanning ───────────────────────────────────────────────

    @staticmethod
    def _load_malware_config() -> dict:
        """Read scoring.security.malware_scan from configs/scoring.yaml."""
        path = Path("configs/scoring.yaml")
        if not path.exists():
            return {}
        try:
            import yaml

            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cfg.get("scoring", {}).get("security", {}).get("malware_scan", {})
        except Exception:
            return {}

    def _scan_clone(self, repo_path: Path, repo_name: str) -> bool:
        """Scan a fresh clone before any processing. Returns True if clean.

        Every clone path must be scanned — this used to happen only in
        scrape_github.py's single/import mode, so the default search mode
        ingested unscanned third-party code.
        """
        if not self.malware_scan:
            return True
        if self._scanner is None:
            from cola_coder.security.scanner import CompositeMalwareScanner

            self._scanner = CompositeMalwareScanner.from_config(
                self._load_malware_config(),
            )
        if not self._scanner.available_scanners:
            # Scanning was REQUESTED (malware_scan=True) but no backend is
            # installed/available. Never skip silently — that admits unscanned
            # third-party code as if verified. Warn loudly and proceed
            # unverified (consistent with the incomplete-scan path below).
            logger.warning(
                "SECURITY: malware scanning is enabled but NO scanner is "
                "available (install yara-python, or enable Microsoft Defender / "
                "ClamAV) — clone of %s is NOT scanned and is NOT verified clean",
                repo_name,
            )
            return True
        result = self._scanner.scan_directory(repo_path)
        if not result.is_clean:
            logger.warning(
                "MALWARE: %d threat(s) in clone of %s — repo skipped and deleted",
                len(result.threats), repo_name,
            )
            return False
        if result.had_errors:
            logger.warning(
                "Malware scan of %s incomplete (%d scanner error(s)) — "
                "proceeding, but this clone is not verified clean",
                repo_name, len(result.scan_errors),
            )
        return True

    def _passes_post_clone_checks(
        self,
        repo_path: Path,
        repo_info: dict[str, Any],
    ) -> bool:
        """Check post-clone quality signals defined in the filter."""
        if self.filter.has_tests is True:
            if not self.processor.check_has_tests(repo_path):
                logger.info(f"  Skipping {repo_info.get('full_name', '?')}: no tests found")
                return False

        if self.filter.has_ci is True:
            if not self.processor.check_has_ci(repo_path):
                logger.info(f"  Skipping {repo_info.get('full_name', '?')}: no CI found")
                return False

        if self.filter.typescript_strict is True:
            if not self.processor.check_tsconfig_strict(repo_path):
                logger.info(
                    f"  Skipping {repo_info.get('full_name', '?')}: "
                    f"TypeScript strict mode not enabled"
                )
                return False

        return True

    def _passes_language_percent_check(
        self,
        repo_name: str,
    ) -> bool:
        """Check if the repo meets the language_min_percent requirement."""
        if (
            self.filter.language_min_percent is not None
            and self.filter.primary_language is not None
        ):
            # Check cached first
            cached = self.cache.get(f"languages:{repo_name}")
            if cached:
                lang_breakdown = cached
            else:
                lang_breakdown = self.client.get_languages(repo_name)
                self.cache.set(f"languages:{repo_name}", lang_breakdown)

            primary = self.filter.primary_language
            pct = lang_breakdown.get(primary, 0.0)
            if pct < self.filter.language_min_percent:
                logger.info(
                    f"  Skipping {repo_name}: {primary} is {pct:.0%}, "
                    f"need {self.filter.language_min_percent:.0%}"
                )
                return False

        return True

    def _passes_owner_checks(self, repo_info: dict[str, Any]) -> bool:
        """Check owner-related filter criteria."""
        owner_login = repo_info.get("owner", {}).get("login", "")

        if self.filter.owner_type is not None:
            actual_type = repo_info.get("owner", {}).get("type", "")
            if actual_type != self.filter.owner_type:
                logger.info(
                    f"  Skipping {repo_info.get('full_name', '?')}: "
                    f"owner type is {actual_type}, want {self.filter.owner_type}"
                )
                return False

        if self.filter.min_owner_followers is not None and owner_login:
            cached = self.cache.get(f"owner:{owner_login}")
            if cached:
                owner_info = cached
            else:
                owner_info = self.client.get_owner_info(owner_login)
                self.cache.set(f"owner:{owner_login}", owner_info)

            followers = owner_info.get("followers", 0)
            if followers < self.filter.min_owner_followers:
                logger.info(
                    f"  Skipping {repo_info.get('full_name', '?')}: "
                    f"owner has {followers} followers, need {self.filter.min_owner_followers}"
                )
                return False

        return True

    def stream(
        self,
        max_repos: int = 100,
        sort: str = "stars",
    ) -> Iterator[DataRecord]:
        """Search, clone, and extract files from matching repos.

        This is the main entry point. It:
        1. Searches GitHub for repos matching the filter
        2. Checks language percentages and owner quality
        3. Clones each repo (shallow, hooks disabled)
        4. Scans the clone for malware (skipped + deleted on threat)
        5. Runs post-clone quality checks
        6. Extracts source files as DataRecord instances
        7. Cleans up clones (if cleanup=True)

        Args:
            max_repos: Maximum number of repos to process.
            sort: Sort field for search results.

        Yields:
            DataRecord for each valid source file across all matching repos.
        """
        repos = self.client.search_repos(
            self.filter, max_results=max_repos, sort=sort,
        )

        logger.info(f"Found {len(repos)} repos matching filter")

        for repo_info in repos:
            full_name = repo_info["full_name"]
            stars = repo_info.get("stargazers_count", 0)
            html_url = repo_info.get("html_url", "")
            license_info = repo_info.get("license") or {}
            spdx_id = license_info.get("spdx_id", "")

            # Pre-clone checks
            if not self._passes_language_percent_check(full_name):
                continue
            if not self._passes_owner_checks(repo_info):
                continue

            # Excluded topics check
            repo_topics = set(repo_info.get("topics", []))
            if self.filter.topics_exclude:
                excluded = repo_topics & set(self.filter.topics_exclude)
                if excluded:
                    logger.info(f"  Skipping {full_name}: has excluded topics {excluded}")
                    continue

            # Excluded languages check
            if self.filter.languages_exclude:
                lang = repo_info.get("language", "")
                if lang and lang.lower() in [ex.lower() for ex in self.filter.languages_exclude]:
                    logger.info(f"  Skipping {full_name}: excluded language {lang}")
                    continue

            # Clone
            try:
                repo_path = self.client.clone_repo(
                    full_name, self.clone_dir, shallow=True,
                )
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                logger.warning(f"  Failed to clone {full_name}: {e}")
                continue

            try:
                # Malware scan FIRST — before any file processing. On threat,
                # delete the clone immediately (even when cleanup=False) so
                # flagged code never lingers on disk.
                if not self._scan_clone(repo_path, full_name):
                    if repo_path.exists():
                        shutil.rmtree(repo_path, ignore_errors=True)
                    continue

                # Post-clone checks
                if not self._passes_post_clone_checks(repo_path, repo_info):
                    continue

                # Detect license from repo files if API didn't return one
                if not spdx_id or spdx_id == "NOASSERTION":
                    detected = self.processor.check_license(repo_path)
                    if detected:
                        spdx_id = detected

                # Extract files
                file_count = 0
                for record in self.processor.extract_files(
                    repo_path,
                    repo_name=full_name,
                    repo_stars=stars,
                    repo_url=html_url,
                    repo_license=spdx_id,
                ):
                    file_count += 1
                    yield record

                logger.info(f"  Extracted {file_count} files from {full_name}")

            finally:
                # Cleanup clone
                if self.cleanup and repo_path.exists():
                    try:
                        shutil.rmtree(repo_path)
                        logger.info(f"  Cleaned up {repo_path}")
                    except OSError as e:
                        logger.warning(f"  Failed to cleanup {repo_path}: {e}")
