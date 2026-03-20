# Plan: Enhanced Repository Filtering & Metadata Scraping

## Problem

The current scraper plan (03) has basic search by language and stars. To build truly competitive training data, we need DEEP repo-level filtering on every available signal — like building a search engine for code quality.

## Repo Metadata to Scrape & Filter On

### GitHub API Fields (per repo)

| Signal | API Field | Filter Use |
|--------|-----------|-----------|
| Stars | `stargazers_count` | Quality proxy (>100 = decent, >1000 = good) |
| Forks | `forks_count` | Community adoption |
| Watchers | `subscribers_count` | Active interest |
| Open Issues | `open_issues_count` | Maintenance signal |
| Size (KB) | `size` | Skip enormous monorepos |
| Language breakdown | `GET /repos/{owner}/{repo}/languages` | % TypeScript, filter by composition |
| License | `license.spdx_id` | Legal compliance |
| Created date | `created_at` | Skip ancient repos |
| Last push | `pushed_at` | Recency/maintenance |
| Default branch | `default_branch` | Clone target |
| Has wiki | `has_wiki` | Documentation quality |
| Has pages | `has_pages` | Documentation quality |
| Topics | `topics` | Domain classification |
| Archived | `archived` | Skip dead projects |
| Fork flag | `fork` | Skip forks (duplicates) |
| Description | `description` | Quality signal |
| Homepage | `homepage` | Maintained project signal |

### Owner/User Metadata

| Signal | API Field | Filter Use |
|--------|-----------|-----------|
| User type | `owner.type` | "Organization" vs "User" |
| Followers | `GET /users/{user}` → `followers` | Author reputation |
| Public repos | `public_repos` | Prolific author signal |
| Other repo stars | Aggregate from user's repos | Author quality signal |
| Created date | `created_at` | Account age |

### Computed Signals (derived from raw data)

| Signal | How to Compute | Filter Use |
|--------|---------------|-----------|
| Language % | `languages` endpoint → `ts_bytes / total_bytes` | e.g., "repos >60% TypeScript" |
| Commit frequency | `GET /repos/{owner}/{repo}/stats/commit_activity` | Active development signal |
| Contributors count | `GET /repos/{owner}/{repo}/contributors?per_page=1&anon=true` → Link header | Team size |
| Has CI | Check for `.github/workflows/`, `.circleci/`, etc. | Quality signal |
| Has tests | Check for `test/`, `__tests__/`, `*.test.*`, `*.spec.*` | Test coverage proxy |
| Package.json deps | Parse package.json → dependency list | Modern stack detection |
| TypeScript strict | Parse tsconfig.json → `strict: true` | Code quality signal |
| README quality | Length + structure of README.md | Documentation quality |

## Filter Configuration

```python
@dataclass
class RepoFilter:
    """Rich filtering criteria for GitHub repo discovery."""

    # Star / fork / watcher thresholds
    min_stars: int = 0
    max_stars: int | None = None
    min_forks: int = 0
    max_forks: int | None = None

    # Language composition
    primary_language: str | None = None          # GitHub's detected primary language
    language_min_percent: float = 0.0            # e.g., 0.6 = "at least 60% TypeScript"
    languages_include: list[str] | None = None   # Must contain these languages
    languages_exclude: list[str] | None = None   # Must NOT contain these

    # Licensing
    licenses: list[str] | None = None            # Allowed SPDX IDs
    require_license: bool = True                 # Reject repos with no license

    # Recency & maintenance
    pushed_after: str | None = None              # ISO date: "2024-01-01"
    created_after: str | None = None
    min_commits_last_year: int = 0
    not_archived: bool = True

    # Owner quality
    owner_type: str | None = None                # "Organization" or "User"
    min_owner_followers: int = 0
    min_owner_repos: int = 0
    min_owner_total_stars: int = 0               # Sum of stars across all their repos

    # Repo quality signals
    is_fork: bool | None = False                 # None=any, False=originals only
    has_description: bool = False
    has_readme: bool = False
    min_contributors: int = 0
    has_ci: bool = False
    has_tests: bool = False

    # TypeScript-specific
    typescript_strict: bool | None = None        # Require strict mode in tsconfig
    has_package_json: bool = False

    # Size limits
    max_repo_size_kb: int = 500_000              # Skip repos > 500MB
    min_files: int = 0
    max_files: int | None = None

    # Content
    topics_include: list[str] | None = None
    topics_exclude: list[str] | None = None

    def to_github_query(self) -> str:
        """Convert filter to GitHub search API query string.

        GitHub search supports: language, stars, forks, pushed, created,
        license, topic, archived, fork, size, user, org
        """
        parts = []
        if self.primary_language:
            parts.append(f"language:{self.primary_language}")
        if self.min_stars:
            max_s = self.max_stars or ""
            parts.append(f"stars:{self.min_stars}..{max_s}")
        if self.min_forks:
            parts.append(f"forks:>={self.min_forks}")
        if self.pushed_after:
            parts.append(f"pushed:>={self.pushed_after}")
        if self.created_after:
            parts.append(f"created:>={self.created_after}")
        if self.licenses:
            for lic in self.licenses:
                parts.append(f"license:{lic.lower()}")
        if self.topics_include:
            for topic in self.topics_include:
                parts.append(f"topic:{topic}")
        if self.not_archived:
            parts.append("archived:false")
        if self.is_fork is False:
            parts.append("fork:false")
        if self.max_repo_size_kb:
            parts.append(f"size:<={self.max_repo_size_kb}")
        return " ".join(parts)
```

## Filter Presets (for CLI menu)

```python
FILTER_PRESETS = {
    "typescript_elite": RepoFilter(
        primary_language="TypeScript",
        language_min_percent=0.6,
        min_stars=500,
        licenses=["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"],
        pushed_after="2024-01-01",
        not_archived=True,
        is_fork=False,
        has_tests=True,
        typescript_strict=True,
        max_repo_size_kb=200_000,
    ),
    "typescript_good": RepoFilter(
        primary_language="TypeScript",
        language_min_percent=0.4,
        min_stars=50,
        licenses=["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"],
        pushed_after="2023-01-01",
        not_archived=True,
        is_fork=False,
    ),
    "python_elite": RepoFilter(
        primary_language="Python",
        language_min_percent=0.6,
        min_stars=500,
        licenses=["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"],
        pushed_after="2024-01-01",
        not_archived=True,
        is_fork=False,
        has_tests=True,
    ),
    "popular_any": RepoFilter(
        min_stars=1000,
        licenses=["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"],
        not_archived=True,
        is_fork=False,
    ),
}
```

## Implementation: RepoMetadataCollector

```python
class RepoMetadataCollector:
    """Collect rich metadata for filtering decisions.

    Makes additional API calls beyond basic search to gather
    language breakdown, owner info, repo contents, etc.

    Rate limit strategy:
    - Batch metadata collection (collect 100 repos, then filter)
    - Cache results to disk (JSON file per repo)
    - Conditional requests (ETag/If-None-Match) for updates
    - Sleep on 403 rate limit (respect X-RateLimit-Reset header)
    """

    def collect(self, repo_name: str) -> RepoMetadata:
        """Collect all available metadata for a repo."""
        basic = self._get_repo_info(repo_name)        # 1 API call
        languages = self._get_languages(repo_name)     # 1 API call
        owner = self._get_owner_info(basic.owner)      # 1 API call (cached)
        contents = self._check_repo_contents(repo_name)  # 1 API call

        return RepoMetadata(
            name=repo_name,
            stars=basic.stars,
            forks=basic.forks,
            watchers=basic.watchers,
            license=basic.license,
            language_breakdown=languages,      # {"TypeScript": 0.72, "JavaScript": 0.15, ...}
            owner_followers=owner.followers,
            owner_total_stars=owner.total_stars,
            has_ci=contents.has_ci,
            has_tests=contents.has_tests,
            has_readme=contents.has_readme,
            tsconfig_strict=contents.tsconfig_strict,
            ...
        )

    def passes_filter(self, metadata: RepoMetadata, filter: RepoFilter) -> tuple[bool, str]:
        """Check if repo metadata passes the filter. Returns (pass, reason)."""
```

## CLI Integration

Add to the interactive scraper menu:

```
Step 2/5 · Repo Quality Filter
  [1] Elite       — >500 stars, strict TS, has tests, MIT/Apache
  [2] Good        — >50 stars, any TS, permissive license
  [3] Broad       — >10 stars, any language, any license
  [4] Custom...   — Configure individual filters
```

Custom mode shows sub-menus for each filter category.

## Metadata Cache

```python
class MetadataCache:
    """Cache repo metadata to avoid redundant API calls.

    Stores as JSON files in data/github_cache/{owner}/{repo}.json
    with TTL-based expiry (default 7 days).

    Why cache?
    - GitHub API rate limit: 5000 req/hr
    - 3 API calls per repo × 1000 repos = 3000 API calls
    - Without cache: can only process ~1600 repos/hr
    - With cache: subsequent runs are instant for cached repos
    """
```

## Dependencies

Same as plan 03 plus:
```
requests>=2.31.0      # HTTP client (already likely installed)
```

No new heavy dependencies — all metadata comes from GitHub REST API.
