# Plan: GitHub Scraper for Custom Training Data

## Problem

StarCoderData is good but generic. To build a truly competitive model, you need:

- **Curated high-quality repos** (not random GitHub noise)
- **Latest code** (StarCoderData snapshots can be months/years old)
- **Your own selection criteria** (e.g., only repos with >100 stars, good test coverage)
- **Private/proprietary code** (your own repos, your company's code)

## Architecture

### Script: `scripts/scrape_github.py`

Interactive CLI for discovering, downloading, and processing GitHub code into
training-ready format.

### Three Modes

#### 1. Curated Repo List (`--repos`)

```bash
python scripts/scrape_github.py --repos data/curated_repos.txt --output data/github/
```

Where `curated_repos.txt` is:
```
vercel/next.js
microsoft/TypeScript
prisma/prisma
trpc/trpc
drizzle-team/drizzle-orm
colinhacks/zod
tanstack/query
# Comments are ignored
```

#### 2. Search Discovery (`--search`)

```bash
python scripts/scrape_github.py --search \
  --language typescript \
  --min-stars 100 \
  --min-forks 20 \
  --license MIT,Apache-2.0 \
  --max-repos 500 \
  --output data/github/
```

Uses GitHub Search API to find repos matching criteria, then clones them.

#### 3. Topic/Ecosystem Discovery (`--topic`)

```bash
python scripts/scrape_github.py --topic "nextjs,react,graphql" \
  --min-stars 50 \
  --output data/github/
```

Discovers repos by GitHub topic tags.

### Pipeline Flow

```
GitHub API / git clone
  |
  v
[Clone repos to temp dir]
  |
  v
[Extract code files by extension]
  |
  v
[License check - reject non-permissive]
  |
  v
[Metadata enrichment]
  |  - repo stars, forks, last commit date
  |  - file path, language, size
  |  - git blame age (how recently modified)
  |
  v
[Quality filtering (reuse existing filters)]
  |
  v
[Deduplication against existing data]
  |
  v
[Save as DataRecord stream -> .npy]
```

### Key Components

#### GitHubClient

```python
# src/cola_coder/data/sources/github.py

class GitHubClient:
    """Handles GitHub API interaction and repo cloning."""

    def __init__(self, token: str | None = None):
        """Token from GITHUB_TOKEN env var or --token flag."""
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.rate_limit_remaining = 5000

    def search_repos(
        self,
        language: str,
        min_stars: int = 0,
        min_forks: int = 0,
        license: list[str] | None = None,
        topics: list[str] | None = None,
        max_results: int = 100,
        sort: str = "stars",  # "stars", "forks", "updated"
        created_after: str | None = None,  # "2024-01-01"
    ) -> list[RepoInfo]:
        """Search GitHub API for repos matching criteria.

        Uses paginated search: GET /search/repositories
        Respects rate limits (30 req/min for search, 5000/hr for REST).
        Returns RepoInfo dataclass with name, stars, license, topics, etc.
        """

    def clone_repo(
        self,
        repo: str,  # "owner/name"
        dest: Path,
        shallow: bool = True,  # --depth 1
        branch: str | None = None,
    ) -> Path:
        """Git clone a repo. Shallow by default for speed."""

    def get_repo_info(self, repo: str) -> RepoInfo:
        """Fetch metadata for a single repo."""
```

#### RepoProcessor

```python
class RepoProcessor:
    """Extract and process code files from a cloned repo."""

    LANGUAGE_EXTENSIONS = {
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx", ".mjs"],
        "python": [".py"],
        "go": [".go"],
        "rust": [".rs"],
        "java": [".java"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".hh"],
    }

    IGNORE_PATTERNS = [
        "node_modules/", "vendor/", "dist/", "build/", ".git/",
        "__pycache__/", ".next/", "coverage/", ".turbo/",
        "*.min.js", "*.min.css", "*.bundle.js", "*.map",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "*.d.ts",  # Type declarations (generated, low signal)
    ]

    def __init__(self, languages: list[str], max_file_size: int = 100_000):
        ...

    def extract_files(self, repo_path: Path) -> Iterator[DataRecord]:
        """Walk repo tree, yield DataRecords with metadata.

        Metadata includes:
        - source: "github"
        - repo: "owner/name"
        - path: relative file path
        - language: detected language
        - license: repo license
        - stars: repo star count
        - last_modified: git log date for this file
        """

    def check_license(self, repo_path: Path) -> str | None:
        """Read LICENSE file and classify.

        Returns license SPDX identifier or None if not found.
        Uses simple pattern matching on LICENSE/LICENSE.md content.
        """
```

#### Deduplication Against Existing Data

```python
class CrossDeduplicator:
    """Detect duplicates across existing training data and new GitHub data.

    Uses MinHash LSH (Locality-Sensitive Hashing) for near-duplicate detection.
    This is the same technique used by BigCode/StarCoder and The Stack v2.

    How it works (for a TS dev):
    1. Each file → set of character n-grams (like substrings of length 5)
    2. Each set → MinHash signature (compact fingerprint, ~128 values)
    3. LSH index: group similar signatures into buckets
    4. Files in the same bucket are candidate duplicates
    5. Verify with actual Jaccard similarity

    Think of it like: shingle the text, hash the shingles, compare hashes.
    Two files with >80% matching shingles are near-duplicates.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        from datasketch import MinHash, MinHashLSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def build_index(self, existing_data_path: str):
        """Index existing training data for dedup checking."""

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a near-duplicate of indexed data."""

    def compute_minhash(self, content: str) -> MinHash:
        """Compute MinHash signature for a code file."""
```

### CLI Interface

```
+-----------------------------------------------------------------------------+
|  Cola-Coder  GitHub Scraper                                                  |
+-----------------------------------------------------------------------------+

  What would you like to do?
+------+---------------------------+----------------------------------------------+
|    # | Option                    | Details                                      |
+------+---------------------------+----------------------------------------------+
|    1 | Search by language        | Find top repos by language + stars            |
|    2 | Search by topic           | Find repos by GitHub topic tags               |
|    3 | Import repo list          | Clone repos from a curated text file          |
|    4 | Clone single repo         | Clone and process one specific repo           |
+------+---------------------------+----------------------------------------------+

  Select [1-4]: 1

  Language: typescript
  Minimum stars [100]: 500
  License filter [MIT,Apache-2.0,BSD]: MIT,Apache-2.0
  Max repos [100]: 200
  Sort by [stars/forks/updated]: stars

Step 1/4 - Searching GitHub...
  Found 200 TypeScript repos with >500 stars and MIT/Apache license

Step 2/4 - Cloning repos...
  [1/200] vercel/next.js (120k stars, MIT) ... done (2.1s)
  [2/200] microsoft/TypeScript (97k stars, Apache-2.0) ... done (3.4s)
  ...

Step 3/4 - Extracting & filtering code...
  Processing 200 repos...
  Files extracted: 45,000
  After quality filter: 38,000 (84.4% kept)
  After dedup vs existing data: 35,200 (92.6% unique)

Step 4/4 - Saving dataset...
  Output: data/processed/github_ts_stars500_200repos.npy
  Chunks: 12,400 x 2048 tokens = 25,395,200 total tokens
  Manifest: data/processed/github_ts_stars500_200repos.manifest.yaml

+--------------------------------------------------------------------+
|  Complete                                                           |
|                                                                     |
|  v GitHub scrape complete!                                          |
|                                                                     |
|    Output: data/processed/github_ts_stars500_200repos.npy           |
|    Repos: 200 cloned, 195 had permissive licenses                   |
|    Files: 35,200 unique, high-quality code files                    |
|    Tokens: 25,395,200                                               |
|    Next: python scripts/train.py --config configs/tiny.yaml         |
+--------------------------------------------------------------------+
```

### Rate Limiting & Robustness

```python
class RateLimiter:
    """Respect GitHub API rate limits.

    Unauthenticated: 60 requests/hour (useless)
    Authenticated: 5,000 requests/hour (REST), 30/min (search)

    Strategy:
    1. Track remaining requests from response headers
    2. When low, sleep until reset time
    3. Use conditional requests (If-None-Match) to avoid counting cached responses
    4. Clone via git (no API call), only use API for search and metadata
    """
```

### Dependencies

```
datasketch>=1.6.0     # MinHash LSH for deduplication
gitpython>=3.1.0      # Git operations (clone, log)
# OR just shell out to `git clone` which is simpler and faster
```

### Output Format

Same .npy format as existing pipeline. The GitHub scraper produces data that
is directly usable by `train.py` — no conversion needed. It can be used
standalone or mixed with HuggingFace data via the pipeline system.

### Security Considerations

- Never clone repos with executable hooks (use `--config core.hooksPath=/dev/null`)
- Sanitize repo names to prevent path traversal
- Limit total disk usage (configurable max, default 50GB for clones)
- Clean up cloned repos after extraction (keep only the .npy output)
- Never include `.env`, credentials, or secrets in training data

### Future: GH Archive (Massive Scale)

For truly massive data collection (billions of files), GH Archive + BigQuery is
the path used by BigCode for The Stack. This is a separate effort:

1. Query BigQuery for `github_repos` table (needs GCP account)
2. Filter by license, language, stars
3. Download content via `contents` table
4. Process locally

This is how StarCoder was built. We can add this as a DataSource plugin later.
