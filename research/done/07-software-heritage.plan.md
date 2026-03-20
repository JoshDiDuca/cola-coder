# 07 — Software Heritage as a Data Source

## What is Software Heritage?

Software Heritage (SWH) is the largest archive of source code in the world, run by
Inria (French national research institute). Its mission is to collect, preserve, and
share all publicly available source code. Think of it as the Internet Archive, but
specifically for code.

Key facts:
- **18+ billion unique source files** (deduplicated by content hash)
- **250+ million origins** (repos from GitHub, GitLab, Bitbucket, PyPI, CRAN, etc.)
- Archives snapshots over time, not just latest state
- Nonprofit, public-interest infrastructure

## How SWH differs from GitHub

| Aspect | GitHub | Software Heritage |
|--------|--------|-------------------|
| Purpose | Hosting & collaboration | Archival & preservation |
| Deduplication | None (same file in 1000 repos = 1000 copies) | By content hash (SHA-256 via SWHID) |
| Scope | GitHub repos only | GitHub + GitLab + Bitbucket + PyPI + CRAN + Debian + ... |
| Historical data | Only if repo still exists | Archived even after repo deletion |
| Identifiers | Owner/repo/path (mutable) | SWHID (content-addressable, immutable) |
| Access | REST API + git clone | REST API + bulk dataset exports + graph dataset |
| Rate limits | 5000 req/hr (authenticated) | 1200 req/hr (unauth), 12000 req/hr (auth) |

The content-addressable storage is the killer feature for training data: if the same
utility function appears in 500 repos, SWH stores it once. This makes deduplication
essentially free at the archive level.

## Data Access Methods

### 1. REST API (archive.softwareheritage.org/api/1/)

The most accessible method. Endpoints follow the SWH data model:

```
Origin (repo URL)
  → Visit (point-in-time snapshot of the origin)
    → Snapshot (set of branches/tags)
      → Revision (commit)
        → Directory (tree)
          → Content (file blob)
```

Key endpoints:
- `GET /api/1/origin/{url}/visits/` — list visits for an origin
- `GET /api/1/snapshot/{id}/` — get branches from a snapshot
- `GET /api/1/revision/{id}/` — get commit metadata
- `GET /api/1/revision/{id}/directory/` — list files at a commit
- `GET /api/1/directory/{id}/` — list entries in a directory
- `GET /api/1/content/sha1:{hash}/raw/` — download raw file content
- `GET /api/1/origin/search/{query}/` — search for origins by URL pattern

Rate limits:
- Unauthenticated: **1200 requests/hour** (20/min)
- Authenticated: **12000 requests/hour** (200/min)
- Response headers: `X-RateLimit-Remaining`, `X-RateLimit-Limit`, `X-RateLimit-Reset`
- On 429: `Retry-After` header tells you when to retry

Authentication:
- Get a token at https://archive.softwareheritage.org (account required)
- Pass as `Authorization: Bearer <token>` header

Practical throughput:
- With auth, ~3 req/sec sustained
- To get one file: need ~4-5 requests (origin → visit → snapshot → directory → content)
- So ~0.6 files/sec, or ~2000 files/hour

### 2. SWH Dataset Exports (dataset.softwareheritage.org)

Bulk compressed exports of the full archive, hosted on Amazon S3:
- **Graph dataset**: Node/edge tables for the entire dependency graph (~1TB compressed)
- **Content dataset**: All unique file contents (~hundreds of TB)
- Format: Apache ORC or CSV, compressed
- Updated periodically (roughly every 6 months)

This is what The Stack v2 used. The process:
1. Download the graph dataset to find Python/TypeScript/etc. files
2. Filter by file extension, license, etc. using the graph metadata
3. Fetch actual content by SHA hash from the content dataset

For a small project: **way too much data and infrastructure overhead.**

### 3. SWH Graph (Compressed Graph Representation)

The WebGraph-compressed version of the SWH relationship graph:
- Billions of nodes (origins, snapshots, revisions, directories, contents)
- Billions of edges (parent-child relationships)
- Compressed from ~1TB to ~100GB using WebGraph framework
- Needs Java + significant RAM to traverse

Used for research at scale, not practical for small projects.

## How The Stack v2 Used SWH

The Stack v2 (by BigCode / ServiceNow) is the largest open code dataset, used to train
StarCoder2. Their process:

1. **Started with SWH graph dataset** (2023-09-06 export)
2. **Filtered to code files** by extension (~600 programming languages)
3. **Near-dedup** using MinHash + LSH (same as The Stack v1)
4. **License detection** using ScanCode on a sample, then github-linguist for language
5. **PII removal** using regex patterns
6. **Opted-out repos** removed based on am-i-in-the-stack.huggingface.co

Result: **~3.3TB of deduplicated, licensed code** across 600+ languages.

Key insight: they did NOT use the SWH API. They downloaded the bulk dataset exports
and processed them on a cluster. This is not feasible for a small project.

## Practical Considerations for cola-coder

### What makes sense for us

For our scale (training 50M-350M parameter models on a single GPU), we have two
practical options:

**Option A: Use The Stack v2 via HuggingFace (recommended for volume)**
- Already available as `bigcode/the-stack-v2-dedup` on HuggingFace
- Already SWH-derived, deduplicated, licensed
- Our HuggingFaceSource already supports this
- Best for bulk training data

**Option B: Use SWH API for targeted retrieval (this data source)**
- Good for fetching specific repos/files not in StarCoderData
- Good for getting historical versions of code
- Good for supplementing training data with specific projects
- Rate-limited but sufficient for targeted use

### When to use the SWH API source

1. **Specific project retrieval**: Want code from a specific open-source project
   that's been archived but maybe deleted from GitHub
2. **Historical code**: Want old versions of popular libraries to train on API evolution
3. **Cross-platform origins**: Want code from GitLab/Bitbucket/etc. repos that aren't
   on GitHub
4. **Dedup-guaranteed supplementation**: Every file from SWH is guaranteed unique by
   content hash

### When NOT to use it

1. **Bulk training data**: Use HuggingFace source with StarCoderData or The Stack v2
2. **Latest code**: SWH crawls with some delay; GitHub source is better for fresh code
3. **High throughput**: 2000 files/hour is too slow for building a large dataset

## Implementation Plan

Build `SoftwareHeritageSource` with:
- `SWHClient` class handling HTTP, auth, and rate limiting
- Stream files from specified origins (repo URLs)
- Filter by file extension
- Respect rate limits via `X-RateLimit-*` headers
- Handle 429 with exponential backoff
- Optional API token via constructor or `SWH_API_TOKEN` env var
- `requests` as an optional dependency (handle ImportError)

The source follows the same `DataSource.stream() -> Iterator[DataRecord]` pattern
as our other sources, making it composable in pipelines.

## References

- Software Heritage: https://www.softwareheritage.org/
- SWH API docs: https://archive.softwareheritage.org/api/1/
- SWH identifiers (SWHID): https://docs.softwareheritage.org/devel/swh-model/persistent-identifiers.html
- The Stack v2 paper: https://arxiv.org/abs/2402.19173
- SWH dataset exports: https://docs.softwareheritage.org/devel/swh-dataset/
