# Plan: Extensible Data Pipeline Architecture

## Problem

The current pipeline is hardcoded to one source (StarCoderData via HuggingFace) and one
filter system. To compete with top models, we need:

- Multiple data sources (HuggingFace, GitHub, local files, synthetic data)
- Pluggable filters (quality, language, dedup, PII, license)
- Data mixing (ratio control across sources)
- Pipeline composition (chain sources → filters → transforms → output)

## Architecture: DataSource + FilterPlugin + Transform Pipeline

### Core Abstractions

```python
# src/cola_coder/data/pipeline.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class DataRecord:
    """A single code file flowing through the pipeline."""
    content: str                          # The code
    metadata: dict = field(default_factory=dict)  # Extensible metadata
    # metadata examples:
    #   source: "github", "huggingface", "local", "synthetic"
    #   language: "typescript"
    #   repo: "vercel/next.js"
    #   license: "MIT"
    #   stars: 45000
    #   path: "src/server/router.ts"
    #   quality_score: 0.87
    #   dedup_hash: "abc123"


class DataSource(ABC):
    """Base class for all data sources."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def stream(self) -> Iterator[DataRecord]:
        """Yield DataRecord objects."""
        ...

    def estimate_size(self) -> int | None:
        """Optional: estimated number of files. Used for progress bars."""
        return None


class FilterPlugin(ABC):
    """Base class for all filters."""

    @abstractmethod
    def name(self) -> str:
        """Filter name for stats tracking."""
        ...

    @abstractmethod
    def check(self, record: DataRecord) -> tuple[bool, str]:
        """Return (keep, reason). reason only used when keep=False."""
        ...

    def setup(self, config: dict) -> None:
        """Optional: configure the filter from YAML config."""
        pass


class Transform(ABC):
    """Base class for data transforms (modify records, don't filter)."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def apply(self, record: DataRecord) -> DataRecord:
        """Transform a record. Return modified record."""
        ...
```

### Built-in Data Sources

```python
# src/cola_coder/data/sources/

class HuggingFaceSource(DataSource):
    """Stream from any HuggingFace dataset (StarCoderData, The Stack v2, etc.)."""
    def __init__(self, dataset: str, languages: list[str], split: str = "train"):
        ...

class GitHubSource(DataSource):
    """Clone and stream from GitHub repos (see github-scraper plan)."""
    def __init__(self, repos: list[str] | str, languages: list[str]):
        ...

class LocalFileSource(DataSource):
    """Stream from local directories of code files."""
    def __init__(self, paths: list[str], extensions: list[str]):
        ...

class SyntheticSource(DataSource):
    """Generate synthetic training data via LLM API."""
    def __init__(self, generator_config: dict):
        ...

class MixedSource(DataSource):
    """Combine multiple sources with configurable ratios."""
    def __init__(self, sources: list[tuple[DataSource, float]]):
        # e.g., [(hf_source, 0.7), (github_source, 0.2), (synthetic_source, 0.1)]
        ...
```

### Built-in Filters

```python
# src/cola_coder/data/filters/

class QualityFilter(FilterPlugin):
    """The existing conservative/strict filter, wrapped as a plugin."""

class DeduplicationFilter(FilterPlugin):
    """MinHash-based near-duplicate detection."""
    # Uses datasketch library for MinHash LSH
    # Configurable: exact, near-duplicate (Jaccard threshold), or both

class LicenseFilter(FilterPlugin):
    """Only keep files with permissive licenses (MIT, Apache, BSD)."""
    # Checks repo-level LICENSE file, not per-file
    # Uses license metadata from DataRecord.metadata["license"]

class PIIFilter(FilterPlugin):
    """Remove files containing PII (emails, API keys, passwords)."""
    # Regex-based + optional presidio integration

class LanguageSyntaxFilter(FilterPlugin):
    """Language-aware syntax validation using tree-sitter."""
    # Full AST parse, not just heuristics
    # Supports: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++

class LengthFilter(FilterPlugin):
    """Configurable min/max line count and file size."""

class ContentFilter(FilterPlugin):
    """Reject files matching content patterns (ads, spam, boilerplate)."""
```

### Built-in Transforms

```python
# src/cola_coder/data/transforms/

class StripComments(Transform):
    """Remove comments from code (optional, for some training strategies)."""

class NormalizeWhitespace(Transform):
    """Normalize indentation (tabs → spaces, trailing whitespace)."""

class AddMetadata(Transform):
    """Enrich DataRecord with computed metadata (language detection, complexity score)."""

class TruncateToMaxTokens(Transform):
    """Truncate very long files to max_tokens_per_file."""
```

### Pipeline Composition

```python
# src/cola_coder/data/pipeline.py

@dataclass
class PipelineConfig:
    """Loaded from YAML."""
    sources: list[dict]       # [{type: "huggingface", dataset: "...", weight: 0.7}, ...]
    filters: list[dict]       # [{type: "quality", mode: "conservative"}, ...]
    transforms: list[dict]    # [{type: "normalize_whitespace"}, ...]
    output: dict              # {dir: "./data/processed", name: "auto"}

class DataPipeline:
    """Compose sources, filters, and transforms into a processing pipeline."""

    def __init__(self, config: PipelineConfig):
        self.sources = self._build_sources(config.sources)
        self.filters = self._build_filters(config.filters)
        self.transforms = self._build_transforms(config.transforms)
        self.stats = PipelineStats()

    def stream(self) -> Iterator[DataRecord]:
        """Yield processed records through the full pipeline."""
        for record in self._mix_sources():
            # Run all filters
            keep = True
            for f in self.filters:
                passed, reason = f.check(record)
                if not passed:
                    self.stats.record_rejection(f.name(), reason)
                    keep = False
                    break
            if not keep:
                continue

            # Run all transforms
            for t in self.transforms:
                record = t.apply(record)

            self.stats.record_kept()
            yield record

    def content_stream(self) -> Iterator[str]:
        """Convenience: yield just content strings (for tokenizer)."""
        for record in self.stream():
            yield record.content
```

### Pipeline YAML Config

```yaml
# configs/pipeline.yaml — defines a complete data pipeline

pipeline:
  sources:
    - type: huggingface
      dataset: "bigcode/starcoderdata"
      languages: ["typescript", "javascript"]
      weight: 0.7

    - type: github
      repo_list: "data/curated_repos.txt"  # One repo per line
      languages: ["typescript"]
      weight: 0.2

    - type: local
      paths: ["./data/custom/my_typescript_code/"]
      extensions: [".ts", ".tsx"]
      weight: 0.1

  filters:
    - type: quality
      mode: conservative
    - type: deduplication
      method: minhash
      threshold: 0.8
    - type: license
      allowed: ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"]
    - type: pii
      enabled: true

  transforms:
    - type: normalize_whitespace
    - type: add_metadata

  output:
    dir: "./data/processed"
    name: auto  # Auto-generated from sources + filters
```

### Registry Pattern for Plugins

```python
# src/cola_coder/data/registry.py

_SOURCE_REGISTRY: dict[str, type[DataSource]] = {}
_FILTER_REGISTRY: dict[str, type[FilterPlugin]] = {}
_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}

def register_source(name: str):
    """Decorator to register a DataSource class."""
    def decorator(cls):
        _SOURCE_REGISTRY[name] = cls
        return cls
    return decorator

def register_filter(name: str):
    """Decorator to register a FilterPlugin class."""
    def decorator(cls):
        _FILTER_REGISTRY[name] = cls
        return cls
    return decorator

# Usage:
@register_source("huggingface")
class HuggingFaceSource(DataSource):
    ...

@register_filter("quality")
class QualityFilter(FilterPlugin):
    ...
```

## Implementation Order

1. **Phase 1: Core abstractions** (DataRecord, DataSource, FilterPlugin, Transform)
2. **Phase 2: Wrap existing code** (HuggingFaceSource wraps download.py, QualityFilter wraps quality_filter.py)
3. **Phase 3: Pipeline composer** (PipelineConfig, DataPipeline, registry)
4. **Phase 4: New sources** (GitHubSource, LocalFileSource)
5. **Phase 5: New filters** (Dedup, License, PII, tree-sitter syntax)
6. **Phase 6: Pipeline YAML** (Load pipeline from config file)

## File Structure

```
src/cola_coder/data/
  pipeline.py          # DataRecord, DataPipeline, PipelineConfig
  registry.py          # Plugin registry (register_source, register_filter, etc.)
  sources/
    __init__.py
    huggingface.py     # HuggingFaceSource (wraps existing download.py)
    github.py          # GitHubSource (see github-scraper plan)
    local.py           # LocalFileSource
    synthetic.py       # SyntheticSource
    mixed.py           # MixedSource (weighted combination)
  filters/
    __init__.py
    quality.py         # QualityFilter (wraps existing quality_filter.py)
    dedup.py           # MinHash deduplication
    license.py         # License checking
    pii.py             # PII detection
    syntax.py          # Tree-sitter syntax validation
    length.py          # Length-based filtering
  transforms/
    __init__.py
    whitespace.py      # Normalize whitespace
    metadata.py        # Enrich metadata
    truncate.py        # Truncate long files
  # Existing files (kept, wrapped by plugins):
  download.py          # Raw HF download logic (used by HuggingFaceSource)
  quality_filter.py    # Raw filter logic (used by QualityFilter)
  preprocess.py        # Tokenization + chunking (unchanged)
  dataset.py           # PyTorch Dataset (unchanged)
  collator.py          # Batch collation (unchanged)
```

## Backwards Compatibility

The existing `prepare_data.py` CLI continues to work as-is. The new pipeline
system is an alternative way to configure data preparation, not a replacement.
Power users can use `pipeline.yaml`, beginners can use the simple CLI flags.

## Industry Comparison

| Feature | Cola-Coder (this plan) | datatrove (HF) | dolma (AI2) | RedPajama |
|---------|----------------------|-----------------|-------------|-----------|
| Plugin registry | Yes | Yes | Yes | No |
| YAML config | Yes | JSON | JSON | YAML |
| Streaming | Yes | Yes | Yes | No |
| Parallel filter | Yes (ProcessPool) | Yes (multiproc) | Yes (Ray) | Yes (Spark) |
| Mixed sources | Yes (weighted) | Yes | Yes | Yes |
| Dedup | MinHash | MinHash | MinHash+Bloom | MinHash |
| License filter | Yes | No | Yes | Yes |
| PII removal | Regex+optional | Regex | Regex+NER | Regex |
| Tree-sitter syntax | Yes | No | No | No |
| Data manifest | Yes (YAML) | No | Yes (JSON) | No |
