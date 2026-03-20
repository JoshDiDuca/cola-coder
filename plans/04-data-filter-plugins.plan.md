# Plan: Data Filter Plugin System

## Problem

The current quality filter is a monolithic file with hardcoded checks. To compete
with top models, we need:

- Easy to add new filters without touching existing code
- Per-language filter configurations
- Composable filter chains (run filter A, then B, then C)
- Community-contributed filters
- Filter benchmarking (measure impact of each filter on model quality)

## Architecture

### Filter Plugin Interface

Every filter implements `FilterPlugin` from the extensible pipeline plan:

```python
class FilterPlugin(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(self, record: DataRecord) -> tuple[bool, str]: ...

    def setup(self, config: dict) -> None: ...
```

### Built-in Filters (Priority Order)

#### Tier 1: Essential (ship first)

1. **QualityFilter** — Wrap existing conservative/strict checks
2. **LengthFilter** — Min/max lines, configurable per language
3. **DeduplicationFilter** — MinHash near-duplicate removal
4. **SyntaxFilter** — Tree-sitter full AST parse (replaces heuristic checks)

#### Tier 2: Important (ship second)

5. **LicenseFilter** — Permissive license enforcement
6. **PIIFilter** — Remove files with emails, API keys, secrets
7. **AutogenFilter** — Detect generated code (protobuf, swagger, etc.)
8. **DataFileFilter** — Reject JSON dumps, CSV, configs

#### Tier 3: Advanced (competitive edge)

9. **LLMJudgeFilter** — Use Claude/GPT to score code quality (expensive but powerful)
10. **ComplexityFilter** — Cyclomatic complexity, cognitive complexity
11. **TestCoverageFilter** — Prefer files from repos with good test coverage
12. **RecencyFilter** — Prefer recently updated files (fresher patterns)
13. **DependencyFilter** — Prefer files using modern dependencies

### Tree-Sitter Syntax Filter (Key Innovation)

The current parser is heuristic-based (regex for Python, brace counting for JS).
Tree-sitter gives us **real AST parsing** for 100+ languages with consistent API:

```python
@register_filter("syntax")
class SyntaxFilter(FilterPlugin):
    """Full AST validation using tree-sitter.

    Tree-sitter is a parser generator that builds concrete syntax trees.
    It's the same parser used by GitHub for code navigation, Neovim for
    syntax highlighting, and Zed editor for everything.

    Benefits over heuristic parsing:
    - Catches ALL syntax errors, not just obvious ones
    - Language-agnostic API (same code handles Python, TS, Go, Rust)
    - Can extract structural info (functions, classes, imports)
    - Fast: ~10ms per file (written in C)
    """

    LANGUAGE_MAP = {
        "typescript": "tree_sitter_typescript",
        "javascript": "tree_sitter_javascript",
        "python": "tree_sitter_python",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
        "java": "tree_sitter_java",
    }

    def __init__(self, languages: list[str], max_error_ratio: float = 0.05):
        """
        Args:
            languages: Languages to validate.
            max_error_ratio: Max fraction of ERROR nodes in AST.
                0.0 = perfect parse only
                0.05 = allow up to 5% error nodes (tolerant)
        """
        import tree_sitter
        self.parsers = {}
        for lang in languages:
            if lang in self.LANGUAGE_MAP:
                ts_lang = tree_sitter.Language(self.LANGUAGE_MAP[lang])
                parser = tree_sitter.Parser(ts_lang)
                self.parsers[lang] = parser

    def check(self, record: DataRecord) -> tuple[bool, str]:
        lang = record.metadata.get("language")
        if not lang or lang not in self.parsers:
            return True, ""  # Can't check, pass through

        parser = self.parsers[lang]
        tree = parser.parse(record.content.encode())

        # Count ERROR nodes
        error_count = self._count_errors(tree.root_node)
        total_nodes = self._count_nodes(tree.root_node)

        if total_nodes == 0:
            return False, "empty_ast"

        error_ratio = error_count / total_nodes
        if error_ratio > self.max_error_ratio:
            return False, f"syntax_errors ({error_count}/{total_nodes} = {error_ratio:.1%})"

        return True, ""
```

### MinHash Deduplication Filter

```python
@register_filter("dedup")
class DeduplicationFilter(FilterPlugin):
    """Near-duplicate removal using MinHash LSH.

    This is THE standard technique used by:
    - BigCode (StarCoder, The Stack)
    - AI2 (Dolma, OLMo)
    - Meta (LLaMA data)
    - RedPajama

    How it works:
    1. Tokenize file into character n-grams (5-grams by default)
    2. Compute MinHash signature (128 hash functions)
    3. Insert into LSH index (hash table of hash tables)
    4. Query: any file whose signature lands in the same bucket is a candidate duplicate
    5. If Jaccard similarity > threshold (0.8), reject as duplicate

    Memory: ~500 bytes per file. 10M files = ~5GB RAM.
    Speed: ~10,000 files/sec for insert+query.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 5,
    ):
        from datasketch import MinHash, MinHashLSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self._counter = 0

    def check(self, record: DataRecord) -> tuple[bool, str]:
        mh = self._compute_minhash(record.content)

        # Check for existing near-duplicates
        result = self.lsh.query(mh)
        if result:
            return False, f"near_duplicate (matches {len(result)} existing)"

        # Not a duplicate — add to index
        self._counter += 1
        self.lsh.insert(f"doc_{self._counter}", mh)
        return True, ""

    def _compute_minhash(self, content: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        # Character n-grams
        for i in range(len(content) - self.ngram_size + 1):
            ngram = content[i:i + self.ngram_size]
            mh.update(ngram.encode("utf-8"))
        return mh
```

### LLM-as-Judge Filter (Tier 3, High Impact)

```python
@register_filter("llm_judge")
class LLMJudgeFilter(FilterPlugin):
    """Use an LLM to score code quality.

    This is the phi-1/phi-2 "textbook quality" approach:
    - Feed code to an LLM with a scoring prompt
    - LLM rates: 1 (garbage) to 5 (textbook quality)
    - Keep only files scoring >= threshold (default 3)

    EXPENSIVE: ~$0.001-0.01 per file depending on length and model.
    For 10M files = $10K-100K. Use only on pre-filtered data.

    Strategy: Run cheap filters first (length, syntax, dedup) to reduce
    to ~1M files, then run LLM judge on the survivors.

    Can use:
    - Claude API (haiku for speed, sonnet for quality)
    - Local model (if you have one trained already — bootstrap!)
    - OpenAI API (gpt-4o-mini for cost)
    """

    SCORING_PROMPT = '''Rate this code file on a scale of 1-5:
1 = Garbage (random, broken, autogenerated noise)
2 = Poor (works but terrible style, no structure)
3 = Average (functional, some organization)
4 = Good (clean, well-structured, follows conventions)
5 = Excellent (textbook quality, educational, exemplary)

Respond with ONLY the number.

```{language}
{code}
```'''

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        threshold: int = 3,
        max_tokens_to_send: int = 2000,
        batch_size: int = 10,
    ):
        ...

    def check(self, record: DataRecord) -> tuple[bool, str]:
        score = self._score(record.content, record.metadata.get("language", ""))
        record.metadata["quality_score"] = score
        if score < self.threshold:
            return False, f"llm_score_{score}"
        return True, ""
```

### Filter Configuration YAML

```yaml
# configs/filters/typescript_strict.yaml
filters:
  - name: length
    min_lines: 10
    max_lines: 5000

  - name: syntax
    languages: ["typescript"]
    max_error_ratio: 0.02  # Very strict: <2% parse errors

  - name: quality
    mode: strict

  - name: dedup
    threshold: 0.8
    num_perm: 128

  - name: license
    allowed: ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"]

  - name: pii
    enabled: true

  # Optional expensive filter — only run on small curated datasets
  # - name: llm_judge
  #   model: claude-3-haiku-20240307
  #   threshold: 3
```

### Filter Benchmarking

```python
# scripts/benchmark_filters.py

def benchmark_filter(
    filter_plugin: FilterPlugin,
    test_data: list[DataRecord],
    labels: list[bool],  # True = good code, False = bad code (human-labeled)
) -> dict:
    """Evaluate a filter's precision and recall.

    Returns:
        {
            "precision": 0.95,  # Of files kept, what % are actually good?
            "recall": 0.82,     # Of good files, what % were kept?
            "f1": 0.88,
            "keep_rate": 0.63,
            "rejection_reasons": {...},
            "false_positives": [...],   # Good files incorrectly rejected
            "false_negatives": [...],   # Bad files incorrectly kept
        }
    """
```

### Dependencies

```
tree-sitter>=0.22.0        # Core parser
tree-sitter-typescript      # TS grammar
tree-sitter-javascript      # JS grammar
tree-sitter-python          # Python grammar
datasketch>=1.6.0           # MinHash LSH
presidio-analyzer>=2.2.0    # PII detection (optional, heavy)
```

## Implementation Priority

1. Wrap existing filters in FilterPlugin interface
2. Add tree-sitter SyntaxFilter (biggest quality improvement)
3. Add MinHash DeduplicationFilter (biggest data efficiency improvement)
4. Add LicenseFilter and PIIFilter
5. Add LLMJudgeFilter (for curated/small datasets)
6. Filter benchmarking tool
