# 61 - Multi-File Context Training

## Overview

Train Cola-Coder on pairs of related TypeScript/JavaScript files simultaneously, teaching the model to understand cross-file relationships. Instead of treating each file as an isolated training sample, pair files that share a semantic relationship and train the model to understand the connection between them.

**Feature flag:** `--enable-multi-file-context` / `config.multi_file_context.enabled`

---

## Motivation

Real-world TypeScript code is never written in isolation. A component always has a test file. An API route always has associated type definitions. A module always has an index that re-exports it. Training on single files teaches the model syntax and patterns, but training on file pairs teaches it:

- How types defined in one file are consumed in another
- How a component's props shape its test structure
- How an interface constrains its implementation
- How barrel exports reflect module organization

This is a significant capability gap between current Cola-Coder generations and production-quality completions. A model that has seen `UserService.ts` alongside `UserService.test.ts` will generate tests that actually match the service's API surface.

**Expected improvement:** Better coherence in generated code that references external types, more realistic mock/stub patterns in tests, correct import paths.

---

## Architecture / Design

### File Pair Types

```
PairType = Literal[
    "component_test",      # Button.tsx + Button.test.tsx
    "api_route_types",     # users.ts (route) + users.types.ts
    "module_index",        # utils/string.ts + utils/index.ts
    "interface_impl",      # IRepository.ts + UserRepository.ts
    "schema_resolver",     # schema.graphql + resolvers.ts (bonus)
    "model_migration",     # User.model.ts + 20240101_add_user.ts
]
```

### Training Format

```
[FILE_1_PATH]
{file1_content}
[FILE_SEPARATOR]
[FILE_2_PATH]
{file2_content}
[END]
```

Token layout:
```
<bos> [FILE1: src/components/Button.tsx]
... button source tokens ...
[SEP]
[FILE2: src/components/Button.test.tsx]
... test source tokens ...
<eos>
```

The separator token `[SEP]` and path tokens `[FILE1:]` / `[FILE2:]` are added to the tokenizer's special tokens vocabulary.

### Context Window Requirements

Single-file training uses `max_seq_len = 1024` (or config value). Multi-file pairs require `2x` that length plus separator overhead (~20 tokens). The system automatically doubles the context window when multi-file context is enabled, or uses a separately configured value:

```yaml
multi_file_context:
  enabled: true
  max_seq_len: 2048          # overrides global max_seq_len for this dataset
  separator_token: "[SEP]"
  include_file_paths: true   # prepend [FILE1: path] headers
  pair_types:
    - component_test
    - api_route_types
    - module_index
    - interface_impl
  min_pair_tokens: 100       # skip pairs where either file is too small
  max_pair_tokens: 1900      # skip pairs that overflow even 2x context
  ast_verification: true     # use AST analysis to confirm actual relationship
```

---

## Implementation Steps

### Step 1: File Pair Discovery (`data/pair_finder.py`)

```python
import ast
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tree_sitter_typescript as ts_ts
from tree_sitter import Language, Parser

@dataclass
class FilePair:
    file1: Path
    file2: Path
    pair_type: str
    confidence: float          # 0.0-1.0 from heuristics
    verified_by_ast: bool = False

class FilePairFinder:
    def __init__(self, repo_root: Path, config: dict):
        self.root = repo_root
        self.config = config
        self._build_import_graph()

    def _build_import_graph(self):
        """Parse all TS/JS files and extract import relationships."""
        self.import_graph: dict[Path, set[Path]] = {}
        for f in self.root.rglob("*.ts"):
            imports = self._extract_imports(f)
            self.import_graph[f] = imports

    def _extract_imports(self, file: Path) -> set[Path]:
        """Extract resolved import paths from a TypeScript file."""
        imports = set()
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            # Regex for static imports: import ... from './path'
            pattern = r'''from\s+['"]([^'"]+)['"]'''
            for match in re.finditer(pattern, content):
                raw = match.group(1)
                if raw.startswith("."):
                    resolved = (file.parent / raw).resolve()
                    # Try .ts, .tsx extensions
                    for ext in [".ts", ".tsx", ".js", ""]:
                        candidate = resolved.with_suffix(ext) if ext else resolved
                        if candidate.exists():
                            imports.add(candidate)
                            break
        except Exception:
            pass
        return imports

    def find_component_test_pairs(self) -> list[FilePair]:
        pairs = []
        for f in self.root.rglob("*.ts"):
            if f.suffix not in (".ts", ".tsx"):
                continue
            # Skip test files themselves
            if ".test." in f.name or ".spec." in f.name:
                continue
            # Look for test counterpart
            for test_suffix in [".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"]:
                test_file = f.with_name(f.stem + test_suffix)
                if test_file.exists():
                    pairs.append(FilePair(
                        file1=f,
                        file2=test_file,
                        pair_type="component_test",
                        confidence=0.95
                    ))
        return pairs

    def find_api_route_type_pairs(self) -> list[FilePair]:
        """Find API route files paired with their type definition files."""
        pairs = []
        for f in self.root.rglob("*.ts"):
            stem = f.stem
            # Common patterns: users.ts + users.types.ts, users.ts + types/users.ts
            for type_pattern in [
                f.with_name(stem + ".types.ts"),
                f.parent / "types" / f.name,
                f.parent / "types" / (stem + ".ts"),
            ]:
                if type_pattern.exists() and type_pattern != f:
                    pairs.append(FilePair(
                        file1=f,
                        file2=type_pattern,
                        pair_type="api_route_types",
                        confidence=0.8
                    ))
        return pairs

    def find_module_index_pairs(self) -> list[FilePair]:
        """Match modules to their barrel index.ts."""
        pairs = []
        for index_file in self.root.rglob("index.ts"):
            parent = index_file.parent
            # Find siblings that are imported by index.ts
            imported = self.import_graph.get(index_file, set())
            for imp in imported:
                if imp.parent == parent and imp != index_file:
                    pairs.append(FilePair(
                        file1=imp,
                        file2=index_file,
                        pair_type="module_index",
                        confidence=0.9
                    ))
        return pairs

    def find_interface_impl_pairs(self) -> list[FilePair]:
        """Find interface files and their implementations using AST."""
        pairs = []
        interface_files = []

        for f in self.root.rglob("*.ts"):
            content = f.read_text(encoding="utf-8", errors="ignore")
            # Files starting with 'I' that export interfaces
            if re.search(r'export\s+interface\s+\w+', content):
                interface_files.append(f)

        for iface_file in interface_files:
            iface_name = iface_file.stem  # e.g., IUserRepository
            # Look for implementing class
            impl_name = iface_name.lstrip("I")  # UserRepository
            for candidate in self.root.rglob(f"{impl_name}.ts"):
                content = candidate.read_text(encoding="utf-8", errors="ignore")
                if re.search(rf'implements\s+{re.escape(iface_name)}', content):
                    pairs.append(FilePair(
                        file1=iface_file,
                        file2=candidate,
                        pair_type="interface_impl",
                        confidence=0.85,
                    ))
        return pairs

    def verify_with_ast(self, pair: FilePair) -> FilePair:
        """Use tree-sitter to confirm the relationship is real."""
        # Check that file2 actually imports from or references file1
        if pair.file2 in self.import_graph:
            if pair.file1 in self.import_graph[pair.file2]:
                pair.verified_by_ast = True
            # Also check reverse direction
        if pair.file1 in self.import_graph:
            if pair.file2 in self.import_graph[pair.file1]:
                pair.verified_by_ast = True
        return pair

    def find_all_pairs(self) -> list[FilePair]:
        all_pairs: list[FilePair] = []
        finders = [
            self.find_component_test_pairs,
            self.find_api_route_type_pairs,
            self.find_module_index_pairs,
            self.find_interface_impl_pairs,
        ]
        for finder in finders:
            pairs = finder()
            if self.config.get("ast_verification", True):
                pairs = [self.verify_with_ast(p) for p in pairs]
            all_pairs.extend(pairs)
        return all_pairs
```

### Step 2: Pair Dataset Builder (`data/multi_file_dataset.py`)

```python
class MultiFileDataset:
    def __init__(self, pairs: list[FilePair], tokenizer, config: dict):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.sep_token_id = tokenizer.encode("[SEP]")[0]

    def encode_pair(self, pair: FilePair) -> Optional[list[int]]:
        content1 = pair.file1.read_text(encoding="utf-8", errors="ignore")
        content2 = pair.file2.read_text(encoding="utf-8", errors="ignore")

        header1 = f"[FILE1: {pair.file1.name}]\n"
        header2 = f"[FILE2: {pair.file2.name}]\n"

        tokens1 = self.tokenizer.encode(header1 + content1)
        tokens2 = self.tokenizer.encode(header2 + content2)

        combined = tokens1 + [self.sep_token_id] + tokens2

        if len(combined) > self.max_seq_len:
            return None  # Skip oversized pairs

        # Pad or truncate to max_seq_len
        combined = combined[:self.max_seq_len]
        return combined
```

### Step 3: Config Integration (`config/training.yaml`)

```yaml
data:
  multi_file_context:
    enabled: false          # opt-in
    repo_scan_paths:
      - ./training_repos
    pair_types:
      - component_test
      - module_index
    ast_verification: true
    mix_ratio: 0.3          # 30% of batches use paired samples
    max_seq_len: 2048
```

### Step 4: Trainer Integration

In `training/trainer.py`, when building the DataLoader, mix single-file and multi-file samples according to `mix_ratio`. The paired samples use the extended context window.

---

## Key Files to Modify

- `data/prepare.py` - Add `--multi-file` flag, invoke `FilePairFinder`
- `data/pair_finder.py` - New file: pair discovery logic
- `data/multi_file_dataset.py` - New file: pair encoding
- `tokenizer/special_tokens.py` - Add `[SEP]`, `[FILE1:]`, `[FILE2:]`
- `config/training.yaml` - Add `multi_file_context` section
- `training/trainer.py` - Mixed DataLoader construction
- `cli/prepare_cmd.py` - Expose `--enable-multi-file-context` flag

---

## Testing Strategy

1. **Unit tests** for `FilePairFinder`: run against a small fixture repo with known file pairs, assert all expected pairs found and no false positives.
2. **AST verification test**: create a pair where file2 imports file1, verify `verified_by_ast=True`; create a pair that merely matches by name but has no import, verify `verified_by_ast=False`.
3. **Token length test**: encode a known oversized pair, assert it returns `None`.
4. **Integration test**: run a 10-step training loop on a batch of paired samples, assert loss decreases.
5. **Regression test**: ensure `mix_ratio=0.0` produces identical results to single-file training.

---

## Performance Considerations

- `_build_import_graph()` is O(N * avg_imports) where N is file count. For repos with 10k files this takes ~30s. Cache the graph to disk as `pair_cache.json`.
- Tree-sitter parsing is fast (~1ms/file) but still runs N times. Parallelize with `concurrent.futures.ThreadPoolExecutor`.
- `max_seq_len=2048` doubles VRAM usage per sample compared to 1024. Reduce batch size accordingly (halve it). On RTX 3080 (10GB) with 1024 context, typical batch size is 8; with 2048 drop to 4.
- Consider gradient checkpointing when enabling 2x context.

---

## Dependencies

```
tree-sitter>=0.21.0
tree-sitter-typescript>=0.21.0
```

No new heavy dependencies. Import graph building only uses stdlib `re` and `pathlib`.

---

## Estimated Complexity

**Medium-High.** The pair discovery logic is straightforward but the integration with the training loop (mixed batching, dynamic sequence lengths) requires care. AST verification adds robustness at the cost of a tree-sitter dependency. Estimated implementation time: 3-5 days.

---

## 2026 Best Practices

- **AST-first relationship verification**: Never rely solely on naming conventions; naming is a hint, imports are the ground truth. Tree-sitter's incremental parsing makes this practical at scale.
- **Configurable mix ratio**: Don't force all batches to be pairs. A 30% mix is a reasonable default; too high can destabilize training if pair quality is inconsistent.
- **Content-addressed pair cache**: Hash file contents, not paths, when caching discovered pairs. Paths change across machines; content hashes are portable and enable dataset reproducibility.
- **Graceful degradation**: If a file in a pair is deleted or modified between discovery and training, log a warning and fall back to single-file training for that sample rather than crashing.
- **Pair type weighting**: Down-weight pair types with lower AST verification rates (`api_route_types` tends to have more false positives than `component_test`). Expose weights in config.
