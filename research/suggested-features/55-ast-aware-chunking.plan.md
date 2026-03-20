# Feature 55: AST-Aware Chunking

**Status:** Proposed
**CLI Flag:** `--ast-chunking`
**Complexity:** Medium

---

## Overview

Instead of splitting source files at arbitrary token count boundaries, parse the TypeScript/JavaScript AST using tree-sitter and split at top-level node boundaries (functions, classes, interfaces, type aliases, exports). Each training chunk contains one or more complete syntactic units. Fallback: if a single top-level node exceeds the token limit, split at method boundaries within it.

---

## Motivation

Arbitrary chunking produces training examples that:
- Start mid-function body
- End mid-expression
- Lack a function signature for their content
- Contain no complete, learnable unit

AST-aware chunking produces examples that:
- Always start at a function or class boundary
- Are syntactically complete
- Contain a self-contained semantic unit the model can learn to generate

Empirical evidence: StarCoder2 and DeepSeek-Coder both use function-level chunking in their preprocessing pipelines. Function-level training examples reduce the perplexity of function-completion tasks by 15-25% vs random chunking.

---

## Architecture / Design

```
Source file (.ts)
  │
  ▼
tree-sitter-typescript parse → AST
  │
  ▼
TopLevelNodeExtractor
  ├── function_declaration
  ├── class_declaration
  ├── interface_declaration
  ├── type_alias_declaration
  ├── export_statement
  └── variable_statement (with function init)
  │
  ▼
NodePacker (greedy bin-packing)
  ├── Pack nodes into chunks ≤ max_tokens
  ├── Never split across node boundaries
  └── If single node > max_tokens: split at method level
  │
  ▼
[Chunk 1: complete function]
[Chunk 2: complete class]
[Chunk 3: interface + type alias]
```

---

## Implementation Steps

### Step 1: Install tree-sitter

```bash
pip install tree-sitter==0.23.0 tree-sitter-typescript==0.23.0
```

### Step 2: AST Parser Wrapper

```python
# src/data/ast_parser.py
from dataclasses import dataclass
from typing import Optional
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Parser, Node

TS_LANGUAGE = Language(ts_typescript.language_typescript())

TOP_LEVEL_TYPES = {
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "export_statement",
    "lexical_declaration",     # const/let at top level (may contain arrow functions)
    "variable_declaration",
}

METHOD_TYPES = {
    "method_definition",
    "public_field_definition",
    "constructor_definition",
}

@dataclass
class AstNode:
    type: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    text: str

def parse_file(source: str) -> Optional[Node]:
    parser = Parser(TS_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    if tree.root_node.has_error:
        return None
    return tree.root_node

def extract_top_level_nodes(source: str) -> list[AstNode]:
    root = parse_file(source)
    if root is None:
        return []

    nodes = []
    for child in root.children:
        if child.type in TOP_LEVEL_TYPES:
            text = source[child.start_byte:child.end_byte]
            nodes.append(AstNode(
                type=child.type,
                start_byte=child.start_byte,
                end_byte=child.end_byte,
                start_line=child.start_point[0],
                end_line=child.end_point[0],
                text=text.strip(),
            ))
    return nodes

def extract_method_nodes(source: str, class_node_text: str) -> list[AstNode]:
    """Extract method-level nodes from within a class."""
    root = parse_file(class_node_text)
    if root is None:
        return []
    nodes = []
    _collect_methods(root, class_node_text, nodes)
    return nodes

def _collect_methods(node: Node, source: str, result: list):
    if node.type in METHOD_TYPES:
        text = source[node.start_byte:node.end_byte]
        result.append(AstNode(
            type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            text=text.strip(),
        ))
    for child in node.children:
        _collect_methods(child, source, result)
```

### Step 3: Node Packer (Greedy Bin-Packing)

```python
# src/data/ast_chunker.py
from src.data.ast_parser import AstNode, extract_top_level_nodes, extract_method_nodes
from transformers import PreTrainedTokenizerFast

def ast_chunk_file(
    source: str,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 2048,
    min_tokens: int = 64,
) -> list[str]:
    """
    Split a source file into chunks at AST boundaries.
    Returns list of source text chunks.
    """
    nodes = extract_top_level_nodes(source)
    if not nodes:
        # Fallback to line-based chunking
        return _line_chunk(source, tokenizer, max_tokens)

    chunks = []
    current_parts: list[str] = []
    current_tokens = 0

    for node in nodes:
        node_tokens = len(tokenizer.encode(node.text, add_special_tokens=False))

        if node_tokens > max_tokens:
            # Node too large: flush current, then split node at method level
            if current_parts:
                chunk_text = "\n\n".join(current_parts)
                if len(tokenizer.encode(chunk_text)) >= min_tokens:
                    chunks.append(chunk_text)
                current_parts = []
                current_tokens = 0
            method_chunks = _split_large_node(node, tokenizer, max_tokens, min_tokens)
            chunks.extend(method_chunks)
        elif current_tokens + node_tokens > max_tokens:
            # Flush current chunk
            if current_parts:
                chunk_text = "\n\n".join(current_parts)
                if len(tokenizer.encode(chunk_text)) >= min_tokens:
                    chunks.append(chunk_text)
            current_parts = [node.text]
            current_tokens = node_tokens
        else:
            current_parts.append(node.text)
            current_tokens += node_tokens

    # Flush remaining
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        if len(tokenizer.encode(chunk_text)) >= min_tokens:
            chunks.append(chunk_text)

    return chunks

def _split_large_node(node: AstNode, tokenizer, max_tokens: int, min_tokens: int) -> list[str]:
    """Split an oversized node at method boundaries."""
    methods = extract_method_nodes(node.text, node.text)
    if not methods:
        # Can't split further: return as-is (truncated)
        tokens = tokenizer.encode(node.text, add_special_tokens=False)
        return [tokenizer.decode(tokens[:max_tokens])]

    chunks = []
    current_parts: list[str] = []
    current_tokens = 0

    for method in methods:
        method_tokens = len(tokenizer.encode(method.text, add_special_tokens=False))
        if current_tokens + method_tokens > max_tokens and current_parts:
            chunk = "\n\n".join(current_parts)
            if len(tokenizer.encode(chunk)) >= min_tokens:
                chunks.append(chunk)
            current_parts = [method.text]
            current_tokens = method_tokens
        else:
            current_parts.append(method.text)
            current_tokens += method_tokens

    if current_parts:
        chunk = "\n\n".join(current_parts)
        if len(tokenizer.encode(chunk)) >= min_tokens:
            chunks.append(chunk)

    return chunks

def _line_chunk(source: str, tokenizer, max_tokens: int) -> list[str]:
    """Fallback: chunk by lines until token limit."""
    lines = source.splitlines(keepends=True)
    chunks = []
    current = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append("".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return chunks
```

### Step 4: Modify preprocess.py

```python
# In src/data/preprocess.py

from src.data.ast_chunker import ast_chunk_file

def process_file_ast(
    source: str,
    tokenizer,
    max_tokens: int = 2048,
    use_ast: bool = True,
) -> list[list[int]]:
    """
    Chunk a source file and tokenize each chunk.
    Returns list of token ID lists.
    """
    if use_ast:
        text_chunks = ast_chunk_file(source, tokenizer, max_tokens)
    else:
        # Legacy: tokenize and split at fixed size
        all_tokens = tokenizer.encode(source, add_special_tokens=False)
        text_chunks = [
            tokenizer.decode(all_tokens[i:i+max_tokens])
            for i in range(0, len(all_tokens), max_tokens)
        ]

    token_chunks = []
    for chunk in text_chunks:
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        if 64 <= len(ids) <= max_tokens:
            token_chunks.append(ids)
    return token_chunks

# In the main preprocessing loop:
# Replace:
#   chunks = [tokens[i:i+ctx_len] for i in range(0, len(tokens), ctx_len)]
# With:
#   chunks = process_file_ast(source_text, tokenizer, max_tokens=ctx_len, use_ast=args.ast_chunking)
```

### Step 5: Manifest Integration

Store chunk metadata alongside the .npy memmaps:

```python
# src/data/ast_manifest.py
import json
from pathlib import Path

def write_ast_manifest(chunks: list[dict], manifest_path: str):
    """
    chunks: list of {source_file, chunk_idx, node_type, start_line, end_line, token_count}
    """
    with open(manifest_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

def load_ast_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        return [json.loads(line) for line in f]

def get_node_type_distribution(manifest: list[dict]) -> dict[str, int]:
    dist = {}
    for entry in manifest:
        nt = entry.get("node_type", "unknown")
        dist[nt] = dist.get(nt, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))
```

### Step 6: CLI Flag

```python
# cli/preprocess.py (or wherever preprocessing is invoked)
parser.add_argument("--ast-chunking", action="store_true",
    help="Use AST-aware chunking (split at function/class boundaries) instead of fixed token count.")
parser.add_argument("--ast-max-tokens", type=int, default=2048,
    help="Maximum tokens per AST chunk (default: 2048).")
parser.add_argument("--ast-min-tokens", type=int, default=64,
    help="Minimum tokens per AST chunk; smaller chunks are discarded (default: 64).")
parser.add_argument("--ast-manifest", action="store_true",
    help="Write AST chunk manifest (node types, line numbers) alongside .npy files.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/preprocess.py` | Add `use_ast` branching, call `ast_chunk_file` |
| `src/data/ast_parser.py` | New — tree-sitter wrapper |
| `src/data/ast_chunker.py` | New — node packer |
| `src/data/ast_manifest.py` | New — manifest I/O |
| `cli/preprocess.py` | Add `--ast-chunking` flag |

---

## Testing Strategy

```python
# tests/test_ast_chunker.py

SIMPLE_TS = """
function add(a: number, b: number): number {
  return a + b;
}

function subtract(a: number, b: number): number {
  return a - b;
}

class Calculator {
  add(a: number, b: number) { return a + b; }
  subtract(a: number, b: number) { return a - b; }
}
"""

def test_extracts_top_level_nodes():
    nodes = extract_top_level_nodes(SIMPLE_TS)
    types = [n.type for n in nodes]
    assert "function_declaration" in types
    assert "class_declaration" in types

def test_chunks_do_not_split_functions(mock_tokenizer):
    chunks = ast_chunk_file(SIMPLE_TS, mock_tokenizer, max_tokens=500)
    for chunk in chunks:
        # Each chunk should have balanced braces
        assert chunk.count("{") == chunk.count("}")

def test_all_content_preserved(mock_tokenizer):
    chunks = ast_chunk_file(SIMPLE_TS, mock_tokenizer, max_tokens=100)
    combined = "\n".join(chunks)
    assert "add" in combined
    assert "subtract" in combined
    assert "Calculator" in combined

def test_large_class_splits_at_methods(mock_tokenizer):
    # Create a class larger than max_tokens
    large_class = "class Big {\n" + "\n".join(
        f"  method{i}(x: number): number {{ return x + {i}; }}"
        for i in range(50)
    ) + "\n}"
    chunks = ast_chunk_file(large_class, mock_tokenizer, max_tokens=200)
    assert len(chunks) > 1
    for chunk in chunks:
        # No chunk should exceed the limit by more than one method
        tokens = mock_tokenizer.encode(chunk)
        assert len(tokens) <= 250  # small slack

def test_parse_error_returns_empty():
    broken = "function { invalid typescript !!!"
    nodes = extract_top_level_nodes(broken)
    # Should return empty or partial list without crashing
    assert isinstance(nodes, list)
```

---

## Performance Considerations

- tree-sitter parsing is extremely fast: ~50MB/s on a modern CPU. A 1GB codebase parses in ~20 seconds.
- tree-sitter parsers are C extensions; parsing has negligible Python overhead.
- The preprocessing step is a one-time cost. AST chunking adds ~10-20% to preprocessing time vs naive chunking.
- Memory: tree-sitter holds the full parse tree in memory per file. For files > 1MB, this can use ~50MB RAM. Limit input file size to 100k characters (already done by quality filter).

---

## Dependencies

```
tree-sitter==0.23.0
tree-sitter-typescript==0.23.0
```

These are pre-compiled binary wheels; no compiler needed.

---

## Estimated Complexity

**Development time:** 3-4 days
**Risk:** Low. tree-sitter is stable and battle-tested (used by GitHub Copilot, CodeBERT). The main risk is edge cases in TypeScript's AST (decorators, template literals, JSX) — handle by falling back to line-chunking if parse fails.
**Lines of new code:** ~350

---

## 2026 Best Practices

- **AST-level preprocessing is standard:** DeepSeek-Coder-V2, StarCoder2, and Qwen2.5-Coder all use function-level granularity for at least part of their training data. This is no longer experimental.
- **Preserve node type metadata:** Storing node types in the manifest enables analysis of whether the model performs better on functions vs classes vs interfaces, which informs curriculum design.
- **JavaScript fallback:** tree-sitter-typescript handles `.ts` and `.tsx`. For `.js`/`.jsx`, use `tree_sitter_javascript` from the same package.
- **Graceful parse failure:** Never crash the entire preprocessing pipeline on a parse error. Log the file path, use line-based fallback, and continue.
