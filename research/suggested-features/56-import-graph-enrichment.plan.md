# Feature 56: Import Graph Enrichment

**Status:** Proposed
**CLI Flag:** `--import-enrichment`
**Complexity:** Medium-High

---

## Overview

For each source file in the training set, parse its imports and prepend the types/interfaces from imported local modules as context. This teaches the model to generate code that correctly uses the APIs defined in imported files. A separator token (`<|imports|>`) marks the boundary between imported context and the main file. To control context explosion, only the imported *symbols* (type signatures, interfaces, function signatures) are included, not full file bodies.

---

## Motivation

A major failure mode of code generation models is incorrect API usage: calling functions with wrong parameter types, misusing interface fields, or using methods that don't exist. This happens because the model only sees the current file, not its imports.

Import graph enrichment addresses this by including a "type stub" of imported modules as context. This is similar to how TypeScript's language server works — it loads `.d.ts` type definitions to provide autocomplete and type checking.

Expected benefits:
- Fewer type errors on code using local modules
- Better function call syntax (correct argument count and types)
- Improved interface implementation completeness

---

## Architecture / Design

```
File: src/services/userService.ts
  import { User, UserRole } from './models/user';
  import { hashPassword } from './utils/crypto';

ImportResolver
  ├── resolve('./models/user') → src/models/user.ts
  │     extract: interface User { id: string; name: string; role: UserRole }
  │              type UserRole = 'admin' | 'user' | 'guest'
  └── resolve('./utils/crypto') → src/utils/crypto.ts
        extract: function hashPassword(password: string): Promise<string>

Enriched training example:
  <|imports|>
  interface User { id: string; name: string; role: UserRole }
  type UserRole = 'admin' | 'user' | 'guest'
  function hashPassword(password: string): Promise<string>
  <|main|>
  [content of userService.ts]
```

---

## Implementation Steps

### Step 1: Special Separator Tokens

```python
# src/tokenizer/extend_tokenizer.py (extend existing)
IMPORT_ENRICHMENT_TOKENS = ["<|imports|>", "<|main|>"]

def add_import_tokens(tokenizer):
    existing = set(tokenizer.additional_special_tokens)
    new = [t for t in IMPORT_ENRICHMENT_TOKENS if t not in existing]
    if new:
        tokenizer.add_special_tokens({"additional_special_tokens": new})
    return tokenizer
```

### Step 2: Import Parser

```python
# src/data/import_parser.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ImportSpec:
    specifiers: list[str]    # ["User", "UserRole"]
    source: str              # "./models/user"
    is_local: bool           # True if starts with ./ or ../
    resolved_path: Optional[str] = None

IMPORT_PATTERN = re.compile(
    r"import\s+(?:type\s+)?\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"
    r"|import\s+(?:type\s+)?(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)

def parse_imports(source: str) -> list[ImportSpec]:
    imports = []
    for match in IMPORT_PATTERN.finditer(source):
        if match.group(1):
            # Named imports: { User, UserRole }
            specifiers = [s.strip().split(" as ")[0].strip()
                          for s in match.group(1).split(",")
                          if s.strip()]
            source_path = match.group(2)
        else:
            # Default import
            specifiers = [match.group(3)]
            source_path = match.group(4)

        is_local = source_path.startswith(".") or source_path.startswith("/")
        imports.append(ImportSpec(
            specifiers=specifiers,
            source=source_path,
            is_local=is_local,
        ))
    return imports

def resolve_import(
    import_spec: ImportSpec,
    current_file: str,
    project_root: str,
) -> Optional[str]:
    if not import_spec.is_local:
        return None  # Package import; use type stubs instead

    current_dir = Path(current_file).parent
    base = (current_dir / import_spec.source).resolve()

    for ext in [".ts", ".tsx", "/index.ts", "/index.tsx"]:
        candidate = Path(str(base) + ext) if not ext.startswith("/") else Path(str(base) + ext)
        if candidate.exists():
            return str(candidate)

    return None
```

### Step 3: Symbol Extractor

```python
# src/data/symbol_extractor.py
import re
from src.data.import_parser import ImportSpec

EXPORT_PATTERNS = [
    # interface
    (re.compile(r"export\s+(?:default\s+)?interface\s+(\w+)\s*(?:extends[^{]*)?\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", re.DOTALL), "interface"),
    # type alias
    (re.compile(r"export\s+type\s+(\w+)\s*=\s*([^;]+);", re.DOTALL), "type"),
    # function signature (no body)
    (re.compile(r"export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{;]+))?(?=\s*[\{;])", re.DOTALL), "function"),
    # class
    (re.compile(r"export\s+(?:abstract\s+)?class\s+(\w+)(?:[^{]*)?\{", re.DOTALL), "class"),
    # const with type
    (re.compile(r"export\s+const\s+(\w+)\s*:\s*([^=]+)=", re.DOTALL), "const"),
    # enum
    (re.compile(r"export\s+(?:const\s+)?enum\s+(\w+)\s*\{([^}]*)\}", re.DOTALL), "enum"),
]

def extract_symbols(source: str, requested_specifiers: list[str]) -> list[str]:
    """Extract type/interface/function signatures for requested names."""
    symbols = []
    requested = set(requested_specifiers)

    for pattern, kind in EXPORT_PATTERNS:
        for match in pattern.finditer(source):
            name = match.group(1)
            if name in requested:
                if kind == "interface":
                    # Include full interface body (limited)
                    body = match.group(2)[:500]  # cap at 500 chars
                    symbols.append(f"interface {name} {{{body}}}")
                elif kind == "type":
                    val = match.group(2).strip()[:200]
                    symbols.append(f"type {name} = {val};")
                elif kind == "function":
                    params = match.group(2)
                    ret = match.group(3).strip() if match.group(3) else "void"
                    symbols.append(f"function {name}({params}): {ret};")
                elif kind == "class":
                    symbols.append(f"class {name} {{ /* ... */ }}")
                elif kind == "const":
                    type_ann = match.group(2).strip()[:100]
                    symbols.append(f"const {name}: {type_ann};")
                elif kind == "enum":
                    body = match.group(2)[:200]
                    symbols.append(f"enum {name} {{{body}}}")

    return symbols
```

### Step 4: Import Context Builder

```python
# src/data/import_context_builder.py
from src.data.import_parser import parse_imports, resolve_import
from src.data.symbol_extractor import extract_symbols
from pathlib import Path

MAX_IMPORT_CONTEXT_TOKENS = 512  # hard cap on imported context

def build_import_context(
    source: str,
    source_file: str,
    project_root: str,
    tokenizer,
    max_context_tokens: int = MAX_IMPORT_CONTEXT_TOKENS,
) -> str:
    """
    Parse imports in `source`, resolve local ones, extract symbols,
    and return a context string to prepend.
    """
    imports = parse_imports(source)
    local_imports = [imp for imp in imports if imp.is_local]

    if not local_imports:
        return ""

    context_parts = []
    used_tokens = 0

    for imp in local_imports:
        resolved = resolve_import(imp, source_file, project_root)
        if not resolved or not Path(resolved).exists():
            continue

        try:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                imported_source = f.read()
        except OSError:
            continue

        symbols = extract_symbols(imported_source, imp.specifiers)
        for sym in symbols:
            sym_tokens = len(tokenizer.encode(sym, add_special_tokens=False))
            if used_tokens + sym_tokens > max_context_tokens:
                break
            context_parts.append(sym)
            used_tokens += sym_tokens

        if used_tokens >= max_context_tokens:
            break

    if not context_parts:
        return ""

    return "// Imported types:\n" + "\n".join(context_parts)

def format_enriched_example(
    import_context: str,
    main_source: str,
    imports_token: str = "<|imports|>",
    main_token: str = "<|main|>",
) -> str:
    if not import_context:
        return main_source
    return f"{imports_token}\n{import_context}\n{main_token}\n{main_source}"
```

### Step 5: Integrate into Preprocessing

```python
# In src/data/preprocess.py

from src.data.import_context_builder import build_import_context, format_enriched_example

def process_file_with_imports(
    source: str,
    source_file: str,
    project_root: str,
    tokenizer,
    max_tokens: int = 2048,
    import_context_tokens: int = 512,
    use_import_enrichment: bool = True,
) -> list[list[int]]:
    if use_import_enrichment:
        context = build_import_context(
            source, source_file, project_root, tokenizer, import_context_tokens
        )
        enriched = format_enriched_example(context, source)
    else:
        enriched = source

    all_tokens = tokenizer.encode(enriched, add_special_tokens=False)
    chunks = [
        all_tokens[i:i+max_tokens]
        for i in range(0, len(all_tokens), max_tokens)
        if len(all_tokens[i:i+max_tokens]) >= 64
    ]
    return chunks
```

### Step 6: CLI Integration

```python
# cli/preprocess.py
parser.add_argument("--import-enrichment", action="store_true",
    help="Prepend imported type context to each training file.")
parser.add_argument("--import-context-tokens", type=int, default=512,
    help="Max tokens to use for import context (default: 512).")
parser.add_argument("--project-root", type=str, default=".",
    help="Root directory for resolving relative imports.")
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/import_parser.py` | New |
| `src/data/symbol_extractor.py` | New |
| `src/data/import_context_builder.py` | New |
| `src/data/preprocess.py` | Call import context builder when flag enabled |
| `src/tokenizer/extend_tokenizer.py` | Add `<|imports|>`, `<|main|>` tokens |
| `cli/preprocess.py` | Add CLI flags |

---

## Testing Strategy

```python
# tests/test_import_enrichment.py

MODELS_TS = '''
export interface User {
  id: string;
  name: string;
  role: UserRole;
}
export type UserRole = 'admin' | 'user';
export function hashPassword(password: string): Promise<string>;
'''

SERVICE_TS = '''
import { User, UserRole } from './models/user';
import { hashPassword } from './utils/crypto';

export async function createUser(name: string): Promise<User> {
  return { id: "1", name, role: "user" };
}
'''

def test_parse_imports():
    imports = parse_imports(SERVICE_TS)
    assert len(imports) == 2
    assert imports[0].specifiers == ["User", "UserRole"]
    assert imports[0].source == "./models/user"
    assert imports[0].is_local is True

def test_extract_symbols():
    symbols = extract_symbols(MODELS_TS, ["User", "UserRole"])
    assert any("User" in s and "id" in s for s in symbols)
    assert any("UserRole" in s and "admin" in s for s in symbols)

def test_package_imports_skipped():
    src = "import { useState } from 'react';\nconst x = 1;"
    imports = parse_imports(src)
    local = [i for i in imports if i.is_local]
    assert len(local) == 0

def test_context_token_cap(mock_tokenizer, tmp_path):
    # Write a large models.ts
    models_file = tmp_path / "models.ts"
    models_file.write_text(MODELS_TS)
    context = build_import_context(
        SERVICE_TS.replace("./models/user", str(models_file)),
        str(tmp_path / "service.ts"),
        str(tmp_path),
        mock_tokenizer,
        max_context_tokens=50,
    )
    tokens = mock_tokenizer.encode(context)
    assert len(tokens) <= 55  # small slack
```

---

## Performance Considerations

- Import resolution requires file I/O: ~1ms per import per file. For a 100k file dataset with average 5 imports each, total I/O = ~500s. Mitigate by caching resolved symbol tables per unique file path:

```python
from functools import lru_cache

@lru_cache(maxsize=4096)
def _load_and_extract(resolved_path: str, specifiers_key: str) -> list[str]:
    specifiers = specifiers_key.split(",")
    with open(resolved_path, "r", errors="ignore") as f:
        src = f.read()
    return extract_symbols(src, specifiers)
```

- The context cap of 512 tokens ensures that import enrichment adds at most 25% overhead to a 2048-token context window.
- During inference, import enrichment requires the user to provide the project root. For standalone generation, disable it.

---

## Dependencies

No new pip dependencies — uses standard library (`re`, `pathlib`).

---

## Estimated Complexity

**Development time:** 4-5 days
**Risk:** Medium. Import resolution has many edge cases (path aliases, barrel exports, re-exports). Start with simple relative imports and expand incrementally.
**Lines of new code:** ~450

---

## 2026 Best Practices

- **Repository-level context is standard:** GitHub Copilot, Cursor, and Codeium all use repository-level context (imports, related files) as standard features. Training with import context aligns the model with how it will be used.
- **Type stubs over full files:** Including full imported files would create context lengths of 10k+ tokens. Type stubs (function signatures + interfaces only) give 90% of the benefit at 5% of the cost.
- **Separator tokens:** Using dedicated `<|imports|>` and `<|main|>` tokens is cleaner than natural language delimiters. The model can learn to use the import context section without confusing it with the main file.
- **Evaluate type error rate:** The primary metric for this feature is TypeScript type error rate on generated code that uses imported types. Measure before/after on a held-out set.
