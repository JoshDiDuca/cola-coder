"""Repository context scanner and retriever.

Scans a TypeScript/JavaScript project and builds structured context that the
model can use to generate code that fits seamlessly into the existing codebase.

The key insight: a model that knows about your imports, types, and similar files
will generate code that actually compiles — right types, right function signatures,
right import paths — rather than plausible-looking code that breaks at runtime.

For a TS dev: this is your IDE's language server, but distilled into a compact
context string that gets prepended to the generation prompt.

Pipeline:
  1. RepoScanner.scan() — run once at session start (reads package.json, tsconfig,
     parses imports for all .ts/.tsx/.js/.jsx files)
  2. RepoScanner.get_context_for_file() — call per completion request; assembles
     a <|repo|>...</|repo|> block trimmed to a token budget
  3. ContextAwareGenerator wraps CodeGenerator and handles step 2 automatically
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ── Skip lists ────────────────────────────────────────────────────────────────

_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", "dist", "build", ".next",
    ".nuxt", "out", "coverage", ".turbo", ".cache", ".venv", "venv",
    ".svelte-kit", "storybook-static",
})

_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs",
})

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ImportRef:
    """A single import statement parsed from a source file.

    For a TS dev: equivalent to a resolved entry in your tsconfig paths map,
    but also covering bare package imports like 'react' or 'zod'.
    """

    names: list[str]       # imported names e.g. ["User", "UserRole"]
    source: str            # raw import specifier e.g. "./types/user" or "react"
    is_relative: bool      # True for ./path or ../path, False for packages


@dataclass
class RepoContext:
    """Immutable snapshot of a repo produced by RepoScanner.scan().

    Cheap to pass around; all expensive I/O happens once during scan().
    """

    root: Path
    file_tree: list[str]                          # compact relative paths
    package_info: dict                            # from package.json
    tsconfig: dict | None                         # from tsconfig.json, or None
    import_graph: dict[str, list[ImportRef]]      # abs_path -> its imports
    framework_versions: dict[str, str]            # e.g. {"react": "18.2.0"}


# ── Regex helpers ─────────────────────────────────────────────────────────────

# Named-import block:  { X, Y as Z }
_RE_NAMED = re.compile(r"\{([^}]*)\}")
# Default import:      import X from ...  (captured as first word before 'from')
_RE_DEFAULT = re.compile(r"^import\s+(\w+)\s+from")
# Namespace import:    import * as X from ...
_RE_NAMESPACE = re.compile(r"import\s+\*\s+as\s+(\w+)\s+from")

# Full import patterns — listed from most specific to most general so the
# first match wins without ambiguity.
_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    # import type { X, Y } from 'path'
    re.compile(r'import\s+type\s+\{([^}]*)\}\s+from\s+["\']([^"\']+)["\']'),
    # import { X, Y } from 'path'
    re.compile(r'import\s+\{([^}]*)\}\s+from\s+["\']([^"\']+)["\']'),
    # import * as X from 'path'
    re.compile(r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']'),
    # import X from 'path'  (default import, no braces)
    re.compile(r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']'),
    # import 'path'  (side-effect)
    re.compile(r'import\s+["\']([^"\']+)["\']'),
    # const X = require('path')
    re.compile(r'(?:const|let|var)\s+(?:\{[^}]*\}|\w+)\s*=\s*require\s*\(\s*["\']([^"\']+)["\']\s*\)'),
]

# Export patterns for signature extraction
_EXPORT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # export interface Foo { ... }  — capture up to the first closing brace
    ("interface", re.compile(
        r'export\s+(?:default\s+)?interface\s+\w[\w<, \[\]]*?\s*\{[^}]*\}',
        re.DOTALL,
    )),
    # export type Foo = Bar | Baz;
    ("type", re.compile(
        r'export\s+type\s+\w+\s*(?:<[^>]*>)?\s*=\s*[^;]+;',
        re.DOTALL,
    )),
    # export enum Foo { ... }
    ("enum", re.compile(
        r'export\s+(?:const\s+)?enum\s+\w+\s*\{[^}]*\}',
        re.DOTALL,
    )),
    # export function foo(args): ReturnType — signature only (first line / up to '{')
    ("function", re.compile(
        r'export\s+(?:async\s+)?function\s+\w+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[\w<>\[\], |&]+)?',
    )),
    # export const foo = ...   (one line)
    ("const", re.compile(
        r'export\s+const\s+\w+\s*(?::\s*[\w<>\[\], |&]+)?\s*=\s*[^\n;]+',
    )),
    # export default function/class Foo
    ("default", re.compile(
        r'export\s+default\s+(?:async\s+)?(?:function|class)\s+\w+\s*(?:<[^>]*>)?\s*\([^)]*\)?',
    )),
]


# ── Core parsing functions ────────────────────────────────────────────────────


def parse_imports(content: str) -> list[ImportRef]:
    """Extract import statements from TS/JS source code (regex-based, no AST).

    Handles:
    - import { X, Y } from './path'
    - import X from 'package'
    - import * as X from './path'
    - import type { X } from './path'
    - const X = require('package')
    - import 'side-effect'

    Args:
        content: Raw source code text.

    Returns:
        List of ImportRef objects, one per import statement found.
    """
    refs: list[ImportRef] = []
    seen: set[tuple[str, str]] = set()  # (names_key, source) dedup

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        for pattern in _IMPORT_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue

            groups = m.groups()

            # Pattern: import 'side-effect'  or  require('pkg')  — 1 group
            if len(groups) == 1:
                source = groups[0]
                names: list[str] = []
            else:
                # Groups are (names_or_identifier, source)
                raw_names, source = groups[0], groups[1]
                # Named imports come in as "X, Y as Z" — strip aliases
                if "," in raw_names or raw_names.strip().startswith("{"):
                    names = [
                        part.split(" as ")[0].strip()
                        for part in raw_names.split(",")
                        if part.strip()
                    ]
                else:
                    names = [raw_names.strip()] if raw_names.strip() else []

            is_relative = source.startswith(".")
            key = (",".join(sorted(names)), source)
            if key not in seen:
                seen.add(key)
                refs.append(ImportRef(names=names, source=source, is_relative=is_relative))
            break  # first matching pattern wins for this line

    return refs


def extract_exports(content: str) -> str:
    """Extract exported type/interface/function signatures from TS/JS source.

    Returns only the signatures — not full implementations — so we stay within
    the token budget while still giving the model complete type information.

    Args:
        content: Raw source code text.

    Returns:
        Newline-separated export signatures, each ending with either ';' or '{ ... }'.
    """
    signatures: list[str] = []
    seen_sigs: set[str] = set()

    for _label, pattern in _EXPORT_PATTERNS:
        for m in pattern.finditer(content):
            raw = m.group(0).strip()

            # For function signatures captured without the body: add ' { ... }'
            if _label == "function" and not raw.endswith("{"):
                raw = raw + " { ... }"

            # For const, trim to one meaningful line
            if _label == "const":
                raw = raw.split("\n")[0].rstrip(",;") + ";"

            # Collapse whitespace runs for readability
            sig = " ".join(raw.split())

            # Cap length — some interface bodies can be enormous
            if len(sig) > 300:
                sig = sig[:297] + "..."

            if sig not in seen_sigs:
                seen_sigs.add(sig)
                signatures.append(sig)

    return "\n".join(signatures)


# ── Similarity helpers ─────────────────────────────────────────────────────────


def jaccard_similarity(tokens_a: set[int], tokens_b: set[int]) -> float:
    """Compute Jaccard similarity (token overlap) between two token sets.

    Jaccard = |A ∩ B| / |A ∪ B|.  Returns 0.0 when both sets are empty.

    For a TS dev: think of this as "how many words do these two files share?"
    — a cheap proxy for semantic similarity that doesn't require embeddings.

    Args:
        tokens_a: Token IDs for file A.
        tokens_b: Token IDs for file B.

    Returns:
        Float in [0, 1], higher means more similar.
    """
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / max(union, 1)


def find_similar_files(
    target_tokens: set[int],
    file_tokens: dict[str, set[int]],
    top_k: int = 3,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Find the top-K files most similar to the target by token overlap.

    Args:
        target_tokens: Token ID set for the file being completed.
        file_tokens: Mapping of file path -> token ID set for the whole repo.
        top_k: Number of similar files to return.
        exclude: File paths to exclude (e.g. the target file itself).

    Returns:
        List of (path, similarity_score) tuples, highest similarity first.
    """
    if not target_tokens:
        return []

    exclude = exclude or set()
    scored: list[tuple[str, float]] = []

    for path, tokens in file_tokens.items():
        if path in exclude or not tokens:
            continue
        score = jaccard_similarity(target_tokens, tokens)
        if score > 0.0:
            scored.append((path, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── File tree ─────────────────────────────────────────────────────────────────


def build_file_tree(root: Path, max_depth: int = 4) -> list[str]:
    """Build a compact file listing respecting .gitignore-style skip rules.

    Skips node_modules, .git, dist, build, and other noise directories.
    Returns relative POSIX paths for platform-agnostic display.

    Args:
        root: Repository root directory.
        max_depth: Maximum directory depth to traverse (0 = root files only).

    Returns:
        Sorted list of relative paths (POSIX format).
    """
    paths: list[str] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name not in {".env.example"}:
                continue
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                _walk(entry, depth + 1)
            elif entry.is_file():
                try:
                    rel = entry.relative_to(root)
                    paths.append(rel.as_posix())
                except ValueError:
                    pass

    _walk(root, 0)
    return paths


# ── RepoScanner ───────────────────────────────────────────────────────────────


class RepoScanner:
    """Scan a TypeScript/JavaScript repository and build context for generation.

    Usage:
        scanner = RepoScanner(Path("/path/to/myapp"))
        context = scanner.scan()              # run once at session start
        ctx_str = scanner.get_context_for_file("src/api/users.ts")
        full_prompt = ctx_str + user_prompt

    The scanner caches token sets for all files after scan() so that repeated
    calls to get_context_for_file() are fast (no disk I/O, just set arithmetic).
    """

    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root.resolve()
        self._context: RepoContext | None = None
        # path -> token ID set (populated lazily from tokenizer, or from word tokens)
        self._file_tokens: dict[str, set[int]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self) -> RepoContext:
        """Full repository scan.  Run once at session start.

        Steps:
          1. Build file tree (respects _SKIP_DIRS)
          2. Read package.json for dependencies and versions
          3. Read tsconfig.json for path aliases
          4. Parse imports for all .ts/.tsx/.js/.jsx files
          5. Build word-token sets for Jaccard similarity

        Returns:
            Populated RepoContext (also cached on self._context).
        """
        logger.info("Scanning repo: %s", self.root)

        file_tree = build_file_tree(self.root)
        package_info, framework_versions = self._read_package_json()
        tsconfig = self._read_tsconfig()
        import_graph = self._build_import_graph(file_tree)
        self._build_file_tokens(file_tree)

        self._context = RepoContext(
            root=self.root,
            file_tree=file_tree,
            package_info=package_info,
            tsconfig=tsconfig,
            import_graph=import_graph,
            framework_versions=framework_versions,
        )

        logger.info(
            "Scan complete: %d files, %d imports parsed",
            len(file_tree),
            sum(len(v) for v in import_graph.values()),
        )
        return self._context

    def get_context_for_file(self, file_path: str, max_tokens: int = 2048) -> str:
        """Build the <|repo|>...</|repo|> context block for a specific file.

        Steps:
          1. Resolve the file's imports
          2. For relative imports: read the imported file, extract exports
          3. For package imports: include version from package.json
          4. Find 3 similar files via Jaccard similarity
          5. Assemble the context block
          6. Trim to max_tokens budget (~4 chars per token heuristic)

        Args:
            file_path: Path to the file being completed (relative or absolute).
            max_tokens: Token budget for the context block.

        Returns:
            Formatted context string, or empty string if scanning hasn't run yet.
        """
        if self._context is None:
            logger.warning("get_context_for_file() called before scan(); running scan now")
            self.scan()

        ctx = self._context
        abs_path = self._resolve_file_path(file_path)

        # ── 1. Header: project name + key framework versions ────────────────
        header = self._build_header(ctx)

        # ── 2. Collect import-based context files ───────────────────────────
        file_blocks: list[str] = []
        char_budget = max_tokens * 4  # rough 4-chars-per-token heuristic

        imports = ctx.import_graph.get(abs_path, [])
        for imp_ref in imports:
            if imp_ref.is_relative:
                imp_abs = self._resolve_import(abs_path, imp_ref.source)
                if imp_abs and imp_abs.exists():
                    exports_str = self._safe_extract_exports(imp_abs)
                    if exports_str:
                        block = self._file_block(self._to_rel(imp_abs), exports_str)
                        file_blocks.append(block)
            else:
                # Package import: just note the version
                version = ctx.framework_versions.get(imp_ref.source, "")
                if version:
                    file_blocks.append(
                        self._file_block(imp_ref.source, f"// version: {version}")
                    )

        # ── 3. Similar files via Jaccard similarity ──────────────────────────
        target_tokens = self._file_tokens.get(abs_path, set())
        exclude = {abs_path}
        # Also exclude files already included from imports
        for imp_ref in imports:
            if imp_ref.is_relative:
                imp_abs = self._resolve_import(abs_path, imp_ref.source)
                if imp_abs:
                    exclude.add(str(imp_abs))

        similar = find_similar_files(
            target_tokens=target_tokens,
            file_tokens=self._file_tokens,
            top_k=3,
            exclude=exclude,
        )
        for sim_path, _score in similar:
            sim_abs = Path(sim_path)
            if sim_abs.exists():
                exports_str = self._safe_extract_exports(sim_abs)
                if exports_str:
                    block = self._file_block(self._to_rel(sim_abs), exports_str)
                    file_blocks.append(block)

        # ── 4. Assemble and trim to budget ───────────────────────────────────
        body_parts: list[str] = []
        used_chars = len(header) + 20  # leave room for tags

        for block in file_blocks:
            if used_chars + len(block) <= char_budget:
                body_parts.append(block)
                used_chars += len(block)
            else:
                # Partial block: fill remaining budget
                remaining = char_budget - used_chars - 30
                if remaining > 100:
                    truncated = block[:remaining] + "\n// ... truncated\n"
                    body_parts.append(truncated)
                break

        body = "\n".join(body_parts)
        return f"<|repo|>\n{header}\n{body}\n<|/repo|>\n"

    def get_repo_summary(self) -> str:
        """Return a human-readable repo summary for CLI display.

        Returns:
            Multi-line summary string (no ANSI codes, plain text).
        """
        if self._context is None:
            return "Repository not scanned yet. Call scan() first."

        ctx = self._context
        ts_files = sum(1 for p in ctx.file_tree if Path(p).suffix in _CODE_EXTENSIONS)
        total_imports = sum(len(v) for v in ctx.import_graph.values())

        lines = [
            f"Repository: {ctx.root.name}",
            f"Files: {len(ctx.file_tree)} total, {ts_files} TS/JS",
            f"Imports parsed: {total_imports}",
        ]

        if ctx.framework_versions:
            fw_str = ", ".join(
                f"{k}@{v}" for k, v in list(ctx.framework_versions.items())[:6]
            )
            lines.append(f"Frameworks: {fw_str}")

        if ctx.tsconfig:
            lines.append("tsconfig.json: found")

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_package_json(self) -> tuple[dict, dict[str, str]]:
        """Read package.json and extract name, deps, and framework versions."""
        pkg_path = self.root / "package.json"
        if not pkg_path.exists():
            return {}, {}

        try:
            data = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not read package.json: %s", exc)
            return {}, {}

        all_deps: dict[str, str] = {}
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            all_deps.update(data.get(key, {}))

        # Strip semver range chars to get clean versions
        framework_versions: dict[str, str] = {}
        for pkg_name, version_spec in all_deps.items():
            clean = version_spec.lstrip("^~>=<").split(" ")[0]
            framework_versions[pkg_name] = clean

        package_info = {
            "name": data.get("name", ""),
            "version": data.get("version", ""),
            "deps": all_deps,
        }
        return package_info, framework_versions

    def _read_tsconfig(self) -> dict | None:
        """Read tsconfig.json if present, stripping JS comments first."""
        for name in ("tsconfig.json", "tsconfig.base.json"):
            path = self.root / name
            if path.exists():
                try:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    # Strip single-line comments (tsconfig allows them)
                    stripped = re.sub(r"//[^\n]*", "", raw)
                    return json.loads(stripped)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Could not read %s: %s", name, exc)
        return None

    def _build_import_graph(self, file_tree: list[str]) -> dict[str, list[ImportRef]]:
        """Parse imports for all code files in the repo."""
        graph: dict[str, list[ImportRef]] = {}

        for rel_path in file_tree:
            if Path(rel_path).suffix not in _CODE_EXTENSIONS:
                continue
            abs_path = str(self.root / rel_path)
            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                graph[abs_path] = parse_imports(content)
            except OSError as exc:
                logger.debug("Could not read %s: %s", abs_path, exc)
                graph[abs_path] = []

        return graph

    def _build_file_tokens(self, file_tree: list[str]) -> None:
        """Build word-based token sets for Jaccard similarity.

        We use simple word tokenization (split on non-alphanumeric chars) rather
        than the model's BPE tokenizer here because: (a) we don't want to require
        the tokenizer to be loaded just for scanning, and (b) word overlap is a
        perfectly good proxy for "similar code style/domain".

        For a TS dev: think of this as a fast TF-IDF approximation without the IDF.
        """
        for rel_path in file_tree:
            if Path(rel_path).suffix not in _CODE_EXTENSIONS:
                continue
            abs_path = str(self.root / rel_path)
            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                # Split on anything that isn't alphanumeric — yields identifiers,
                # keywords, type names as "tokens"
                words = set(re.split(r"[^a-zA-Z0-9_$]+", content))
                words.discard("")
                # Use hash of each word string as the integer "token ID"
                self._file_tokens[abs_path] = {hash(w) for w in words}
            except OSError:
                pass

    def _build_header(self, ctx: RepoContext) -> str:
        """Build the Project: line for the context block."""
        name = ctx.package_info.get("name", ctx.root.name) or ctx.root.name

        # Pick the most recognisable framework versions for the header
        _priority = ["next", "react", "vue", "svelte", "angular", "astro",
                     "express", "fastify", "hono", "trpc", "prisma", "drizzle",
                     "zod", "typescript"]
        shown: list[str] = []
        for fw in _priority:
            if fw in ctx.framework_versions:
                shown.append(f"{fw}@{ctx.framework_versions[fw]}")
            if len(shown) >= 5:
                break

        if shown:
            return f"Project: {name} ({', '.join(shown)})"
        return f"Project: {name}"

    def _resolve_file_path(self, file_path: str) -> str:
        """Resolve a file_path argument to an absolute path string."""
        p = Path(file_path)
        if p.is_absolute():
            return str(p.resolve())
        return str((self.root / p).resolve())

    def _resolve_import(self, source_abs: str, import_spec: str) -> Path | None:
        """Resolve a relative import specifier to an actual file path."""
        base_dir = Path(source_abs).parent
        candidate = base_dir / import_spec
        extensions = ["", ".ts", ".tsx", ".js", ".jsx", ".mts", ".cts",
                      "/index.ts", "/index.tsx", "/index.js"]
        for ext in extensions:
            resolved = Path(str(candidate) + ext)
            if resolved.exists() and resolved.is_file():
                return resolved.resolve()
        return None

    def _safe_extract_exports(self, path: Path) -> str:
        """Read a file and extract its exports, catching all errors."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return extract_exports(content)
        except OSError as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return ""

    def _to_rel(self, path_or_str: str | Path) -> str:
        """Convert an absolute path to a POSIX-format relative path."""
        try:
            return Path(path_or_str).relative_to(self.root).as_posix()
        except ValueError:
            return str(path_or_str)

    @staticmethod
    def _file_block(rel_path: str, content: str) -> str:
        """Wrap content in <|file|>...</|file|> tags."""
        return f"<|file|>{rel_path}\n{content}\n<|/file|>"
