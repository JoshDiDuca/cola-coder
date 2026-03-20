"""Test-Code Pair Extractor: extract (test, implementation) pairs for training data.

Scans codebases to find test files and match them to their corresponding source
files using naming-convention heuristics. Creates paired training examples in
both directions:
  - [test → source]: teach TDD — write implementation that passes the tests
  - [source → test]: teach test generation — write tests for existing code

Research precedent: AlphaCode and CodeContests use (problem + test cases →
implementation) pairs and see 30-50% improvement on competitive programming vs
source-only training.

For a TS dev: similar to how Jest auto-discovers *.test.ts files, but run in
reverse — we walk the whole repo and match each test file back to the module it
exercises.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Test-file naming patterns
# ---------------------------------------------------------------------------

# (compiled regex for the filename, resolution strategy)
_FILE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Python
    (re.compile(r"^test_(.+)\.py$"), "strip_prefix"),       # test_calculator.py → calculator.py
    (re.compile(r"^(.+)_test\.py$"), "strip_suffix"),       # calculator_test.py → calculator.py
    # TypeScript / JavaScript
    (re.compile(r"^(.+)\.test\.(ts|tsx|js|jsx)$"), "same_dir"),   # foo.test.ts → foo.ts
    (re.compile(r"^(.+)\.spec\.(ts|tsx|js|jsx)$"), "same_dir"),   # foo.spec.ts → foo.ts
    (re.compile(r"^(.+)-test\.(ts|tsx|js|jsx)$"),  "same_dir"),   # foo-test.ts → foo.ts
    (re.compile(r"^(.+)_test\.(ts|tsx|js|jsx)$"),  "same_dir"),   # foo_test.ts → foo.ts
]

# Directory names that indicate a test folder and where to look for sources
_DIR_PATTERNS: list[tuple[str, str]] = [
    ("__tests__", ".."),       # __tests__/foo.ts  → ../foo.ts
    ("__test__",  ".."),
    ("tests",     "../src"),   # tests/foo.py      → ../src/foo.py
    ("test",      "../src"),   # test/foo.ts       → ../src/foo.ts
]

_SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".cache",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TestCodePair:
    """A matched (test file, source file) pair."""
    test_file: str       # path relative to root (or absolute if no root)
    source_file: str     # path relative to root (or absolute if no root)
    test_content: str
    source_content: str
    language: str        # "python", "typescript", "javascript"
    confidence: float    # 0.0 – 1.0; higher = more certain the files are related


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    return "javascript"


def _is_test_filename(name: str) -> bool:
    for pattern, _ in _FILE_PATTERNS:
        if pattern.match(name):
            return True
    return False


def _is_in_test_dir(parts: tuple[str, ...]) -> bool:
    for test_dir, _ in _DIR_PATTERNS:
        if test_dir in parts:
            return True
    return False


def _extract_function_names(code: str, language: str) -> set[str]:
    """Return all function/method names defined in the source."""
    if language == "python":
        pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.MULTILINE)
        return {m.group(1) for m in pattern.finditer(code)}
    # TypeScript / JavaScript
    pattern = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)"
        r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(?[^=]*\)?\s*=>"
        r"|(?:public|private|protected|static)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:{]",
        re.MULTILINE,
    )
    names: set[str] = set()
    noise = {"function", "const", "let", "var", "async", "return", "if", "for",
              "while", "switch", "class", "new", "import", "export"}
    for m in pattern.finditer(code):
        for g in m.groups():
            if g and len(g) > 1 and g not in noise:
                names.add(g)
    return names


def _extract_test_references(code: str, language: str) -> set[str]:
    """Return identifiers called / imported in the test file."""
    names: set[str] = set()
    noise = {
        "describe", "it", "test", "expect", "beforeEach", "afterEach",
        "beforeAll", "afterAll", "jest", "vi", "assert", "should",
        "assertEqual", "assertTrue", "assertFalse", "assertRaises",
        "setUp", "tearDown", "self", "cls", "super", "print", "len",
        "range", "str", "int", "float", "list", "dict", "set", "tuple",
    }
    # import references
    import_pat = re.compile(r"import\s*\{([^}]+)\}", re.MULTILINE)
    for m in import_pat.finditer(code):
        for tok in m.group(1).split(","):
            name = tok.strip().split(" as ")[0].strip()
            if name and name not in noise:
                names.add(name)
    # function calls
    call_pat = re.compile(r"\b(\w{2,})\s*\(")
    for m in call_pat.finditer(code):
        name = m.group(1)
        if name not in noise:
            names.add(name)
    return names


def _compute_confidence(test_content: str, source_content: str, language: str) -> float:
    """Estimate how likely the test actually exercises the source (0-1)."""
    src_fns = _extract_function_names(source_content, language)
    test_refs = _extract_test_references(test_content, language)
    if not src_fns:
        return 0.3  # source has no detected functions; uncertain
    overlap = src_fns & test_refs
    return min(1.0, len(overlap) / max(1, len(src_fns)))


# ---------------------------------------------------------------------------
# PairExtractor
# ---------------------------------------------------------------------------

class PairExtractor:
    """Scan a codebase and extract (test, source) file pairs for training.

    Usage::

        extractor = PairExtractor()
        pairs = extractor.find_pairs("/path/to/repo")
        training_data = extractor.create_training_pairs(pairs)
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def find_pairs(self, root_dir: str) -> list[TestCodePair]:
        """Recursively scan *root_dir* and return matched (test, source) pairs.

        Args:
            root_dir: Absolute or relative path to the repository root.

        Returns:
            List of TestCodePair; one entry per unique (test_file, source_file)
            match.  Unresolvable test files are skipped silently.
        """
        root = Path(root_dir).resolve()
        all_source_paths: list[str] = []

        # Collect all source files (non-test) for the matcher
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune irrelevant directories in-place
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix in _SOURCE_EXTENSIONS:
                    all_source_paths.append(str(fpath))

        # Identify test files and attempt to match each to a source file
        seen: set[tuple[str, str]] = set()
        pairs: list[TestCodePair] = []

        for candidate in all_source_paths:
            cpath = Path(candidate)
            if not (_is_test_filename(cpath.name) or _is_in_test_dir(cpath.parts)):
                continue

            # Try to find the matching source
            source_str = self.match_test_to_source(str(cpath), all_source_paths)
            if source_str is None:
                continue

            key = (str(cpath), source_str)
            if key in seen:
                continue
            seen.add(key)

            try:
                test_content = cpath.read_text(encoding="utf-8", errors="ignore")
                source_content = Path(source_str).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if not test_content.strip() or not source_content.strip():
                continue

            language = _detect_language(source_str)
            confidence = _compute_confidence(test_content, source_content, language)

            # Use paths relative to root when possible
            try:
                test_rel = str(cpath.relative_to(root))
                src_rel = str(Path(source_str).relative_to(root))
            except ValueError:
                test_rel = str(cpath)
                src_rel = source_str

            pairs.append(TestCodePair(
                test_file=test_rel,
                source_file=src_rel,
                test_content=test_content,
                source_content=source_content,
                language=language,
                confidence=confidence,
            ))

        return pairs

    def match_test_to_source(self, test_path: str, source_paths: list[str]) -> str | None:
        """Find the source file that corresponds to *test_path*.

        Tries naming-convention heuristics in priority order and returns the
        first match found among *source_paths*.

        Args:
            test_path: Path (str) to the test file.
            source_paths: All candidate source file paths to search.

        Returns:
            The matched source path string, or ``None`` if no match found.
        """
        tpath = Path(test_path)
        name = tpath.name
        parent = tpath.parent
        parts = tpath.parts

        # Build a quick lookup: basename → [full_path, ...]
        by_name: dict[str, list[str]] = {}
        for sp in source_paths:
            bn = Path(sp).name
            by_name.setdefault(bn, []).append(sp)

        # Strategy 1: strip test prefix/suffix from the filename
        for pattern, strategy in _FILE_PATTERNS:
            m = pattern.match(name)
            if not m:
                continue

            if strategy == "strip_prefix":
                # test_calculator.py → calculator.py
                base = m.group(1)
                ext = tpath.suffix
                candidate_name = f"{base}{ext}"
            elif strategy == "strip_suffix":
                # calculator_test.py → calculator.py
                base = m.group(1)
                ext = tpath.suffix
                candidate_name = f"{base}{ext}"
            else:
                # same_dir: foo.test.ts → foo.ts
                base = m.group(1)
                ext_lang = m.group(2) if m.lastindex and m.lastindex >= 2 else tpath.suffix.lstrip(".")
                candidate_name = f"{base}.{ext_lang}"

            # Look in same directory first
            same_dir_candidate = str(parent / candidate_name)
            if same_dir_candidate in source_paths or Path(same_dir_candidate).exists():
                # Verify it's actually in our source_paths list (or is a real file)
                for sp in source_paths:
                    if Path(sp).name == candidate_name and Path(sp) != tpath:
                        if _is_same_or_near_dir(Path(sp).parent, parent):
                            return sp

            # Fall back to any file with that name
            if candidate_name in by_name:
                candidates = [sp for sp in by_name[candidate_name] if sp != test_path]
                if candidates:
                    # Prefer the one closest in directory depth
                    candidates.sort(key=lambda sp: _dir_distance(Path(sp).parent, parent))
                    return candidates[0]

        # Strategy 2: __tests__ / test / tests directory pattern
        for test_dir, relative_to in _DIR_PATTERNS:
            if test_dir not in parts:
                continue
            idx = list(parts).index(test_dir)
            # Reconstruct path without the test directory component
            parts_before = parts[:idx]
            parts_after = parts[idx + 1:]
            if parts_before:
                base_path = Path(*parts_before).joinpath(*parts_after) if parts_after else Path(*parts_before)
            else:
                base_path = Path(*parts_after) if parts_after else Path(".")

            # Try the file as-is or with common source extensions
            for ext in ["", ".py", ".ts", ".tsx", ".js", ".jsx"]:
                candidate = base_path.with_suffix(ext) if ext else base_path
                cname = candidate.name
                if cname in by_name:
                    hits = [sp for sp in by_name[cname] if sp != test_path]
                    if hits:
                        return hits[0]

        return None

    def extract_test_functions(self, code: str, language: str) -> list[str]:
        """Extract individual test function bodies from *code*.

        Returns a list of strings, each containing one complete test function
        (including its signature line).

        Args:
            code: Full source of a test file.
            language: "python", "typescript", or "javascript".
        """
        if language == "python":
            return self._extract_python_test_functions(code)
        return self._extract_ts_test_functions(code)

    def create_training_pairs(self, pairs: list[TestCodePair]) -> list[dict]:
        """Format matched pairs into training records (Alpaca-style).

        Generates **two** records per pair:
        - ``test_to_source``: given tests, implement the source
        - ``source_to_test``: given source, write the tests

        Args:
            pairs: List of TestCodePair objects (from :meth:`find_pairs`).

        Returns:
            List of dicts with keys ``instruction``, ``input``, ``output``,
            ``metadata``.
        """
        records: list[dict] = []
        for pair in pairs:
            lang_label = pair.language

            # Direction A: test → source
            records.append({
                "instruction": (
                    f"Given the following {lang_label} test file, implement the source code "
                    f"that would make all tests pass.\n\n"
                    f"Tests:\n```{lang_label}\n{pair.test_content[:3000]}\n```\n\n"
                    f"Implement the source code:"
                ),
                "input": "",
                "output": f"```{lang_label}\n{pair.source_content[:4000]}\n```",
                "metadata": {
                    "test_file": pair.test_file,
                    "source_file": pair.source_file,
                    "direction": "test_to_source",
                    "language": pair.language,
                    "confidence": pair.confidence,
                },
            })

            # Direction B: source → test
            records.append({
                "instruction": (
                    f"Write comprehensive {lang_label} tests for the following code:\n\n"
                    f"Source:\n```{lang_label}\n{pair.source_content[:3000]}\n```\n\n"
                    f"Tests:"
                ),
                "input": "",
                "output": f"```{lang_label}\n{pair.test_content[:4000]}\n```",
                "metadata": {
                    "test_file": pair.test_file,
                    "source_file": pair.source_file,
                    "direction": "source_to_test",
                    "language": pair.language,
                    "confidence": pair.confidence,
                },
            })

        return records

    def summary(self, pairs: list[TestCodePair]) -> dict:
        """Return a summary statistics dict for a list of pairs.

        Args:
            pairs: List of TestCodePair objects.

        Returns:
            Dict with counts and per-language breakdown.
        """
        by_lang: dict[str, int] = {}
        confidences: list[float] = []
        for p in pairs:
            by_lang[p.language] = by_lang.get(p.language, 0) + 1
            confidences.append(p.confidence)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        high_conf = sum(1 for c in confidences if c >= 0.5)

        return {
            "total_pairs": len(pairs),
            "by_language": by_lang,
            "avg_confidence": round(avg_conf, 3),
            "high_confidence_pairs": high_conf,   # confidence >= 0.5
            "training_records": len(pairs) * 2,   # both directions
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_python_test_functions(self, code: str) -> list[str]:
        """Extract functions whose names start with ``test_``."""
        lines = code.splitlines()
        functions: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect a test function definition
            stripped = line.lstrip()
            if re.match(r"(?:async\s+)?def\s+test_\w+", stripped):
                indent = len(line) - len(stripped)
                func_lines = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.lstrip()
                    # End of function: same or lesser indent with content
                    if next_stripped and (len(next_line) - len(next_stripped)) <= indent:
                        # Check it's not a continuation (decorator, nested def at same level)
                        if not next_stripped.startswith("@"):
                            break
                    func_lines.append(next_line)
                    j += 1
                functions.append("\n".join(func_lines))
                i = j
            else:
                i += 1
        return functions

    def _extract_ts_test_functions(self, code: str) -> list[str]:
        """Extract ``test(...)`` / ``it(...)`` / ``describe(...)`` blocks."""
        # Simple line-based extraction: find test/it/describe calls and collect
        # until the matching closing brace.
        test_start = re.compile(
            r"^\s*(?:(?:export\s+)?(?:async\s+)?(?:test|it|describe)\s*\()"
        )
        blocks: list[str] = []
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            if test_start.match(lines[i]):
                brace_depth = 0
                block_lines = []
                j = i
                started = False
                while j < len(lines):
                    block_lines.append(lines[j])
                    for ch in lines[j]:
                        if ch == "{":
                            brace_depth += 1
                            started = True
                        elif ch == "}":
                            brace_depth -= 1
                    if started and brace_depth <= 0:
                        j += 1
                        break
                    j += 1
                blocks.append("\n".join(block_lines))
                i = j
            else:
                i += 1
        return blocks


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _is_same_or_near_dir(a: Path, b: Path) -> bool:
    """Return True if *a* and *b* are the same directory or siblings."""
    try:
        return a == b or a.parent == b or b.parent == a or a.parent == b.parent
    except Exception:
        return False


def _dir_distance(a: Path, b: Path) -> int:
    """Count how many directory components differ between *a* and *b*."""
    try:
        a_parts = a.resolve().parts
        b_parts = b.resolve().parts
        # Count common prefix length
        common = 0
        for x, y in zip(a_parts, b_parts):
            if x == y:
                common += 1
            else:
                break
        return (len(a_parts) - common) + (len(b_parts) - common)
    except Exception:
        return 999
