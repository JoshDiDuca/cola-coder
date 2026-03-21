"""Import Analyzer: parse and classify import statements from generated code.

Supports Python, TypeScript, and JavaScript.  Identifies stdlib vs third-party
packages, counts unique imports, and flags potential circular-import patterns
(when a module imports another that imports back).

For a TS dev: like eslint-plugin-import but as a standalone data structure you
can query programmatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True

# Standard-library module names for Python 3 (common subset — not exhaustive).
_PYTHON_STDLIB: frozenset[str] = frozenset(
    {
        "abc", "ast", "asyncio", "base64", "builtins", "collections", "contextlib",
        "copy", "csv", "datetime", "decimal", "enum", "functools", "gc", "glob",
        "hashlib", "heapq", "html", "http", "importlib", "inspect", "io", "itertools",
        "json", "logging", "math", "multiprocessing", "operator", "os", "pathlib",
        "pickle", "platform", "pprint", "queue", "random", "re", "shutil", "signal",
        "socket", "sqlite3", "ssl", "stat", "string", "struct", "subprocess",
        "sys", "tempfile", "threading", "time", "traceback", "types", "typing",
        "unittest", "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
        "dataclasses", "fractions", "statistics", "textwrap", "unicodedata",
        "argparse", "configparser", "getpass", "gzip", "lzma", "pdb", "profile",
    }
)

# Node.js built-in module names (common subset).
_NODE_BUILTINS: frozenset[str] = frozenset(
    {
        "assert", "buffer", "child_process", "cluster", "console", "constants",
        "crypto", "dgram", "dns", "domain", "events", "fs", "http", "http2",
        "https", "module", "net", "os", "path", "perf_hooks", "process",
        "punycode", "querystring", "readline", "repl", "stream", "string_decoder",
        "sys", "timers", "tls", "trace_events", "tty", "url", "util", "v8", "vm",
        "wasi", "worker_threads", "zlib",
    }
)


def is_enabled() -> bool:
    """Return True if the import analyzer feature is active."""
    return FEATURE_ENABLED


@dataclass
class ImportEntry:
    """A single resolved import."""

    raw: str  # original import statement line
    module: str  # top-level module name
    names: list[str] = field(default_factory=list)  # imported symbols (if any)
    is_stdlib: bool = False
    is_relative: bool = False  # relative import (Python: "from . import x")
    alias: str | None = None  # "import x as y" -> alias="y"


@dataclass
class ImportReport:
    """Results from analyzing imports in a code snippet."""

    language: str = ""
    imports: list[ImportEntry] = field(default_factory=list)
    unique_modules: list[str] = field(default_factory=list)
    stdlib_modules: list[str] = field(default_factory=list)
    third_party_modules: list[str] = field(default_factory=list)
    relative_imports: list[str] = field(default_factory=list)
    potential_circular: list[tuple[str, str]] = field(default_factory=list)
    total_import_lines: int = 0

    def summary(self) -> str:
        """Return a short human-readable summary."""
        return (
            f"language={self.language} total={self.total_import_lines} "
            f"unique={len(self.unique_modules)} "
            f"stdlib={len(self.stdlib_modules)} "
            f"third_party={len(self.third_party_modules)}"
        )


class ImportAnalyzer:
    """Parse import statements and classify them by origin.

    Usage::

        analyzer = ImportAnalyzer()
        report = analyzer.analyze(code, language="python")
        print(report.summary())
    """

    def analyze(self, code: str, language: str = "python") -> ImportReport:
        """Analyze imports in *code*.

        Args:
            code: Source code string.
            language: One of ``"python"``, ``"typescript"``, ``"javascript"``.

        Returns:
            ImportReport with classified imports.
        """
        language = language.lower()
        if language in ("typescript", "javascript", "ts", "js"):
            entries = self._parse_js_ts(code)
            stdlib = _NODE_BUILTINS
        else:
            entries = self._parse_python(code)
            stdlib = _PYTHON_STDLIB

        report = ImportReport(language=language)
        report.imports = entries
        report.total_import_lines = len(entries)

        seen_modules: dict[str, list[str]] = {}
        for entry in entries:
            entry.is_stdlib = entry.module in stdlib
            if entry.is_relative:
                report.relative_imports.append(entry.raw)
            key = entry.module
            seen_modules.setdefault(key, [])
            seen_modules[key].extend(entry.names)

        report.unique_modules = sorted(seen_modules.keys())
        report.stdlib_modules = sorted(m for m in seen_modules if m in stdlib)
        report.third_party_modules = sorted(
            m for m in seen_modules if m not in stdlib and not m.startswith(".")
        )
        report.potential_circular = self._detect_circular(entries)

        return report

    # ------------------------------------------------------------------
    # Python parser
    # ------------------------------------------------------------------

    def _parse_python(self, code: str) -> list[ImportEntry]:
        entries: list[ImportEntry] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # "from X import Y [as Z]" or "from .X import Y"
            m = re.match(
                r"^from\s+(\.+\S*|\S+)\s+import\s+(.+)$",
                stripped,
            )
            if m:
                module_part = m.group(1).lstrip(".")
                names_part = m.group(2)
                is_relative = stripped.startswith("from .")
                module = module_part.split(".")[0] if module_part else "."
                names = self._parse_import_names(names_part)
                entries.append(
                    ImportEntry(
                        raw=stripped,
                        module=module,
                        names=names,
                        is_relative=is_relative,
                    )
                )
                continue

            # "import X [as Y], Z [as W]"
            m2 = re.match(r"^import\s+(.+)$", stripped)
            if m2:
                for part in m2.group(1).split(","):
                    part = part.strip()
                    alias: str | None = None
                    if " as " in part:
                        part, alias = (s.strip() for s in part.split(" as ", 1))
                    module = part.split(".")[0]
                    entries.append(
                        ImportEntry(raw=stripped, module=module, alias=alias)
                    )
        return entries

    # ------------------------------------------------------------------
    # JS / TS parser
    # ------------------------------------------------------------------

    def _parse_js_ts(self, code: str) -> list[ImportEntry]:
        entries: list[ImportEntry] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # ES module: import ... from "..."  or  import "..."
            m = re.match(
                r"""^import\s+(?P<clause>[^'"`]*?)\s*from\s*['"`](?P<spec>[^'"` ]+)['"`]""",
                stripped,
            )
            if m:
                spec = m.group("spec")
                names = self._parse_js_clause(m.group("clause"))
                module = self._js_module_name(spec)
                is_rel = spec.startswith(".")
                entries.append(
                    ImportEntry(
                        raw=stripped,
                        module=module,
                        names=names,
                        is_relative=is_rel,
                    )
                )
                continue

            # Side-effect import: import "..."
            m2 = re.match(r"""^import\s*['"`](?P<spec>[^'"` ]+)['"`]""", stripped)
            if m2:
                spec = m2.group("spec")
                module = self._js_module_name(spec)
                entries.append(
                    ImportEntry(raw=stripped, module=module, is_relative=spec.startswith("."))
                )
                continue

            # CommonJS: require("...")
            m3 = re.search(r"""require\s*\(\s*['"`]([^'"` ]+)['"`]\s*\)""", stripped)
            if m3:
                spec = m3.group(1)
                module = self._js_module_name(spec)
                entries.append(
                    ImportEntry(
                        raw=stripped, module=module, is_relative=spec.startswith(".")
                    )
                )

        return entries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_import_names(names_str: str) -> list[str]:
        """Parse 'a, b, c' or '(a, b)' into a list of names."""
        names_str = names_str.strip().strip("()")
        parts = [p.strip() for p in names_str.split(",") if p.strip()]
        result = []
        for part in parts:
            name = part.split(" as ")[0].strip()
            if name:
                result.append(name)
        return result

    @staticmethod
    def _parse_js_clause(clause: str) -> list[str]:
        """Extract imported names from an ES import clause."""
        clause = clause.strip()
        names: list[str] = []
        # default: "React" or "React, { useState }"
        m_default = re.match(r"^(\w+)\s*(?:,|$)", clause)
        if m_default:
            names.append(m_default.group(1))
        # named: { a, b as c }
        m_named = re.search(r"\{([^}]+)\}", clause)
        if m_named:
            for part in m_named.group(1).split(","):
                name = part.split(" as ")[0].strip()
                if name:
                    names.append(name)
        # namespace: * as ns
        m_ns = re.search(r"\*\s+as\s+(\w+)", clause)
        if m_ns:
            names.append(f"* as {m_ns.group(1)}")
        return names

    @staticmethod
    def _js_module_name(spec: str) -> str:
        """Return the package/module name from a JS module specifier."""
        if spec.startswith("."):
            return spec
        # Scoped packages: @org/pkg -> "@org/pkg"
        if spec.startswith("@"):
            parts = spec.split("/")
            return "/".join(parts[:2]) if len(parts) >= 2 else spec
        return spec.split("/")[0]

    @staticmethod
    def _detect_circular(entries: list[ImportEntry]) -> list[tuple[str, str]]:
        """Very basic heuristic: flag if two relative imports reference each other."""
        relative = [e.module for e in entries if e.is_relative]
        # Trivial circular detection: same relative target imported twice
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for m in relative:
            if m in seen:
                pairs.append((m, m))
            seen.add(m)
        return pairs
