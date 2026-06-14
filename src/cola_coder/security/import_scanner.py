"""Hallucinated-import / slopsquatting scanner (single source of truth).

2026 supply-chain threat: code models HALLUCINATE package names (5-21% of samples;
USENIX Security 2025) that don't exist. Attackers register the fabricated name on
PyPI/npm with malware, so an LLM's invented `import foo` becomes an install-time
compromise ("slopsquatting"; the `huggingface-cli` PyPI incident got 30k+ installs).

This statically extracts the imported package roots from generated/scraped code and
flags any that are NOT in a curated known-safe allowlist (stdlib + popular packages).
An "unknown" import is a REVIEW signal — it may be a legit niche package OR a
hallucination/typosquat — so callers should warn/annotate, not hard-block. Static,
no execution. Mirrors security/code_patterns.py (different threat class).
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from enum import Enum

# stdlib module names (frozenset, 3.10+) — authoritative, no hardcoding needed.
_PY_STDLIB = set(getattr(sys, "stdlib_module_names", frozenset()))

# Curated popular PyPI packages, by IMPORT name (PIL not pillow, sklearn not
# scikit-learn, yaml not pyyaml, cv2 not opencv, bs4 not beautifulsoup4).
_PY_POPULAR = {
    "requests", "numpy", "pandas", "scipy", "torch", "torchvision", "tensorflow",
    "keras", "sklearn", "matplotlib", "seaborn", "PIL", "cv2", "flask", "django",
    "fastapi", "starlette", "pydantic", "uvicorn", "gunicorn", "pytest", "sqlalchemy",
    "alembic", "boto3", "botocore", "aiohttp", "httpx", "click", "typer", "rich",
    "tqdm", "yaml", "dotenv", "redis", "celery", "transformers", "datasets",
    "tokenizers", "accelerate", "safetensors", "huggingface_hub", "openai",
    "anthropic", "langchain", "bs4", "lxml", "jwt", "cryptography", "passlib",
    "jinja2", "markupsafe", "werkzeug", "setuptools", "pip", "wheel", "attr", "attrs",
    "dateutil", "pytz", "six", "google", "grpc", "psycopg2", "pymongo", "elasticsearch",
}
_PY_KNOWN = _PY_STDLIB | _PY_POPULAR

# Node builtins + curated popular npm packages (scoped names kept as @scope/name).
_JS_KNOWN = {
    # node builtins
    "fs", "path", "http", "https", "os", "crypto", "util", "stream", "events",
    "child_process", "url", "querystring", "assert", "buffer", "process", "zlib",
    "net", "tls", "dns", "readline", "cluster", "worker_threads", "perf_hooks",
    "node:fs", "node:path", "node:http", "node:crypto", "node:os", "node:util",
    # popular npm
    "react", "react-dom", "next", "vue", "svelte", "@angular/core", "rxjs", "redux",
    "@reduxjs/toolkit", "lodash", "axios", "express", "koa", "fastify", "jest",
    "vitest", "typescript", "zod", "prisma", "@prisma/client", "tailwindcss",
    "eslint", "prettier", "webpack", "vite", "esbuild", "rollup", "dotenv", "chalk",
    "commander", "yargs", "moment", "dayjs", "uuid", "classnames", "styled-components",
    "@emotion/react", "graphql", "@apollo/client", "mongoose", "pg", "mysql2",
    "socket.io", "ws", "jsonwebtoken", "bcrypt", "cors", "body-parser", "nodemon",
    "ts-node", "@types/node", "@types/react", "react-router-dom", "zustand", "swr",
    "@tanstack/react-query",
}

_JS_IMPORT_RE = re.compile(
    r"""(?:import\b[^'"]*?from\s*|import\s*|require\s*\(\s*|import\s*\(\s*)['"]([^'"]+)['"]"""
)


def _python_import_roots(code: str) -> set[str]:
    """Top-level package roots imported by Python `code` (absolute imports only)."""
    roots: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Partial/garbled generations: fall back to a line regex.
        for m in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z0-9_.]+)", code, re.MULTILINE):
            roots.add(m.group(1).split(".")[0])
        return roots
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level>0 = relative (local) import — not a package, skip.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _js_import_specs(code: str) -> set[str]:
    """Package specifiers imported by JS/TS `code` (relative/local paths excluded)."""
    pkgs: set[str] = set()
    for m in _JS_IMPORT_RE.finditer(code):
        spec = m.group(1)
        if spec.startswith(".") or spec.startswith("/"):
            continue  # relative / absolute-local import, not a package
        if spec.startswith("@"):
            parts = spec.split("/")
            pkgs.add("/".join(parts[:2]))  # @scope/name
        else:
            pkgs.add(spec.split("/")[0])   # pkg from pkg/sub/path
    return pkgs


def extract_imports(code: str, language: str | None = None) -> set[str]:
    """Extract imported package names from `code`. Routes by `language`."""
    if not code:
        return set()
    if language == "typescript" or language == "javascript":
        return _js_import_specs(code)
    if language == "python":
        return _python_import_roots(code)
    # Unknown/auto: union both (cheap, and a TS file rarely parses as Python).
    return _python_import_roots(code) | _js_import_specs(code)


def scan_unknown_imports(code: str, language: str | None = None) -> list[str]:
    """Return imported packages NOT in the known-safe allowlist (sorted).

    These are slopsquatting REVIEW candidates — possibly legit niche packages,
    possibly hallucinated/typosquatted. Empty list = all imports recognized.
    """
    if not code:
        return []
    imports = extract_imports(code, language)
    known = _JS_KNOWN if language in ("typescript", "javascript") else (
        _PY_KNOWN if language == "python" else _PY_KNOWN | _JS_KNOWN
    )
    return sorted(imp for imp in imports if imp not in known)


def has_unknown_imports(code: str, language: str | None = None) -> bool:
    """True if `code` imports any package outside the known-safe allowlist."""
    return bool(scan_unknown_imports(code, language))


# ---------------------------------------------------------------------------
# Typosquat / slopsquat triage (SEC-023)
#
# `scan_unknown_imports` treats every out-of-allowlist import identically — a
# legit niche package and a malicious typosquat of a popular package get the
# same flat "unknown" verdict. ConfuGuard (arXiv:2502.20528) frames the goal as
# distinguishing CONFUSION attacks from legitimate packages; the established
# cheap first-line screen (SpellBound; IQTLabs/pypi-scan; the 98.4%-accuracy
# Damerau-Levenshtein study) is string-distance against the popular set,
# combined with separator normalization and homoglyph/homophone substitution.
# This module has no network/metadata access, so we implement exactly that
# offline screen and emit a TRIAGED risk verdict, never a hard block.
# ---------------------------------------------------------------------------


# Visually/typographically confusable single-character substitutions used by
# squatters (the "1"/"l", "0"/"o" tricks). Applied symmetrically when comparing
# a candidate to a known name so e.g. ``l0dash`` collapses onto ``lodash``.
_HOMOGLYPHS: dict[str, str] = {
    "0": "o",
    "1": "l",
    "5": "s",
    "3": "e",
    "rn": "m",
    "vv": "w",
}


class ImportRisk(str, Enum):
    """Triage verdict for an out-of-allowlist import."""

    TYPOSQUAT = "typosquat"  # confusably close to a popular package
    UNKNOWN = "unknown"  # not close to anything known (legit niche OR fabricated)


@dataclass
class SuspectImport:
    """A single out-of-allowlist import with its triage verdict.

    Attributes:
        name: the imported package specifier as written in the code.
        risk: TYPOSQUAT (close to a popular name) or UNKNOWN.
        nearest: the popular package it most resembles (None when UNKNOWN).
        distance: normalized edit distance to ``nearest`` (None when UNKNOWN).
    """

    name: str
    risk: ImportRisk
    nearest: str | None = None
    distance: int | None = None


@dataclass
class ImportTriageReport:
    """Aggregated triage over a code sample's unknown imports."""

    typosquats: list[SuspectImport] = field(default_factory=list)
    unknown: list[SuspectImport] = field(default_factory=list)

    @property
    def has_typosquat(self) -> bool:
        """True if any unknown import is a likely typosquat of a popular package."""
        return bool(self.typosquats)


def _normalize_name(name: str) -> str:
    """Canonicalize a package name for confusion comparison.

    Lowercases, unifies the ``- _ .`` separators, applies homoglyph folding, and
    drops separators entirely so ``mysql-import`` and ``mysql_import`` and
    ``mysqlimport`` collapse to one comparable token (the "swapped/joined words"
    and separator-confusion squats the typosquatting literature flags).
    """
    folded = name.lower()
    for glyph, canonical in _HOMOGLYPHS.items():
        folded = folded.replace(glyph, canonical)
    for sep in ("-", "_", "."):
        folded = folded.replace(sep, "")
    return folded


def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string-alignment (Damerau-Levenshtein) edit distance.

    Counts insertions, deletions, substitutions, and ADJACENT TRANSPOSITIONS as
    single edits — the latter being the dominant typo class (``recieve`` for
    ``receive``) plain Levenshtein over-penalizes.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_prev: list[int] = []
    prev: list[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            cur[j + 1] = min(
                cur[j] + 1,  # insertion
                prev[j + 1] + 1,  # deletion
                prev[j] + cost,  # substitution
            )
            if (
                i > 0
                and j > 0
                and ca == b[j - 1]
                and a[i - 1] == cb
            ):
                cur[j + 1] = min(cur[j + 1], prev_prev[j - 1] + 1)  # transposition
        prev_prev, prev = prev, cur
    return prev[len(b)]


def _strip_scope(spec: str) -> str:
    """Reduce an npm scoped specifier (``@scope/name``) to its ``name`` part."""
    if spec.startswith("@") and "/" in spec:
        return spec.split("/", 1)[1]
    return spec


def _nearest_popular(name: str, popular: set[str]) -> tuple[str | None, int]:
    """Find the popular package nearest to ``name`` and the normalized distance.

    Returns ``(None, 0)`` when ``name`` normalizes to a popular name exactly (a
    separator/homoglyph confusion is itself the squat — distance 0 on the
    normalized form but the raw names differ).
    """
    target = _normalize_name(_strip_scope(name))
    best_name: str | None = None
    # Rank by (distance, raw-name length, raw name) so ties resolve
    # deterministically to the simplest canonical neighbor — e.g. 'reactt'
    # resolves to 'react', not the equidistant '@emotion/react'. ``set``
    # iteration order is otherwise arbitrary.
    best_key: tuple[int, int, str] | None = None
    for pop in popular:
        cand = _normalize_name(_strip_scope(pop))
        if not cand:
            continue
        dist = _damerau_levenshtein(target, cand)
        key = (dist, len(pop), pop)
        if best_key is None or key < best_key:
            best_key, best_name = key, pop
    return best_name, (0 if best_key is None else best_key[0])


def _is_typosquat(
    name: str,
    nearest: str | None,
    distance: int,
    *,
    max_distance: int,
    min_length: int,
) -> bool:
    """Decide whether ``name`` is close enough to ``nearest`` to be a squat.

    A confusion requires the candidate to be a DIFFERENT raw name from its
    neighbor (an exact match is in the allowlist and never reaches here), long
    enough that closeness is meaningful (short names collide by chance), and
    within ``max_distance`` normalized edits.
    """
    if nearest is None:
        return False
    if _normalize_name(_strip_scope(name)) == _normalize_name(_strip_scope(nearest)):
        # Same normalized form, different raw name -> separator/homoglyph squat.
        return name != nearest
    if len(_normalize_name(_strip_scope(name))) < min_length:
        return False
    return 0 < distance <= max_distance


def classify_unknown_imports(
    code: str,
    language: str | None = None,
    *,
    max_distance: int = 1,
    min_length: int = 4,
) -> ImportTriageReport:
    """Triage a sample's out-of-allowlist imports into typosquats vs unknowns.

    Reuses :func:`scan_unknown_imports` for the allowlist screen (DRY), then
    classifies each survivor by string distance to the popular package set:
    a close confusable name is a TYPOSQUAT (high-risk slopsquatting candidate),
    anything else is UNKNOWN (possibly a legitimate niche package).

    Args:
        code: source to scan.
        language: ``"python"`` / ``"typescript"`` / ``"javascript"`` / None (both).
        max_distance: max normalized Damerau-Levenshtein edits to count as a squat.
        min_length: ignore confusion for names shorter than this (chance collisions).

    Returns:
        An :class:`ImportTriageReport` with the typosquat and unknown lists.
    """
    report = ImportTriageReport()
    unknowns = scan_unknown_imports(code, language)
    if not unknowns:
        return report
    if language in ("typescript", "javascript"):
        popular = _JS_KNOWN
    elif language == "python":
        popular = _PY_KNOWN
    else:
        popular = _PY_KNOWN | _JS_KNOWN
    for name in unknowns:
        nearest, distance = _nearest_popular(name, popular)
        if _is_typosquat(
            name, nearest, distance, max_distance=max_distance, min_length=min_length
        ):
            report.typosquats.append(
                SuspectImport(name=name, risk=ImportRisk.TYPOSQUAT, nearest=nearest,
                              distance=distance)
            )
        else:
            report.unknown.append(SuspectImport(name=name, risk=ImportRisk.UNKNOWN))
    return report
