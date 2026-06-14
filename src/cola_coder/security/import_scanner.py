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
