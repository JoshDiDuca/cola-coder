"""Hardened tsconfig.json generation for safe TypeScript compilation.

Security measures:
- plugins: []       -> prevents tsc from loading compiler plugins (arbitrary code execution)
- types: []         -> prevents automatic @types/* package loading
- typeRoots: []     -> prevents type root directory scanning
- Explicit include  -> only named files, no wildcards matching injected files
- No paths/baseUrl  -> prevents path traversal outside the temp dir
"""

from __future__ import annotations

import json
from pathlib import Path


def create_hardened_tsconfig(
    strict: bool = True,
    include_files: list[str] | None = None,
) -> dict:
    """Create a tsconfig.json that is safe to use with untrusted code.

    Args:
        strict: Enable TypeScript strict mode.
        include_files: Explicit list of .ts files to check. Defaults to ["*.ts"].

    Returns:
        Dict suitable for json.dumps() -> tsconfig.json.
    """
    return {
        "compilerOptions": {
            "strict": strict,
            "noEmit": True,
            "target": "ES2022",
            "module": "ESNext",
            "moduleResolution": "bundler",
            "skipLibCheck": True,
            "plugins": [],       # SECURITY: no plugin execution
            "types": [],         # SECURITY: no @types resolution
            "typeRoots": [],     # SECURITY: no type root scanning
            # No "paths" or "baseUrl" — prevents path traversal
        },
        "include": include_files or ["*.ts"],
        "exclude": ["node_modules", "**/*.js", "**/*.cjs", "**/*.mjs"],
    }


def write_hardened_tsconfig(
    directory: str | Path,
    strict: bool = True,
    include_files: list[str] | None = None,
) -> Path:
    """Write a hardened tsconfig.json to the given directory.

    Returns:
        Path to the written tsconfig.json.
    """
    tsconfig = create_hardened_tsconfig(strict, include_files)
    path = Path(directory) / "tsconfig.json"
    path.write_text(json.dumps(tsconfig, indent=2), encoding="utf-8")
    return path
