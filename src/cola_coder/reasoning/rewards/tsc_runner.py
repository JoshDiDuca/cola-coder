"""Unified tsc execution engine -- always through SandboxedRunner.

Single Responsibility: manages temp files, hardened tsconfig, subprocess execution.
Used by both TscScorer (data scoring) and TypeCheckReward (RL training).

All tsc execution in the entire codebase goes through this class.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig


@dataclass
class TscError:
    """A single tsc diagnostic."""
    file: str
    line: int
    col: int
    severity: str  # "error" or "warning"
    code: str      # e.g. "TS2322"
    message: str


class TscRunner:
    """Unified sandboxed tsc execution for the entire codebase.

    Manages temp files, writes hardened tsconfig.json (plugins=[], types=[],
    typeRoots=[]), runs tsc through SandboxedRunner, parses errors.

    Used by:
    - TscScorer (data quality scoring)
    - TypeCheckReward (RL training rewards)
    - BatchTypeChecker (batch RL evaluation)
    """

    # Regex for tsc error output: "filename(line,col): error TSxxxx: message"
    _ERROR_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$",
        re.MULTILINE,
    )

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 10,
        runner: SandboxedRunner | None = None,
        cache_size: int = 256,
    ) -> None:
        self._strict = strict
        self._timeout = timeout
        self._runner = runner or SandboxedRunner(timeout=timeout)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[TscError]] = OrderedDict()
        # Resolve the full tsc path (needed on Windows where tsc is a .CMD file
        # and subprocess.run won't find it without the full path or shell=True)
        self._tsc_path = shutil.which("tsc") or "tsc"

    def check(self, code: str) -> list[TscError]:
        """Type-check a single TypeScript file. Returns list of errors.

        Results are cached by MD5 hash.
        """
        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()

        # Check cache
        if code_hash in self._cache:
            self._cache.move_to_end(code_hash)
            return self._cache[code_hash]

        with tempfile.TemporaryDirectory(prefix="cola_tsc_") as tmpdir:
            # Write code file
            code_path = Path(tmpdir) / "check.ts"
            code_path.write_text(code, encoding="utf-8")

            # Write hardened tsconfig
            tsconfig = create_hardened_tsconfig(
                strict=self._strict,
                include_files=["check.ts"],
            )
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )

            # Run through SandboxedRunner
            result = self._runner.run(
                [self._tsc_path, "--project", ".", "--pretty", "false"],
                cwd=tmpdir,
                label="tsc",
                file_hash=code_hash,
            )

            all_output = (result.stdout or "") + "\n" + (result.stderr or "")
            errors = self._parse_errors(all_output)

            # Cache
            self._cache[code_hash] = errors
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

            return errors

    def check_batch(self, codes: list[str]) -> dict[int, list[TscError]]:
        """Type-check multiple files in a single tsc invocation.

        Args:
            codes: List of TypeScript source strings.

        Returns:
            Dict mapping index -> list of errors for that file.
        """
        if not codes:
            return {}

        with tempfile.TemporaryDirectory(prefix="cola_tsc_batch_") as tmpdir:
            filenames: list[str] = []
            for i, code in enumerate(codes):
                filename = f"check_{i}.ts"
                filepath = Path(tmpdir) / filename
                filepath.write_text(code, encoding="utf-8")
                filenames.append(filename)

            # Write hardened tsconfig with explicit include list
            tsconfig = create_hardened_tsconfig(
                strict=self._strict,
                include_files=filenames,
            )
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )

            # Run tsc ONCE through SandboxedRunner
            result = self._runner.run(
                [self._tsc_path, "--project", ".", "--pretty", "false"],
                cwd=tmpdir,
                label="tsc_batch",
            )

            all_output = (result.stdout or "") + "\n" + (result.stderr or "")
            per_file = self._parse_per_file_errors(all_output)

            # Map back to indices
            result_map: dict[int, list[TscError]] = {}
            for i, filename in enumerate(filenames):
                result_map[i] = per_file.get(filename, [])

            return result_map

    def _parse_errors(self, output: str) -> list[TscError]:
        """Parse tsc error output into structured error list."""
        errors: list[TscError] = []
        for match in self._ERROR_PATTERN.finditer(output):
            errors.append(TscError(
                file=match.group(1),
                line=int(match.group(2)),
                col=int(match.group(3)),
                severity=match.group(4),
                code=match.group(5),
                message=match.group(6),
            ))
        return errors

    def _parse_per_file_errors(self, output: str) -> dict[str, list[TscError]]:
        """Parse tsc output grouped by filename."""
        per_file: dict[str, list[TscError]] = {}
        for match in self._ERROR_PATTERN.finditer(output):
            filepath = match.group(1)
            filename = Path(filepath).name
            error = TscError(
                file=filename,
                line=int(match.group(2)),
                col=int(match.group(3)),
                severity=match.group(4),
                code=match.group(5),
                message=match.group(6),
            )
            per_file.setdefault(filename, []).append(error)
        return per_file

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        return shutil.which("tsc") is not None
