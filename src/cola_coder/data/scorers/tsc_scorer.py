"""TypeScript compiler scorer -- runs tsc via SandboxedRunner with hardened tsconfig."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig


class TscScorer:
    """Score TypeScript files using tsc --noEmit via SandboxedRunner.

    Security: runs tsc in an isolated temp directory with a hardened tsconfig.json
    that blocks plugin execution, @types loading, and path traversal.
    """

    name: str = "tsc"

    # Score mapping: error count -> quality score
    _SCORE_MAP: list[tuple[int, float]] = [
        (0, 1.0),     # No errors = perfect
        (1, 0.8),     # 1 error = good
        (3, 0.6),     # 2-3 errors = decent
        (5, 0.4),     # 4-5 errors = average
        (10, 0.2),    # 6-10 errors = poor
    ]
    # 11+ errors = 0.1

    # Regex to parse tsc error output: "filename(line,col): error TSxxxx: message"
    _ERROR_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$",
        re.MULTILINE,
    )

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 10,
        cache_size: int = 256,
        runner: SandboxedRunner | None = None,
    ) -> None:
        self._strict = strict
        self._timeout = timeout
        self._cache_size = cache_size
        self._runner = runner or SandboxedRunner(timeout=timeout)
        self._cache: OrderedDict[str, list[dict[str, object]]] = OrderedDict()

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        """Score a single TypeScript code sample."""
        if not self._is_typescript(code, metadata):
            return ScorerResult(
                score=0.5, scorer_name=self.name,
                details={"skipped": True, "reason": "not_typescript"},
            )

        errors = self._sandboxed_tsc(code)
        num_errors = len(errors) if errors is not None else 0
        has_syntax = any(
            e.get("code", "").startswith("TS1") for e in (errors or [])
        )

        score = self._errors_to_score(num_errors)
        if has_syntax:
            score = min(score, 0.3)  # Penalize syntax errors more heavily

        return ScorerResult(
            score=score,
            scorer_name=self.name,
            details={
                "num_errors": num_errors,
                "has_syntax_errors": has_syntax,
                "errors": errors[:5] if errors else [],  # First 5 for debugging
            },
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score multiple samples -- write all to temp dir, run tsc ONCE through runner."""
        if not items:
            return []

        # Separate TS from non-TS
        results: list[ScorerResult | None] = [None] * len(items)
        ts_indices: list[int] = []

        for i, (code, metadata) in enumerate(items):
            if not self._is_typescript(code, metadata):
                results[i] = ScorerResult(
                    score=0.5, scorer_name=self.name,
                    details={"skipped": True, "reason": "not_typescript"},
                )
            else:
                ts_indices.append(i)

        if not ts_indices:
            return [r for r in results if r is not None]

        # Write all TS files to a single temp dir
        with tempfile.TemporaryDirectory(prefix="cola_tsc_batch_") as tmpdir:
            file_map: dict[str, int] = {}  # filename -> original index

            for i in ts_indices:
                code, _ = items[i]
                filename = f"check_{i}.ts"
                filepath = Path(tmpdir) / filename
                filepath.write_text(code, encoding="utf-8")
                file_map[filename] = i

            # Write hardened tsconfig
            include_files = list(file_map.keys())
            tsconfig = create_hardened_tsconfig(
                strict=self._strict,
                include_files=include_files,
            )
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )

            # Run tsc ONCE through SandboxedRunner
            result = self._runner.run(
                ["tsc", "--project", ".", "--pretty", "false"],
                cwd=tmpdir,
                label="tsc_batch",
            )

            # Parse per-file errors
            all_output = (result.stdout or "") + "\n" + (result.stderr or "")
            per_file_errors = self._parse_per_file_errors(all_output)

            for filename, idx in file_map.items():
                file_errors = per_file_errors.get(filename, [])
                num_errors = len(file_errors)
                has_syntax = any(
                    e.get("code", "").startswith("TS1") for e in file_errors
                )
                score = self._errors_to_score(num_errors)
                if has_syntax:
                    score = min(score, 0.3)

                results[idx] = ScorerResult(
                    score=score,
                    scorer_name=self.name,
                    details={
                        "num_errors": num_errors,
                        "has_syntax_errors": has_syntax,
                        "errors": file_errors[:3],
                    },
                )

        return [r if r is not None else ScorerResult(score=0.5, scorer_name=self.name) for r in results]

    def _sandboxed_tsc(self, code: str) -> list[dict[str, object]] | None:
        """Write code to temp dir, write hardened tsconfig, run tsc through runner."""
        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()

        # Check cache
        if code_hash in self._cache:
            self._cache.move_to_end(code_hash)
            return self._cache[code_hash]

        with tempfile.TemporaryDirectory(prefix="cola_tsc_") as tmpdir:
            # Write the code file
            code_path = Path(tmpdir) / "check.ts"
            code_path.write_text(code, encoding="utf-8")

            # Write HARDENED tsconfig (plugins=[], types=[], typeRoots=[])
            tsconfig = create_hardened_tsconfig(
                strict=self._strict,
                include_files=["check.ts"],
            )
            (Path(tmpdir) / "tsconfig.json").write_text(
                json.dumps(tsconfig), encoding="utf-8",
            )

            # Run through SandboxedRunner (NOT subprocess.run)
            result = self._runner.run(
                ["tsc", "--project", ".", "--pretty", "false"],
                cwd=tmpdir,
                label="tsc",
                file_hash=code_hash,
            )

            all_output = (result.stdout or "") + "\n" + (result.stderr or "")
            errors = self._parse_errors(all_output)

            # Cache result
            self._cache[code_hash] = errors
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

            return errors

    def _parse_errors(self, output: str) -> list[dict[str, object]]:
        """Parse tsc error output into structured error list."""
        errors: list[dict[str, object]] = []
        for match in self._ERROR_PATTERN.finditer(output):
            errors.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "col": int(match.group(3)),
                "severity": match.group(4),
                "code": match.group(5),
                "message": match.group(6),
            })
        return errors

    def _parse_per_file_errors(self, output: str) -> dict[str, list[dict[str, object]]]:
        """Parse tsc output into per-file error lists."""
        per_file: dict[str, list[dict[str, object]]] = {}
        for match in self._ERROR_PATTERN.finditer(output):
            # Extract just the filename from the path
            filepath = match.group(1)
            filename = Path(filepath).name
            error: dict[str, object] = {
                "file": filename,
                "line": int(match.group(2)),
                "col": int(match.group(3)),
                "severity": match.group(4),
                "code": match.group(5),
                "message": match.group(6),
            }
            per_file.setdefault(filename, []).append(error)
        return per_file

    @classmethod
    def _errors_to_score(cls, num_errors: int) -> float:
        """Map error count to 0.0-1.0 score."""
        for threshold, score in cls._SCORE_MAP:
            if num_errors <= threshold:
                return score
        return 0.1  # 11+ errors

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        return shutil.which("tsc") is not None

    @staticmethod
    def _is_typescript(code: str, metadata: dict[str, object] | None) -> bool:
        """Detect if code is TypeScript."""
        if metadata:
            lang = str(metadata.get("language", "")).lower()
            if lang in ("typescript", "ts", "tsx"):
                return True
            file_path = str(metadata.get("file_path", ""))
            if file_path:
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                if ext in ("ts", "tsx", "mts", "cts"):
                    return True

        # Heuristic: look for TypeScript-specific patterns
        ts_indicators = [": string", ": number", ": boolean", "interface ",
                         ": void", "as const", "<T>", "readonly ", "enum "]
        matches = sum(1 for ind in ts_indicators if ind in code)
        return matches >= 2
