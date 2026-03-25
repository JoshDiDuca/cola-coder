"""ESLint scorer — score TypeScript/JavaScript files using ESLint."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner


class EslintScorer:
    """Score TypeScript/JavaScript code quality using ESLint."""

    name: str = "eslint"

    # Score mapping: error/warning count -> quality score
    _SCORE_MAP: list[tuple[int, float]] = [
        (0, 1.0),     # 0 issues = perfect
        (2, 0.9),     # 1-2 issues = great
        (5, 0.7),     # 3-5 issues = good
        (10, 0.5),    # 6-10 issues = average
        (20, 0.3),    # 11-20 issues = poor
    ]
    # 21+ issues = 0.1

    def __init__(
        self,
        timeout: int = 15,
        runner: SandboxedRunner | None = None,
    ) -> None:
        self._timeout = timeout
        self._runner = runner or SandboxedRunner(timeout=timeout)

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        """Score a single code sample with ESLint."""
        if not self._is_js_ts(code, metadata):
            return ScorerResult(
                score=0.5, scorer_name=self.name,
                details={"skipped": True, "reason": "not_js_ts"},
            )

        ext = self._detect_extension(metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / f"file{ext}"
            filepath.write_text(code, encoding="utf-8")

            result = self._run_eslint(tmpdir, [str(filepath)])
            if result is None:
                return ScorerResult(
                    score=0.5, scorer_name=self.name,
                    details={"skipped": True, "reason": "eslint_failed"},
                )

            return self._parse_result(result)

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score multiple samples — writes all to temp dir, runs eslint ONCE."""
        if not items:
            return []

        # Separate JS/TS from non-JS/TS
        js_ts_indices: list[int] = []
        results: list[ScorerResult | None] = [None] * len(items)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_map: dict[str, int] = {}  # filename -> original index

            for i, (code, metadata) in enumerate(items):
                if not self._is_js_ts(code, metadata):
                    results[i] = ScorerResult(
                        score=0.5, scorer_name=self.name,
                        details={"skipped": True, "reason": "not_js_ts"},
                    )
                    continue

                ext = self._detect_extension(metadata)
                filename = f"file_{i}{ext}"
                filepath = Path(tmpdir) / filename
                filepath.write_text(code, encoding="utf-8")
                file_map[str(filepath)] = i
                js_ts_indices.append(i)

            if not js_ts_indices:
                return [r for r in results if r is not None]

            # Run eslint once on the entire temp directory
            eslint_result = self._run_eslint(tmpdir, list(file_map.keys()))

            if eslint_result is None:
                # ESLint failed entirely — return neutral for all
                for i in js_ts_indices:
                    results[i] = ScorerResult(
                        score=0.5, scorer_name=self.name,
                        details={"skipped": True, "reason": "eslint_failed"},
                    )
            else:
                # Parse per-file results
                per_file = self._parse_per_file(eslint_result)
                for filepath_str, idx in file_map.items():
                    if filepath_str in per_file:
                        results[idx] = per_file[filepath_str]
                    else:
                        # File not in output — assume clean
                        results[idx] = ScorerResult(
                            score=1.0, scorer_name=self.name,
                            details={"error_count": 0, "warning_count": 0},
                        )

        return [r if r is not None else ScorerResult(score=0.5, scorer_name=self.name) for r in results]

    def _run_eslint(self, cwd: str, files: list[str]) -> str | None:
        """Run eslint on files and return JSON stdout, or None on failure."""
        # Try eslint directly, then fall back to npx
        for cmd_prefix in [["eslint"], ["npx", "eslint"]]:
            cmd = [
                *cmd_prefix,
                "--format", "json",
                "--no-eslintrc",
                *files,
            ]
            result = self._runner.run(cmd, cwd=cwd)
            # eslint returns exit code 1 when there are lint errors (not a failure)
            if result.returncode in (0, 1) and result.stdout.strip():
                return result.stdout
        return None

    def _parse_result(self, json_output: str) -> ScorerResult:
        """Parse eslint JSON output for a single file."""
        try:
            data = json.loads(json_output)
            if not isinstance(data, list) or not data:
                return ScorerResult(score=0.5, scorer_name=self.name, details={"parse_error": True})

            file_result = data[0]
            error_count = int(file_result.get("errorCount", 0))
            warning_count = int(file_result.get("warningCount", 0))
            total = error_count + warning_count

            return ScorerResult(
                score=self._issues_to_score(total),
                scorer_name=self.name,
                details={
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "total_issues": total,
                },
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            return ScorerResult(score=0.5, scorer_name=self.name, details={"parse_error": True})

    def _parse_per_file(self, json_output: str) -> dict[str, ScorerResult]:
        """Parse eslint JSON output with multiple files."""
        result_map: dict[str, ScorerResult] = {}
        try:
            data = json.loads(json_output)
            if not isinstance(data, list):
                return result_map

            for file_result in data:
                filepath = file_result.get("filePath", "")
                error_count = int(file_result.get("errorCount", 0))
                warning_count = int(file_result.get("warningCount", 0))
                total = error_count + warning_count

                result_map[filepath] = ScorerResult(
                    score=self._issues_to_score(total),
                    scorer_name=self.name,
                    details={
                        "error_count": error_count,
                        "warning_count": warning_count,
                        "total_issues": total,
                    },
                )
        except (json.JSONDecodeError, KeyError):
            pass
        return result_map

    @classmethod
    def _issues_to_score(cls, total_issues: int) -> float:
        """Map total issue count to 0.0-1.0 score."""
        for threshold, score in cls._SCORE_MAP:
            if total_issues <= threshold:
                return score
        return 0.1  # 21+ issues

    @staticmethod
    def is_available() -> bool:
        """Check if eslint is installed."""
        return (
            shutil.which("eslint") is not None
            or shutil.which("npx") is not None
        )

    @staticmethod
    def _is_js_ts(code: str, metadata: dict[str, object] | None) -> bool:
        """Detect if code is JavaScript/TypeScript."""
        if metadata:
            lang = str(metadata.get("language", "")).lower()
            if lang in ("typescript", "javascript", "ts", "js", "tsx", "jsx"):
                return True
            file_path = str(metadata.get("file_path", ""))
            if file_path:
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                if ext in ("ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"):
                    return True

        ts_indicators = ["const ", "let ", "import ", "export ", "=> {", "function "]
        matches = sum(1 for ind in ts_indicators if ind in code)
        return matches >= 2

    @staticmethod
    def _detect_extension(metadata: dict[str, object] | None) -> str:
        """Detect appropriate file extension from metadata."""
        if metadata:
            file_path = str(metadata.get("file_path", ""))
            if file_path and "." in file_path:
                ext = "." + file_path.rsplit(".", 1)[-1].lower()
                if ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"):
                    return ext
            lang = str(metadata.get("language", "")).lower()
            if lang in ("typescript", "ts"):
                return ".ts"
            if lang in ("javascript", "js"):
                return ".js"
        return ".ts"  # Default to TypeScript
