"""Batch type checking for GRPO groups.

Instead of spawning tsc per file, writes ALL generated files to a temp
directory with a tsconfig.json and runs tsc ONCE.

Speed: ~200ms for 16 files (vs ~800ms spawning 16 processes).

This is the preferred method during GRPO training where we generate
groups of 8-16 solutions and need to score them all.

For a TS dev: this is like having a monorepo with 16 files and running
tsc once — much faster than running tsc 16 times separately.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile

from .type_check import TypeCheckReward, _ERROR_LINE_RE, SYNTAX_ERROR_RANGE

logger = logging.getLogger(__name__)


class BatchTypeChecker:
    """Fast batch type checking for GRPO groups.

    Writes all generated files to a temp directory with a tsconfig.json
    and runs tsc ONCE, then parses per-file errors from the output.
    """

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 30,
    ):
        """Initialize the batch type checker.

        Args:
            strict: Use --strict mode.
            timeout: Timeout in seconds for the batch tsc run.
        """
        self.strict = strict
        self.timeout = timeout
        self._tsc_path = self._find_tsc()

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        return TypeCheckReward.is_available()

    def score_batch(self, codes: list[str]) -> list[float]:
        """Type-check a batch of code files simultaneously.

        Args:
            codes: List of TypeScript code strings.

        Returns:
            List of float scores (same length as codes).
        """
        results = self.detailed_batch(codes)
        return [r["score"] for r in results]

    def detailed_batch(self, codes: list[str]) -> list[dict]:
        """Return detailed diagnostics for each file in batch.

        Args:
            codes: List of TypeScript code strings.

        Returns:
            List of dicts with score, num_errors, errors, etc.
        """
        if self._tsc_path is None:
            logger.warning("tsc not available — returning zero scores")
            return [
                {"score": 0.0, "num_errors": -1, "errors": [], "tsc_failed": True}
                for _ in codes
            ]

        if not codes:
            return []

        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="cola_batch_")

            # Write all files with predictable names
            file_names = []
            for i, code in enumerate(codes):
                fname = f"gen_{i}.ts"
                file_names.append(fname)
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(code)

            # Write tsconfig.json
            tsconfig = {
                "compilerOptions": {
                    "strict": self.strict,
                    "noEmit": True,
                    "target": "ES2022",
                    "module": "ESNext",
                    "moduleResolution": "bundler",
                    "skipLibCheck": True,
                },
                "include": ["*.ts"],
            }
            tsconfig_path = os.path.join(tmpdir, "tsconfig.json")
            with open(tsconfig_path, "w", encoding="utf-8") as f:
                json.dump(tsconfig, f)

            # Run tsc once on the whole project
            cmd = [self._tsc_path, "--project", tmpdir, "--pretty", "false"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tmpdir,
            )

            # Parse per-file errors
            per_file_errors = self._parse_batch_output(
                result.stdout + result.stderr, file_names
            )

            # Check if any file had syntax errors — if so, tsc may have
            # skipped type-checking other files in the batch. In that case,
            # fall back to individual checking for files with 0 reported errors.
            any_syntax_errors = any(
                any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)
                for errors in per_file_errors.values()
                if errors
            )

            # Convert to result dicts
            results = []
            single_checker = None
            for i, fname in enumerate(file_names):
                errors = per_file_errors.get(fname, [])

                # If batch had syntax errors in other files and this file
                # reported 0 errors, re-check individually to be accurate
                if any_syntax_errors and not errors:
                    if single_checker is None:
                        single_checker = TypeCheckReward(
                            strict=self.strict, timeout=self.timeout
                        )
                    individual = single_checker._run_tsc(codes[i])
                    if individual is not None:
                        errors = individual

                score = self._errors_to_score(errors)
                has_syntax = any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)
                results.append({
                    "score": score,
                    "num_errors": len(errors),
                    "errors": errors,
                    "error_codes": [f"TS{e['code']}" for e in errors],
                    "has_syntax_errors": has_syntax,
                    "tsc_failed": False,
                })

            return results

        except subprocess.TimeoutExpired:
            logger.warning("Batch tsc timed out after %d seconds", self.timeout)
            return [
                {"score": 0.0, "num_errors": -1, "errors": [], "tsc_failed": True}
                for _ in codes
            ]
        except (FileNotFoundError, OSError) as e:
            logger.error("Failed to run batch tsc: %s", e)
            return [
                {"score": 0.0, "num_errors": -1, "errors": [], "tsc_failed": True}
                for _ in codes
            ]
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass

    def _parse_batch_output(
        self, output: str, file_names: list[str]
    ) -> dict[str, list[dict]]:
        """Parse tsc output and group errors by file.

        Returns:
            Dict mapping filename to list of error dicts.
        """
        per_file: dict[str, list[dict]] = {f: [] for f in file_names}

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _ERROR_LINE_RE.match(line)
            if match:
                raw_file = match.group(1)
                # Extract just the filename (tsc may give absolute or relative path)
                basename = os.path.basename(raw_file)
                error = {
                    "file": basename,
                    "line": int(match.group(2)),
                    "col": int(match.group(3)),
                    "code": int(match.group(4)),
                    "message": match.group(5),
                }
                if basename in per_file:
                    per_file[basename].append(error)

        return per_file

    @staticmethod
    def _errors_to_score(errors: list[dict]) -> float:
        """Convert error list to a score (same logic as TypeCheckReward)."""
        if not errors:
            return 1.0

        has_syntax = any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)
        if has_syntax:
            return -0.5

        n = len(errors)
        if n <= 2:
            return 0.7
        elif n <= 5:
            return 0.3
        else:
            return 0.0

    @staticmethod
    def _find_tsc() -> str | None:
        """Find the tsc binary path."""
        tsc = shutil.which("tsc")
        if tsc:
            return tsc
        npx = shutil.which("npx")
        if npx:
            try:
                result = subprocess.run(
                    ["npx", "tsc", "--version"],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    return "tsc"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        return None
