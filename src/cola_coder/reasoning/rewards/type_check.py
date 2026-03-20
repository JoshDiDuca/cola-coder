"""TypeScript compiler as GRPO reward function.

Uses `tsc --noEmit --strict` to score generated TypeScript code.
This is a FREE, FAST (~50ms per file), DETERMINISTIC reward signal.

Requirements:
- Node.js installed
- TypeScript installed: npm install -g typescript

If tsc is not available, the reward function degrades gracefully.

Error severity classification:
- Syntax errors (TS1XXX): Code doesn't even parse — worst score
- Type errors (TS2XXX): Type system violations — graduated penalty
- Semantic errors (TS7XXX): Implicit any, unused vars — minor penalty

For a TS dev: think of this like running your CI type-check step, but
as a reward signal for RL. The model learns to write code that passes
tsc --strict, just like you learn to write code that passes CI.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict

logger = logging.getLogger(__name__)

# --- Error classification ---

# Syntax errors: code doesn't parse at all
SYNTAX_ERROR_RANGE = range(1000, 2000)

# Type errors: the meat of type checking
TYPE_ERROR_RANGE = range(2000, 3000)

# Semantic / strictness errors (implicit any, etc.)
SEMANTIC_ERROR_RANGE = range(7000, 8000)

# Error code regex: "error TS2322:" -> 2322
_ERROR_CODE_RE = re.compile(r"error TS(\d+):")

# Full error line regex: "file.ts(line,col): error TS2322: message"
_ERROR_LINE_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+error\s+TS(\d+):\s+(.+)$"
)


class TypeCheckReward:
    """Use TypeScript compiler as GRPO reward function.

    Scores generated TypeScript code by running tsc --noEmit --strict.

    Score mapping:
        1.0  = no errors (perfect type check)
        0.7  = 1-2 errors (minor issues)
        0.3  = 3-5 errors (moderate issues)
        0.0  = 6+ errors (major issues)
       -0.5  = syntax errors (doesn't parse)

    The score is designed for GRPO: the group of solutions will have
    variance in scores, allowing the policy gradient to learn which
    code patterns produce type-safe output.
    """

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 10,
        cache_size: int = 256,
    ):
        """Initialize the type check reward.

        Args:
            strict: Use --strict mode (recommended for training).
            timeout: Timeout in seconds for tsc subprocess.
            cache_size: Max number of cached scores (LRU).
        """
        self.strict = strict
        self.timeout = timeout
        self._tsc_path = self._find_tsc()
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._cache_size = cache_size

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed and accessible."""
        tsc = shutil.which("tsc")
        if tsc:
            return True
        # Also check npx tsc
        npx = shutil.which("npx")
        if npx:
            try:
                result = subprocess.run(
                    ["npx", "tsc", "--version"],
                    capture_output=True, text=True, timeout=15,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        return False

    def score(self, code: str) -> float:
        """Score generated TypeScript code.

        Returns a float reward value:
            1.0 (perfect), 0.7 (minor), 0.3 (moderate),
            0.0 (major), -0.5 (syntax error)
        """
        errors = self._run_tsc(code)

        if errors is None:
            # tsc crashed or timed out
            return 0.0

        if not errors:
            return 1.0

        # Check for syntax errors (TS1XXX range)
        has_syntax_error = any(
            e["code"] in SYNTAX_ERROR_RANGE for e in errors
        )
        if has_syntax_error:
            return -0.5

        num_errors = len(errors)
        if num_errors <= 2:
            return 0.7
        elif num_errors <= 5:
            return 0.3
        else:
            return 0.0

    def detailed_score(self, code: str) -> dict:
        """Return score + detailed diagnostics.

        Useful for analysis: which type errors does the model make most?
        The error_codes field maps to TypeScript diagnostics, e.g.:
        - TS2322: Type 'X' is not assignable to type 'Y'
        - TS2339: Property 'X' does not exist on type 'Y'
        - TS2345: Argument type mismatch
        - TS7006: Parameter implicitly has 'any' type
        """
        errors = self._run_tsc(code)

        if errors is None:
            return {
                "score": 0.0,
                "num_errors": -1,
                "errors": [],
                "error_codes": [],
                "has_syntax_errors": False,
                "tsc_failed": True,
            }

        score = self.score(code)
        has_syntax = any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)

        return {
            "score": score,
            "num_errors": len(errors),
            "errors": errors,
            "error_codes": [f"TS{e['code']}" for e in errors],
            "has_syntax_errors": has_syntax,
            "tsc_failed": False,
        }

    def _run_tsc(self, code: str) -> list[dict] | None:
        """Write code to temp file, run tsc --noEmit --strict, parse output.

        Returns:
            List of error dicts, or None if tsc crashed/timed out.
        """
        if self._tsc_path is None:
            logger.warning("tsc not available — returning None")
            return None

        # Check cache
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash in self._cache:
            self._cache.move_to_end(code_hash)
            return self._cache[code_hash]

        tmp_path = None
        try:
            # Write to temp file (don't auto-delete so tsc can read it)
            fd, tmp_path = tempfile.mkstemp(suffix=".ts", prefix="cola_check_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            # Build tsc command
            cmd = [self._tsc_path, "--noEmit", "--pretty", "false"]
            if self.strict:
                cmd.append("--strict")
            # Target modern ES to avoid downlevel errors
            cmd.extend(["--target", "ES2022"])
            cmd.extend(["--module", "ESNext"])
            cmd.extend(["--moduleResolution", "bundler"])
            # Skip lib check to avoid errors from .d.ts files
            cmd.append("--skipLibCheck")
            cmd.append(tmp_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )

            errors = self._parse_errors(result.stdout + result.stderr)

            # Cache the result
            self._cache[code_hash] = errors
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

            return errors

        except subprocess.TimeoutExpired:
            logger.warning("tsc timed out after %d seconds", self.timeout)
            return None
        except (FileNotFoundError, OSError) as e:
            logger.error("Failed to run tsc: %s", e)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _parse_errors(self, output: str) -> list[dict]:
        """Parse tsc output into structured error list.

        tsc output format (with --pretty false):
            file.ts(line,col): error TS2322: Type 'X' is not assignable to type 'Y'
        """
        errors = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _ERROR_LINE_RE.match(line)
            if match:
                errors.append({
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "col": int(match.group(3)),
                    "code": int(match.group(4)),
                    "message": match.group(5),
                })
            elif "error TS" in line:
                # Catch errors without file location
                code_match = _ERROR_CODE_RE.search(line)
                if code_match:
                    errors.append({
                        "file": "",
                        "line": 0,
                        "col": 0,
                        "code": int(code_match.group(1)),
                        "message": line,
                    })
        return errors

    @staticmethod
    def _find_tsc() -> str | None:
        """Find the tsc binary path."""
        # Try direct tsc first (globally installed)
        tsc = shutil.which("tsc")
        if tsc:
            return tsc
        # Fall back to npx tsc (slower but works)
        npx = shutil.which("npx")
        if npx:
            # Verify it works
            try:
                result = subprocess.run(
                    ["npx", "tsc", "--version"],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    return "tsc"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        logger.warning(
            "TypeScript compiler (tsc) not found. "
            "Install with: npm install -g typescript"
        )
        return None
