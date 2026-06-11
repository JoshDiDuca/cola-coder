"""Safe tool execution with timeouts and sandboxing.

Executes tool calls from the model in a controlled environment.
All execution happens in subprocesses with timeouts to prevent
runaway processes.

SECURITY: Never executes arbitrary shell commands. Only registered
tools with specific handlers are allowed.
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0.0


class ToolExecutor:
    """Safely execute tool calls.

    Each tool has a specific handler function. No arbitrary
    command execution is allowed.
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        timeout: int = 30,
        max_output_chars: int = 5000,
    ):
        """
        Args:
            project_root: Root directory for relative paths
            timeout: Max execution time per tool call (seconds)
            max_output_chars: Truncate output beyond this
        """
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        self.max_output_chars = max_output_chars

        # Built-in handlers
        self._handlers: dict[str, Any] = {
            "run_tests": self._handle_run_tests,
            "lint": self._handle_lint,
            "typecheck": self._handle_typecheck,
            "read_file": self._handle_read_file,
            "git_diff": self._handle_git_diff,
            "git_log": self._handle_git_log,
            "search_code": self._handle_search_code,
        }

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments dict

        Returns:
            ToolResult with output or error
        """
        start = time.perf_counter()

        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        try:
            output = handler(arguments)
            elapsed = (time.perf_counter() - start) * 1000

            # Truncate long output
            if len(output) > self.max_output_chars:
                output = output[: self.max_output_chars] + "\n... (truncated)"

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                execution_time_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool timed out after {self.timeout}s",
                execution_time_ms=self.timeout * 1000,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=str(e),
                execution_time_ms=elapsed,
            )

    def _run_subprocess(self, cmd: list[str], cwd: str | Path | None = None) -> str:
        """Run a subprocess with timeout. Returns stdout."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=cwd or self.project_root,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve a file path safely.

        Prevents path traversal outside project root.

        SECURITY: must use Path containment, NOT str.startswith — a string
        prefix check lets a SIBLING directory bypass it (e.g. project root
        ".../cola-coder" would accept ".../cola-coder-secrets/x" because the
        string starts with the root). is_relative_to compares path components,
        so a sibling is correctly rejected.
        """
        resolved = (self.project_root / path).resolve()
        if resolved != self.project_root and not resolved.is_relative_to(self.project_root):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    # --- Tool handlers ---

    def _handle_run_tests(self, args: dict) -> str:
        """Run pytest on specified path."""
        test_path = args.get("test_path", "tests/")
        verbose = args.get("verbose", False)

        self._validate_path(test_path)

        cmd = [str(self.project_root / ".venv" / "Scripts" / "pytest"), test_path]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "--no-header", "-q"])

        return self._run_subprocess(cmd)

    def _handle_lint(self, args: dict) -> str:
        """Run ruff linter."""
        file_path = args.get("file_path", "src/")
        fix = args.get("fix", False)

        self._validate_path(file_path)

        cmd = [str(self.project_root / ".venv" / "Scripts" / "ruff"), "check", file_path]
        if fix:
            cmd.append("--fix")

        return self._run_subprocess(cmd)

    def _handle_typecheck(self, args: dict) -> str:
        """Run TypeScript type checking on code string."""
        code = args.get("code", "")
        strict = args.get("strict", True)

        if not code:
            return "Error: no code provided"

        # Write to temp file and run tsc
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            cmd = ["npx", "tsc", "--noEmit", temp_path]
            if strict:
                cmd.insert(2, "--strict")
            return self._run_subprocess(cmd)
        finally:
            os.unlink(temp_path)

    def _handle_read_file(self, args: dict) -> str:
        """Read a file's contents."""
        path = args.get("path", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        if not path:
            return "Error: no path provided"

        resolved = self._validate_path(path)

        if not resolved.exists():
            return f"Error: file not found: {path}"

        content = resolved.read_text(encoding="utf-8", errors="ignore")

        if start_line or end_line:
            lines = content.split("\n")
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            content = "\n".join(lines[start:end])

        return content

    def _handle_git_diff(self, args: dict) -> str:
        """Show git diff."""
        ref = args.get("ref", "HEAD")
        file_path = args.get("file_path")

        # Sanitize ref to prevent injection. A leading "-" is rejected because
        # `git diff <ref>` would treat it as a FLAG, not a ref (e.g. --ext-diff
        # can run an external diff driver) — argument injection even though the
        # char set is otherwise restricted.
        if not ref or ref.startswith("-"):
            return "Error: invalid git ref"
        if not all(c.isalnum() or c in ".-_/~^" for c in ref):
            return "Error: invalid git ref"

        cmd = ["git", "diff", ref]
        if file_path:
            self._validate_path(file_path)
            cmd.extend(["--", file_path])

        return self._run_subprocess(cmd)

    def _handle_git_log(self, args: dict) -> str:
        """Show git log."""
        count = min(args.get("count", 5), 20)  # Cap at 20

        cmd = [
            "git",
            "log",
            f"-{count}",
            "--oneline",
            "--no-color",
        ]

        return self._run_subprocess(cmd)

    def _handle_search_code(self, args: dict) -> str:
        """Search code using git grep."""
        query = args.get("query", "")
        max_results = min(args.get("max_results", 5), 20)

        if not query:
            return "Error: no query provided"

        # Sanitize query for safety
        safe_query = query.replace(";", "").replace("|", "").replace("&", "")

        cmd = [
            "git",
            "grep",
            "-n",
            "-I",
            "--max-count",
            str(max_results),
            safe_query,
        ]

        return self._run_subprocess(cmd)
