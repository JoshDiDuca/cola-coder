"""Sandboxed code execution for evaluating generated code.

When the model generates code (HumanEval, GRPO python_exec rewards), we need
to actually RUN it to see if it works. Running model-generated code is
dangerous in principle — the model is trained on untrusted internet data and
its output could touch files or the network.

Execution goes through the shared SandboxedRunner (the same isolation layer
used for scoring untrusted data):

- **native** mode (default): isolated temp dir, hard timeout, restricted
  environment (empty PATH). The subprocess still runs as your user — this
  limits accidents, not a determined attacker.
- **docker** mode (set ``scoring.security.mode: docker`` in
  configs/scoring.yaml): full container isolation — no network, read-only
  rootfs, memory/pids limits, all capabilities dropped, runs as nobody.

For a TS dev: native mode is like a worker thread with a timeout; docker
mode is the actual VM-style sandbox.
"""

import functools
import os
import sys
import tempfile
import textwrap
from pathlib import Path

from ..data.scorers.sandbox import SandboxedRunner
from .humaneval import CodingProblem

# Image used when docker mode is enabled. Overridable via
# scoring.security.python_docker_image in configs/scoring.yaml.
_PYTHON_DOCKER_IMAGE = "python:3.12-alpine"


@functools.lru_cache(maxsize=1)
def get_execution_runner() -> SandboxedRunner:
    """Shared SandboxedRunner for model-generated Python execution.

    Reads the ``security`` section of configs/scoring.yaml (same config that
    governs tsc/eslint scoring isolation): ``mode: docker`` switches from
    native temp-dir isolation to full container isolation. Cached because
    Docker availability detection costs a subprocess call and GRPO calls
    execute_code thousands of times per epoch.
    """
    mode = "native"
    memory_mb = 512
    image = _PYTHON_DOCKER_IMAGE
    cfg_path = Path("configs/scoring.yaml")
    if cfg_path.exists():
        try:
            import yaml

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            scoring = raw.get("scoring", raw) or {}
            security = scoring.get("security", {}) or {}
            mode = security.get("mode", "native")
            memory_mb = security.get("memory_mb", 512)
            image = security.get("python_docker_image", _PYTHON_DOCKER_IMAGE)
        except Exception:
            pass  # Fall back to native defaults — execution must not crash eval
    return SandboxedRunner(
        use_docker=(mode == "docker"),
        timeout=10,
        memory_mb=memory_mb,
        docker_image=image,
    )


def execute_code(
    code: str,
    timeout: float = 10.0,
    runner: SandboxedRunner | None = None,
) -> tuple[bool, str]:
    """Execute Python code in the sandbox.

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds.
        runner: Optional SandboxedRunner override (defaults to the shared
            runner configured by configs/scoring.yaml).

    Returns:
        (success, output) tuple.
        success: True if the code ran without errors.
        output: stdout + stderr from the code execution.
    """
    runner = runner or get_execution_runner()

    # Restricted environment for native mode (Docker isolates on its own):
    # empty PATH blocks resolving system commands by name. SYSTEMROOT must
    # survive on Windows or the Python interpreter itself misbehaves.
    env = {"PATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    if sys.platform == "win32" and os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    try:
        with tempfile.TemporaryDirectory(
            prefix="cola_exec_", ignore_cleanup_errors=True
        ) as tmpdir:
            (Path(tmpdir) / "main.py").write_text(code, encoding="utf-8")
            # Inside the container the interpreter is plain `python`;
            # natively we reuse the current interpreter.
            interpreter = "python" if runner.use_docker else sys.executable
            result = runner.run(
                [interpreter, "main.py"],
                cwd=tmpdir,
                label="python_exec",
                env=env,
                timeout=max(1, int(timeout)),
            )
    except Exception as e:
        return False, f"EXECUTION ERROR: {str(e)}"

    if result.returncode == -1:
        return False, f"TIMEOUT: Code execution exceeded {timeout}s"
    if result.returncode == -2:
        return False, f"EXECUTION ERROR: {result.stderr}"

    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def evaluate_solution(
    problem: CodingProblem,
    generated_code: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Evaluate a generated solution against test cases.

    The generated code should complete the function body. We combine:
    1. The model's generated code (the function implementation)
    2. The test cases from the problem definition

    Args:
        problem: The coding problem with test cases.
        generated_code: The model's generated code (function body).
        timeout: Maximum execution time.

    Returns:
        (passed, output) tuple.
        passed: True if all test cases pass.
        output: Execution output or error messages.
    """
    # A problem with no test code can't be verified — running just the
    # generated code with no assertions would exit 0 and falsely count as a
    # pass, inflating pass@k. Treat "unverifiable" as NOT passed.
    if not problem.test_code or not problem.test_code.strip():
        return False, "NO TESTS: empty test_code — solution cannot be verified"

    # Combine the generated code with the test cases
    full_code = generated_code + "\n\n" + textwrap.dedent(problem.test_code)

    return execute_code(full_code, timeout=timeout)


def extract_function(generated_text: str, entry_point: str) -> str:
    """Extract the generated function from the model's output.

    The model generates text that starts with the function signature (from the prompt)
    and continues with the implementation. We need to extract just the complete
    function definition.

    Args:
        generated_text: The full text output from the model.
        entry_point: The function name to look for.

    Returns:
        The extracted function code.
    """
    lines = generated_text.split("\n")
    result_lines = []
    in_function = False
    indent_level = None

    for line in lines:
        # Find the start of our function
        if f"def {entry_point}" in line:
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue

        if in_function:
            # Check if we've left the function (non-empty line at same or lower indent)
            stripped = line.lstrip()
            if stripped and not line.startswith(" " * (indent_level + 1)):
                # Could be a new top-level definition or class
                if stripped.startswith("def ") or stripped.startswith("class "):
                    break
                # Allow if it's a continuation at top level
                if len(line) - len(stripped) <= indent_level and stripped:
                    break

            result_lines.append(line)

    return "\n".join(result_lines)
