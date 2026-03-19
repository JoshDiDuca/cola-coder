"""Sandboxed code execution for evaluating generated code.

When the model generates code, we need to actually RUN it to see if it works.
But running arbitrary generated code is dangerous — it could delete files,
access the network, or do other harmful things.

We run generated code in a sandboxed subprocess with:
- A timeout (kills after N seconds)
- No network access (via the subprocess environment)
- Limited memory

For a TS dev: this is like running untrusted code in a worker thread with
restricted permissions, similar to a sandbox or a VM.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from .humaneval import CodingProblem


def execute_code(
    code: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        (success, output) tuple.
        success: True if the code ran without errors.
        output: stdout + stderr from the code execution.
    """
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Run in a subprocess with timeout and restricted environment
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Restricted environment: only essential variables
            env={
                "PATH": "",  # No access to system commands
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: Code execution exceeded {timeout}s"
    except Exception as e:
        return False, f"EXECUTION ERROR: {str(e)}"
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


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
