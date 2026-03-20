"""Nano Benchmark: 10 ultra-simple TypeScript coding problems.

These problems are designed to give signal even for small models (50M params)
early in training. Each problem tests basic TypeScript/JavaScript patterns
that should emerge before the model can handle HumanEval-level tasks.

Problems are scored on:
1. Syntax validity (does it parse?)
2. Type correctness (does tsc accept it?)
3. Test execution (does it produce correct output?)
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED

@dataclass
class NanoProblem:
    """A single nano-difficulty coding problem."""
    task_id: str
    description: str
    prompt: str  # What the model sees
    expected_pattern: str  # Regex or substring to check in output
    test_code: str  # TypeScript test code
    difficulty: int  # 1-3 (1=trivial, 3=easy)

# 10 ultra-simple problems
NANO_PROBLEMS: list[NanoProblem] = [
    NanoProblem(
        task_id="nano_hello",
        description="Return 'Hello, World!'",
        prompt='function hello(): string {\n  // Return "Hello, World!"\n',
        expected_pattern="Hello, World!",
        test_code='console.log(hello() === "Hello, World!" ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_add",
        description="Add two numbers",
        prompt="function add(a: number, b: number): number {\n  // Return the sum of a and b\n",
        expected_pattern="return",
        test_code='console.log(add(2, 3) === 5 ? "PASS" : "FAIL");\nconsole.log(add(0, 0) === 0 ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_identity",
        description="Return the input unchanged",
        prompt="function identity<T>(x: T): T {\n  // Return x unchanged\n",
        expected_pattern="return x",
        test_code='console.log(identity(42) === 42 ? "PASS" : "FAIL");\nconsole.log(identity("hi") === "hi" ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_is_even",
        description="Check if a number is even",
        prompt="function isEven(n: number): boolean {\n  // Return true if n is even\n",
        expected_pattern="return",
        test_code='console.log(isEven(4) === true ? "PASS" : "FAIL");\nconsole.log(isEven(3) === false ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_max",
        description="Return the larger of two numbers",
        prompt="function max(a: number, b: number): number {\n  // Return the larger of a and b\n",
        expected_pattern="return",
        test_code='console.log(max(3, 7) === 7 ? "PASS" : "FAIL");\nconsole.log(max(10, 2) === 10 ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_length",
        description="Return the length of a string",
        prompt="function strLength(s: string): number {\n  // Return the length of the string\n",
        expected_pattern="length",
        test_code='console.log(strLength("hello") === 5 ? "PASS" : "FAIL");\nconsole.log(strLength("") === 0 ? "PASS" : "FAIL");',
        difficulty=1,
    ),
    NanoProblem(
        task_id="nano_array_first",
        description="Return the first element of an array",
        prompt="function first(arr: number[]): number | undefined {\n  // Return the first element\n",
        expected_pattern="return",
        test_code='console.log(first([1, 2, 3]) === 1 ? "PASS" : "FAIL");\nconsole.log(first([]) === undefined ? "PASS" : "FAIL");',
        difficulty=2,
    ),
    NanoProblem(
        task_id="nano_reverse",
        description="Reverse a string",
        prompt="function reverseStr(s: string): string {\n  // Return the string reversed\n",
        expected_pattern="return",
        test_code='console.log(reverseStr("abc") === "cba" ? "PASS" : "FAIL");\nconsole.log(reverseStr("") === "" ? "PASS" : "FAIL");',
        difficulty=2,
    ),
    NanoProblem(
        task_id="nano_sum_array",
        description="Sum all numbers in an array",
        prompt="function sumArray(arr: number[]): number {\n  // Return the sum of all elements\n",
        expected_pattern="return",
        test_code='console.log(sumArray([1, 2, 3]) === 6 ? "PASS" : "FAIL");\nconsole.log(sumArray([]) === 0 ? "PASS" : "FAIL");',
        difficulty=2,
    ),
    NanoProblem(
        task_id="nano_contains",
        description="Check if array contains a value",
        prompt="function contains(arr: number[], target: number): boolean {\n  // Return true if arr contains target\n",
        expected_pattern="return",
        test_code='console.log(contains([1, 2, 3], 2) === true ? "PASS" : "FAIL");\nconsole.log(contains([1, 2, 3], 4) === false ? "PASS" : "FAIL");',
        difficulty=2,
    ),
]


@dataclass
class NanoResult:
    """Result from running one nano problem."""
    task_id: str
    has_output: bool  # Did the model generate anything?
    has_syntax: bool  # Is it syntactically valid?
    has_return: bool  # Does it contain a return statement?
    has_types: bool  # Does it pass tsc?
    tests_passed: int  # How many test assertions passed
    tests_total: int  # Total test assertions
    generated_code: str  # What the model generated


class NanoBenchmark:
    """Run the nano benchmark suite on a model."""

    def __init__(self):
        self.results: list[NanoResult] = []

    def evaluate_generation(self, problem: NanoProblem, generated: str) -> NanoResult:
        """Evaluate a single generation against a nano problem."""
        # Basic checks
        has_output = len(generated.strip()) > 0
        has_return = "return" in generated

        # Extract just the function body (after the prompt)
        full_code = problem.prompt + generated

        # Close the function if needed
        if full_code.count("{") > full_code.count("}"):
            full_code += "\n" + "}" * (full_code.count("{") - full_code.count("}"))

        # Syntax check via tsc
        has_syntax = False
        has_types = False
        tests_passed = 0
        tests_total = problem.test_code.count("PASS")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ts_file = Path(tmpdir) / "test.ts"
                ts_file.write_text(full_code + "\n\n" + problem.test_code, encoding="utf-8")

                # Type check
                result = subprocess.run(
                    ["npx", "tsc", "--noEmit", "--strict", "--target", "ES2020", str(ts_file)],
                    capture_output=True, text=True, timeout=10,
                    cwd=tmpdir,
                )
                has_types = result.returncode == 0
                has_syntax = has_types or ("error TS1" not in result.stderr)  # TS1xxx = syntax errors

                # If types pass, try running tests
                if has_types:
                    # Compile and run
                    js_file = Path(tmpdir) / "test.js"
                    compile_result = subprocess.run(
                        ["npx", "tsc", "--outDir", tmpdir, "--target", "ES2020", str(ts_file)],
                        capture_output=True, text=True, timeout=10,
                        cwd=tmpdir,
                    )
                    if compile_result.returncode == 0 and js_file.exists():
                        run_result = subprocess.run(
                            ["node", str(js_file)],
                            capture_output=True, text=True, timeout=5,
                        )
                        tests_passed = run_result.stdout.count("PASS")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return NanoResult(
            task_id=problem.task_id,
            has_output=has_output,
            has_syntax=has_syntax,
            has_return=has_return,
            has_types=has_types,
            tests_passed=tests_passed,
            tests_total=tests_total,
            generated_code=generated[:200],
        )

    def run(self, generator, problems: list[NanoProblem] | None = None) -> list[NanoResult]:
        """Run all nano problems through the generator.

        Args:
            generator: CodeGenerator instance (or any object with generate(prompt) method)
            problems: Optional subset of problems. Defaults to all.
        """
        problems = problems or NANO_PROBLEMS
        self.results = []

        cli.header("Cola-Coder", "Nano Benchmark")
        cli.info("Problems", len(problems))
        cli.rule()

        for i, problem in enumerate(problems):
            cli.substep(f"[{i+1}/{len(problems)}] {problem.description}")

            try:
                generated = generator.generate(
                    prompt=problem.prompt,
                    max_new_tokens=128,
                    temperature=0.3,  # Low temperature for benchmarking
                    top_k=40,
                    top_p=0.9,
                )
                # Extract only the new tokens (after the prompt)
                new_text = generated[len(problem.prompt):] if generated.startswith(problem.prompt) else generated
            except Exception as e:
                new_text = ""
                cli.warn(f"  Generation failed: {e}")

            result = self.evaluate_generation(problem, new_text)
            self.results.append(result)

        self.print_report()
        return self.results

    def run_without_model(self, generations: dict[str, str]) -> list[NanoResult]:
        """Run evaluation on pre-generated outputs (for testing without a model).

        Args:
            generations: Dict mapping task_id -> generated code string
        """
        self.results = []
        for problem in NANO_PROBLEMS:
            generated = generations.get(problem.task_id, "")
            result = self.evaluate_generation(problem, generated)
            self.results.append(result)
        self.print_report()
        return self.results

    def print_report(self):
        """Print a summary report of results."""
        if not self.results:
            cli.warn("No results to report.")
            return

        cli.rule("Nano Benchmark Results")

        total = len(self.results)
        has_output = sum(1 for r in self.results if r.has_output)
        has_syntax = sum(1 for r in self.results if r.has_syntax)
        has_return = sum(1 for r in self.results if r.has_return)
        has_types = sum(1 for r in self.results if r.has_types)
        tests_passed = sum(r.tests_passed for r in self.results)
        tests_total = sum(r.tests_total for r in self.results)

        cli.kv_table({
            "Has output": f"{has_output}/{total} ({100*has_output/total:.0f}%)",
            "Valid syntax": f"{has_syntax}/{total} ({100*has_syntax/total:.0f}%)",
            "Has return": f"{has_return}/{total} ({100*has_return/total:.0f}%)",
            "Type correct": f"{has_types}/{total} ({100*has_types/total:.0f}%)",
            "Tests passed": f"{tests_passed}/{tests_total} ({100*tests_passed/tests_total:.0f}%)" if tests_total > 0 else "0/0",
        }, title="Nano Benchmark Report")

        # Per-problem detail
        for r in self.results:
            status = "PASS" if r.tests_passed == r.tests_total and r.tests_total > 0 else "FAIL"
            marker = "[green]PASS[/green]" if status == "PASS" else "[red]FAIL[/red]"
            cli.print(f"  {marker} {r.task_id}: syntax={'Y' if r.has_syntax else 'N'} types={'Y' if r.has_types else 'N'} tests={r.tests_passed}/{r.tests_total}")

    def to_dict(self) -> dict:
        """Export results as a dictionary for saving."""
        return {
            "benchmark": "nano",
            "num_problems": len(self.results),
            "summary": {
                "output_rate": sum(1 for r in self.results if r.has_output) / max(len(self.results), 1),
                "syntax_rate": sum(1 for r in self.results if r.has_syntax) / max(len(self.results), 1),
                "type_rate": sum(1 for r in self.results if r.has_types) / max(len(self.results), 1),
                "test_pass_rate": sum(r.tests_passed for r in self.results) / max(sum(r.tests_total for r in self.results), 1),
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "has_output": r.has_output,
                    "has_syntax": r.has_syntax,
                    "has_types": r.has_types,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                }
                for r in self.results
            ],
        }
