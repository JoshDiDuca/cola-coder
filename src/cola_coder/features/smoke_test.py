"""Quick Smoke Test: instant validation that a model learned something.

After any training run, generate 5 canonical prompts and check if the
outputs look reasonable. This catches obvious failures:
- Empty output
- Repetition loops
- Garbage/random tokens
- All-whitespace output
"""

from dataclasses import dataclass
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED

# Canonical smoke test prompts - these are common TypeScript patterns
# that even a barely-trained model should start to complete sensibly
SMOKE_PROMPTS = [
    {
        "name": "Function definition",
        "prompt": "function add(a: number, b: number): number {\n",
        "expect_contains": ["return", "}"],  # Should at least close the function
    },
    {
        "name": "Interface declaration",
        "prompt": "interface User {\n",
        "expect_contains": [":"],  # Should declare at least one property
    },
    {
        "name": "Async function",
        "prompt": "const fetchData = async (",
        "expect_contains": [")", "=>", "{"],  # Should form a valid arrow function
    },
    {
        "name": "Class definition",
        "prompt": "class Logger {\n  constructor(",
        "expect_contains": [")"],  # Should complete the constructor
    },
    {
        "name": "Export function",
        "prompt": "export function validateEmail(email: string): boolean {\n",
        "expect_contains": ["return"],  # Should have a return statement
    },
]


@dataclass
class SmokeResult:
    """Result from one smoke test prompt."""
    name: str
    prompt: str
    generated: str
    is_empty: bool
    has_repetition: bool
    has_garbage: bool
    has_expected: list[bool]  # Which expected patterns were found
    score: float  # 0.0 - 1.0


def detect_repetition(text: str, min_pattern_len: int = 5, min_repeats: int = 3) -> bool:
    """Detect if text contains excessive repetition."""
    if len(text) < min_pattern_len * min_repeats:
        return False

    # Check for repeated substrings
    for pattern_len in range(min_pattern_len, min(50, len(text) // min_repeats)):
        for start in range(len(text) - pattern_len * min_repeats):
            pattern = text[start:start + pattern_len]
            if pattern.strip() == "":
                continue
            count = 0
            pos = start
            while pos <= len(text) - pattern_len:
                if text[pos:pos + pattern_len] == pattern:
                    count += 1
                    pos += pattern_len
                else:
                    break
            if count >= min_repeats:
                return True
    return False


def detect_garbage(text: str) -> bool:
    """Detect if text is mostly non-code garbage."""
    if not text.strip():
        return True

    # High ratio of non-printable or unusual characters
    printable = sum(1 for c in text if c.isprintable() or c in '\n\t\r')
    if len(text) > 0 and printable / len(text) < 0.8:
        return True

    # No recognizable code tokens
    code_indicators = ['(', ')', '{', '}', ';', ':', '=', 'function', 'const', 'let', 'return', 'if', 'for']
    has_code = any(indicator in text for indicator in code_indicators)
    if not has_code and len(text) > 20:
        return True

    return False


class SmokeTest:
    """Run quick smoke tests on a model."""

    def __init__(self):
        self.results: list[SmokeResult] = []

    def run(self, generator, prompts: list[dict] | None = None) -> list[SmokeResult]:
        """Run smoke tests.

        Args:
            generator: CodeGenerator instance with generate(prompt) method.
            prompts: Optional custom prompts. Defaults to SMOKE_PROMPTS.
        """
        prompts = prompts or SMOKE_PROMPTS
        self.results = []

        cli.header("Cola-Coder", "Quick Smoke Test")
        cli.info("Prompts", len(prompts))
        cli.rule()

        for i, prompt_info in enumerate(prompts):
            name = prompt_info["name"]
            prompt = prompt_info["prompt"]
            expect = prompt_info.get("expect_contains", [])

            cli.substep(f"[{i+1}/{len(prompts)}] {name}")

            try:
                full_output = generator.generate(
                    prompt=prompt,
                    max_new_tokens=150,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9,
                )
                generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
            except Exception as e:
                generated = ""
                cli.warn(f"  Generation failed: {e}")

            # Analyze
            is_empty = len(generated.strip()) == 0
            has_repetition = detect_repetition(generated)
            has_garbage = detect_garbage(generated)
            has_expected = [pattern in generated for pattern in expect]

            # Score: 0-1
            score = 0.0
            if not is_empty:
                score += 0.2
            if not has_repetition:
                score += 0.2
            if not has_garbage:
                score += 0.2
            if expect:
                score += 0.4 * (sum(has_expected) / len(has_expected))

            result = SmokeResult(
                name=name,
                prompt=prompt,
                generated=generated[:300],
                is_empty=is_empty,
                has_repetition=has_repetition,
                has_garbage=has_garbage,
                has_expected=has_expected,
                score=score,
            )
            self.results.append(result)

            # Show output preview
            preview = generated[:100].replace('\n', '\\n')
            if is_empty:
                cli.print("    [red]EMPTY OUTPUT[/red]")
            elif has_repetition:
                cli.print(f"    [yellow]REPETITION DETECTED[/yellow]: {preview}")
            elif has_garbage:
                cli.print(f"    [yellow]GARBAGE OUTPUT[/yellow]: {preview}")
            else:
                cli.print(f"    [green]OK[/green]: {preview}")

        self.print_report()
        return self.results

    def run_without_model(self, generations: dict[str, str]) -> list[SmokeResult]:
        """Run on pre-generated outputs (testing without model).

        Args:
            generations: Dict mapping prompt name -> generated text
        """
        self.results = []
        for prompt_info in SMOKE_PROMPTS:
            name = prompt_info["name"]
            generated = generations.get(name, "")
            expect = prompt_info.get("expect_contains", [])

            is_empty = len(generated.strip()) == 0
            has_repetition = detect_repetition(generated)
            has_garbage = detect_garbage(generated)
            has_expected = [pattern in generated for pattern in expect]

            score = 0.0
            if not is_empty:
                score += 0.2
            if not has_repetition:
                score += 0.2
            if not has_garbage:
                score += 0.2
            if expect:
                score += 0.4 * (sum(has_expected) / len(has_expected))

            self.results.append(SmokeResult(
                name=name,
                prompt=prompt_info["prompt"],
                generated=generated[:300],
                is_empty=is_empty,
                has_repetition=has_repetition,
                has_garbage=has_garbage,
                has_expected=has_expected,
                score=score,
            ))

        self.print_report()
        return self.results

    def print_report(self):
        """Print smoke test report."""
        if not self.results:
            cli.warn("No results to report.")
            return

        cli.rule("Smoke Test Results")

        avg_score = sum(r.score for r in self.results) / len(self.results)
        passed = sum(1 for r in self.results if r.score >= 0.6)

        cli.kv_table({
            "Average score": f"{avg_score:.1%}",
            "Passed (>=60%)": f"{passed}/{len(self.results)}",
            "Empty outputs": str(sum(1 for r in self.results if r.is_empty)),
            "Repetition detected": str(sum(1 for r in self.results if r.has_repetition)),
            "Garbage detected": str(sum(1 for r in self.results if r.has_garbage)),
        }, title="Smoke Test Summary")

        # Overall assessment
        if avg_score >= 0.8:
            cli.success("Model looks healthy!")
        elif avg_score >= 0.5:
            cli.warn("Model shows some learning but needs more training.")
        elif avg_score >= 0.2:
            cli.warn("Model barely learned. Continue training or check data quality.")
        else:
            cli.error("Model appears untrained. Check training pipeline.")

    def to_dict(self) -> dict:
        """Export results for saving."""
        return {
            "benchmark": "smoke_test",
            "num_prompts": len(self.results),
            "average_score": sum(r.score for r in self.results) / max(len(self.results), 1),
            "results": [
                {
                    "name": r.name,
                    "score": r.score,
                    "is_empty": r.is_empty,
                    "has_repetition": r.has_repetition,
                    "has_garbage": r.has_garbage,
                }
                for r in self.results
            ],
        }
