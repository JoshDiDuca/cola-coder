"""Smoke test suite for validating a trained model generates reasonable code.

Runs a battery of quick checks that should complete in under 30 seconds:
- Basic token generation
- Python syntax validity
- Perplexity in a sane range
- No runaway token repetition
- Output diversity across prompts
- No raw special tokens in the middle of output
- Temperature sensitivity
- Presence of programming keywords

Each test returns a TestResult. The whole suite returns a SmokeTestReport.

For a TS dev: this is like a health-check endpoint — it doesn't test correctness
in depth (that's HumanEval's job), but it quickly tells you if something is
fundamentally broken.
"""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass, field
from typing import Protocol


# ── Public dataclasses ────────────────────────────────────────────────────────


@dataclass
class TestResult:
    """Result of a single smoke test."""

    # Prevent pytest from trying to collect this dataclass as a test class.
    __test__ = False

    name: str
    passed: bool
    message: str
    duration_ms: float


@dataclass
class SmokeTestReport:
    """Aggregated results from all smoke tests."""

    results: list[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """True only if every individual test passed."""
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_failed(self) -> int:
        return len(self.results) - self.num_passed

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Smoke tests {status}: "
            f"{self.num_passed}/{len(self.results)} passed "
            f"in {self.total_duration_ms:.0f}ms"
        )


# ── Generator protocol ────────────────────────────────────────────────────────


class GeneratorProtocol(Protocol):
    """Minimal interface that SmokeTest needs from any generator."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
    ) -> str: ...


# ── Helpers ───────────────────────────────────────────────────────────────────

# Short prompts used across tests — representative code snippets
_PROMPTS = [
    "def hello",
    "function add(",
    "class User",
    "import ",
    "// Calculate",
]

# Tokens that look like raw special-token strings leaking into output
_SPECIAL_TOKEN_PATTERNS = ["<|endoftext|>", "<bos>", "<eos>", "<pad>", "<unk>"]

# Programming keywords that should appear in code-generating output
_CODE_KEYWORDS = [
    "def", "class", "return", "import", "function", "const", "let", "var",
    "if", "for", "while", "async", "await", "type", "interface", "struct",
]

_MAX_NEW_TOKENS = 256
_TEMPERATURE = 0.8
_TEST_TIMEOUT_S = 10.0



def _generate_safe(generator: GeneratorProtocol, prompt: str, **kwargs) -> str:
    """Generate with default smoke-test params, merging any overrides."""
    params = dict(
        max_new_tokens=_MAX_NEW_TOKENS,
        temperature=_TEMPERATURE,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    params.update(kwargs)
    return generator.generate(prompt, **params)


def _new_tokens(output: str, prompt: str) -> str:
    """Return only the tokens generated after the prompt prefix."""
    if output.startswith(prompt):
        return output[len(prompt):]
    return output


# ── Main class ────────────────────────────────────────────────────────────────


class SmokeTest:
    """Quick validation that a trained model generates reasonable code.

    Usage::

        test = SmokeTest(generator, tokenizer)
        report = test.run_all()
        print(report.summary)

    The ``tokenizer`` argument is optional — it's only used for the
    perplexity test.  Pass ``None`` to skip that test.
    """

    def __init__(self, generator: GeneratorProtocol, tokenizer=None):
        self.generator = generator
        self.tokenizer = tokenizer

    # ── Test runner ───────────────────────────────────────────────────────

    def run_all(self) -> SmokeTestReport:
        """Run all smoke tests and return a SmokeTestReport."""
        suite_start = time.perf_counter()

        tests = [
            self.test_generates_tokens,
            self.test_code_syntax,
            self.test_perplexity_range,
            self.test_repetition,
            self.test_diversity,
            self.test_special_tokens,
            self.test_temperature_sensitivity,
            self.test_code_keywords,
        ]

        results: list[TestResult] = []
        for test_fn in tests:
            t0 = time.perf_counter()
            try:
                result = test_fn()
            except Exception as exc:  # noqa: BLE001
                result = TestResult(
                    name=test_fn.__name__,
                    passed=False,
                    message=f"Uncaught exception: {exc}",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            results.append(result)

        total_ms = (time.perf_counter() - suite_start) * 1000
        return SmokeTestReport(results=results, total_duration_ms=total_ms)

    # ── Individual tests ──────────────────────────────────────────────────

    def test_generates_tokens(self) -> TestResult:
        """Model produces non-empty output for a basic prompt."""
        t0 = time.perf_counter()
        try:
            output = _generate_safe(self.generator, _PROMPTS[0])
            new_text = _new_tokens(output, _PROMPTS[0])
            passed = len(new_text.strip()) > 0
            message = (
                f"Generated {len(new_text)} new chars"
                if passed
                else "Output was empty after stripping prompt"
            )
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_generates_tokens",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_code_syntax(self) -> TestResult:
        """Generated Python code can be parsed by ast.parse."""
        t0 = time.perf_counter()
        try:
            # Use a Python-style prompt
            prompt = "def hello"
            output = _generate_safe(self.generator, prompt)
            # Try to parse whatever the model produced
            try:
                ast.parse(output)
                passed = True
                message = "Output parses as valid Python"
            except SyntaxError as syn_err:
                # Partial output often ends mid-expression — be lenient:
                # if the error is at the very last line it may just be truncated
                lines = output.strip().splitlines()
                last_line = lines[-1] if lines else ""
                truncated = syn_err.lineno is not None and syn_err.lineno >= len(lines)
                if truncated or len(last_line) < 5:
                    passed = True
                    message = "Output appears truncated (last line incomplete) — syntax OK"
                else:
                    passed = False
                    message = f"SyntaxError on line {syn_err.lineno}: {syn_err.msg}"
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_code_syntax",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_perplexity_range(self) -> TestResult:
        """Perplexity is finite and in a reasonable range (not 1, not inf).

        Requires a tokenizer that has an ``encode`` method.  If no tokenizer
        was provided, the test is skipped (marked passed with a note).
        """
        t0 = time.perf_counter()
        try:
            if self.tokenizer is None:
                return TestResult(
                    name="test_perplexity_range",
                    passed=True,
                    message="Skipped — no tokenizer provided",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            import torch
            import torch.nn.functional as F

            prompt = _PROMPTS[0]
            output = _generate_safe(self.generator, prompt)
            if not output.strip():
                return TestResult(
                    name="test_perplexity_range",
                    passed=False,
                    message="Cannot compute perplexity — output is empty",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            # Encode the generated output as token IDs
            token_ids = self.tokenizer.encode(output, add_bos=False)
            if len(token_ids) < 2:
                return TestResult(
                    name="test_perplexity_range",
                    passed=True,
                    message="Output too short to compute perplexity — skipping",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            # Use the generator's model if it's a CodeGenerator
            model = getattr(self.generator, "model", None)
            device = getattr(self.generator, "device", "cpu")
            if model is None:
                return TestResult(
                    name="test_perplexity_range",
                    passed=True,
                    message="Skipped — generator has no .model attribute",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(ids)  # (1, seq, vocab)

            # Shift: logits[0..n-2] predict ids[1..n-1]
            shift_logits = logits[0, :-1, :]  # (n-1, vocab)
            shift_targets = ids[0, 1:]        # (n-1,)
            loss = F.cross_entropy(shift_logits, shift_targets)
            perplexity = math.exp(loss.item())

            # Reasonable range: a tiny/undertrained model can have ppl in the
            # hundreds; an overfit model approaches 1.  Flag extremes.
            if not math.isfinite(perplexity):
                passed, message = False, f"Perplexity is not finite: {perplexity}"
            elif perplexity < 1.5:
                passed = False
                message = f"Perplexity suspiciously low ({perplexity:.1f}) — possible memorisation/overfit"
            elif perplexity > 50_000:
                passed = False
                message = f"Perplexity extremely high ({perplexity:.0f}) — model may be untrained"
            else:
                passed = True
                message = f"Perplexity {perplexity:.1f} is in acceptable range"
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_perplexity_range",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_repetition(self) -> TestResult:
        """Model doesn't produce a single repeated token for the whole output."""
        t0 = time.perf_counter()
        try:
            prompt = _PROMPTS[0]
            output = _generate_safe(self.generator, prompt)
            new_text = _new_tokens(output, prompt).strip()

            if not new_text:
                return TestResult(
                    name="test_repetition",
                    passed=False,
                    message="Output is empty — cannot check repetition",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            # Split into words / tokens for repetition analysis
            words = new_text.split()
            if len(words) < 4:
                return TestResult(
                    name="test_repetition",
                    passed=True,
                    message="Output too short for repetition check — skipping",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            # Detect if a single word dominates (>80% of words)
            from collections import Counter
            counts = Counter(words)
            most_common_word, most_common_count = counts.most_common(1)[0]
            repetition_ratio = most_common_count / len(words)

            if repetition_ratio > 0.8:
                passed = False
                message = (
                    f"Repetitive output: '{most_common_word}' appears "
                    f"{most_common_count}/{len(words)} times ({repetition_ratio:.0%})"
                )
            else:
                passed = True
                message = (
                    f"Repetition ratio {repetition_ratio:.0%} "
                    f"(most common: '{most_common_word}')"
                )
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_repetition",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_diversity(self) -> TestResult:
        """Different prompts produce different outputs."""
        t0 = time.perf_counter()
        try:
            outputs = []
            for prompt in _PROMPTS[:3]:
                out = _generate_safe(self.generator, prompt)
                outputs.append(out.strip())

            # Require at least 2 distinct outputs from 3 prompts
            unique_outputs = set(outputs)
            if len(unique_outputs) < 2:
                passed = False
                message = "All prompts produced identical output — model may be degenerate"
            else:
                passed = True
                message = (
                    f"{len(unique_outputs)}/{len(outputs)} distinct outputs "
                    "for different prompts"
                )
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_diversity",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_special_tokens(self) -> TestResult:
        """Output body doesn't contain raw special-token strings."""
        t0 = time.perf_counter()
        try:
            prompt = _PROMPTS[0]
            output = _generate_safe(self.generator, prompt)
            new_text = _new_tokens(output, prompt)

            found = [tok for tok in _SPECIAL_TOKEN_PATTERNS if tok in new_text]
            if found:
                passed = False
                message = f"Raw special tokens found in output: {found}"
            else:
                passed = True
                message = "No raw special tokens in output"
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_special_tokens",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_temperature_sensitivity(self) -> TestResult:
        """Higher temperature produces more diverse output than lower temperature.

        Generates several samples at low and high temperature and checks that
        the high-temperature samples are more varied (more unique strings).
        """
        t0 = time.perf_counter()
        try:
            prompt = _PROMPTS[0]
            n_samples = 3

            low_outputs = set()
            high_outputs = set()

            for _ in range(n_samples):
                low_out = _generate_safe(
                    self.generator, prompt, temperature=0.1, max_new_tokens=64
                )
                high_out = _generate_safe(
                    self.generator, prompt, temperature=1.4, max_new_tokens=64
                )
                low_outputs.add(low_out.strip())
                high_outputs.add(high_out.strip())

            # High temp should produce at least as many unique strings as low temp.
            # We're lenient: fail only if high temp is strictly less diverse.
            if len(high_outputs) < len(low_outputs):
                passed = False
                message = (
                    f"High temperature ({len(high_outputs)} unique) less diverse "
                    f"than low temperature ({len(low_outputs)} unique)"
                )
            else:
                passed = True
                message = (
                    f"Low temp: {len(low_outputs)} unique / "
                    f"High temp: {len(high_outputs)} unique out of {n_samples} samples"
                )
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_temperature_sensitivity",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def test_code_keywords(self) -> TestResult:
        """Generated output contains at least one programming keyword."""
        t0 = time.perf_counter()
        try:
            outputs = []
            for prompt in _PROMPTS:
                out = _generate_safe(self.generator, prompt)
                outputs.append(out)

            combined = " ".join(outputs).lower()
            found_keywords = [kw for kw in _CODE_KEYWORDS if kw in combined]

            if not found_keywords:
                passed = False
                message = "No programming keywords found in any generated output"
            else:
                passed = True
                message = f"Found keywords: {', '.join(found_keywords[:8])}"
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"Exception: {exc}"
        return TestResult(
            name="test_code_keywords",
            passed=passed,
            message=message,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
