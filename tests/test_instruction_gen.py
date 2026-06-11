"""Tests for instruction_gen.py (the code->instruction SFT generator).

This data-quality-critical module had ZERO test coverage. The headline guard is
score_quality's language bias: the "parses" bonus used to be Python-only, so a
TypeScript/JS pair scored 0.2 lower than an identical Python one — biasing the
keep/reject filter AGAINST the project's primary language. These tests lock the
language-aware fix and the dedup/threshold behavior.
"""

import tempfile
from pathlib import Path

from cola_coder.data.sources.instruction_gen import (
    CodeToInstructionGenerator,
    _detect_language,
    score_quality,
)


class TestScoreQualityLanguageParity:
    def test_balanced_brace_ts_gets_syntax_bonus(self):
        instr = "Write a TypeScript function that adds two numbers together."
        ts = "function add(a: number, b: number): number {\n  return a + b;\n}"
        # The TS response is NOT valid Python, but balanced braces earn the
        # same +0.2 "parses" bonus a Python response would.
        assert score_quality(instr, ts) >= 0.8

    def test_python_and_ts_score_equally_for_equivalent_pairs(self):
        instr = "Write a function that adds two numbers together cleanly."
        py = "def add(a, b):\n    return a + b\n"
        ts = "function add(a, b) {\n  return a + b;\n}"
        # Within a small margin — neither language is penalized.
        assert abs(score_quality(instr, py) - score_quality(instr, ts)) <= 0.05

    def test_unbalanced_braces_no_bonus(self):
        instr = "Fix this broken snippet please, it is incomplete somehow."
        broken = "function add(a, b) {\n  return a + b;\n"  # missing }
        # No Python parse, unbalanced braces → no syntax bonus.
        assert score_quality(instr, broken) < 0.8

    def test_empty_inputs_zero(self):
        assert score_quality("", "x") == 0.0
        assert score_quality("x", "") == 0.0

    def test_too_short_response_zero(self):
        assert score_quality("Write something useful here please.", "ok") == 0.0


class TestDetectLanguage:
    def test_python(self):
        assert _detect_language(Path("a.py")) == "python"

    def test_typescript(self):
        assert _detect_language(Path("a.ts")) == "typescript"
        assert _detect_language(Path("a.tsx")) == "typescript"

    def test_js_default(self):
        assert _detect_language(Path("a.js")) == "javascript"


class TestGenerate:
    def test_generates_chatml_pairs_from_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.py"
            f.write_text(
                "def fibonacci(n):\n"
                "    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n",
                encoding="utf-8",
            )
            gen = CodeToInstructionGenerator(source_file=str(f))
            examples = gen.generate(num_samples=20, quality_threshold=0.5)
            assert len(examples) >= 1
            msgs = examples[0]["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            # Quality score is popped from the output dict (internal only).
            assert "_quality" not in examples[0]

    def test_dedup_by_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.py"
            body = (
                "def add(a, b):\n"
                "    \"\"\"Add two numbers.\"\"\"\n"
                "    return a + b\n"
            )
            f.write_text(body, encoding="utf-8")
            gen = CodeToInstructionGenerator(source_file=str(f))
            examples = gen.generate(num_samples=50, quality_threshold=0.4)
            users = [e["messages"][1]["content"][:200].lower().strip() for e in examples]
            assert len(users) == len(set(users))  # no duplicate instructions

    def test_no_source_raises(self):
        import pytest
        with pytest.raises(ValueError):
            CodeToInstructionGenerator()


class TestBugInjectionNoGlobalMutation:
    def test_fix_pair_does_not_mutate_global(self):
        from cola_coder.data.sources import instruction_gen as ig

        before = list(ig._BUG_INJECTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.py"
            f.write_text(
                "def f(x):\n    if x == 1:\n        return True\n    return False\n",
                encoding="utf-8",
            )
            CodeToInstructionGenerator(source_file=str(f)).generate(
                num_samples=10, quality_threshold=0.0
            )
        # The module-level injection list must be unchanged (shuffle a copy).
        assert ig._BUG_INJECTIONS == before
