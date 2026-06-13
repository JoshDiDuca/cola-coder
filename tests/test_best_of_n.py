"""Tests for best-of-N generation with sandboxed verification.

Uses fake generators and injected verifiers so no GPU, tsc install, or
sandbox is needed. The Python-syntax path uses the real compile() check
(static — never executes candidate code).
"""

from pathlib import Path

import pytest

from cola_coder.inference.best_of_n import (
    BestOfNResult,
    detect_language,
    generate_best_of_n,
    _strip_prompt,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════


class FakeGroupGenerator:
    """Generator exposing generate_group, returning canned candidates."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.group_calls: list[dict] = []

    def generate_group(self, prompt, num_completions, **kwargs):
        self.group_calls.append({"prompt": prompt,
                                 "num_completions": num_completions, **kwargs})
        return self.texts[:num_completions]


class FakeSerialGenerator:
    """Generator WITHOUT generate_group (like ContextAwareGenerator)."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.generate_calls = 0

    def generate(self, prompt, **kwargs):
        text = self.texts[self.generate_calls % len(self.texts)]
        self.generate_calls += 1
        return text


class FakeHybridGenerator:
    """Generator exposing BOTH generate_group (batched) and generate (serial).

    Records which path was taken so tests can assert that n-gram blocking
    forces the serial path (the batched sampler can't honor it).
    """

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.group_calls: list[dict] = []
        self.serial_calls: list[dict] = []

    def generate_group(self, prompt, num_completions, **kwargs):
        self.group_calls.append({"num_completions": num_completions, **kwargs})
        return self.texts[:num_completions]

    def generate(self, prompt, **kwargs):
        self.serial_calls.append(kwargs)
        return self.texts[len(self.serial_calls) % len(self.texts) - 1]


class FakeTscRunner:
    """check_batch stub: maps candidate index -> list of error strings."""

    def __init__(self, errors_by_index: dict[int, list[str]]):
        self.errors_by_index = errors_by_index
        self.checked: list[list[str]] = []

    def check_batch(self, codes):
        self.checked.append(codes)
        return self.errors_by_index


# ══════════════════════════════════════════════════════════════════════════════
# Language detection & prompt stripping
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_detect_language_typescript(self):
        code = "interface User { name: string }\nconst x: number = 1;"
        assert detect_language(code) == "typescript"

    def test_detect_language_python(self):
        code = "def add(a, b):\n    return a + b"
        assert detect_language(code) == "python"

    def test_strip_prompt_exact_prefix(self):
        assert _strip_prompt("def f():\n    pass", "def f():\n") == "    pass"

    def test_strip_prompt_bpe_drift_falls_back_to_common_prefix(self):
        # decode(encode(prompt)) lost a trailing space — strip the longest
        # common prefix ("def f(): " including the single space)
        assert _strip_prompt("def f(): pass", "def f():  ") == "pass"


# ══════════════════════════════════════════════════════════════════════════════
# Python verification paths
# ══════════════════════════════════════════════════════════════════════════════


class TestPythonSyntaxPath:
    def test_valid_python_beats_syntax_error(self):
        good = "def add(a, b):\n    return a + b"
        bad = "def add(a, b:\n    return a +"
        gen = FakeGroupGenerator([bad, good])

        result = generate_best_of_n(gen, "def add", num_candidates=2,
                                    language="python")

        assert isinstance(result, BestOfNResult)
        assert result.verifier == "python_syntax"
        assert result.best.text == good
        assert result.best.verified is True

    def test_candidates_sorted_best_first(self):
        good = "def add(a, b):\n    return a + b"
        bad = "def broken(:"
        gen = FakeGroupGenerator([bad, good])

        result = generate_best_of_n(gen, "def", num_candidates=2,
                                    language="python")

        assert result.candidates[0] is result.best
        assert [c.verified for c in result.candidates] == [True, False]
        assert result.candidates[0].score > result.candidates[1].score

    def test_syntax_check_never_executes(self):
        # If this were executed, it would create a file — verify it doesn't.
        marker = Path("BEST_OF_N_EXECUTED.marker")
        evil = f"open({str(marker)!r}, 'w').write('x')"
        gen = FakeGroupGenerator([evil])

        generate_best_of_n(gen, "x", num_candidates=1, language="python")

        assert not marker.exists()


class TestPythonExecPath:
    def test_passing_tests_wins(self):
        passing = "def add(a, b):\n    return a + b"
        failing = "def add(a, b):\n    return a - b"

        def fake_execute(code, timeout):
            ok = "a + b" in code
            return ok, "ok" if ok else "AssertionError"

        gen = FakeGroupGenerator([failing, passing])
        result = generate_best_of_n(
            gen, "def add", num_candidates=2, language="python",
            tests="assert add(1, 2) == 3", execute_fn=fake_execute,
        )

        assert result.verifier == "python_exec"
        assert result.best.text == passing
        assert result.best.details["tests_passed"] is True
        assert result.candidates[1].details["tests_passed"] is False

    def test_tests_are_appended_to_candidate(self):
        seen = []

        def fake_execute(code, timeout):
            seen.append(code)
            return True, ""

        gen = FakeGroupGenerator(["def f(): pass"])
        generate_best_of_n(
            gen, "def f", num_candidates=1, language="python",
            tests="assert f() is None", execute_fn=fake_execute,
        )

        assert "def f(): pass" in seen[0]
        assert "assert f() is None" in seen[0]


# ══════════════════════════════════════════════════════════════════════════════
# TypeScript verification path
# ══════════════════════════════════════════════════════════════════════════════


class TestTypeScriptPath:
    def test_clean_tsc_candidate_wins(self):
        texts = ["const x: number = 'oops';", "const x: number = 1;"]
        gen = FakeGroupGenerator(texts)
        tsc = FakeTscRunner({0: ["TS2322: type mismatch"], 1: []})

        result = generate_best_of_n(gen, "const x", num_candidates=2,
                                    language="typescript", tsc_runner=tsc)

        assert result.verifier == "tsc"
        assert result.best.text == texts[1]
        assert result.best.verified is True
        assert result.best.details["tsc_errors"] == 0
        assert result.candidates[1].details["tsc_errors"] == 1
        # All candidates were checked in one batch
        assert tsc.checked == [texts]

    def test_fewer_errors_rank_higher_when_none_verify(self):
        texts = ["a", "b", "c"]
        gen = FakeGroupGenerator(texts)
        tsc = FakeTscRunner({0: ["e"] * 7, 1: ["e"], 2: ["e"] * 4})

        result = generate_best_of_n(gen, "interface X {}", num_candidates=3,
                                    language="typescript", tsc_runner=tsc)

        assert not result.best.verified
        assert result.best.text == "b"  # 1 error beats 4 and 7

    def test_tsc_unavailable_falls_back_to_heuristics(self, monkeypatch):
        from cola_coder.reasoning.rewards import tsc_runner as tsc_mod

        monkeypatch.setattr(tsc_mod.TscRunner, "is_available", lambda: False)
        gen = FakeGroupGenerator(["const x: number = 1;"])

        result = generate_best_of_n(gen, "const x", num_candidates=1,
                                    language="typescript", tsc_runner=None)

        assert result.verifier == "heuristic"
        assert result.best.details.get("heuristic_only") is True


# ══════════════════════════════════════════════════════════════════════════════
# Generation plumbing
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerationPlumbing:
    def test_uses_generate_group_with_sampling_params(self):
        gen = FakeGroupGenerator(["def f(): pass"] * 4)
        generate_best_of_n(
            gen, "def f", num_candidates=4, language="python",
            max_new_tokens=99, temperature=0.5, top_k=10, top_p=0.8, min_p=0.05,
        )

        call = gen.group_calls[0]
        assert call["num_completions"] == 4
        assert call["max_new_tokens"] == 99
        assert call["temperature"] == 0.5
        assert call["top_k"] == 10
        assert call["top_p"] == 0.8
        assert call["min_p"] == 0.05

    def test_no_repeat_ngram_forces_serial_path(self):
        # INFER-018: the batched sampler can't track per-sequence n-grams, so a
        # positive no_repeat_ngram_size must route through serial generate()
        # (which honors it) instead of being silently dropped by generate_group.
        gen = FakeHybridGenerator(["def f(): pass"] * 3)
        generate_best_of_n(
            gen, "def f", num_candidates=3, language="python",
            no_repeat_ngram_size=3,
        )
        assert gen.group_calls == [], "n-gram blocking must NOT use the batched path"
        assert len(gen.serial_calls) == 3
        assert all(c["no_repeat_ngram_size"] == 3 for c in gen.serial_calls)

    def test_no_ngram_block_keeps_batched_fast_path(self):
        # Default (no n-gram blocking) must still use the fast batched path.
        gen = FakeHybridGenerator(["def f(): pass"] * 3)
        generate_best_of_n(gen, "def f", num_candidates=3, language="python")
        assert len(gen.group_calls) == 1
        assert gen.serial_calls == []

    def test_serial_fallback_without_generate_group(self):
        gen = FakeSerialGenerator(["def f(): pass", "def g(): pass"])
        result = generate_best_of_n(gen, "def", num_candidates=3,
                                    language="python")

        assert gen.generate_calls == 3
        assert len(result.candidates) == 3

    def test_num_candidates_must_be_positive(self):
        gen = FakeGroupGenerator(["x"])
        with pytest.raises(ValueError):
            generate_best_of_n(gen, "x", num_candidates=0, language="python")

    def test_completion_is_prompt_stripped(self):
        prompt = "def add(a, b):\n"
        gen = FakeGroupGenerator([prompt + "    return a + b"])
        result = generate_best_of_n(gen, prompt, num_candidates=1,
                                    language="python")

        assert result.best.completion == "    return a + b"

    def test_auto_language_routes_python(self):
        gen = FakeGroupGenerator(["def f(): pass"])
        result = generate_best_of_n(gen, "def f():", num_candidates=1,
                                    language="auto")
        assert result.language == "python"
        assert result.verifier == "python_syntax"


# ══════════════════════════════════════════════════════════════════════════════
# Feature wrapper
# ══════════════════════════════════════════════════════════════════════════════


class TestFeatureWrapper:
    def test_wrapper_delegates_to_core(self):
        from cola_coder.features import best_of_n_verification as feature

        assert feature.is_enabled() is True
        gen = FakeGroupGenerator(["def f(): pass"])
        result = feature.generate_best_of_n(gen, "def f", num_candidates=1,
                                            language="python")
        assert isinstance(result, BestOfNResult)


class TestSecurityAwareRanking:
    """IDEA-008/SEC-017: among equally-verified candidates, prefer the SECURE one."""

    def test_secure_candidate_preferred_among_verified(self):
        insecure = "const out = eval(userInput);"   # tsc-clean but dangerous (eval)
        secure = "const total = 1 + 2;"
        gen = FakeGroupGenerator([insecure, secure])
        runner = FakeTscRunner({})  # no errors for any index -> both verify clean
        result = generate_best_of_n(
            gen, "// prompt", num_candidates=2,
            language="typescript", tsc_runner=runner,
        )
        # Both pass the hard verifier, but the secure one must rank first.
        assert all(c.verified for c in result.candidates)
        assert result.best.details["secure"] is True
        assert "eval" not in result.best.completion
        flagged = [c for c in result.candidates if c.details.get("secure") is False]
        assert flagged, "the eval() candidate should be marked insecure"
        assert "eval() usage" in flagged[0].details.get("dangerous_patterns", [])

    def test_security_only_breaks_ties_not_beats_verified(self):
        # An UNVERIFIED-but-secure candidate must NOT beat a VERIFIED-but-insecure one;
        # functional correctness dominates, security is only a secondary key.
        insecure_ok = "const out = eval(x);"   # verifies (index 0, no errors)
        secure_bad = "const y: number = 'str';"  # does NOT verify (index 1 has an error)
        gen = FakeGroupGenerator([insecure_ok, secure_bad])
        runner = FakeTscRunner({1: ["TS2322: type error"]})  # only index 1 fails
        result = generate_best_of_n(
            gen, "// p", num_candidates=2, language="typescript", tsc_runner=runner,
        )
        assert result.best.verified is True
        assert "eval" in result.best.completion  # verified insecure beats unverified secure


class TestAdaptiveBudget:
    """IDEA-009: adaptive best-of-N grows the budget only as needed."""

    def test_early_stop_when_first_batch_verifies(self):
        from cola_coder.inference.best_of_n import generate_best_of_n_adaptive
        gen = FakeGroupGenerator(["const x: number = 1;", "const y: number = 2;"])
        runner = FakeTscRunner({})  # clean -> both verify
        result = generate_best_of_n_adaptive(
            gen, "// p", initial_candidates=2, max_candidates=6,
            language="typescript", tsc_runner=runner,
        )
        assert len(result.candidates) == 2      # stopped after the first batch
        assert len(gen.group_calls) == 1
        assert result.best.verified

    def test_expands_when_none_verify(self):
        from cola_coder.inference.best_of_n import generate_best_of_n_adaptive
        gen = FakeGroupGenerator(["const x: number = 1;", "const y: number = 2;"])
        runner = FakeTscRunner({0: ["TS2322: e"], 1: ["TS2322: e"]})  # fail every batch
        result = generate_best_of_n_adaptive(
            gen, "// p", initial_candidates=2, max_candidates=6,
            language="typescript", tsc_runner=runner,
        )
        assert len(result.candidates) == 6      # grew to the cap (2 -> 4 -> 6)
        assert len(gen.group_calls) == 3
        assert not result.best.verified

    def test_prefers_secure_among_verified_on_early_stop(self):
        from cola_coder.inference.best_of_n import generate_best_of_n_adaptive
        gen = FakeGroupGenerator(["const a = eval(x);", "const b: number = 1;"])
        runner = FakeTscRunner({})  # both verify clean
        result = generate_best_of_n_adaptive(
            gen, "// p", initial_candidates=2, max_candidates=6,
            language="typescript", tsc_runner=runner,
        )
        assert result.best.details["secure"] is True
        assert "eval" not in result.best.completion
