"""Hermetic tests for the function-step process-credit profiler (EVAL-034).

No GPU, no model, no real subprocess: every test injects a FAKE ``execute_fn``
that returns scripted pass/fail verdicts. Inputs are plain strings.
"""

from __future__ import annotations

from cola_coder.evaluation.process_credit import (
    FunctionStep,
    StepProfile,
    decompose_functions,
    function_step_scores,
    profile_candidates,
)


# ---------------------------------------------------------------------------
# Fake executors
# ---------------------------------------------------------------------------


def always_pass(code: str, timeout: float) -> tuple[bool, str]:
    return True, ""


def always_fail(code: str, timeout: float) -> tuple[bool, str]:
    return False, "boom"


class ScriptedExecutor:
    """Returns False when the executed code contains any 'fail marker' substring."""

    def __init__(self, fail_markers: list[str]) -> None:
        self.fail_markers = fail_markers
        self.calls: list[str] = []

    def __call__(self, code: str, timeout: float) -> tuple[bool, str]:
        self.calls.append(code)
        if any(m in code for m in self.fail_markers):
            return False, "scripted-fail"
        return True, "ok"


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_decompose_single_function() -> None:
    code = "def foo(x):\n    return x + 1\n"
    steps = decompose_functions(code)
    assert len(steps) == 1
    assert steps[0].name == "foo"
    assert isinstance(steps[0], FunctionStep)
    assert steps[0].lineno == 1


def test_decompose_nested_function() -> None:
    code = "def outer(x):\n    def inner(y):\n        return y\n    return inner(x)\n"
    names = {s.name for s in decompose_functions(code)}
    assert names == {"outer", "inner"}


def test_decompose_async_function() -> None:
    code = "async def fetch(url):\n    return await get(url)\n"
    steps = decompose_functions(code)
    assert len(steps) == 1
    assert steps[0].name == "fetch"


def test_decompose_empty_returns_empty() -> None:
    assert decompose_functions("") == []
    assert decompose_functions("   \n  ") == []


def test_decompose_unparseable_returns_empty() -> None:
    # Missing colon / dangling — not valid Python.
    assert decompose_functions("def broken(:\n  pass") == []


def test_decompose_multiple_top_level_in_order() -> None:
    code = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    steps = decompose_functions(code)
    assert [s.name for s in steps] == ["a", "b"]
    assert steps[0].lineno < steps[1].lineno


def test_ts_regex_fallback_function_decl() -> None:
    code = "function add(a, b) {\n  return a + b;\n}\n"
    steps = decompose_functions(code, language="typescript")
    assert [s.name for s in steps] == ["add"]


def test_ts_regex_fallback_arrow() -> None:
    code = "const double = (x) => {\n  return x * 2;\n};\n"
    steps = decompose_functions(code, language="typescript")
    assert "double" in {s.name for s in steps}


def test_auto_language_routes_ts() -> None:
    code = "function f(): number { return 1; }\nconst x: number = 2;\n"
    steps = decompose_functions(code, language="auto")
    assert any(s.name == "f" for s in steps)


# ---------------------------------------------------------------------------
# Attribution + scoring
# ---------------------------------------------------------------------------


def test_all_pass_gives_score_one() -> None:
    code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    profile = function_step_scores(code, tests, always_pass)
    assert isinstance(profile, StepProfile)
    assert profile.process_score == 1.0
    assert profile.steps[0]["name"] == "add"
    assert profile.steps[0]["n_tests"] == 2
    assert profile.steps[0]["score"] == 1.0


def test_named_assert_attributed_to_function() -> None:
    code = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    tests = "assert add(1, 2) == 3\nassert add(2, 2) == 4\n"
    profile = function_step_scores(code, tests, always_pass)
    by_name = {s["name"]: s for s in profile.steps}
    # add got both asserts; sub got none → falls back to executability probe.
    assert by_name["add"]["n_tests"] == 2
    assert by_name["sub"]["n_tests"] == 0


def test_one_dead_function_lowers_score() -> None:
    # add is tested+passes; helper is untested → probe pass → neutral 0.5.
    code = "def add(a, b):\n    return a + b\n\ndef helper():\n    return 0\n"
    tests = "assert add(1, 2) == 3\n"
    profile = function_step_scores(code, tests, always_pass)
    # Length-normalized: add weight 1.0 score 1.0; helper (untested, 2 lines)
    # weight 0.5 score 0.5 → (1.0 + 0.25) / 1.5 = 0.833...
    assert profile.process_score < 1.0
    assert profile.process_score > 0.5


def test_failing_attributed_test_lowers_step_score() -> None:
    code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\nassert add(2, 2) == 4\n"
    # Fail the assert mentioning '== 4'.
    execu = ScriptedExecutor(fail_markers=["== 4"])
    profile = function_step_scores(code, tests, execu)
    assert profile.steps[0]["score"] == 0.5  # 1 of 2 asserts pass
    assert profile.process_score == 0.5


def test_executability_probe_for_untested_function() -> None:
    # No asserts mention 'orphan' → probe. Probe defines the candidate + 'pass'.
    code = "def orphan():\n    return 42\n"
    tests = "x = 1\n"  # no asserts referencing orphan
    profile = function_step_scores(code, tests, always_pass)
    step = profile.steps[0]
    assert step["n_tests"] == 0
    assert step["executable"] is True
    assert step["score"] == 0.5


def test_non_executable_untested_function_scores_zero() -> None:
    code = "def orphan():\n    return 42\n"
    tests = "y = 2\n"
    profile = function_step_scores(code, tests, always_fail)
    step = profile.steps[0]
    assert step["executable"] is False
    assert step["score"] == 0.0


# ---------------------------------------------------------------------------
# Length normalization (verbosity-hack resistance)
# ---------------------------------------------------------------------------


def test_length_normalization_resists_verbose_vacuous_function() -> None:
    short_helper = "def helper():\n    return 0\n"
    long_helper = "def helper():\n" + "".join(f"    x{i} = {i}\n" for i in range(40)) + "    return 0\n"
    tested = "def add(a, b):\n    return a + b\n\n"
    tests = "assert add(1, 2) == 3\n"

    short_profile = function_step_scores(tested + short_helper, tests, always_pass)
    long_profile = function_step_scores(tested + long_helper, tests, always_pass)

    # The verbose vacuous helper must NOT inflate the score above the short one —
    # in fact its length discount drags the overall score DOWN (closer to add's
    # full-weight 1.0 the less the helper weighs).
    assert long_profile.process_score > short_profile.process_score
    # And both stay below 1.0 (a dead helper is present in both).
    assert short_profile.process_score < 1.0
    assert long_profile.process_score < 1.0


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------


def test_fragile_flags_dead_function_when_overall_passes() -> None:
    code = "def add(a, b):\n    return a + b\n\ndef never_called():\n    return 99\n"
    tests = "assert add(1, 2) == 3\n"
    profile = function_step_scores(code, tests, always_pass)
    # Overall passes (always_pass), never_called is untested → fragile.
    assert "never_called" in profile.fragile_functions
    assert "add" not in profile.fragile_functions


def test_no_fragile_when_overall_fails() -> None:
    code = "def add(a, b):\n    return a + b\n\ndef helper():\n    return 0\n"
    tests = "assert add(1, 2) == 999\n"
    execu = ScriptedExecutor(fail_markers=["== 999"])
    profile = function_step_scores(code, tests, execu)
    # Overall test fails → nothing is "riding along" on a passing candidate.
    assert profile.fragile_functions == []


# ---------------------------------------------------------------------------
# Determinism + batch
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    code = "def add(a, b):\n    return a + b\n\ndef helper():\n    return 0\n"
    tests = "assert add(1, 2) == 3\n"
    p1 = function_step_scores(code, tests, always_pass)
    p2 = function_step_scores(code, tests, always_pass)
    assert p1.process_score == p2.process_score
    assert p1.steps == p2.steps
    assert p1.fragile_functions == p2.fragile_functions


def test_profile_candidates_maps_over_list() -> None:
    candidates = [
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a - b\n",
    ]
    tests = "assert add(1, 2) == 3\n"
    execu = ScriptedExecutor(fail_markers=["return a - b"])
    rows = profile_candidates(candidates, tests, execu)
    assert len(rows) == 2
    assert rows[0]["index"] == 0
    assert rows[1]["index"] == 1
    # First candidate passes; second's add body fails its assert.
    assert rows[0]["process_score"] > rows[1]["process_score"]


def test_security_flag_on_dangerous_function() -> None:
    code = "def run(cmd):\n    return eval(cmd)\n"
    tests = "assert run('1+1') == 2\n"
    profile = function_step_scores(code, tests, always_pass, scan_security=True)
    assert profile.steps[0]["dangerous"]  # eval() flagged


def test_security_scan_can_be_disabled() -> None:
    code = "def run(cmd):\n    return eval(cmd)\n"
    tests = "assert run('1+1') == 2\n"
    profile = function_step_scores(code, tests, always_pass, scan_security=False)
    assert profile.steps[0]["dangerous"] == []
