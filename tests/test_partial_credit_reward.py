"""MODEL-041: fractional (partial-credit) test reward — EGCA-style finer credit.

Splits the test block into individual asserts and returns the fraction passed.
Uses an injected fake executor so no real sandbox/GPU is needed.
"""

from cola_coder.reasoning.rewards.partial_credit import (
    fractional_python_reward,
    split_test_cases,
)


def _fake_exec(pass_substrings):
    """Executor that 'passes' (True) iff the code contains any of the given marks."""
    def run(code: str, timeout: float):
        ok = any(s in code for s in pass_substrings)
        return ok, "ok" if ok else "AssertionError"
    return run


class TestSplitTestCases:
    def test_splits_asserts_and_setup(self):
        asserts, setup = split_test_cases(
            "import math\nx = 2\nassert add(1, 1) == 2\nassert add(2, 2) == 4"
        )
        assert len(asserts) == 2
        assert "import math" in setup and "x = 2" in setup

    def test_no_asserts_returns_empty(self):
        asserts, setup = split_test_cases("check(candidate)")  # custom harness, no assert
        assert asserts == []

    def test_syntax_error_returns_empty(self):
        assert split_test_cases("def broken(:\n  pass")[0] == []


class TestFractionalReward:
    def test_all_pass_is_one(self):
        # Executor passes any assert -> fraction 1.0.
        r, info = fractional_python_reward("sol", "assert a\nassert b", _fake_exec(["assert"]))
        assert r == 1.0
        assert info["num_tests"] == 2 and info["num_passed"] == 2
        assert info["mode"] == "fractional"

    def test_partial_pass_is_fractional(self):
        # Only the assert containing 'good' passes -> 1 of 2.
        test = "assert good_case()\nassert bad_case()"
        r, info = fractional_python_reward("sol", test, _fake_exec(["good"]))
        assert r == 0.5
        assert info["num_passed"] == 1 and info["num_tests"] == 2

    def test_none_pass_is_zero(self):
        r, info = fractional_python_reward("sol", "assert a\nassert b", _fake_exec(["NOPE"]))
        assert r == 0.0
        assert info["num_passed"] == 0

    def test_setup_is_shared_across_cases(self):
        # The case only passes if the SETUP line is present in the executed code.
        test = "helper = 1\nassert uses_helper()"
        seen = {}

        def run(code: str, timeout: float):
            seen["had_setup"] = "helper = 1" in code
            return seen["had_setup"], ""

        fractional_python_reward("sol", test, run)
        assert seen["had_setup"] is True

    def test_binary_fallback_without_asserts(self):
        # No top-level asserts -> whole-block all-or-nothing.
        r, info = fractional_python_reward("sol", "check(candidate)", _fake_exec(["check"]))
        assert r == 1.0
        assert info["mode"] == "binary_fallback"
        assert info["num_tests"] == 1


class TestRegistryWiring:
    def test_python_partial_registered(self):
        from cola_coder.reasoning.reward_registry import RewardRegistry
        assert RewardRegistry.is_registered("python_partial")
        fn = RewardRegistry.get("python_partial")
        assert callable(fn)
