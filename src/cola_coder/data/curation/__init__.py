"""Test-driven data curation: score training data by whether the code actually works.

This module provides tools to:
1. Detect test frameworks in repositories (jest, vitest, pytest, go test, cargo test)
2. Run test suites in sandboxed environments (Docker or subprocess)
3. Score repositories based on test results and coverage
4. Weight training data by verified quality tiers

Usage:
    from cola_coder.data.curation import TestRunner, TestScorer, RepoScore, TestResult

    runner = TestRunner(mode="subprocess", timeout=300)
    result = runner.run_tests(Path("./my-repo"))
    score = runner.score_repo(Path("./my-repo"))
"""

from cola_coder.data.curation.test_scorer import RepoScore, TestResult, TestScorer
from cola_coder.data.curation.test_runner import TestRunner

__all__ = [
    "TestRunner",
    "TestScorer",
    "TestResult",
    "RepoScore",
]
