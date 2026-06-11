"""Score repositories based on test execution results.

Quality tiers:
    verified — Tests pass with >80% coverage (highest quality)
    tested   — Has tests and some pass
    detected — Tests exist but couldn't be run successfully
    none     — No tests found
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestResult:
    """Result of running a test suite."""

    framework: str  # "jest", "vitest", "pytest", "go_test", "cargo_test", etc.
    total_tests: int
    passed: int
    failed: int
    skipped: int
    error: str | None = None
    coverage: float | None = None  # 0.0-1.0 if available
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed (excluding skipped)."""
        runnable = self.total_tests - self.skipped
        if runnable <= 0:
            return 0.0
        return self.passed / runnable

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total_tests - self.skipped and self.passed > 0


@dataclass
class RepoScore:
    """Quality score for a repo based on test execution."""

    tests_detected: bool
    tests_pass: bool
    test_result: TestResult | None
    quality_tier: str  # "verified", "tested", "detected", "none"
    score: float  # 0.0-1.0 composite score
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        result_dict = None
        if self.test_result is not None:
            result_dict = {
                "framework": self.test_result.framework,
                "total_tests": self.test_result.total_tests,
                "passed": self.test_result.passed,
                "failed": self.test_result.failed,
                "skipped": self.test_result.skipped,
                "error": self.test_result.error,
                "coverage": self.test_result.coverage,
                "duration_seconds": self.test_result.duration_seconds,
            }
        return {
            "tests_detected": self.tests_detected,
            "tests_pass": self.tests_pass,
            "quality_tier": self.quality_tier,
            "score": self.score,
            "test_result": result_dict,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RepoScore:
        """Deserialize from a JSON-friendly dict."""
        result = None
        if d.get("test_result"):
            r = d["test_result"]
            result = TestResult(
                framework=r["framework"],
                total_tests=r["total_tests"],
                passed=r["passed"],
                failed=r["failed"],
                skipped=r["skipped"],
                error=r.get("error"),
                coverage=r.get("coverage"),
                duration_seconds=r.get("duration_seconds", 0.0),
            )
        return cls(
            tests_detected=d["tests_detected"],
            tests_pass=d["tests_pass"],
            test_result=result,
            quality_tier=d["quality_tier"],
            score=d["score"],
            details=d.get("details", {}),
        )


class TestScorer:
    """Score repos/files based on test execution results."""

    QUALITY_TIERS = {
        "verified": 1.0,  # Tests pass, >80% coverage
        "tested": 0.7,    # Has tests, some pass
        "detected": 0.4,  # Tests exist but couldn't run
        "none": 0.2,      # No tests found
    }

    # Training weight multipliers per tier
    TIER_WEIGHTS = {
        "verified": 3.0,
        "tested": 2.0,
        "detected": 1.0,
        "none": 0.5,
    }

    def score(self, test_result: TestResult | None, tests_detected: bool = False) -> RepoScore:
        """Score a repo based on its test execution results.

        Args:
            test_result: Result from running tests, or None if tests couldn't run.
            tests_detected: Whether test files/config were found even if tests
                           couldn't execute.

        Returns:
            RepoScore with quality tier and composite score.
        """
        if test_result is None:
            if tests_detected:
                return RepoScore(
                    tests_detected=True,
                    tests_pass=False,
                    test_result=None,
                    quality_tier="detected",
                    score=self.QUALITY_TIERS["detected"],
                    details={"reason": "Tests detected but could not be executed"},
                )
            return RepoScore(
                tests_detected=False,
                tests_pass=False,
                test_result=None,
                quality_tier="none",
                score=self.QUALITY_TIERS["none"],
                details={"reason": "No tests found"},
            )

        # Tests ran — determine tier
        if test_result.all_passed:
            if test_result.coverage is not None and test_result.coverage >= 0.8:
                tier = "verified"
            elif test_result.coverage is not None and test_result.coverage >= 0.5:
                # Good coverage but not great — still verified tier but lower score
                tier = "verified"
            else:
                # Tests pass but no/low coverage info
                tier = "tested"
        elif test_result.pass_rate >= 0.5:
            tier = "tested"
        elif test_result.total_tests > 0:
            tier = "detected"
        else:
            tier = "detected" if tests_detected else "none"

        # Compute composite score (0.0-1.0)
        base = self.QUALITY_TIERS[tier]
        score = base

        # Adjust score within tier based on pass rate and coverage
        if test_result.total_tests > 0:
            pass_bonus = test_result.pass_rate * 0.2  # up to +0.2
            score = min(1.0, base + pass_bonus)

        if test_result.coverage is not None:
            coverage_bonus = test_result.coverage * 0.1  # up to +0.1
            score = min(1.0, score + coverage_bonus)

        details = {
            "pass_rate": test_result.pass_rate,
            "coverage": test_result.coverage,
        }
        if test_result.error:
            details["error"] = test_result.error

        return RepoScore(
            tests_detected=True,
            tests_pass=test_result.all_passed,
            test_result=test_result,
            quality_tier=tier,
            score=round(score, 3),
            details=details,
        )

    def file_weight(self, score: RepoScore) -> float:
        """Training weight multiplier for files from this repo.

        Returns a multiplier (0.5-3.0) that can be used to weight training
        examples from this repo during data loading.
        """
        return self.TIER_WEIGHTS.get(score.quality_tier, 1.0)
