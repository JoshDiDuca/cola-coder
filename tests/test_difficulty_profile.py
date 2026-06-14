"""EVAL-026: verifier-effort difficulty profiling — tier classification + report."""

from dataclasses import dataclass

from cola_coder.evaluation.difficulty_profile import (
    profile_difficulty,
    verifier_effort_tier,
)


@dataclass
class _Res:
    candidates_used: int
    solved: bool


class TestVerifierEffortTier:
    def test_unsolved(self):
        assert verifier_effort_tier(8, 8, solved=False) == "unsolved"

    def test_easy_solved_cheaply(self):
        assert verifier_effort_tier(2, 8, solved=True) == "easy"      # 25%

    def test_medium(self):
        assert verifier_effort_tier(4, 8, solved=True) == "medium"    # 50%

    def test_hard_needed_most_of_budget(self):
        assert verifier_effort_tier(8, 8, solved=True) == "hard"      # 100%

    def test_boundaries(self):
        assert verifier_effort_tier(1, 8, True) == "easy"            # 12.5%
        assert verifier_effort_tier(3, 8, True) == "medium"          # 37.5%
        assert verifier_effort_tier(6, 8, True) == "hard"            # 75%

    def test_zero_max_candidates_safe(self):
        # No division-by-zero; solved with any usage falls to hard.
        assert verifier_effort_tier(1, 0, True) == "hard"


class TestProfileDifficulty:
    def test_distribution_and_rates(self):
        results = [
            _Res(2, True),    # easy
            _Res(2, True),    # easy
            _Res(4, True),    # medium
            _Res(8, True),    # hard
            _Res(8, False),   # unsolved
        ]
        report = profile_difficulty(results, max_candidates=8)
        assert report["n"] == 5
        assert report["tiers"] == {"easy": 2, "medium": 1, "hard": 1, "unsolved": 1}
        assert report["solve_rate"] == 0.8
        assert report["mean_candidates"] == (2 + 2 + 4 + 8 + 8) / 5

    def test_empty_results(self):
        report = profile_difficulty([], max_candidates=8)
        assert report["n"] == 0
        assert report["solve_rate"] == 0.0
        assert report["mean_candidates"] == 0.0
        assert report["tiers"]["easy"] == 0


class TestBestOfNExposesEffort:
    def test_fixed_n_populates_effort_fields(self):
        # End-to-end through generate_best_of_n with fakes (no GPU/tsc).
        import sys
        sys.path.insert(0, "tests")
        from test_best_of_n import FakeGroupGenerator, FakeTscRunner
        from cola_coder.inference.best_of_n import generate_best_of_n

        gen = FakeGroupGenerator(["const x: number = 1;", "const y: number = 2;"])
        runner = FakeTscRunner({})  # both verify
        result = generate_best_of_n(gen, "// p", num_candidates=2,
                                    language="typescript", tsc_runner=runner)
        assert result.candidates_used == 2
        assert result.rounds == 1
        assert result.solved is True
        assert result.final_temperature > 0

    def test_adaptive_counts_rounds_and_solved_false(self):
        import sys
        sys.path.insert(0, "tests")
        from test_best_of_n import FakeGroupGenerator, FakeTscRunner
        from cola_coder.inference.best_of_n import generate_best_of_n_adaptive

        gen = FakeGroupGenerator(["const x: number = 1;", "const y: number = 2;"])
        runner = FakeTscRunner({0: ["e"], 1: ["e"]})  # never verifies
        result = generate_best_of_n_adaptive(
            gen, "// p", initial_candidates=2, max_candidates=6, growth=2,
            language="typescript", tsc_runner=runner,
        )
        assert result.rounds == 3          # 2 -> 4 -> 6
        assert result.candidates_used == 6
        assert result.solved is False
