"""MODEL-042: verifier-effort E2H curriculum scheduler.

Pure logic — tracks per-problem verified pass-rate, re-tags difficulty, fades
mastered problems while keeping a minimum active set.
"""

import pytest

from cola_coder.reasoning.curriculum_scheduler import VerifierEffortCurriculum


def _problems(keys):
    return [{"prompt": k, "test_code": "", "difficulty": "medium"} for k in keys]


class TestRecordingAndTier:
    def test_tier_reflects_latest_rate(self):
        c = VerifierEffortCurriculum(easy_below=0.8, hard_above=0.2)
        c.record("easyP", 1.0)
        c.record("hardP", 0.0)
        c.record("midP", 0.5)
        c.end_epoch()
        assert c.tier_for("easyP") == "easy"
        assert c.tier_for("hardP") == "hard"
        assert c.tier_for("midP") == "medium"

    def test_unseen_problem_is_medium(self):
        assert VerifierEffortCurriculum().tier_for("never") == "medium"

    def test_epoch_mean_of_multiple_records(self):
        c = VerifierEffortCurriculum()
        c.record("p", 1.0)
        c.record("p", 0.0)   # mean 0.5
        c.end_epoch()
        assert c.latest_rate("p") == pytest.approx(0.5)


class TestMastery:
    def test_not_mastered_before_streak(self):
        c = VerifierEffortCurriculum(mastery_threshold=0.9, mastery_streak=2)
        c.record("p", 1.0)
        c.end_epoch()
        assert c.is_mastered("p") is False           # only 1 epoch

    def test_mastered_after_streak(self):
        c = VerifierEffortCurriculum(mastery_threshold=0.9, mastery_streak=2)
        for _ in range(2):
            c.record("p", 1.0)
            c.end_epoch()
        assert c.is_mastered("p") is True

    def test_streak_resets_on_dip(self):
        c = VerifierEffortCurriculum(mastery_threshold=0.9, mastery_streak=2)
        c.record("p", 1.0)
        c.end_epoch()
        c.record("p", 0.3)                            # dipped
        c.end_epoch()
        assert c.is_mastered("p") is False


class TestE2HFadeOut:
    def test_mastered_problems_faded(self):
        c = VerifierEffortCurriculum(mastery_threshold=0.9, mastery_streak=1, min_active=1)
        probs = _problems(["a", "b", "c"])
        # a mastered, b/c struggling.
        for k, r in (("a", 1.0), ("b", 0.2), ("c", 0.1)):
            c.record(k, r)
        c.end_epoch()
        active = c.active(probs, key_fn=lambda p: p["prompt"])
        keys = {p["prompt"] for p in active}
        assert "a" not in keys and keys == {"b", "c"}

    def test_min_active_floor_keeps_hardest_faded(self):
        # All mastered, but min_active=2 must re-include the 2 least-mastered.
        c = VerifierEffortCurriculum(mastery_threshold=0.9, mastery_streak=1, min_active=2)
        probs = _problems(["a", "b", "c"])
        for k, r in (("a", 1.0), ("b", 0.95), ("c", 0.92)):
            c.record(k, r)
        c.end_epoch()
        active = c.active(probs, key_fn=lambda p: p["prompt"])
        assert len(active) == 2
        # The two LEAST-mastered (lowest rate: c=0.92, b=0.95) are re-included.
        assert {p["prompt"] for p in active} == {"b", "c"}

    def test_never_empties(self):
        c = VerifierEffortCurriculum(mastery_threshold=0.5, mastery_streak=1, min_active=1)
        probs = _problems(["only"])
        c.record("only", 1.0)
        c.end_epoch()
        active = c.active(probs, key_fn=lambda p: p["prompt"])
        assert len(active) >= 1

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            VerifierEffortCurriculum(mastery_threshold=1.5)
        with pytest.raises(ValueError):
            VerifierEffortCurriculum(mastery_streak=0)
