"""EVAL-027: verifier-anchored LLM-judge calibration.

Uses the sandbox verifier as bias-free ground truth to measure/correct an
LLM-judge's reliability. Pure logic — no model/sandbox.
"""

import pytest

from cola_coder.evaluation.judge_calibration import (
    agreement_stats,
    best_score_threshold,
    confusion_counts,
    corrected_prevalence,
)


class TestConfusion:
    def test_counts(self):
        judge = [True, True, False, False]
        verif = [True, False, True, False]
        tp, fp, tn, fn = confusion_counts(judge, verif)
        assert (tp, fp, tn, fn) == (1, 1, 1, 1)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            confusion_counts([True], [True, False])


class TestAgreementStats:
    def test_perfect_judge(self):
        v = [True, True, False, False]
        s = agreement_stats(v, v)
        assert s["tpr"] == 1.0 and s["fpr"] == 0.0
        assert s["accuracy"] == 1.0 and s["kappa"] == 1.0

    def test_biased_judge_high_fpr(self):
        # Judge passes everything (verbosity bias) — verifier fails half.
        judge = [True, True, True, True]
        verif = [True, True, False, False]
        s = agreement_stats(judge, verif)
        assert s["tpr"] == 1.0       # catches all real passes
        assert s["fpr"] == 1.0       # but also passes all real fails
        assert s["accuracy"] == 0.5

    def test_tpr_none_when_no_verifier_positives(self):
        s = agreement_stats([True, False], [False, False])
        assert s["tpr"] is None      # no positives → TPR undefined
        assert s["fpr"] is not None


class TestCorrectedPrevalence:
    def test_recovers_true_rate(self):
        # Judge with TPR=0.9, FPR=0.2 observed a 0.55 pass-rate.
        # true = (0.55 - 0.2) / (0.9 - 0.2) = 0.35 / 0.7 = 0.5
        assert corrected_prevalence(0.55, tpr=0.9, fpr=0.2) == pytest.approx(0.5)

    def test_clamped_to_unit_interval(self):
        assert corrected_prevalence(0.1, tpr=0.9, fpr=0.2) == 0.0   # would be negative
        assert corrected_prevalence(0.99, tpr=0.9, fpr=0.2) == 1.0  # would exceed 1

    def test_no_signal_judge_returns_observed(self):
        # TPR == FPR → judge carries no information.
        assert corrected_prevalence(0.4, tpr=0.5, fpr=0.5) == pytest.approx(0.4)


class TestBestThreshold:
    def test_finds_separating_threshold(self):
        # Scores cleanly separate at 0.5; verifier passes the high-scored ones.
        scores = [0.1, 0.2, 0.8, 0.9]
        verif = [False, False, True, True]
        t, acc = best_score_threshold(scores, verif, metric="accuracy")
        assert acc == 1.0
        assert 0.2 < t <= 0.8

    def test_youden_metric(self):
        scores = [0.1, 0.4, 0.6, 0.95]
        verif = [False, False, True, True]
        t, j = best_score_threshold(scores, verif, metric="youden")
        assert j == pytest.approx(1.0)   # perfect separation → TPR-FPR = 1

    def test_empty(self):
        assert best_score_threshold([], []) == (0.0, 0.0)

    def test_bad_metric(self):
        with pytest.raises(ValueError, match="metric"):
            best_score_threshold([0.5], [True], metric="f1")
