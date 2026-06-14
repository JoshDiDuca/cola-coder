"""Verifier-anchored LLM-judge calibration (EVAL-027).

2026 eval consensus: LLM-as-a-judge is scalable but biased (verbosity, position,
authority cues) and noisy — judge imperfections can invalidate statistical
guarantees unless calibrated against ground truth ("Noisy but Valid",
arXiv:2601.20913). For CODE, cola-coder owns the bias-free ground truth most
projects lack: the sandbox VERIFIER (tsc/tests pass = objectively correct).

This measures the LLM-judge against the verifier and corrects for its noise:
- ``agreement_stats``: the judge's True/False Positive Rate vs the verifier oracle,
  plus accuracy and Cohen's κ.
- ``corrected_prevalence``: Rogan-Gladen — recover the TRUE pass-rate from the
  judge's observed (noisy) pass-rate given its TPR/FPR, so a verbosity-biased judge
  doesn't inflate the reported quality of a corpus.
- ``best_score_threshold``: calibrate the judge's continuous score cut-point to best
  match the verifier (the threshold the project's LlmJudge / train_judge_classifier
  should distill at).

Pure logic — no model/sandbox — so it runs and tests with no GPU. Inputs are the
two verdict lists you already have after scoring a calibration set with BOTH the
judge and the verifier.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def confusion_counts(
    judge_pass: Sequence[bool], verifier_pass: Sequence[bool]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) treating the VERIFIER as ground truth.

    tp = judge says pass AND verifier passes; fp = judge says pass AND verifier fails.
    """
    if len(judge_pass) != len(verifier_pass):
        raise ValueError("judge_pass and verifier_pass must be the same length")
    tp = fp = tn = fn = 0
    for j, v in zip(judge_pass, verifier_pass):
        if v and j:
            tp += 1
        elif not v and j:
            fp += 1
        elif not v and not j:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def agreement_stats(
    judge_pass: Sequence[bool], verifier_pass: Sequence[bool]
) -> dict:
    """Judge reliability vs the verifier oracle: TPR, FPR, accuracy, Cohen's κ.

    TPR/FPR are None when the verifier has no positives / no negatives respectively
    (rate undefined). κ is None for a degenerate single-class set.
    """
    tp, fp, tn, fn = confusion_counts(judge_pass, verifier_pass)
    n = tp + fp + tn + fn
    pos = tp + fn   # verifier positives
    neg = fp + tn   # verifier negatives
    tpr = tp / pos if pos else None
    fpr = fp / neg if neg else None
    accuracy = (tp + tn) / n if n else 0.0

    kappa: float | None = None
    if n:
        p_o = (tp + tn) / n
        p_yes = ((tp + fp) / n) * (pos / n)
        p_no = ((fn + tn) / n) * (neg / n)
        p_e = p_yes + p_no
        kappa = (p_o - p_e) / (1 - p_e) if p_e != 1.0 else None

    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr, "accuracy": accuracy, "kappa": kappa,
    }


def corrected_prevalence(observed_pass_rate: float, tpr: float, fpr: float) -> float:
    """Rogan-Gladen: recover the TRUE pass-rate from a noisy judge's observed rate.

    true ≈ (observed − FPR) / (TPR − FPR), clamped to [0, 1]. Undefined when
    TPR == FPR (the judge carries no signal) → returns the observed rate unchanged.
    """
    denom = tpr - fpr
    if denom == 0:
        return max(0.0, min(1.0, observed_pass_rate))
    return max(0.0, min(1.0, (observed_pass_rate - fpr) / denom))


def best_score_threshold(
    judge_scores: Iterable[float],
    verifier_pass: Sequence[bool],
    metric: str = "accuracy",
) -> tuple[float, float]:
    """Find the judge-score cut-point (pass = score ≥ t) that best matches the verifier.

    Args:
        judge_scores: the LLM-judge's continuous scores for the calibration set.
        verifier_pass: the verifier's ground-truth verdicts (same order).
        metric: "accuracy" or "youden" (TPR − FPR, prevalence-robust).

    Returns:
        (threshold, best_metric_value). Empty input → (0.0, 0.0).
    """
    scores = list(judge_scores)
    if len(scores) != len(verifier_pass):
        raise ValueError("judge_scores and verifier_pass must be the same length")
    if not scores:
        return 0.0, 0.0
    if metric not in ("accuracy", "youden"):
        raise ValueError("metric must be 'accuracy' or 'youden'")

    best_t, best_val = scores[0], -1.0
    # Candidate cut-points: each unique score (pass when score >= t).
    for t in sorted(set(scores)):
        judged = [s >= t for s in scores]
        stats = agreement_stats(judged, verifier_pass)
        if metric == "accuracy":
            val = stats["accuracy"]
        else:
            tpr = stats["tpr"] if stats["tpr"] is not None else 0.0
            fpr = stats["fpr"] if stats["fpr"] is not None else 0.0
            val = tpr - fpr
        if val > best_val:
            best_val, best_t = val, t
    return best_t, best_val
