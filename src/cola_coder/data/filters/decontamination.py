"""Benchmark-decontamination filter plugin (DATA-065).

Contamination — eval/benchmark problems leaking into the training corpus — inflates
eval scores and is the central threat to honest evaluation (2026 surveys: n-gram /
containment overlap is the standard screen). The project already has a
`DataLeakageDetector` (MinHash + containment) but only as an offline REPORT
(scripts/check_contamination.py); it never DROPS contaminated samples during prep.

This wires the same primitives (`_shingles` / `_containment`, reused DRY) into the
data-prep filter chain: index the eval/benchmark texts once, then drop any training
record that CONTAINS one (containment of the eval doc's shingles within the train
record ≥ threshold — the right metric for a short eval problem embedded in a larger
scraped file). Opt-in like every filter; a no-op when no eval texts are configured.

Note: containment/n-gram catches verbatim + near-verbatim copies, NOT paraphrased /
rephrased leakage (arXiv:2311.04850) — for that, embedding/LLM detection is needed
(deferred). This is the cheap, high-precision first layer.
"""

from __future__ import annotations

from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter
# Reuse the canonical shingling + containment from the leakage detector (DRY).
from cola_coder.features.data_leakage_detector import _containment, _shingles


def _load_benchmark_texts() -> list[str]:
    """Best-effort load of built-in eval problem prompts (empty on any failure)."""
    try:
        from cola_coder.evaluation.problem_loader import get_all_problems
        return [getattr(p, "prompt", "") for p in get_all_problems() if getattr(p, "prompt", "")]
    except Exception:  # noqa: BLE001 — decontamination must never crash data prep
        return []


@register_filter("decontamination")
class DecontaminationFilter(FilterPlugin):
    """Drop training records that overlap eval/benchmark problems.

    Config options (via setup() or YAML):
        threshold: containment (|eval∩train|/|eval|) at/above which a record is
            contaminated and dropped (default 0.8).
        shingle_size: character n-gram size for shingling (default 5).
        eval_texts: explicit list of benchmark texts to decontaminate against.
        benchmark: when truthy, also load the built-in eval problem prompts.
    """

    def __init__(
        self,
        eval_texts: list[str] | None = None,
        threshold: float = 0.8,
        shingle_size: int = 5,
    ) -> None:
        self._threshold = threshold
        self._shingle_size = shingle_size
        self._eval_shingles = [_shingles(t, shingle_size) for t in (eval_texts or [])]

    def name(self) -> str:
        return f"decontamination(thr={self._threshold}, refs={len(self._eval_shingles)})"

    def setup(self, config: dict[str, Any]) -> None:
        self._threshold = float(config.get("threshold", self._threshold))
        self._shingle_size = int(config.get("shingle_size", self._shingle_size))
        texts = list(config.get("eval_texts", []))
        if config.get("benchmark"):
            texts += _load_benchmark_texts()
        self._eval_shingles = [_shingles(t, self._shingle_size) for t in texts if t]

    def check(self, record: DataRecord) -> tuple[bool, str]:
        if not self._eval_shingles:
            return True, ""  # nothing to decontaminate against → no-op
        train_sh = _shingles(record.content, self._shingle_size)
        if not train_sh:
            return True, ""
        worst = max(
            (_containment(ev, train_sh) for ev in self._eval_shingles), default=0.0
        )
        if worst >= self._threshold:
            return False, f"benchmark_contamination (containment={worst:.2f})"
        return True, ""
