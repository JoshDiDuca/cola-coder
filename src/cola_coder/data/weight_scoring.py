"""Quality-weight scoring for tokenized .npy datasets.

Shared by ``prepare_data.py`` (single-source) and ``collect_data.py``
(multi-source) so both produce IDENTICAL quality-weight semantics from one
implementation. Each chunk is decoded back to text and scored with the
``code_scorer`` feature; the resulting per-chunk weights are written to the
prepare_data-convention ``<stem>.weights.npy`` sidecar, aligned 1:1 with the
(already deduped) chunks.

CRITICAL ordering: scoring must run AFTER any dedup that mutates the .npy, so
``weights[i]`` lines up with the surviving ``data[i]`` (the same invariant
prepare_data documents for its dedup-before-score step).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_chunk_weights(
    npy_path: str | Path,
    tokenizer,
    scorer,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Score every chunk of a tokenized .npy → a float32 weight array.

    The caller owns the ``scorer`` (a ``code_scorer.CodeScorer``) so this stays
    a pure, feature-gate-free loop. Returns a weights array aligned 1:1 with the
    rows of ``npy_path`` (``scorer.score_to_weight(scorer.score(text))`` per
    decoded chunk).
    """
    data = np.load(str(npy_path), mmap_mode="r")
    n = len(data)
    weights = np.zeros(n, dtype=np.float32)
    it: range | object = range(n)
    if progress:
        from tqdm import tqdm

        it = tqdm(range(n), desc="Scoring")
    for i in it:
        text = tokenizer.decode(data[i].tolist())
        weights[i] = scorer.score_to_weight(scorer.score(text))
    return weights


def score_npy_to_weights(
    npy_path: str | Path,
    tokenizer,
    *,
    progress: bool = False,
) -> tuple[str, np.ndarray] | tuple[None, None]:
    """Convenience: build a scorer (honoring the feature gate), score the file,
    and write ``<stem>.weights.npy``.

    Returns ``(weights_path, weights_array)`` on success, or ``(None, None)``
    when the ``code_scorer`` feature is disabled or unavailable — so a caller
    can surface that quality-weighted training will NOT be active rather than
    silently writing a meaningless sidecar.
    """
    try:
        from cola_coder.features.code_scorer import CodeScorer, is_enabled
    except ImportError:
        return None, None
    if not is_enabled():
        return None, None

    weights = compute_chunk_weights(npy_path, tokenizer, CodeScorer(), progress=progress)
    weights_path = str(Path(npy_path).with_suffix(".weights.npy"))
    np.save(weights_path, weights)
    return weights_path, weights
