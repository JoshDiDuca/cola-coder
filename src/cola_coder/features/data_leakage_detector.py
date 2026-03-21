"""Data Leakage Detector — feature 46.

Checks whether evaluation/test data appears in training data using
MinHash-style fingerprinting for approximate near-duplicate detection.

Algorithm overview:
    1. Each document is shingle-hashed: split into overlapping n-grams of
       *words* (or character n-grams), hash each shingle.
    2. For each document compute a MinHash signature (``num_hashes`` minimum
       hash values over the shingle set).
    3. Two documents are "near-duplicates" (potential contamination) if their
       Jaccard similarity estimate exceeds ``similarity_threshold``.

This is a pure-Python implementation — no external libraries required — so it
can be tested in isolation.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → detector reports no contamination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if data leakage detection is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContaminationMatch:
    """A single suspected contamination pair."""

    eval_doc_id: int
    """Index of the evaluation/test document."""
    train_doc_id: int
    """Index of the training document most similar to it."""
    similarity: float
    """Estimated Jaccard similarity (0.0–1.0)."""
    eval_preview: str
    """First 100 chars of the evaluation document."""
    train_preview: str
    """First 100 chars of the training document."""

    def summary(self) -> str:
        return (
            f"eval[{self.eval_doc_id}] ↔ train[{self.train_doc_id}] "
            f"similarity={self.similarity:.3f}"
        )


@dataclass
class LeakageReport:
    """Results of a data leakage check."""

    num_eval_docs: int = 0
    num_train_docs: int = 0
    num_contaminated: int = 0
    contamination_rate: float = 0.0
    matches: List[ContaminationMatch] = field(default_factory=list)
    similarity_threshold: float = 0.8

    def has_leakage(self) -> bool:
        return self.num_contaminated > 0

    def summary(self) -> str:
        return (
            f"LeakageReport: eval={self.num_eval_docs} train={self.num_train_docs} "
            f"contaminated={self.num_contaminated} "
            f"rate={self.contamination_rate:.2%}"
        )


# ---------------------------------------------------------------------------
# MinHash implementation
# ---------------------------------------------------------------------------


def _fnv1a_hash(data: bytes, seed: int = 0) -> int:
    """FNV-1a hash with an integer seed mixed in."""
    h = 2166136261 ^ (seed * 0x9e3779b9 & 0xFFFFFFFF)
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _shingles(text: str, n: int) -> Set[str]:
    """Generate character-level n-gram shingles from text."""
    text = " ".join(text.lower().split())  # normalise whitespace
    if len(text) < n:
        return {text} if text else set()
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def _minhash(shingles: Set[str], num_hashes: int) -> List[int]:
    """Compute MinHash signature for a set of shingles."""
    if not shingles:
        return [0xFFFFFFFF] * num_hashes

    sig = [0xFFFFFFFF] * num_hashes
    for shingle in shingles:
        encoded = shingle.encode("utf-8", errors="replace")
        for seed in range(num_hashes):
            h = _fnv1a_hash(encoded, seed)
            if h < sig[seed]:
                sig[seed] = h
    return sig


def _jaccard_from_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


class DataLeakageDetector:
    """Detects training/eval data overlap using MinHash fingerprints.

    Usage::

        detector = DataLeakageDetector(similarity_threshold=0.8)
        detector.index_train(train_documents)
        report = detector.check_eval(eval_documents)
        if report.has_leakage():
            print(report.summary())
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        num_hashes: int = 128,
        shingle_size: int = 5,
    ) -> None:
        """
        Args:
            similarity_threshold: Documents with Jaccard similarity above this
                are flagged as potential contamination.
            num_hashes: Number of hash functions in the MinHash signature.
                More → more accurate but slower.
            shingle_size: Character n-gram size for shingling.
        """
        self.similarity_threshold = similarity_threshold
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self._train_docs: List[str] = []
        self._train_sigs: List[List[int]] = []

    def index_train(self, documents: Sequence[str]) -> None:
        """Build MinHash signatures for training documents.

        Args:
            documents: Training corpus documents.
        """
        if not FEATURE_ENABLED:
            return
        self._train_docs = list(documents)
        self._train_sigs = [
            _minhash(_shingles(doc, self.shingle_size), self.num_hashes)
            for doc in documents
        ]

    def check_eval(
        self, eval_documents: Sequence[str]
    ) -> LeakageReport:
        """Check evaluation documents against indexed training data.

        Args:
            eval_documents: Evaluation/test corpus documents.

        Returns:
            LeakageReport listing suspected contamination matches.
        """
        if not FEATURE_ENABLED:
            return LeakageReport(
                num_eval_docs=len(eval_documents),
                num_train_docs=len(self._train_docs),
            )

        report = LeakageReport(
            num_eval_docs=len(eval_documents),
            num_train_docs=len(self._train_docs),
            similarity_threshold=self.similarity_threshold,
        )

        for eval_idx, eval_doc in enumerate(eval_documents):
            eval_sig = _minhash(_shingles(eval_doc, self.shingle_size), self.num_hashes)
            best_sim = 0.0
            best_train_idx = -1

            for train_idx, train_sig in enumerate(self._train_sigs):
                sim = _jaccard_from_minhash(eval_sig, train_sig)
                if sim > best_sim:
                    best_sim = sim
                    best_train_idx = train_idx

            if best_sim >= self.similarity_threshold and best_train_idx >= 0:
                match = ContaminationMatch(
                    eval_doc_id=eval_idx,
                    train_doc_id=best_train_idx,
                    similarity=best_sim,
                    eval_preview=eval_doc[:100],
                    train_preview=self._train_docs[best_train_idx][:100],
                )
                report.matches.append(match)

        report.num_contaminated = len(report.matches)
        report.contamination_rate = report.num_contaminated / max(
            len(eval_documents), 1
        )
        return report

    def fingerprint(self, text: str) -> List[int]:
        """Return the MinHash fingerprint for a single text (for inspection)."""
        return _minhash(_shingles(text, self.shingle_size), self.num_hashes)

    @property
    def num_train_indexed(self) -> int:
        """Number of training documents indexed."""
        return len(self._train_docs)
