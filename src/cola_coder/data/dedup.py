"""Deduplication for Cola-Coder datasets.

Two levels of deduplication:
1. ExactDeduplicator  - SHA-256 hash-based exact duplicate removal (fast)
2. CrossDatasetDeduplicator - MinHash-based near-duplicate detection (thorough)

The MinHash approach requires the optional `datasketch` package. If it is not
installed, CrossDatasetDeduplicator falls back to exact dedup automatically.

All .npy files are expected to be 2D arrays of shape (num_chunks, chunk_size)
with dtype uint16.

Usage:
    from cola_coder.data.dedup import ExactDeduplicator

    dedup = ExactDeduplicator()
    clean_data, removed = dedup.deduplicate_array(data)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FileDedupResult:
    """Result of deduplicating a .npy file in place."""

    before: int
    after: int
    removed: int
    mode: str  # human-readable label, e.g. "exact (SHA-256)"
    minhash_active: bool  # True only if real MinHash ran (datasketch present)


def dedup_npy_file(
    path: str | Path,
    mode: str = "exact",
    threshold: float = 0.8,
    tokenizer=None,
) -> FileDedupResult:
    """Deduplicate a 2D token .npy file IN PLACE, atomically.

    Loads via mmap (so the full array isn't held in RAM twice), runs the
    requested deduplicator, and — only when rows were removed — rewrites the
    file via a temp file + os.replace.

    CRITICAL (Windows): the mmap handle on ``path`` MUST be released before
    os.replace, or Windows raises PermissionError because the file is still
    open. This function closes the memmap explicitly; callers must not rely on
    GC timing.

    Args:
        path: Path to the .npy file (2D, shape (num_chunks, chunk_size)).
        mode: "exact" (SHA-256) or "minhash" (near-dup; falls back to exact
            when datasketch is unavailable).
        threshold: Jaccard threshold for minhash mode.
        tokenizer: Optional tokenizer for minhash character n-grams.

    Returns:
        FileDedupResult with before/after/removed counts and the mode label.
    """
    path = Path(path)
    data = np.load(path, mmap_mode="r")
    before = len(data)

    if mode == "minhash":
        dedup = CrossDatasetDeduplicator(method="minhash", threshold=threshold)
        deduped, removed = dedup.deduplicate_self_array(data, tokenizer=tokenizer)
        minhash_active = dedup._use_minhash
        label = (
            f"minhash (Jaccard >= {threshold})"
            if minhash_active else "exact (minhash fallback)"
        )
    else:
        deduped, removed = ExactDeduplicator().deduplicate_array(data)
        minhash_active = False
        label = "exact (SHA-256)"

    after = len(deduped)
    if removed > 0:
        # Release the mmap handle BEFORE os.replace (Windows file lock).
        mm = getattr(data, "_mmap", None)
        del data
        if mm is not None:
            mm.close()
        tmp = path.with_suffix(".dedup_tmp.npy")
        np.save(tmp, deduped)
        os.replace(tmp, path)

    return FileDedupResult(
        before=before, after=after, removed=removed,
        mode=label, minhash_active=minhash_active,
    )

# Try to import datasketch for MinHash support
try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation."""
    input_chunks: int
    output_chunks: int
    duplicates_removed: int
    output_path: str = ""
    method: str = "exact"


class ExactDeduplicator:
    """Hash-based exact duplicate removal. Fast first pass.

    Uses SHA-256 of the raw token bytes. Two chunks are considered duplicates
    only if every single token matches. Memory cost: 32 bytes per unique chunk
    hash, so 10M chunks ~ 320MB RAM.
    """

    def __init__(self):
        self.seen_hashes: set[str] = set()

    def hash_chunk(self, tokens: np.ndarray) -> str:
        """SHA-256 hash of a token array (1D)."""
        return hashlib.sha256(tokens.tobytes()).hexdigest()

    def is_duplicate(self, tokens: np.ndarray) -> bool:
        """Check if this chunk is a duplicate of something we've already seen.

        If it's new, records it and returns False.
        If it's a repeat, returns True.
        """
        h = self.hash_chunk(tokens)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def deduplicate_array(self, data: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove exact duplicate chunks from a 2D array.

        Args:
            data: 2D array of shape (num_chunks, chunk_size).

        Returns:
            (deduped_data, num_removed) — the cleaned array and count of
            duplicates that were removed.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        self.seen_hashes.clear()
        keep_mask = np.ones(len(data), dtype=bool)

        for i in range(len(data)):
            if self.is_duplicate(data[i]):
                keep_mask[i] = False

        deduped = np.array(data[keep_mask])
        num_removed = int((~keep_mask).sum())

        logger.info(
            "Exact dedup: %d -> %d chunks (%d removed, %.1f%% dedup rate)",
            len(data), len(deduped), num_removed,
            100.0 * num_removed / max(len(data), 1),
        )

        return deduped, num_removed

    def reset(self):
        """Clear the hash set so this deduplicator can be reused."""
        self.seen_hashes.clear()


class CrossDatasetDeduplicator:
    """MinHash-based near-duplicate detection across datasets.

    Uses datasketch's MinHash LSH for scalable near-duplicate detection.
    If datasketch is not installed, falls back to exact dedup automatically.

    Pipeline:
    1. Build MinHash signatures for all chunks in the primary dataset
    2. For each chunk in the secondary dataset, query the index
    3. Mark near-duplicates for removal
    4. Output deduplicated dataset
    """

    def __init__(
        self,
        method: str = "minhash",
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 5,
    ):
        self.method = method
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size

        # Fall back to exact if datasketch isn't available
        if method == "minhash" and not _HAS_DATASKETCH:
            logger.warning(
                "datasketch not installed — falling back to exact dedup. "
                "Install with: pip install datasketch"
            )
            self.method = "exact"

        self._lsh: MinHashLSH | None = None
        self._index_count = 0

    @property
    def _use_minhash(self) -> bool:
        return self.method == "minhash" and _HAS_DATASKETCH

    def _tokens_to_ngrams(
        self,
        tokens: np.ndarray,
        tokenizer=None,
    ) -> list[str]:
        """Convert a token array to character n-grams for MinHash.

        If a tokenizer is provided, decodes tokens to text first and uses
        character n-grams. Otherwise uses token-level n-grams directly.
        """
        if tokenizer is not None:
            try:
                text = tokenizer.decode(tokens.tolist())
                # Character n-grams
                return [text[i:i + self.ngram_size]
                        for i in range(len(text) - self.ngram_size + 1)]
            except Exception:
                pass

        # Fallback: token-level n-grams (convert to strings)
        tok_strs = [str(t) for t in tokens]
        return [" ".join(tok_strs[i:i + self.ngram_size])
                for i in range(len(tok_strs) - self.ngram_size + 1)]

    def _make_minhash(
        self,
        tokens: np.ndarray,
        tokenizer=None,
    ) -> MinHash:
        """Create a MinHash signature for a chunk."""
        m = MinHash(num_perm=self.num_perm)
        ngrams = self._tokens_to_ngrams(tokens, tokenizer)
        for ng in ngrams:
            m.update(ng.encode("utf-8"))
        return m

    def _load_tokenizer(self, tokenizer_path: str | None):
        """Try to load a tokenizer for text decoding."""
        if tokenizer_path is None:
            return None
        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            return CodeTokenizer(tokenizer_path)
        except Exception:
            logger.warning("Could not load tokenizer from %s, using token n-grams",
                           tokenizer_path)
            return None

    def build_index(
        self,
        dataset_path: str,
        tokenizer_path: str | None = None,
    ) -> int:
        """Index a dataset for near-duplicate queries.

        Args:
            dataset_path: Path to a .npy file.
            tokenizer_path: Optional path to tokenizer.json for text decoding.

        Returns:
            Number of chunks indexed.
        """
        if not self._use_minhash:
            # For exact mode, we don't need an index — handled at query time
            data = np.load(dataset_path, mmap_mode="r")
            self._index_count = len(data)
            self._index_path = dataset_path
            return self._index_count

        tokenizer = self._load_tokenizer(tokenizer_path)
        data = np.load(dataset_path, mmap_mode="r")

        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._index_minhashes: dict[str, MinHash] = {}

        for i in range(len(data)):
            m = self._make_minhash(data[i], tokenizer)
            key = f"primary_{i}"
            self._lsh.insert(key, m)
            self._index_minhashes[key] = m

        self._index_count = len(data)
        logger.info("Built MinHash index with %d chunks from %s",
                     self._index_count, dataset_path)
        return self._index_count

    def find_duplicates(
        self,
        candidate_path: str,
        tokenizer_path: str | None = None,
    ) -> set[int]:
        """Find chunks in candidate that are duplicates of indexed data.

        Args:
            candidate_path: Path to .npy file to check.
            tokenizer_path: Optional tokenizer path.

        Returns:
            Set of chunk indices (in candidate) that are near-duplicates.
        """
        if not self._use_minhash:
            # Exact dedup fallback
            return self._find_exact_duplicates(candidate_path)

        if self._lsh is None:
            raise RuntimeError("Must call build_index() before find_duplicates()")

        tokenizer = self._load_tokenizer(tokenizer_path)
        data = np.load(candidate_path, mmap_mode="r")
        duplicates: set[int] = set()

        for i in range(len(data)):
            m = self._make_minhash(data[i], tokenizer)
            result = self._lsh.query(m)
            if result:
                duplicates.add(i)

        logger.info("Found %d near-duplicates in %s (of %d chunks)",
                     len(duplicates), candidate_path, len(data))
        return duplicates

    def _find_exact_duplicates(self, candidate_path: str) -> set[int]:
        """Fallback: find exact duplicates between indexed data and candidate."""
        if not hasattr(self, "_index_path"):
            raise RuntimeError("Must call build_index() before find_duplicates()")

        primary = np.load(self._index_path, mmap_mode="r")
        candidate = np.load(candidate_path, mmap_mode="r")

        # Build hash set of primary
        primary_hashes: set[str] = set()
        for i in range(len(primary)):
            h = hashlib.sha256(primary[i].tobytes()).hexdigest()
            primary_hashes.add(h)

        # Check candidate against primary
        duplicates: set[int] = set()
        for i in range(len(candidate)):
            h = hashlib.sha256(candidate[i].tobytes()).hexdigest()
            if h in primary_hashes:
                duplicates.add(i)

        logger.info("Found %d exact duplicates in candidate (of %d chunks)",
                     len(duplicates), len(candidate))
        return duplicates

    def deduplicate_self_array(
        self,
        data: np.ndarray,
        tokenizer=None,
    ) -> tuple[np.ndarray, int]:
        """Remove near-duplicate rows WITHIN a single 2D array (keep first seen).

        Mirrors ``ExactDeduplicator.deduplicate_array`` but with MinHash
        near-duplicate semantics: a chunk is dropped if its estimated Jaccard
        similarity to an already-kept chunk meets ``self.threshold``. Because
        identical chunks have Jaccard 1.0, this also subsumes exact dedup.

        When ``datasketch`` is unavailable (``_use_minhash`` is False) this
        transparently falls back to exact dedup — near-dups are NOT caught, so
        the caller should surface that to the user.

        Args:
            data: 2D array of shape (num_chunks, chunk_size).
            tokenizer: Optional tokenizer for character n-grams (else token
                n-grams are used directly — faster, no decode).

        Returns:
            (deduped_data, num_removed).
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        if not self._use_minhash:
            # Exact fallback — no near-dup detection without datasketch.
            return ExactDeduplicator().deduplicate_array(data)

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        keep_mask = np.ones(len(data), dtype=bool)

        for i in range(len(data)):
            m = self._make_minhash(data[i], tokenizer)
            if lsh.query(m):
                # A near-duplicate of a chunk we've already kept → drop it.
                keep_mask[i] = False
            else:
                lsh.insert(f"k_{i}", m)

        deduped = np.array(data[keep_mask])
        num_removed = int((~keep_mask).sum())

        logger.info(
            "MinHash near-dup self-dedup: %d -> %d chunks (%d removed, "
            "%.1f%% dedup rate, threshold=%.2f)",
            len(data), len(deduped), num_removed,
            100.0 * num_removed / max(len(data), 1), self.threshold,
        )
        return deduped, num_removed

    def deduplicate_pair(
        self,
        primary_path: str,
        secondary_path: str,
        tokenizer_path: str | None = None,
        output_path: str | None = None,
    ) -> DeduplicationResult:
        """Remove chunks from secondary that duplicate primary.

        Primary dataset is kept intact. Only secondary is modified.

        Args:
            primary_path: Path to primary .npy (kept as-is).
            secondary_path: Path to secondary .npy (duplicates removed).
            tokenizer_path: Optional tokenizer for text-based MinHash.
            output_path: Where to save deduplicated secondary. If None,
                         overwrites secondary_path.

        Returns:
            DeduplicationResult with stats.
        """
        if output_path is None:
            output_path = secondary_path

        self.build_index(primary_path, tokenizer_path)
        dup_indices = self.find_duplicates(secondary_path, tokenizer_path)

        secondary = np.load(secondary_path, mmap_mode="r")
        input_count = len(secondary)

        if dup_indices:
            keep_mask = np.ones(input_count, dtype=bool)
            for idx in dup_indices:
                keep_mask[idx] = False
            deduped = np.array(secondary[keep_mask])
        else:
            deduped = np.array(secondary)

        np.save(output_path, deduped)

        result = DeduplicationResult(
            input_chunks=input_count,
            output_chunks=len(deduped),
            duplicates_removed=len(dup_indices),
            output_path=output_path,
            method=self.method,
        )

        logger.info(
            "Dedup pair: %d -> %d chunks (%d removed) [%s]",
            result.input_chunks, result.output_chunks,
            result.duplicates_removed, result.method,
        )
        return result
