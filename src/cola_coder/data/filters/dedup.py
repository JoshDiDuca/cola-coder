"""Near-duplicate removal using MinHash LSH.

Uses datasketch for efficient approximate deduplication. Files that are
near-duplicates of previously seen files are rejected.

datasketch is OPTIONAL. If not installed, this filter is a no-op that
passes everything through and logs a warning once.
"""

import logging

from cola_coder.data.registry import register_filter

logger = logging.getLogger(__name__)

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


@register_filter("deduplication")
class DeduplicationFilter:
    """Near-duplicate removal using MinHash LSH from datasketch.

    Uses character n-gram shingling to compute MinHash signatures, then
    queries an LSH index to find near-duplicates. If a file is too similar
    to one already seen, it's rejected.

    This is the same approach used by StarCoder, The Stack, and most
    large-scale code deduplication pipelines.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, ngram_size: int = 5):
        """
        Args:
            threshold: Jaccard similarity threshold. Files more similar
                than this to any previously seen file are rejected.
                0.8 is a good default for code dedup.
            num_perm: Number of permutations for MinHash.
                More = more accurate but slower. 128 is standard.
            ngram_size: Size of character n-grams for shingling.
                5 works well for code (captures variable names, keywords).
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self._warned = False
        self._count = 0

        if _HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        else:
            self.lsh = None

    def name(self) -> str:
        return "deduplication"

    def _compute_minhash(self, content: str) -> "MinHash":
        """Compute MinHash signature from character n-grams."""
        m = MinHash(num_perm=self.num_perm)
        # Use character n-grams (shingles)
        for i in range(len(content) - self.ngram_size + 1):
            shingle = content[i:i + self.ngram_size]
            m.update(shingle.encode("utf-8"))
        return m

    def check(self, record) -> tuple[bool, str]:
        """Check if the file is a near-duplicate of something already seen.

        Args:
            record: Object with .content (str) and .metadata (dict) attributes.

        Returns:
            (keep, reason) tuple.
        """
        if not _HAS_DATASKETCH:
            if not self._warned:
                logger.warning(
                    "DeduplicationFilter: datasketch not available, skipping dedup. "
                    "Install with: pip install datasketch"
                )
                self._warned = True
            return True, ""

        content = record.content
        if not content or len(content) < self.ngram_size:
            return True, ""

        minhash = self._compute_minhash(content)

        # Query LSH for near-duplicates
        try:
            results = self.lsh.query(minhash)
        except ValueError:
            # Can happen if LSH is in a bad state
            results = []

        if results:
            return False, f"near_duplicate (similar to {len(results)} existing file(s))"

        # Not a duplicate — insert into the index
        key = f"doc_{self._count}"
        self._count += 1
        try:
            self.lsh.insert(key, minhash)
        except ValueError:
            # Duplicate key or other issue — still keep the file
            pass

        return True, ""

    def setup(self, config: dict) -> None:
        """Optional setup from config dict."""
        if "threshold" in config:
            self.threshold = config["threshold"]
        if "num_perm" in config:
            self.num_perm = config["num_perm"]
        if "ngram_size" in config:
            self.ngram_size = config["ngram_size"]

    def reset(self) -> None:
        """Clear the dedup index. Useful between processing runs."""
        if _HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            self._count = 0
