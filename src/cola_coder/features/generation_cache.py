"""Generation Cache: LRU cache for repeated prompt completions.

Avoids re-running inference for prompts that have been seen before.
Especially useful when running evaluation suites that contain repeated
prefixes, or when doing interactive development.

For a TS dev: like a Map<string, string> with a max-size eviction policy
(Least-Recently-Used) — exactly the same idea as a browser request cache.

Example::

    cache = GenerationCache(max_size=1000)
    result = cache.get(prompt)
    if result is None:
        result = model.generate(prompt)
        cache.set(prompt, result)
    print(cache.stats())
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached generation."""

    prompt_hash: str
    output: str
    created_at: float
    last_accessed: float
    hit_count: int = 0

    def touch(self) -> None:
        self.last_accessed = time.monotonic()
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cache usage statistics."""

    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float

    def __repr__(self) -> str:
        return (
            f"CacheStats(size={self.size}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"hits={self.hits}, misses={self.misses}, evictions={self.evictions})"
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class GenerationCache:
    """LRU cache for prompt -> completion pairs.

    Thread-safety: single-threaded only (no locking).

    Parameters
    ----------
    max_size:
        Maximum number of entries before LRU eviction kicks in.
    key_hash:
        Whether to hash prompt keys (saves memory and avoids very long dict keys).
    """

    def __init__(self, max_size: int = 512, key_hash: bool = True) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.key_hash = key_hash

        # OrderedDict maintains insertion/access order — most-recent at end
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> str | None:
        """Return cached output for *prompt*, or None if not cached.

        Accessing an entry moves it to the end (MRU position).
        """
        key = self._make_key(prompt)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        entry.touch()
        self._store.move_to_end(key)
        self._hits += 1
        return entry.output

    def set(self, prompt: str, output: str) -> None:
        """Store *output* for *prompt*, evicting LRU entry if at capacity."""
        key = self._make_key(prompt)
        now = time.monotonic()

        if key in self._store:
            # Update existing entry and move to MRU position
            self._store[key].output = output
            self._store[key].last_accessed = now
            self._store.move_to_end(key)
            return

        if len(self._store) >= self.max_size:
            # Evict least-recently-used (first item in OrderedDict)
            self._store.popitem(last=False)
            self._evictions += 1

        self._store[key] = CacheEntry(
            prompt_hash=key,
            output=output,
            created_at=now,
            last_accessed=now,
        )

    def invalidate(self, prompt: str) -> bool:
        """Remove a specific prompt from the cache.  Returns True if found."""
        key = self._make_key(prompt)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return CacheStats(
            size=len(self._store),
            max_size=self.max_size,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            hit_rate=hit_rate,
        )

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, prompt: str) -> bool:
        return self._make_key(prompt) in self._store

    def __repr__(self) -> str:
        s = self.stats()
        return f"GenerationCache(size={s.size}/{s.max_size}, hit_rate={s.hit_rate:.1%})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_key(self, prompt: str) -> str:
        if self.key_hash:
            return hashlib.sha256(prompt.encode()).hexdigest()
        return prompt
