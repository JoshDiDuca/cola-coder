"""Streaming Training: read training data from files on-the-fly.

Instead of loading an entire dataset into memory, this streams data directly
from files — yielding tokenized samples one at a time. Useful for large
datasets that don't fit in RAM.

For a TS dev: this is like using a Node.js ReadableStream or an async
generator over a file, rather than fs.readFileSync() into a giant buffer.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Iterator, List, Optional

from torch.utils.data import DataLoader, IterableDataset

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StreamingConfig:
    """Configuration for streaming dataset behaviour.

    Attributes:
        buffer_size:    Number of tokenized samples to hold in the prefetch
                        buffer.  Larger = more memory; smaller = less
                        randomness when shuffle_buffer > 0.
        shuffle_buffer: If > 0, use reservoir-sampling over this many items
                        before yielding.  Set to 0 to disable shuffling.
        skip_first:     Skip this many samples at the start of the stream
                        (useful for resuming).
        max_samples:    Stop after yielding this many samples total.
                        None means no limit.
    """
    buffer_size: int = 1_000
    shuffle_buffer: int = 256
    skip_first: int = 0
    max_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# Core IterableDataset
# ---------------------------------------------------------------------------

class StreamingDataset(IterableDataset):
    """An IterableDataset that reads text files line-by-line and tokenizes
    each line on-the-fly.

    Compatible with torch.utils.data.DataLoader (set num_workers=0 or
    implement worker-splitting for multi-worker support).

    Args:
        file_paths:   List of paths to text files (one sample per line).
        tokenizer_fn: A callable that takes a str and returns a list[int].
        config:       StreamingConfig controlling buffering / shuffling.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer_fn: Callable[[str], List[int]],
        config: Optional[StreamingConfig] = None,
    ) -> None:
        super().__init__()
        self.file_paths = list(file_paths)
        self.tokenizer_fn = tokenizer_fn
        self.config = config if config is not None else StreamingConfig()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: str) -> Generator[str, None, None]:
        """Generator that yields non-empty stripped lines from *path*."""
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line

    def _shuffle_buffer(
        self,
        items: Iterable[List[int]],
        buffer_size: int,
    ) -> Generator[List[int], None, None]:
        """Reservoir-sampling shuffle over *items* with a fixed *buffer_size*.

        Fills a buffer up to *buffer_size*, then for every new incoming item
        replaces a random position in the buffer and yields the displaced item.
        At the end, the remaining buffer is shuffled and drained.

        This gives approximate shuffling with O(buffer_size) memory, just like
        tf.data.Dataset.shuffle() — or like Fisher-Yates over a sliding window.
        """
        buf: List[List[int]] = []
        for item in items:
            if len(buf) < buffer_size:
                buf.append(item)
            else:
                idx = random.randrange(buffer_size)
                yield buf[idx]
                buf[idx] = item
        # drain remaining buffer in random order
        random.shuffle(buf)
        yield from buf

    def _raw_stream(self) -> Generator[List[int], None, None]:
        """Yield tokenized samples from all files in order."""
        for path in self.file_paths:
            for line in self._read_file(path):
                tokens = self.tokenizer_fn(line)
                if tokens:  # skip empty tokenizations
                    yield tokens

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        """Yield tokenized samples one at a time, applying skip / shuffle /
        max_samples from StreamingConfig."""
        cfg = self.config
        stream: Iterable[List[int]] = self._raw_stream()

        # Optional shuffle via reservoir sampling
        if cfg.shuffle_buffer > 0:
            stream = self._shuffle_buffer(stream, cfg.shuffle_buffer)

        emitted = 0
        skipped = 0

        for sample in stream:
            # Honour skip_first
            if skipped < cfg.skip_first:
                skipped += 1
                continue

            yield sample
            emitted += 1

            # Honour max_samples
            if cfg.max_samples is not None and emitted >= cfg.max_samples:
                return


# ---------------------------------------------------------------------------
# Convenience DataLoader wrapper
# ---------------------------------------------------------------------------

class StreamingDataLoader:
    """Thin wrapper around torch.utils.data.DataLoader that pairs it with a
    StreamingDataset.

    Usage::

        loader = StreamingDataLoader(file_paths, tokenizer_fn, config,
                                     batch_size=32)
        for batch in loader:
            ...  # batch is a list of token-id lists (or tensors if collate_fn given)

    Args:
        file_paths:   Forwarded to StreamingDataset.
        tokenizer_fn: Forwarded to StreamingDataset.
        config:       Forwarded to StreamingDataset.
        batch_size:   DataLoader batch size.
        num_workers:  DataLoader worker processes (use 0 for simple streaming).
        collate_fn:   Optional collate function; defaults to the DataLoader
                      default which returns lists.
        **kwargs:     Any extra keyword arguments forwarded to DataLoader.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer_fn: Callable[[str], List[int]],
        config: Optional[StreamingConfig] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        self.dataset = StreamingDataset(file_paths, tokenizer_fn, config)
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        loader_kwargs.update(kwargs)
        self._loader = DataLoader(self.dataset, **loader_kwargs)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        # IterableDataset has no __len__; raise informative error
        raise TypeError(
            "StreamingDataLoader has no fixed length — use max_samples in "
            "StreamingConfig to cap the stream."
        )


# ---------------------------------------------------------------------------
# Utility: estimate dataset size
# ---------------------------------------------------------------------------

def estimate_dataset_size(file_paths: List[str]) -> int:
    """Estimate the total number of samples (non-empty lines) across all files
    without loading any content into memory.

    This is O(total lines) in time and O(1) in memory — it just counts newlines
    using a buffered read, similar to ``wc -l``.

    Args:
        file_paths: List of paths to text files.

    Returns:
        Estimated number of samples (non-empty lines) across all files.
    """
    total = 0
    for path in file_paths:
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    # Count non-empty lines: each newline is a potential sample.
                    # We approximate by counting newlines; empty lines are rare
                    # in well-formed training data.
                    total += chunk.count(b"\n")
            # If the file doesn't end with a newline, the last line still counts.
            # Re-open cheaply to check last byte.
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size > 0:
                    fh.seek(-1, os.SEEK_END)
                    last = fh.read(1)
                    if last != b"\n":
                        total += 1
        except OSError:
            # File missing or unreadable — skip silently
            pass
    return total
