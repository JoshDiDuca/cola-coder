"""Preprocess raw code data into training-ready token arrays.

The preprocessing pipeline:
1. Stream raw code files from HuggingFace
2. Tokenize each file using our BPE tokenizer
3. Concatenate all tokens into one long sequence
4. Chunk into fixed-length training examples (e.g., 2048 tokens each)
5. Save as memory-mapped numpy arrays for fast random access

Performance architecture (producer-consumer pattern):
    [HF Stream] ---> Queue ---> [Batch Tokenizer] ---> [Chunk Writer]
       (I/O)         (64)        (Rust parallelism)      (memmap)

The producer thread reads from the HuggingFace stream (network I/O bound)
and buffers texts into a queue. The main thread pulls batches from the queue,
tokenizes them using the Rust-backed encode_batch() (CPU parallelized), and
streams chunks directly to a memory-mapped file on disk.

For a TS dev: this is like having a ReadableStream piped through a
TransformStream into a WritableStream, with backpressure via the queue.

Memory-mapped arrays (mmap) are like virtual arrays that live on disk
but behave like they're in RAM. The OS loads pages on demand, so we can
have a 100GB dataset but only use a few GB of RAM at a time.
"""

import os
import signal
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Iterator

import numpy as np
from tqdm import tqdm

from ..manifest import write_data_manifest
from ..tokenizer.tokenizer_utils import CodeTokenizer

# Sentinel value to signal the producer thread is done
_DONE = object()


def _producer_thread(
    text_iterator: Iterator[str],
    queue: Queue,
    batch_size: int,
):
    """Producer: reads from the HF stream and pushes batches into the queue.

    Runs in a separate thread so network I/O doesn't block tokenization.
    Groups texts into batches for efficient batch tokenization.
    """
    batch: list[str] = []
    try:
        for text in text_iterator:
            batch.append(text)
            if len(batch) >= batch_size:
                queue.put(batch)
                batch = []
        # Flush any remaining texts
        if batch:
            queue.put(batch)
    finally:
        queue.put(_DONE)


def tokenize_and_chunk(
    text_iterator: Iterator[str],
    tokenizer: CodeTokenizer,
    chunk_size: int = 2048,
    output_dir: str = "./data/processed",
    max_tokens: int | None = None,
    batch_size: int = 64,
    output_name: str = "train_data",
    *,
    manifest_info: dict | None = None,
) -> str:
    """Tokenize text and save as chunked memory-mapped arrays.

    Uses a producer-consumer pattern for overlapping I/O and tokenization,
    batch tokenization via the Rust backend, and streams chunks directly
    to a memory-mapped file to cap RAM usage.

    Args:
        text_iterator: Iterator yielding code strings (already quality-filtered).
        tokenizer: Trained BPE tokenizer.
        chunk_size: Length of each training example in tokens.
        output_dir: Where to save the processed data.
        max_tokens: Stop after this many tokens (for testing).
        batch_size: Number of files to tokenize at once (higher = more
                    efficient Rust parallelism, more memory). Default 64.
        manifest_info: Optional dict of extra metadata for the data manifest.
            Expected keys: dataset, languages, filter_mode, filter_stats,
            tokenizer_path, vocab_size, workers.

    Returns:
        Path to the saved .npy file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = str(out_path / f"{output_name}.npy")

    # --- Start producer thread to overlap I/O with tokenization ---
    queue: Queue = Queue(maxsize=16)  # Backpressure: max 16 batches buffered
    producer = threading.Thread(
        target=_producer_thread,
        args=(text_iterator, queue, batch_size),
        daemon=True,
    )
    producer.start()

    # --- Pre-allocate memmap for streaming writes ---
    # Start with space for 100k chunks, grow if needed
    initial_capacity = 100_000
    tmp_file = str(out_path / f"{output_name}_tmp.npy")
    mmap_data = _create_memmap(tmp_file, initial_capacity, chunk_size)
    capacity = initial_capacity
    num_chunks = 0

    # --- Graceful shutdown on Ctrl+C ---
    # Instead of losing all progress, finalize whatever we have so far.
    _interrupted = False

    def _handle_interrupt(signum, frame):
        nonlocal _interrupted
        if _interrupted:
            # Second Ctrl+C: hard exit
            raise KeyboardInterrupt
        _interrupted = True
        print("\n⏹ Ctrl+C received — finishing up and saving what we have...")

    old_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    # --- Tokenize and chunk ---
    token_buffer: list[int] = []
    total_tokens = 0
    total_files = 0
    start_time = time.time()

    pbar = tqdm(desc="Processing files", unit=" files")

    while not _interrupted:
        try:
            item = queue.get(timeout=30)
        except Empty:
            # If producer is still alive, keep waiting
            if producer.is_alive():
                continue
            break

        if item is _DONE:
            break

        batch_texts: list[str] = item
        total_files += len(batch_texts)

        # Batch tokenization — Rust backend parallelizes this internally
        token_lists = tokenizer.encode_batch(
            batch_texts, add_bos=True, add_eos=True,
        )

        for tokens in token_lists:
            token_buffer.extend(tokens)
            total_tokens += len(tokens)

        # Extract full chunks from the buffer
        while len(token_buffer) >= chunk_size:
            chunk = token_buffer[:chunk_size]
            token_buffer = token_buffer[chunk_size:]

            # Grow memmap if needed
            if num_chunks >= capacity:
                mmap_data, capacity = _grow_memmap(
                    mmap_data, tmp_file, capacity, chunk_size,
                )

            mmap_data[num_chunks] = np.array(chunk, dtype=np.uint16)
            num_chunks += 1

        pbar.update(len(batch_texts))

        # Update progress bar with throughput
        elapsed = time.time() - start_time
        if elapsed > 0:
            toks_per_sec = total_tokens / elapsed
            pbar.set_postfix(
                tokens=f"{total_tokens:,}",
                rate=f"{toks_per_sec:,.0f} tok/s",
                chunks=f"{num_chunks:,}",
            )

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    pbar.close()
    signal.signal(signal.SIGINT, old_handler)  # Restore original handler
    producer.join(timeout=5)

    # --- Finalize: trim memmap to actual size and save as .npy ---
    if num_chunks == 0:
        print("Warning: No data was processed!")
        data = np.zeros((1, chunk_size), dtype=np.uint16)
    else:
        # Read just the used portion and save as a proper .npy file
        data = np.array(mmap_data[:num_chunks])

    np.save(output_file, data)

    # Clean up temp file
    del mmap_data
    tmp_path = Path(tmp_file)
    if tmp_path.exists():
        tmp_path.unlink()

    # --- Summary ---
    elapsed = time.time() - start_time
    if _interrupted:
        print(f"\n⏹ INTERRUPTED — saved partial data ({num_chunks:,} chunks)")
        print(f"  This is perfectly usable for training!")
    print(f"\nSaved {num_chunks:,} chunks of {chunk_size} tokens each")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Total files processed: {total_files:,}")
    print(f"Data shape: {data.shape}")
    print(f"File size: {Path(output_file).stat().st_size / 1e6:.1f} MB")
    print(f"Wall time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    if elapsed > 0:
        print(f"Throughput: {total_tokens / elapsed:,.0f} tokens/sec, "
              f"{total_files / elapsed:,.0f} files/sec")
    print(f"Saved to: {output_file}")

    # --- Write data manifest ---
    manifest_path = str(out_path / f"{output_name}.manifest.yaml")
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    manifest_kwargs = {
        "output_file": f"{output_name}.npy",
        "output_size_bytes": Path(output_file).stat().st_size,
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_tokens": total_tokens,
        "dtype": "uint16",
        "total_files": total_files,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "wall_time_seconds": elapsed,
        "throughput_tokens_per_sec": throughput,
    }
    if manifest_info:
        manifest_kwargs.update(manifest_info)
    write_data_manifest(manifest_path, **manifest_kwargs)
    print(f"Manifest: {manifest_path}")

    return output_file


def _create_memmap(
    path: str, num_rows: int, num_cols: int,
) -> np.memmap:
    """Create a new memory-mapped file for streaming chunk writes."""
    return np.memmap(
        path, dtype=np.uint16, mode="w+", shape=(num_rows, num_cols),
    )


def _grow_memmap(
    old_mmap: np.memmap,
    path: str,
    old_capacity: int,
    chunk_size: int,
) -> tuple[np.memmap, int]:
    """Double the memmap capacity when we run out of space."""
    new_capacity = old_capacity * 2
    old_mmap.flush()

    # Re-open with larger shape (numpy memmap handles the file resize)
    new_mmap = np.memmap(
        path, dtype=np.uint16, mode="r+", shape=(new_capacity, chunk_size),
    )
    print(f"  [memmap] Grew capacity: {old_capacity:,} → {new_capacity:,} chunks")
    return new_mmap, new_capacity


def load_processed_data(path: str) -> np.ndarray:
    """Load preprocessed data from disk.

    Uses memory-mapping so we don't load the entire file into RAM.

    Args:
        path: Path to the .npy file.

    Returns:
        Memory-mapped numpy array, shape (num_chunks, chunk_size).
    """
    return np.load(path, mmap_mode="r")
