"""Preprocess raw code data into training-ready token arrays.

The preprocessing pipeline:
1. Stream raw code files from HuggingFace
2. Tokenize each file using our BPE tokenizer
3. Concatenate all tokens into one long sequence
4. Chunk into fixed-length training examples (e.g., 2048 tokens each)
5. Save as memory-mapped numpy arrays for fast random access

Memory-mapped arrays (mmap) are like virtual arrays that live on disk
but behave like they're in RAM. The OS loads pages on demand, so we can
have a 100GB dataset but only use a few GB of RAM at a time.

For a TS dev: think of mmap like lazy-loading — the data is on disk but
accessed transparently as if it were in memory.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from ..tokenizer.tokenizer_utils import CodeTokenizer


def tokenize_and_chunk(
    text_iterator: Iterator[str],
    tokenizer: CodeTokenizer,
    chunk_size: int = 2048,
    output_dir: str = "./data/processed",
    max_tokens: int | None = None,
) -> str:
    """Tokenize text and save as chunked memory-mapped arrays.

    Args:
        text_iterator: Iterator yielding code strings.
        tokenizer: Trained BPE tokenizer.
        chunk_size: Length of each training example in tokens.
        output_dir: Where to save the processed data.
        max_tokens: Stop after this many tokens (for testing).

    Returns:
        Path to the saved .npy file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Accumulate tokens in a buffer, then write chunks
    token_buffer: list[int] = []
    chunks: list[np.ndarray] = []
    total_tokens = 0

    print("Tokenizing and chunking data...")
    for text in tqdm(text_iterator, desc="Processing files"):
        # Tokenize the code file
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        token_buffer.extend(tokens)
        total_tokens += len(tokens)

        # Extract full chunks from the buffer
        while len(token_buffer) >= chunk_size:
            chunk = token_buffer[:chunk_size]
            chunks.append(np.array(chunk, dtype=np.uint16))
            token_buffer = token_buffer[chunk_size:]

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    # Convert all chunks to a single numpy array
    if not chunks:
        print("Warning: No data was processed!")
        # Create a minimal dummy array so the pipeline doesn't break
        chunks = [np.zeros(chunk_size, dtype=np.uint16)]

    data = np.stack(chunks)  # Shape: (num_chunks, chunk_size)

    # Save as .npy file (fast to load with np.load)
    output_file = str(out_path / "train_data.npy")
    np.save(output_file, data)

    print(f"Saved {len(chunks)} chunks of {chunk_size} tokens each")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Data shape: {data.shape}")
    print(f"File size: {Path(output_file).stat().st_size / 1e6:.1f} MB")
    print(f"Saved to: {output_file}")

    return output_file


def load_processed_data(path: str) -> np.ndarray:
    """Load preprocessed data from disk.

    Uses memory-mapping so we don't load the entire file into RAM.

    Args:
        path: Path to the .npy file.

    Returns:
        Memory-mapped numpy array, shape (num_chunks, chunk_size).
    """
    return np.load(path, mmap_mode="r")
