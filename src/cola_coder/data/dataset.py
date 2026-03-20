"""PyTorch Dataset for training data.

PyTorch's training loop needs a Dataset object that provides individual
training examples by index. This is like implementing an interface:

    interface Dataset {
        __len__(): number;          // How many examples?
        __getitem__(idx: number): { input_ids: Tensor, labels: Tensor };
    }

The DataLoader then wraps this Dataset and handles batching, shuffling,
and parallel loading (similar to how a web server handles requests).
"""

# Quality-weighted training: when a weights file is available, each training
# example is weighted by its quality score. High-quality code (weight > 1.0)
# contributes more to the loss, while low-quality code (weight < 1.0) contributes
# less. This teaches the model to produce code that looks like the good examples.
# Think of it like a playlist where your favorite songs play more often.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import load_processed_data


class CodeDataset(Dataset):
    """Dataset of tokenized code chunks for language model training.

    Each item is a sequence of token IDs. During training, the model
    tries to predict token[i+1] from token[i] for every position.
    """

    def __init__(self, data_path: str, max_seq_len: int | None = None):
        """
        Args:
            data_path: Path to the .npy file created by preprocess.py.
            max_seq_len: If set, truncate chunks to this length. This handles
                         the case where data was prepared with a larger chunk
                         size than the model's max_seq_len. If None, use full
                         chunk size.
        """
        # Memory-mapped: data stays on disk, loaded on demand
        self.data = load_processed_data(data_path)
        self.num_chunks = self.data.shape[0]
        self.chunk_size = self.data.shape[1]
        self.max_seq_len = max_seq_len

        if max_seq_len and max_seq_len < self.chunk_size:
            print(f"  Note: Data chunks are {self.chunk_size} tokens but model "
                  f"max_seq_len is {max_seq_len}. Truncating to {max_seq_len}.")

    def __len__(self) -> int:
        """Number of training examples."""
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get one training example by index.

        Returns:
            Dictionary with 'input_ids' tensor.
            Labels are the same as input_ids shifted by 1 (handled in the model).
        """
        # Convert from numpy to PyTorch tensor
        # .copy() is needed because mmap arrays are read-only
        chunk = self.data[idx]
        if self.max_seq_len and self.max_seq_len < len(chunk):
            chunk = chunk[:self.max_seq_len]
        tokens = torch.from_numpy(chunk.astype(np.int64).copy())
        return {"input_ids": tokens}


class WeightedCodeDataset(CodeDataset):
    """CodeDataset extended with per-example quality weights.

    Each example gets a scalar weight drawn from a .npy file of quality scores
    (produced by the quality filter pipeline). The weights are normalized so
    their mean is 1.0, which preserves the overall gradient scale while still
    letting high-quality examples contribute proportionally more to training.

    If no weights file is provided, every example gets weight 1.0 and training
    is identical to plain CodeDataset.
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int | None = None,
        weights_path: str | None = None,
    ):
        """
        Args:
            data_path: Path to the .npy file created by preprocess.py.
            max_seq_len: If set, truncate chunks to this length.
            weights_path: Path to a float32 .npy file of shape [num_chunks].
                          Each value is the quality score for the matching chunk.
                          If None, all weights default to 1.0.
        """
        super().__init__(data_path, max_seq_len=max_seq_len)

        if weights_path is not None:
            raw = np.load(weights_path).astype(np.float32)
            if len(raw) != self.num_chunks:
                raise ValueError(
                    f"weights file has {len(raw)} entries but data has "
                    f"{self.num_chunks} chunks — they must match exactly."
                )
            mean = raw.mean()
            if mean == 0:
                raise ValueError("All quality weights are zero; cannot normalize.")
            # Normalize so mean == 1.0, preserving overall gradient scale
            self.weights = torch.from_numpy(raw / mean)
        else:
            self.weights = torch.ones(self.num_chunks, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get one training example with its quality weight.

        Returns:
            Dictionary with 'input_ids' tensor and scalar 'weight' tensor.
        """
        item = super().__getitem__(idx)
        item["weight"] = self.weights[idx]
        return item


class WeightedCodeCollator:
    """Collates weighted examples into batches.

    Like CodeCollator, but also stacks the per-example 'weight' scalars into a
    1-D tensor so the training loop can apply them to the per-example losses.
    """

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Args:
            examples: List of dictionaries from WeightedCodeDataset.__getitem__.

        Returns:
            Batched dictionary:
                "input_ids": tensor of shape (batch_size, seq_len)
                "weights":   tensor of shape (batch_size,)
        """
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        weights = torch.stack([ex["weight"] for ex in examples])
        return {"input_ids": input_ids, "weights": weights}


class CodeCollator:
    """Collates individual examples into batches.

    For a TS dev: this is like a function that takes an array of individual
    items and combines them into a batch object. Since all our sequences
    are the same length (pre-chunked), this is just stacking tensors.
    """

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Args:
            examples: List of dictionaries from CodeDataset.__getitem__.

        Returns:
            Batched dictionary: {"input_ids": tensor of shape (batch_size, seq_len)}
        """
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        return {"input_ids": input_ids}


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int | None = None,
    weights_path: str | None = None,
) -> DataLoader:
    """Create a DataLoader for training.

    The DataLoader handles:
    - Shuffling examples each epoch (so the model sees data in random order)
    - Batching (grouping examples together for parallel processing)
    - Parallel data loading (num_workers threads loading data while GPU trains)
    - Pin memory (pre-loads data to GPU-friendly memory for faster transfer)

    Args:
        data_path: Path to preprocessed .npy file.
        batch_size: How many examples per batch.
        shuffle: Whether to randomize order each epoch.
        num_workers: Parallel data loading threads.
        max_seq_len: Truncate data chunks to this length (if they're longer
                     than the model's max_seq_len).
        weights_path: Optional path to a float32 .npy quality-weights file
                      (shape: [num_chunks]). When provided and the file exists,
                      uses WeightedCodeDataset + WeightedCodeCollator so the
                      training loop can apply per-example loss scaling. If None
                      or the file does not exist, falls back to the plain
                      CodeDataset + CodeCollator (fully backward compatible).

    Returns:
        PyTorch DataLoader ready for training.
    """
    import os

    use_weights = weights_path is not None and os.path.exists(weights_path)

    if use_weights:
        dataset = WeightedCodeDataset(data_path, max_seq_len=max_seq_len, weights_path=weights_path)
        collator = WeightedCodeCollator()
        mean_w = dataset.weights.mean().item()
        print(f"  Using quality-weighted training (mean weight: {mean_w:.2f})")
    else:
        dataset = CodeDataset(data_path, max_seq_len=max_seq_len)
        collator = CodeCollator()

    import torch
    use_pin_memory = torch.cuda.is_available()  # Only pin memory if GPU exists

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=use_pin_memory,  # Faster CPU→GPU transfer (only when GPU available)
        drop_last=True,  # Drop incomplete final batch (simpler training loop)
    )
