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

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import load_processed_data


class CodeDataset(Dataset):
    """Dataset of tokenized code chunks for language model training.

    Each item is a sequence of token IDs. During training, the model
    tries to predict token[i+1] from token[i] for every position.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the .npy file created by preprocess.py.
        """
        # Memory-mapped: data stays on disk, loaded on demand
        self.data = load_processed_data(data_path)
        self.num_chunks = self.data.shape[0]
        self.chunk_size = self.data.shape[1]

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
        tokens = torch.from_numpy(self.data[idx].astype(np.int64).copy())
        return {"input_ids": tokens}


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

    Returns:
        PyTorch DataLoader ready for training.
    """
    dataset = CodeDataset(data_path)
    collator = CodeCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,  # Faster CPU→GPU transfer
        drop_last=True,  # Drop incomplete final batch (simpler training loop)
    )
