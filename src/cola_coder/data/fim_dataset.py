"""FIM-augmented Dataset wrapper.

Wraps any existing CodeDataset (or compatible dataset) and applies
Fill-in-the-Middle transformation on-the-fly during training.

Usage:
    base = CodeDataset("data/processed/train_data.npy")
    fim_ds = FIMDataset(base, tokenizer, fim_rate=0.5)
    loader = DataLoader(fim_ds, batch_size=32, collate_fn=CodeCollator())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .fim import FIMTransform

if TYPE_CHECKING:
    from ..tokenizer.tokenizer_utils import CodeTokenizer


class FIMDataset(Dataset):
    """Wraps an existing CodeDataset and adds on-the-fly FIM transformation.

    NOTE (DATA-025): the TRAINER's canonical dynamic-FIM path is
    ``data.dataset.create_dataloader(fim_rate=...)`` + ``FIMTrainingCollator``
    (weight-preserving, collator-based). This dataset-wrapper variant does the
    same FIM rearrangement per ``__getitem__`` and is provided for standalone
    use; do NOT stack it on top of a dataloader that already applies FIM, or
    each sample gets transformed twice (double-FIM, corrupt training data).

    Each call to __getitem__ independently decides (at random) whether to
    apply FIM, so the model sees a mix of standard and FIM examples every
    epoch — exactly as described in the Bavarian et al. paper.

    The wrapped dataset must return dicts with at least the key "input_ids"
    (a 1-D int64 tensor), which is the format produced by CodeDataset and
    WeightedCodeDataset.

    Args:
        base_dataset: Any dataset that yields {"input_ids": Tensor, ...}.
        tokenizer: CodeTokenizer instance (provides fim_*_id attributes).
        fim_rate: Fraction of examples transformed to FIM format [0, 1].
        psm_rate: Fraction of FIM examples using PSM vs SPM ordering [0, 1].
        seed: Optional seed for the transform's RNG.  When set, every index
              always returns the same transformation result, which is useful
              for deterministic testing but you usually want randomness in
              training (leave as None).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: "CodeTokenizer",
        fim_rate: float = 0.5,
        psm_rate: float = 0.5,
        seed: int | None = None,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.transform = FIMTransform(
            fim_rate=fim_rate,
            psm_rate=psm_rate,
            truncate_or_pad=True,
            seed=seed,
        )

    # Delegate length and pickling to the base dataset
    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getstate__(self) -> dict:
        """Support DataLoader pickling (Windows multiprocessing)."""
        return self.__dict__.copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a (possibly FIM-transformed) training example.

        Args:
            idx: Index into the base dataset.

        Returns:
            Dict matching the base dataset's format, with "input_ids"
            potentially rearranged into FIM format.
        """
        item = self.base[idx]
        input_ids: torch.Tensor = item["input_ids"]

        # Convert to Python list for FIMTransform, then back to tensor
        ids_list: list[int] = input_ids.tolist()
        transformed = self.transform.apply(ids_list, self.tokenizer)

        item = dict(item)  # shallow copy so we don't mutate the original
        item["input_ids"] = torch.tensor(transformed, dtype=torch.int64)

        return item


def create_fim_dataloader(
    data_path: str,
    tokenizer: "CodeTokenizer",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int | None = None,
    weights_path: str | None = None,
    fim_rate: float = 0.5,
    psm_rate: float = 0.5,
) -> "torch.utils.data.DataLoader":
    """Convenience factory: CodeDataset + FIMDataset + DataLoader in one call.

    Mirrors the signature of data.dataset.create_dataloader so it can be
    used as a drop-in replacement.

    Args:
        data_path: Path to preprocessed .npy file.
        tokenizer: CodeTokenizer instance.
        batch_size: Samples per batch.
        shuffle: Randomize order each epoch.
        num_workers: Parallel data-loading workers.
        max_seq_len: Truncate chunks longer than this.
        weights_path: Optional quality-weights .npy file path.
        fim_rate: Fraction of examples to transform to FIM format.
        psm_rate: Fraction of FIM examples using PSM ordering.

    Returns:
        DataLoader ready for training.
    """
    import os
    import torch
    from torch.utils.data import DataLoader

    from .dataset import CodeDataset, WeightedCodeDataset, CodeCollator, WeightedCodeCollator

    use_weights = weights_path is not None and os.path.exists(weights_path)

    if use_weights:
        base = WeightedCodeDataset(data_path, max_seq_len=max_seq_len, weights_path=weights_path)
        collator = WeightedCodeCollator()
    else:
        base = CodeDataset(data_path, max_seq_len=max_seq_len)
        collator = CodeCollator()

    dataset = FIMDataset(base, tokenizer, fim_rate=fim_rate, psm_rate=psm_rate)

    cpu_count = os.cpu_count() or 4
    if num_workers > cpu_count:
        num_workers = cpu_count

    use_pin_memory = torch.cuda.is_available()
    use_persistent = num_workers > 0

    print(f"  FIMDataset: fim_rate={fim_rate:.0%}, psm_rate={psm_rate:.0%}, "
          f"workers={num_workers}, batch_size={batch_size}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
