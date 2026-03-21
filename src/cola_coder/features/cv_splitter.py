"""Cross-Validation Splitter — Feature 97

Create k-fold cross-validation splits for code datasets.

Design principles
-----------------
- Respects **file boundaries**: chunks from the same file always end up in
  the same fold, preventing data leakage between train and validation.
- Stratified option: tries to distribute files evenly by approximate size
  (number of lines / tokens) across folds.
- Deterministic: given the same seed the same splits are produced every time.
- Lightweight: no numpy or torch required — all pure Python.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the cross-validation splitter is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CodeFile:
    """Represents a single source file in the dataset."""

    path: str
    size: int = 0  # approximate size (lines, tokens, bytes — caller decides)
    group: Optional[str] = None  # optional grouping key (e.g. repo name)

    def __hash__(self) -> int:
        return hash(self.path)


@dataclass
class CVFold:
    """A single cross-validation fold."""

    fold_index: int
    train_files: list[CodeFile]
    val_files: list[CodeFile]

    @property
    def n_train(self) -> int:
        return len(self.train_files)

    @property
    def n_val(self) -> int:
        return len(self.val_files)

    @property
    def train_size(self) -> int:
        return sum(f.size for f in self.train_files)

    @property
    def val_size(self) -> int:
        return sum(f.size for f in self.val_files)

    def as_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_size": self.train_size,
            "val_size": self.val_size,
        }


@dataclass
class CVReport:
    """Summary of a k-fold split."""

    k: int
    n_files: int
    folds: list[CVFold]
    seed: Optional[int]

    @property
    def avg_val_fraction(self) -> float:
        if not self.folds:
            return 0.0
        fracs = [
            f.n_val / (f.n_train + f.n_val)
            for f in self.folds
            if f.n_train + f.n_val > 0
        ]
        return sum(fracs) / len(fracs) if fracs else 0.0

    def fold(self, idx: int) -> CVFold:
        return self.folds[idx]

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "n_files": self.n_files,
            "seed": self.seed,
            "avg_val_fraction": self.avg_val_fraction,
            "folds": [f.as_dict() for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class CrossValidationSplitter:
    """Create k-fold CV splits that respect file boundaries."""

    def __init__(self, k: int = 5, seed: Optional[int] = None) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self.k = k
        self.seed = seed

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def split(
        self,
        files: list[CodeFile],
        stratify: bool = False,
    ) -> CVReport:
        """Create k-fold splits.

        Parameters
        ----------
        files:
            List of :class:`CodeFile` objects to split.
        stratify:
            If True, distribute files across folds to balance total size.

        Returns
        -------
        CVReport
        """
        if not files:
            return CVReport(k=self.k, n_files=0, folds=[], seed=self.seed)

        rng = random.Random(self.seed)

        # Group files by their group key if provided, so group members stay together
        groups = self._group_files(files)
        group_list = list(groups.values())
        rng.shuffle(group_list)

        if stratify:
            group_list = self._sort_for_stratification(group_list)

        # Assign groups to folds
        fold_groups: list[list[list[CodeFile]]] = [[] for _ in range(self.k)]
        for i, group in enumerate(group_list):
            fold_groups[i % self.k].append(group)

        # Flatten each fold's files
        fold_files: list[list[CodeFile]] = [
            [f for grp in fg for f in grp] for fg in fold_groups
        ]

        folds: list[CVFold] = []
        for val_fold in range(self.k):
            val = fold_files[val_fold]
            train = [
                f
                for i, fold in enumerate(fold_files)
                for f in fold
                if i != val_fold
            ]
            folds.append(
                CVFold(
                    fold_index=val_fold,
                    train_files=train,
                    val_files=val,
                )
            )

        return CVReport(k=self.k, n_files=len(files), folds=folds, seed=self.seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_files(files: list[CodeFile]) -> dict[str, list[CodeFile]]:
        """Group files by their ``group`` key (or individual path if None)."""
        groups: dict[str, list[CodeFile]] = {}
        for f in files:
            key = f.group if f.group is not None else f.path
            groups.setdefault(key, []).append(f)
        return groups

    @staticmethod
    def _sort_for_stratification(
        groups: list[list[CodeFile]],
    ) -> list[list[CodeFile]]:
        """Sort groups by descending size for a round-robin stratification."""
        return sorted(groups, key=lambda g: sum(f.size for f in g), reverse=True)

    # ------------------------------------------------------------------
    # Determinism helper
    # ------------------------------------------------------------------

    @staticmethod
    def file_fold_id(path: str, k: int) -> int:
        """Deterministically assign a file to a fold by hashing its path.

        This is an alternative to shuffling — useful when folds are computed
        independently (e.g. distributed setting).
        """
        digest = hashlib.md5(path.encode()).hexdigest()
        return int(digest[:8], 16) % k

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    @staticmethod
    def check_leakage(fold: CVFold) -> list[str]:
        """Return paths that appear in both train and val of *fold*."""
        train_paths = {f.path for f in fold.train_files}
        val_paths = {f.path for f in fold.val_files}
        return sorted(train_paths & val_paths)
