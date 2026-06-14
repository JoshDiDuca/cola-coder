"""Curriculum ordering — reorder training data by quality score.

Research shows that training on easy (high-quality) data first, then
progressively harder data, improves model performance (Arctic-SnowCoder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class CurriculumStrategy(str, Enum):
    EASY_TO_HARD = "easy_to_hard"     # High quality first (recommended)
    HARD_TO_EASY = "hard_to_easy"     # Low quality first
    STAGED = "staged"                  # Split into N quality phases
    RANDOM = "random"                  # Shuffle (baseline)
    FOLDING = "folding"                # Repeat easy->hard sweep L times (DELT, 2506.21545)


@dataclass
class CurriculumSchedule:
    """Describes the curriculum ordering and phase boundaries."""
    strategy: str
    total_samples: int
    phases: list[dict[str, object]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save schedule as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": self.strategy,
                "total_samples": self.total_samples,
                "phases": self.phases,
            }, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> CurriculumSchedule:
        """Load schedule from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return CurriculumSchedule(
            strategy=data["strategy"],
            total_samples=data["total_samples"],
            phases=data.get("phases", []),
        )


class CurriculumOrderer:
    """Reorder scored training data by difficulty for curriculum learning."""

    def __init__(
        self,
        strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD,
        num_phases: int = 3,
        num_folds: int = 4,
    ) -> None:
        self.strategy = strategy
        self.num_phases = max(1, num_phases)
        self.num_folds = max(1, num_folds)

    def _folding_order(self, weights: np.ndarray) -> np.ndarray:
        """Folding Ordering (DELT, arXiv:2506.21545).

        Sort easy->hard once, then stride the sorted sequence into L folds and
        concatenate them. Each fold is a strided sample spanning the full
        difficulty range in easy->hard order, so the model revisits the whole
        curriculum L times at fixed intervals. This fixes the forgetting,
        distribution bias, and duplication that pure single-pass sorting hits,
        while keeping every sample exactly once (a true permutation).
        """
        sorted_idx = np.argsort(-weights)  # easy / high-quality first
        n_folds = min(self.num_folds, len(sorted_idx)) or 1
        # Round-robin stride preserves easy->hard order within each fold.
        folds = [sorted_idx[f::n_folds] for f in range(n_folds)]
        return np.concatenate(folds) if folds else sorted_idx

    def reorder(
        self,
        data_path: str | Path,
        weights_path: str | Path,
        output_path: str | Path | None = None,
    ) -> CurriculumSchedule:
        """Reorder data by composite score.

        For easy_to_hard: highest quality first (descending score).
        For hard_to_easy: lowest quality first (ascending score).
        For random: shuffle randomly.

        Modifies both .npy and .weights.npy in place (or writes to output_path).

        Returns:
            CurriculumSchedule with phase information.
        """
        data = np.load(str(data_path))
        weights = np.load(str(weights_path))

        if len(data) != len(weights):
            raise ValueError(
                f"Data ({len(data)}) and weights ({len(weights)}) length mismatch"
            )

        if self.strategy == CurriculumStrategy.EASY_TO_HARD:
            order = np.argsort(-weights)  # Descending: best first
        elif self.strategy == CurriculumStrategy.HARD_TO_EASY:
            order = np.argsort(weights)   # Ascending: worst first
        elif self.strategy == CurriculumStrategy.RANDOM:
            rng = np.random.default_rng(42)
            order = rng.permutation(len(data))
        elif self.strategy == CurriculumStrategy.FOLDING:
            order = self._folding_order(weights)
        elif self.strategy == CurriculumStrategy.STAGED:
            return self._staged_reorder(data, weights, data_path, weights_path)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        reordered_data = data[order]
        reordered_weights = weights[order]

        out_data = str(output_path) if output_path else str(data_path)
        out_weights = str(Path(out_data).with_suffix(".weights.npy"))

        np.save(out_data, reordered_data)
        np.save(out_weights, reordered_weights)

        # Build phase info (split into num_phases for reporting)
        phases = self._compute_phases(reordered_weights)

        schedule = CurriculumSchedule(
            strategy=self.strategy.value,
            total_samples=len(data),
            phases=phases,
        )

        schedule_path = Path(out_data).with_suffix(".curriculum.json")
        schedule.save(schedule_path)

        return schedule

    def staged_reorder(
        self,
        data_path: str | Path,
        weights_path: str | Path,
        output_dir: str | Path,
    ) -> list[str]:
        """Split data into N phases by quality tier.

        Returns:
            List of paths to the N output .npy files.
        """
        data = np.load(str(data_path))
        weights = np.load(str(weights_path))

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Sort by quality descending
        order = np.argsort(-weights)
        data = data[order]
        weights = weights[order]

        # Split into N roughly equal chunks
        chunk_size = len(data) // self.num_phases
        paths: list[str] = []

        for i in range(self.num_phases):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_phases - 1 else len(data)
            phase_data = data[start:end]
            phase_weights = weights[start:end]

            phase_name = f"phase_{i + 1}_of_{self.num_phases}"
            data_out = out / f"{phase_name}.npy"
            weights_out = out / f"{phase_name}.weights.npy"

            np.save(str(data_out), phase_data)
            np.save(str(weights_out), phase_weights)
            paths.append(str(data_out))

        return paths

    def _staged_reorder(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        data_path: str | Path,
        weights_path: str | Path,
    ) -> CurriculumSchedule:
        """Handle staged strategy: sort then annotate phases."""
        order = np.argsort(-weights)  # Best first
        data = data[order]
        weights = weights[order]

        np.save(str(data_path), data)
        np.save(str(weights_path), weights)

        phases = self._compute_phases(weights)

        schedule = CurriculumSchedule(
            strategy="staged",
            total_samples=len(data),
            phases=phases,
        )

        schedule_path = Path(str(data_path)).with_suffix(".curriculum.json")
        schedule.save(schedule_path)

        return schedule

    def _compute_phases(self, weights: np.ndarray) -> list[dict[str, object]]:
        """Compute phase boundaries and statistics."""
        n = len(weights)
        if n == 0:
            return []

        chunk_size = n // self.num_phases
        phases: list[dict[str, object]] = []

        for i in range(self.num_phases):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_phases - 1 else n
            phase_weights = weights[start:end]

            phases.append({
                "phase": i + 1,
                "start_idx": int(start),
                "end_idx": int(end),
                "num_samples": int(end - start),
                "mean_score": float(np.mean(phase_weights)),
                "min_score": float(np.min(phase_weights)),
                "max_score": float(np.max(phase_weights)),
            })

        return phases
