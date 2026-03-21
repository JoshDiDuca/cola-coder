"""Checkpoint averaging for better generalization.

Averaging multiple checkpoints from different points in training can improve
model quality without any additional training cost. The intuition is that
different checkpoints explore different regions of the loss landscape — their
average sits in a flatter, better-generalizing region.

Two methods:
- Uniform average: simple mean of all K checkpoints. Good default.
- EMA (exponential moving average): newer checkpoints get more weight.
  Applies from oldest to newest with: w = decay * w + (1 - decay) * new_w

For a TS dev: think of uniform as Array.reduce((acc, v) => acc + v) / n,
and EMA as a running average where recent values count more.

Weight tying invariant (CRITICAL):
  tok_emb.weight and output.weight share the same tensor. Saved checkpoints
  EXCLUDE output.weight. The averaged checkpoint must also exclude it so the
  model constructor can re-tie them on load.

torch.compile invariant:
  Saved checkpoints always use clean keys (no _orig_mod. prefix). The averager
  works with clean keys throughout and saves clean keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class AverageResult:
    """Result of a checkpoint averaging operation."""

    output_path: str
    num_checkpoints: int
    method: str  # "uniform" or "ema"
    checkpoint_paths: list[str] = field(default_factory=list)


class CheckpointAverager:
    """Average weights from multiple checkpoints for better generalization.

    Usage:
        averager = CheckpointAverager(model_config)
        result = averager.average_last_k("checkpoints/tiny", k=3)
        # Loads the averaged model with: load_checkpoint(result.output_path, model)
    """

    def __init__(self, model_config=None):
        # model_config is accepted but not strictly required for averaging —
        # the averager works directly on saved state dicts.
        self.config = model_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def uniform_average(
        self,
        checkpoint_paths: list[str],
        output_path: str,
    ) -> AverageResult:
        """Simple average: new_weight = mean(weight_1, ..., weight_K).

        Memory-efficient: accumulates in float32 without loading all
        checkpoints simultaneously (each loaded, added to accumulator, freed).

        Args:
            checkpoint_paths: Ordered list of checkpoint directories.
            output_path: Directory to write the averaged checkpoint.

        Returns:
            AverageResult with details about the operation.

        Raises:
            FileNotFoundError: If any checkpoint is missing model.safetensors.
            ValueError: If checkpoints have mismatched state dict keys.
        """
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty")

        k = len(checkpoint_paths)
        resolved = [self._resolve_ckpt_dir(p) for p in checkpoint_paths]
        self._validate_checkpoints(resolved)

        # Load first checkpoint as the accumulator (float32 for numeric stability)
        acc = self._load_state_dict(resolved[0])
        acc = {k_: v.to(torch.float32) for k_, v in acc.items()}

        # Accumulate remaining checkpoints
        for ckpt_dir in resolved[1:]:
            sd = self._load_state_dict(ckpt_dir)
            for key in acc:
                acc[key].add_(sd[key].to(torch.float32))
            del sd

        # Divide by K to get the mean
        for key in acc:
            acc[key].div_(k)

        self._save_averaged(acc, output_path)

        return AverageResult(
            output_path=output_path,
            num_checkpoints=k,
            method="uniform",
            checkpoint_paths=[str(p) for p in resolved],
        )

    def exponential_moving_average(
        self,
        checkpoint_paths: list[str],
        output_path: str,
        decay: float = 0.999,
    ) -> AverageResult:
        """EMA: weight = decay * old_weight + (1 - decay) * new_weight.

        Applied from oldest to newest checkpoint. Newer checkpoints receive
        more influence because each update mixes them into the running average.

        Args:
            checkpoint_paths: Ordered list of checkpoint directories (oldest first).
            output_path: Directory to write the averaged checkpoint.
            decay: EMA decay factor. Higher = more weight on older checkpoints.
                0.999 is a sensible default for 3-10 checkpoints.

        Returns:
            AverageResult with details about the operation.

        Raises:
            FileNotFoundError: If any checkpoint is missing.
            ValueError: If decay not in (0, 1), or mismatched keys.
        """
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty")
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        k = len(checkpoint_paths)
        resolved = [self._resolve_ckpt_dir(p) for p in checkpoint_paths]
        self._validate_checkpoints(resolved)

        # Start from the oldest checkpoint
        ema = self._load_state_dict(resolved[0])
        ema = {k_: v.to(torch.float32) for k_, v in ema.items()}

        # Blend in newer checkpoints
        for ckpt_dir in resolved[1:]:
            sd = self._load_state_dict(ckpt_dir)
            for key in ema:
                # ema = decay * ema + (1 - decay) * new
                ema[key].mul_(decay).add_(sd[key].to(torch.float32) * (1.0 - decay))
            del sd

        self._save_averaged(ema, output_path)

        return AverageResult(
            output_path=output_path,
            num_checkpoints=k,
            method="ema",
            checkpoint_paths=[str(p) for p in resolved],
        )

    def average_last_k(
        self,
        checkpoint_dir: str,
        k: int = 3,
        output_path: str | None = None,
        method: str = "uniform",
        decay: float = 0.999,
    ) -> AverageResult:
        """Average the last K checkpoints in a directory.

        Scans for all step_* directories, sorts by step number (ascending),
        then takes the last K.

        Args:
            checkpoint_dir: Directory containing step_XXXXXXXX subdirectories.
            k: Number of most-recent checkpoints to average.
            output_path: Where to save the averaged checkpoint.
                Defaults to <checkpoint_dir>/averaged_last_{k}.
            method: "uniform" or "ema".
            decay: EMA decay (only used if method="ema").

        Returns:
            AverageResult with details about the operation.

        Raises:
            ValueError: If fewer than k checkpoints exist, or unknown method.
        """
        all_ckpts = self.find_checkpoints(checkpoint_dir)

        if len(all_ckpts) < k:
            raise ValueError(
                f"Requested last {k} checkpoints but only {len(all_ckpts)} found "
                f"in {checkpoint_dir!r}. "
                f"Found: {[Path(p).name for p in all_ckpts]}"
            )

        selected = all_ckpts[-k:]

        if output_path is None:
            output_path = str(Path(checkpoint_dir) / f"averaged_last_{k}")

        if method == "uniform":
            return self.uniform_average(selected, output_path)
        elif method == "ema":
            return self.exponential_moving_average(selected, output_path, decay=decay)
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'uniform' or 'ema'.")

    @staticmethod
    def find_checkpoints(checkpoint_dir: str) -> list[str]:
        """Find and sort all step_* checkpoint directories by step number.

        Args:
            checkpoint_dir: Directory to search for step_XXXXXXXX subdirs.

        Returns:
            List of absolute paths, sorted ascending by step number.
            Only includes directories that contain model.safetensors.
        """
        base = Path(checkpoint_dir)
        if not base.exists():
            return []

        pattern = re.compile(r"^step_(\d+)$")
        step_dirs: list[tuple[int, Path]] = []

        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if m and (entry / "model.safetensors").exists():
                step_dirs.append((int(m.group(1)), entry))

        step_dirs.sort(key=lambda x: x[0])
        return [str(p) for _, p in step_dirs]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ckpt_dir(path: str) -> Path:
        """Resolve a checkpoint path to an existing directory."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path!r}")
        if not p.is_dir():
            raise FileNotFoundError(f"Checkpoint path is not a directory: {path!r}")
        return p

    @staticmethod
    def _load_state_dict(ckpt_dir: Path) -> dict[str, torch.Tensor]:
        """Load model.safetensors from a checkpoint directory.

        Keys are always clean (no _orig_mod. prefix) because checkpoint.py
        strips the prefix on save. We strip it here defensively too.
        """
        model_path = ckpt_dir / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(
                f"model.safetensors not found in {ckpt_dir}. "
                "Is this a valid checkpoint directory?"
            )
        raw = load_file(str(model_path), device="cpu")

        # Defensively strip torch.compile prefix (checkpoint.py already does
        # this on save, but handle edge cases just in case)
        clean: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            clean_key = k.removeprefix("_orig_mod.")
            # Also skip output.weight if somehow present (tied weight alias)
            if clean_key == "output.weight":
                continue
            clean[clean_key] = v

        return clean

    @staticmethod
    def _validate_checkpoints(ckpt_dirs: list[Path]) -> None:
        """Verify all checkpoints have the same state dict keys.

        Raises:
            ValueError: If keys don't match between checkpoints.
        """
        if len(ckpt_dirs) <= 1:
            return

        # Load just keys from each checkpoint
        def _keys(p: Path) -> frozenset[str]:
            from safetensors import safe_open  # type: ignore

            with safe_open(str(p / "model.safetensors"), framework="pt", device="cpu") as f:
                raw_keys = set(f.keys())
            # Strip compile prefix and remove tied weight alias
            clean = frozenset(
                k.removeprefix("_orig_mod.")
                for k in raw_keys
                if k.removeprefix("_orig_mod.") != "output.weight"
            )
            return clean

        reference_keys = _keys(ckpt_dirs[0])
        for ckpt_dir in ckpt_dirs[1:]:
            keys = _keys(ckpt_dir)
            if keys != reference_keys:
                extra = keys - reference_keys
                missing = reference_keys - keys
                parts = []
                if extra:
                    parts.append(f"extra keys in {ckpt_dir.name}: {sorted(extra)}")
                if missing:
                    parts.append(f"missing keys in {ckpt_dir.name}: {sorted(missing)}")
                raise ValueError(
                    "Checkpoint key mismatch — cannot average incompatible models. "
                    + "; ".join(parts)
                )

    @staticmethod
    def _save_averaged(state_dict: dict[str, torch.Tensor], output_path: str) -> None:
        """Save averaged state dict as model.safetensors.

        Creates the output directory if it doesn't exist.
        Converts tensors back to float32 (the natural format for averaged weights;
        callers can cast when loading into a bf16/fp16 model).

        Note: output.weight is intentionally NOT included — the model constructor
        re-ties it to tok_emb.weight on load (weight tying invariant).
        """
        import shutil

        out_dir = Path(output_path)
        tmp_dir = out_dir.parent / f".tmp_{out_dir.name}"

        # Clean up any stale temp dir from a previous failed save
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Ensure all tensors are contiguous float32 before saving
        clean_state = {k: v.to(torch.float32).contiguous() for k, v in state_dict.items()}

        # Double-check output.weight is not present (weight tying invariant)
        clean_state.pop("output.weight", None)

        save_file(clean_state, str(tmp_dir / "model.safetensors"))

        # Atomic rename
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
