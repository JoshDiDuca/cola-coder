"""A/B Checkpoint Comparison: compare two model checkpoints to understand what changed.

Loads two checkpoints side-by-side and computes quantitative metrics:
- Parameter norms (per-layer, before and after)
- Weight differences (L2 distance, cosine similarity per layer)
- Config diffs (what hyperparameters changed between runs)
- Summary and top-k layers with the biggest changes

For a TS dev: this is like a git diff but for model weights — instead of line
changes you see which layers drifted most between checkpoint A and checkpoint B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def load_checkpoint_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dict from a checkpoint file or directory.

    Supports:
    - A ``.pt`` / ``.pth`` file (raw ``torch.save`` checkpoint dict or bare state dict)
    - A ``.safetensors`` file
    - A directory containing ``model.safetensors`` or ``pytorch_model.bin``

    Returns a flat ``{name: tensor}`` state dict (all tensors moved to CPU).
    """
    path = Path(path)

    if path.is_dir():
        # Try safetensors first, then pytorch_model.bin
        safetensors_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            path = safetensors_path
        elif bin_path.exists():
            path = bin_path
        else:
            # Fallback: look for any .pt file
            pt_files = list(path.glob("*.pt")) + list(path.glob("*.pth"))
            if not pt_files:
                raise FileNotFoundError(
                    f"No checkpoint file found in directory: {path}"
                )
            path = pt_files[0]

    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            return {k: v.cpu() for k, v in load_file(str(path)).items()}
        except ImportError as exc:
            raise ImportError(
                "safetensors package is required to load .safetensors files. "
                "Install with: pip install safetensors"
            ) from exc

    # torch.save format — may be a bare state dict or a checkpoint wrapper
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        # Common checkpoint wrapper keys
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                return {k: v.cpu() for k, v in raw[key].items()}
        # Assume it's already a state dict if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return {k: v.cpu() for k, v in raw.items()}
    raise ValueError(
        f"Could not extract a state dict from checkpoint: {path}"
    )


def weight_diff_summary(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Quick comparison of two state dicts.

    Returns a summary dict with:
    - ``total_params``: total parameter count (shared keys only)
    - ``mean_l2``: mean L2 distance across all matching layers
    - ``max_l2``: maximum L2 distance (the most-changed layer)
    - ``mean_cosine_sim``: mean cosine similarity across matching layers
    - ``only_in_a``: keys present in A but not B
    - ``only_in_b``: keys present in B but not A
    """
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    shared = sorted(keys_a & keys_b)

    l2_values: list[float] = []
    cos_values: list[float] = []
    total_params = 0

    for key in shared:
        ta = state_a[key].float().flatten()
        tb = state_b[key].float().flatten()
        if ta.shape != tb.shape:
            continue
        total_params += ta.numel()
        diff = ta - tb
        l2 = diff.norm(p=2).item()
        l2_values.append(l2)

        norm_a = ta.norm(p=2).item()
        norm_b = tb.norm(p=2).item()
        if norm_a > 0 and norm_b > 0:
            cos = torch.dot(ta, tb).item() / (norm_a * norm_b)
            cos_values.append(float(cos))

    return {
        "total_params": total_params,
        "shared_layers": len(shared),
        "mean_l2": float(sum(l2_values) / len(l2_values)) if l2_values else 0.0,
        "max_l2": float(max(l2_values)) if l2_values else 0.0,
        "mean_cosine_sim": float(sum(cos_values) / len(cos_values)) if cos_values else 0.0,
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
    }


# ---------------------------------------------------------------------------
# CheckpointComparison class
# ---------------------------------------------------------------------------

class CheckpointComparison:
    """Compare two model checkpoints to understand what changed.

    Accepts either pre-loaded state dicts (``dict[str, Tensor]``) or file paths
    (``str`` / ``Path``) for both checkpoints.

    Example::

        comp = CheckpointComparison("runs/ckpt_5000.pt", "runs/ckpt_10000.pt")
        print(comp.summary())
        print(comp.biggest_changes(top_k=10))
    """

    def __init__(
        self,
        checkpoint_a: dict[str, torch.Tensor] | str | Path,
        checkpoint_b: dict[str, torch.Tensor] | str | Path,
    ) -> None:
        if isinstance(checkpoint_a, (str, Path)):
            self.state_a = load_checkpoint_weights(checkpoint_a)
            self.path_a: str | None = str(checkpoint_a)
        else:
            self.state_a = {k: v.cpu() for k, v in checkpoint_a.items()}
            self.path_a = None

        if isinstance(checkpoint_b, (str, Path)):
            self.state_b = load_checkpoint_weights(checkpoint_b)
            self.path_b: str | None = str(checkpoint_b)
        else:
            self.state_b = {k: v.cpu() for k, v in checkpoint_b.items()}
            self.path_b = None

        self._shared_keys = sorted(set(self.state_a.keys()) & set(self.state_b.keys()))

    # ------------------------------------------------------------------
    # Core comparison methods
    # ------------------------------------------------------------------

    def compare_weights(self) -> dict[str, dict[str, float]]:
        """Compute L2 distance and cosine similarity between matching parameters.

        Returns a dict keyed by parameter name, each value containing:
        - ``l2_distance``: Euclidean distance between flattened weight vectors
        - ``cosine_similarity``: cosine similarity (1.0 = identical direction)
        - ``shape``: tuple representing the parameter shape
        - ``numel``: number of elements
        """
        result: dict[str, dict[str, float]] = {}

        for key in self._shared_keys:
            ta = self.state_a[key].float().flatten()
            tb = self.state_b[key].float().flatten()
            if ta.shape != tb.shape:
                continue

            diff = ta - tb
            l2 = diff.norm(p=2).item()

            norm_a = ta.norm(p=2).item()
            norm_b = tb.norm(p=2).item()
            if norm_a > 0 and norm_b > 0:
                cos_sim = torch.dot(ta, tb).item() / (norm_a * norm_b)
            else:
                cos_sim = 1.0 if norm_a == norm_b else 0.0

            result[key] = {
                "l2_distance": float(l2),
                "cosine_similarity": float(cos_sim),
                "shape": tuple(self.state_a[key].shape),
                "numel": int(ta.numel()),
            }

        return result

    def compare_norms(self) -> dict[str, dict[str, float]]:
        """Compare parameter norms between the two checkpoints.

        Returns a dict keyed by parameter name, each value containing:
        - ``norm_a``: L2 norm of parameter in checkpoint A
        - ``norm_b``: L2 norm of parameter in checkpoint B
        - ``norm_delta``: absolute difference in norms (norm_b - norm_a)
        - ``norm_ratio``: norm_b / norm_a (>1 means weights grew, <1 means shrunk)
        """
        result: dict[str, dict[str, float]] = {}

        for key in self._shared_keys:
            ta = self.state_a[key].float()
            tb = self.state_b[key].float()
            if ta.shape != tb.shape:
                continue

            norm_a = ta.norm(p=2).item()
            norm_b = tb.norm(p=2).item()
            delta = norm_b - norm_a
            ratio = (norm_b / norm_a) if norm_a > 0 else float("inf")

            result[key] = {
                "norm_a": float(norm_a),
                "norm_b": float(norm_b),
                "norm_delta": float(delta),
                "norm_ratio": float(ratio),
            }

        return result

    def compare_configs(
        self,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Diff two config dicts.

        Returns a dict with:
        - ``changed``: keys whose values differ, with ``{"a": old, "b": new}``
        - ``only_in_a``: keys present in A but missing from B
        - ``only_in_b``: keys present in B but missing from A
        - ``identical``: list of keys that are the same in both configs
        """
        keys_a = set(config_a.keys())
        keys_b = set(config_b.keys())
        shared = keys_a & keys_b

        changed: dict[str, dict[str, Any]] = {}
        identical: list[str] = []

        for key in sorted(shared):
            if config_a[key] != config_b[key]:
                changed[key] = {"a": config_a[key], "b": config_b[key]}
            else:
                identical.append(key)

        return {
            "changed": changed,
            "only_in_a": sorted(keys_a - keys_b),
            "only_in_b": sorted(keys_b - keys_a),
            "identical": identical,
        }

    def summary(self) -> dict[str, Any]:
        """High-level comparison summary.

        Returns a dict containing:
        - ``total_params_a`` / ``total_params_b``: total parameter counts
        - ``shared_layers``: number of layers present in both checkpoints
        - ``layers_only_in_a`` / ``layers_only_in_b``: unmatched layer names
        - ``mean_l2_distance``: average L2 distance across all layers
        - ``max_l2_distance``: largest per-layer L2 distance
        - ``mean_cosine_similarity``: average cosine similarity across layers
        - ``fraction_changed``: fraction of shared layers with l2 > 0
        - ``path_a`` / ``path_b``: source paths if loaded from files
        """
        weight_diffs = self.compare_weights()
        self.compare_norms()

        total_params_a = sum(t.numel() for t in self.state_a.values())
        total_params_b = sum(t.numel() for t in self.state_b.values())

        keys_a = set(self.state_a.keys())
        keys_b = set(self.state_b.keys())

        l2_values = [v["l2_distance"] for v in weight_diffs.values()]
        cos_values = [v["cosine_similarity"] for v in weight_diffs.values()]

        n = len(l2_values)
        mean_l2 = sum(l2_values) / n if n else 0.0
        max_l2 = max(l2_values) if l2_values else 0.0
        mean_cos = sum(cos_values) / n if n else 1.0
        changed_count = sum(1 for v in l2_values if v > 0.0)

        return {
            "total_params_a": total_params_a,
            "total_params_b": total_params_b,
            "shared_layers": len(self._shared_keys),
            "layers_only_in_a": sorted(keys_a - keys_b),
            "layers_only_in_b": sorted(keys_b - keys_a),
            "mean_l2_distance": float(mean_l2),
            "max_l2_distance": float(max_l2),
            "mean_cosine_similarity": float(mean_cos),
            "fraction_changed": float(changed_count / n) if n else 0.0,
            "path_a": self.path_a,
            "path_b": self.path_b,
        }

    def biggest_changes(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k layers with the largest weight changes (by L2 distance).

        Each entry in the returned list contains:
        - ``name``: parameter name
        - ``l2_distance``: L2 distance between the two weight tensors
        - ``cosine_similarity``: cosine similarity
        - ``norm_a`` / ``norm_b``: parameter norms
        - ``numel``: number of elements in the parameter
        """
        weight_diffs = self.compare_weights()
        norms = self.compare_norms()

        entries: list[dict[str, Any]] = []
        for key, wd in weight_diffs.items():
            norm_info = norms.get(key, {})
            entries.append(
                {
                    "name": key,
                    "l2_distance": wd["l2_distance"],
                    "cosine_similarity": wd["cosine_similarity"],
                    "norm_a": norm_info.get("norm_a", 0.0),
                    "norm_b": norm_info.get("norm_b", 0.0),
                    "numel": wd["numel"],
                }
            )

        entries.sort(key=lambda x: x["l2_distance"], reverse=True)
        return entries[:top_k]
