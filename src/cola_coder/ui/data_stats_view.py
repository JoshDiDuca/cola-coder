"""Training-data statistics endpoint helper for the local UI.

Thin wrapper over the shared :func:`cola_coder.data.stats.compute_data_stats` —
the same numbers the CLI ``data_stats.py`` prints. Robust to missing data or a
missing NumPy: returns an ``{"error": ...}`` dict, never raises.
"""

from __future__ import annotations

from dataclasses import asdict

from cola_coder.data.stats import compute_data_stats


def data_stats(
    data_path: str | None = None,
    weights_path: str | None = None,
    estimate_unique: bool = True,
) -> dict:
    """Compute prepared-data statistics, discovering the data file if needed."""
    try:
        result = compute_data_stats(
            data_path, weights_path, estimate_unique=estimate_unique
        )
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except ImportError:
        return {"error": "numpy is required (pip install numpy)"}
    except Exception as exc:  # corrupt .npy, etc.
        return {"error": str(exc)}

    # asdict recurses into the nested WeightTier dataclasses → list[dict].
    return asdict(result)
