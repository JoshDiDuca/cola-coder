"""Training Cost Estimator: estimate GPU hours and cloud costs for model training.

Given a model config and token count, compute realistic estimates for:
- GPU hours required
- Electricity cost
- Cloud provider pricing (AWS, GCP, Lambda Labs)

GPU throughput data is based on measured benchmarks (tokens/sec) from the
project's hardware estimates in CLAUDE.md plus published cloud GPU numbers.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if cost estimation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# GPU database
# ---------------------------------------------------------------------------

# Approximate training throughput in tokens/sec for a ~125M param model at full
# batch.  Scale roughly linearly with model size.
#
# Sources:
#   - RTX 4080: measured cola-coder benchmark (~45 tok/s at 125M)
#   - RTX 3080: measured cola-coder benchmark (~28 tok/s at 125M, fp16)
#   - A100 40GB: ~3-5x RTX 3090 class; widely cited ~200-250 tok/s at 125M bf16
#   - A100 80GB: ~similar throughput, better for larger batches
#   - H100 SXM: ~2x A100; ~400-500 tok/s at 125M
#   - RTX 4090: similar to A100 40GB in absolute compute; ~180-200 tok/s

GPU_SPECS: dict[str, dict[str, Any]] = {
    "rtx_3080": {
        "display_name": "RTX 3080 (10GB)",
        "tokens_per_sec_125m": 28.0,
        "vram_gb": 10,
        "tdp_watts": 320,
        # Cloud on-demand $/hour (approximate, 2024)
        "cloud_prices": {
            "own_hardware": 0.00,
        },
    },
    "rtx_4080": {
        "display_name": "RTX 4080 (16GB)",
        "tokens_per_sec_125m": 45.0,
        "vram_gb": 16,
        "tdp_watts": 320,
        "cloud_prices": {
            "own_hardware": 0.00,
        },
    },
    "rtx_4090": {
        "display_name": "RTX 4090 (24GB)",
        "tokens_per_sec_125m": 190.0,
        "vram_gb": 24,
        "tdp_watts": 450,
        "cloud_prices": {
            "vast_ai": 0.40,
            "lambda_labs": 0.50,
        },
    },
    "a100_40gb": {
        "display_name": "A100 40GB",
        "tokens_per_sec_125m": 230.0,
        "vram_gb": 40,
        "tdp_watts": 400,
        "cloud_prices": {
            "aws_p4d": 3.20,
            "gcp_a2": 3.67,
            "lambda_labs": 1.10,
            "vast_ai": 1.50,
        },
    },
    "a100_80gb": {
        "display_name": "A100 80GB",
        "tokens_per_sec_125m": 250.0,
        "vram_gb": 80,
        "tdp_watts": 400,
        "cloud_prices": {
            "aws_p4de": 4.20,
            "gcp_a2_ultra": 5.00,
            "lambda_labs": 1.29,
        },
    },
    "h100": {
        "display_name": "H100 SXM (80GB)",
        "tokens_per_sec_125m": 480.0,
        "vram_gb": 80,
        "tdp_watts": 700,
        "cloud_prices": {
            "aws_p5": 12.00,
            "gcp_a3": 10.00,
            "lambda_labs": 2.49,
            "vast_ai": 3.00,
        },
    },
}

# Electricity cost USD/kWh (US average 2024)
_DEFAULT_ELECTRICITY_USD_KWH = 0.13

# Overhead factor: PUE (Power Usage Effectiveness) for data centres ~1.2,
# plus CPU/networking overhead
_PUE_FACTOR = 1.2

# Scaling: tokens/sec scales roughly as 125M / n_params (linear approximation)
# More accurate: compute = 6 * params * tokens  (standard ML FLOPs estimate)
_REFERENCE_PARAMS = 125_000_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CostReport:
    """Full cost estimate from CostEstimator.estimate()."""

    gpu_type: str
    gpu_display_name: str
    n_params: int
    total_tokens: int
    tokens_per_sec: float
    training_hours: float
    training_days: float
    electricity_kwh: float
    electricity_cost_usd: float
    cloud_estimates: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            f"Training Cost Estimate — {self.gpu_display_name}",
            f"  Model parameters:  {self.n_params / 1e6:.0f}M",
            f"  Training tokens:   {self.total_tokens / 1e9:.2f}B",
            f"  Throughput:        {self.tokens_per_sec:.0f} tok/s",
            f"  Training time:     {self.training_hours:.1f} h  ({self.training_days:.1f} days)",
            f"  Electricity:       {self.electricity_kwh:.1f} kWh  "
            f"(${self.electricity_cost_usd:.2f} @ ${_DEFAULT_ELECTRICITY_USD_KWH}/kWh)",
        ]
        if self.cloud_estimates:
            lines.append("  Cloud pricing:")
            for provider, cost in sorted(self.cloud_estimates.items(), key=lambda x: x[1]):
                lines.append(f"    {provider:<20s}  ${cost:,.2f}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class CostEstimator:
    """Estimate training cost for different GPU types.

    Usage::

        from cola_coder.features.cost_estimator import CostEstimator

        estimator = CostEstimator()
        report = estimator.estimate(
            config={"model": {"n_layers": 12, "d_model": 768}},
            tokens=10_000_000_000,  # 10B tokens
            gpu_type="a100_40gb",
        )
        print(report.summary())
    """

    def __init__(
        self,
        electricity_usd_kwh: float = _DEFAULT_ELECTRICITY_USD_KWH,
    ) -> None:
        self.electricity_usd_kwh = electricity_usd_kwh

    def available_gpus(self) -> list[str]:
        """Return list of supported GPU type keys."""
        return list(GPU_SPECS.keys())

    def estimate(
        self,
        config: dict[str, Any],
        tokens: int,
        gpu_type: str = "a100_40gb",
    ) -> CostReport:
        """Estimate training cost.

        Parameters
        ----------
        config:
            Model config dict (flat or nested with "model" key).
            Used to estimate parameter count if not directly available.
        tokens:
            Total training tokens (e.g. ``10_000_000_000`` for 10B).
        gpu_type:
            GPU key from :data:`GPU_SPECS`.  Call :meth:`available_gpus`
            for a full list.

        Returns
        -------
        CostReport

        Raises
        ------
        ValueError
            If *gpu_type* is not in :data:`GPU_SPECS`.
        """
        key = gpu_type.lower().replace(" ", "_").replace("-", "_")
        if key not in GPU_SPECS:
            raise ValueError(
                f"Unknown GPU type: {gpu_type!r}. "
                f"Choose from: {list(GPU_SPECS.keys())}"
            )

        spec = GPU_SPECS[key]
        n_params = self._estimate_params(config)
        notes: list[str] = []

        # Scale throughput: larger models are slower (roughly inversely proportional)
        scale = _REFERENCE_PARAMS / max(n_params, 1)
        tokens_per_sec = spec["tokens_per_sec_125m"] * scale
        tokens_per_sec = max(tokens_per_sec, 1.0)

        training_seconds = tokens / tokens_per_sec
        training_hours = training_seconds / 3600.0
        training_days = training_hours / 24.0

        # Electricity
        tdp = spec["tdp_watts"]
        kw = tdp / 1000.0 * _PUE_FACTOR
        electricity_kwh = kw * training_hours
        electricity_cost = electricity_kwh * self.electricity_usd_kwh

        # Cloud estimates
        cloud_estimates: dict[str, float] = {}
        for provider, price_per_hour in spec.get("cloud_prices", {}).items():
            if price_per_hour > 0:
                cloud_estimates[provider] = round(training_hours * price_per_hour, 2)
            else:
                cloud_estimates[provider] = 0.0

        # Notes
        vram = spec.get("vram_gb", 0)
        notes.append(
            f"Throughput scaled from 125M baseline ({spec['tokens_per_sec_125m']:.0f} tok/s) "
            f"to {n_params / 1e6:.0f}M params — actual speed depends on batch size and seq_len."
        )
        if n_params > vram * 1_000_000_000 / 16:
            notes.append(
                f"Model may not fit in {vram}GB VRAM at full precision — "
                "consider gradient checkpointing or lower precision."
            )

        return CostReport(
            gpu_type=key,
            gpu_display_name=spec["display_name"],
            n_params=n_params,
            total_tokens=tokens,
            tokens_per_sec=tokens_per_sec,
            training_hours=training_hours,
            training_days=training_days,
            electricity_kwh=electricity_kwh,
            electricity_cost_usd=electricity_cost,
            cloud_estimates=cloud_estimates,
            notes=notes,
        )

    def compare(
        self,
        config: dict[str, Any],
        tokens: int,
        gpu_types: list[str] | None = None,
    ) -> list[CostReport]:
        """Estimate cost across multiple GPU types for easy comparison.

        Parameters
        ----------
        config:
            Model config dict.
        tokens:
            Total training tokens.
        gpu_types:
            List of GPU type keys.  Defaults to all available GPUs.

        Returns
        -------
        list[CostReport]
            Sorted by training hours (fastest first).
        """
        gpus = gpu_types if gpu_types is not None else self.available_gpus()
        reports = [self.estimate(config, tokens, g) for g in gpus]
        return sorted(reports, key=lambda r: r.training_hours)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_params(config: dict[str, Any]) -> int:
        """Estimate parameter count from a model config dict."""
        cfg = config.get("model", config)
        d_model = cfg.get("d_model", 768)
        n_layers = cfg.get("n_layers", 12)
        n_heads = cfg.get("n_heads", 12)
        n_kv_heads = cfg.get("n_kv_heads", n_heads)
        vocab_size = cfg.get("vocab_size", 32_000)
        d_ffn = cfg.get("d_ffn", d_model * 4)

        if not (d_model and n_layers and vocab_size):
            return _REFERENCE_PARAMS

        head_dim = d_model // n_heads if n_heads else d_model
        emb = vocab_size * d_model
        attn = d_model * d_model + 2 * n_kv_heads * head_dim * d_model + d_model * d_model
        ffn = 3 * d_model * d_ffn
        layer = attn + ffn + 2 * d_model
        return emb + n_layers * layer + d_model
