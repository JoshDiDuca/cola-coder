"""VRAM Estimator: predict GPU memory usage before training.

Estimates peak VRAM usage based on model config, training config, and
available GPU. Helps users choose appropriate configs for their hardware.

Formula breakdown:
- Model weights: params * bytes_per_param
- Optimizer state: params * 8 (Adam stores m and v, each same size as params)
- Gradients: params * bytes_per_param
- Activations: batch_size * seq_len * dim * n_layers * bytes_per_activation
- KV cache (inference): 2 * n_layers * n_kv_heads * head_dim * max_seq_len * bytes
"""

from dataclasses import dataclass
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class VRAMEstimate:
    """Breakdown of estimated VRAM usage."""
    model_weights_gb: float
    optimizer_state_gb: float
    gradients_gb: float
    activations_gb: float
    kv_cache_gb: float  # For inference
    total_training_gb: float
    total_inference_gb: float
    gpu_name: str | None
    gpu_vram_gb: float | None
    fits_training: bool | None
    fits_inference: bool | None


def estimate_vram(
    model_config=None,
    training_config=None,
    config_path: str | None = None,
) -> VRAMEstimate:
    """Estimate VRAM usage for training and inference.

    Args:
        model_config: ModelConfig instance (from cola_coder.model.config)
        training_config: TrainingConfig instance
        config_path: Path to YAML config file (alternative to passing configs)

    Returns:
        VRAMEstimate with detailed breakdown.
    """
    if config_path:
        from cola_coder.model.config import Config
        cfg = Config.from_yaml(config_path)
        model_config = cfg.model
        training_config = cfg.training

    if model_config is None:
        raise ValueError("Must provide model_config or config_path")

    # Bytes per parameter based on precision
    precision = getattr(training_config, 'precision', 'bf16') if training_config else 'bf16'
    if precision in ('bf16', 'fp16'):
        bytes_per_param = 2
    else:
        bytes_per_param = 4

    # Total parameters
    total_params = model_config.total_params

    # Model weights
    model_weights_bytes = total_params * bytes_per_param
    model_weights_gb = model_weights_bytes / 1e9

    # Optimizer state (Adam: m + v, stored in fp32)
    optimizer_state_bytes = total_params * 4 * 2  # m and v in fp32
    optimizer_state_gb = optimizer_state_bytes / 1e9

    # Gradients (same dtype as model)
    gradients_bytes = total_params * bytes_per_param
    gradients_gb = gradients_bytes / 1e9

    # Activations estimate
    batch_size = getattr(training_config, 'batch_size', 32) if training_config else 32
    seq_len = model_config.max_seq_len
    dim = model_config.dim
    n_layers = model_config.n_layers
    hidden_dim = model_config.ffn_hidden_dim
    kv_dim = model_config.n_kv_heads * model_config.head_dim

    # Activation memory per layer for backward pass (PyTorch stores these).
    # Flash Attention never materializes the full attention matrix.
    #
    # Attention saved activations: x, Q, K, V, attn_output
    #   = batch * seq * (3*dim + 2*kv_dim)
    #
    # SwiGLU FFN saved activations: x, gate_proj_out, silu_out, up_proj_out, gate*value
    #   = batch * seq * (dim + 4*hidden_dim)
    #
    # Note: hidden_dim (e.g. 2048) is often much larger than dim (e.g. 768),
    # so the FFN dominates activation memory.
    per_layer_elements = batch_size * seq_len * (
        3 * dim + 2 * kv_dim  # attention
        + dim + 4 * hidden_dim  # FFN
    )
    activation_bytes = per_layer_elements * n_layers * bytes_per_param

    # torch.compile adds ~20% overhead for intermediate buffers
    activation_bytes = int(activation_bytes * 1.2)

    # Gradient checkpointing reduces activation memory by ~60%
    gradient_ckpt = getattr(training_config, 'gradient_checkpointing', False) if training_config else False
    if gradient_ckpt:
        activation_bytes = int(activation_bytes * 0.4)

    activations_gb = activation_bytes / 1e9

    # KV cache for inference (per token, grows with sequence)
    kv_cache_bytes = (
        2 * n_layers * model_config.n_kv_heads * model_config.head_dim
        * seq_len * bytes_per_param
    )
    kv_cache_gb = kv_cache_bytes / 1e9

    # Totals
    total_training = model_weights_gb + optimizer_state_gb + gradients_gb + activations_gb
    total_inference = model_weights_gb + kv_cache_gb

    # Add ~10% overhead for PyTorch internals, CUDA context, etc.
    total_training *= 1.1
    total_inference *= 1.1

    # GPU detection
    gpu_name = None
    gpu_vram_gb = None
    fits_training = None
    fits_inference = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = props.total_memory / 1e9
            fits_training = total_training < gpu_vram_gb * 0.95  # 95% safety margin
            fits_inference = total_inference < gpu_vram_gb * 0.95
    except ImportError:
        pass

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        optimizer_state_gb=optimizer_state_gb,
        gradients_gb=gradients_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        total_training_gb=total_training,
        total_inference_gb=total_inference,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        fits_training=fits_training,
        fits_inference=fits_inference,
    )


def print_vram_estimate(estimate: VRAMEstimate):
    """Print a formatted VRAM estimate."""
    cli.header("Cola-Coder", "VRAM Estimator")

    cli.kv_table({
        "Model weights": f"{estimate.model_weights_gb:.2f} GB",
        "Optimizer state": f"{estimate.optimizer_state_gb:.2f} GB",
        "Gradients": f"{estimate.gradients_gb:.2f} GB",
        "Activations": f"{estimate.activations_gb:.2f} GB",
        "Training total": f"{estimate.total_training_gb:.2f} GB",
        "": "",
        "KV cache (inference)": f"{estimate.kv_cache_gb:.2f} GB",
        "Inference total": f"{estimate.total_inference_gb:.2f} GB",
    }, title="VRAM Breakdown")

    if estimate.gpu_name:
        cli.rule("GPU Check")
        cli.info("GPU", f"{estimate.gpu_name} ({estimate.gpu_vram_gb:.1f} GB)")

        if estimate.fits_training:
            cli.success(f"Training FITS ({estimate.total_training_gb:.1f} GB < {estimate.gpu_vram_gb:.1f} GB)")
        elif estimate.fits_training is False:
            cli.error(f"Training WON'T FIT ({estimate.total_training_gb:.1f} GB > {estimate.gpu_vram_gb:.1f} GB)")
            cli.warn("Suggestions:")
            cli.dim("  1. Reduce batch_size")
            cli.dim("  2. Enable gradient_checkpointing: true")
            cli.dim("  3. Use a smaller model config")
            cli.dim("  4. Increase gradient_accumulation (reduce batch_size proportionally)")

        if estimate.fits_inference:
            cli.success(f"Inference FITS ({estimate.total_inference_gb:.1f} GB < {estimate.gpu_vram_gb:.1f} GB)")
        elif estimate.fits_inference is False:
            cli.error(f"Inference WON'T FIT ({estimate.total_inference_gb:.1f} GB > {estimate.gpu_vram_gb:.1f} GB)")
    else:
        cli.warn("No GPU detected — cannot verify fit")

    return estimate
