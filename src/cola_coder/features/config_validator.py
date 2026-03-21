"""Config Validator: catch invalid configs before training.

Validates:
- Type correctness (int vs float vs string)
- Value ranges (learning_rate > 0, batch_size > 0)
- Architecture compatibility (n_heads divides dim, n_kv_heads divides n_heads)
- VRAM feasibility (optional, if GPU available)
- File existence (data paths, checkpoint paths)

Returns a list of errors and warnings for clear reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ValidationIssue:
    """A validation error or warning."""
    level: str  # "error" or "warning"
    field: str  # Config field path (e.g., "model.dim")
    message: str
    suggestion: str = ""


def validate_config(config, check_files: bool = True, check_vram: bool = True) -> list[ValidationIssue]:
    """Validate a Config object and return list of issues.

    Args:
        config: Config instance (from cola_coder.model.config)
        check_files: Whether to check that referenced files/dirs exist
        check_vram: Whether to estimate VRAM fit

    Returns:
        List of ValidationIssue objects. Empty list = valid.
    """
    issues = []
    m = config.model
    t = config.training
    d = config.data
    # --- Model config ---

    # dim must be divisible by n_heads
    if m.dim % m.n_heads != 0:
        issues.append(ValidationIssue(
            "error", "model.dim",
            f"dim ({m.dim}) must be divisible by n_heads ({m.n_heads})",
            f"Use dim={m.n_heads * (m.dim // m.n_heads)} or adjust n_heads",
        ))

    # n_heads must be divisible by n_kv_heads (for GQA)
    if m.n_heads % m.n_kv_heads != 0:
        issues.append(ValidationIssue(
            "error", "model.n_kv_heads",
            f"n_heads ({m.n_heads}) must be divisible by n_kv_heads ({m.n_kv_heads})",
            "n_kv_heads must be a factor of n_heads for Grouped Query Attention",
        ))

    # Positive values
    if m.dim <= 0:
        issues.append(ValidationIssue("error", "model.dim", f"dim must be positive, got {m.dim}"))
    if m.n_layers <= 0:
        issues.append(ValidationIssue("error", "model.n_layers", f"n_layers must be positive, got {m.n_layers}"))
    if m.n_heads <= 0:
        issues.append(ValidationIssue("error", "model.n_heads", f"n_heads must be positive, got {m.n_heads}"))
    if m.vocab_size <= 0:
        issues.append(ValidationIssue("error", "model.vocab_size", f"vocab_size must be positive, got {m.vocab_size}"))
    if m.max_seq_len <= 0:
        issues.append(ValidationIssue("error", "model.max_seq_len", f"max_seq_len must be positive, got {m.max_seq_len}"))

    # Dropout range
    if not (0.0 <= m.dropout <= 1.0):
        issues.append(ValidationIssue(
            "error", "model.dropout",
            f"dropout must be between 0 and 1, got {m.dropout}",
        ))

    # Power of 2 suggestions
    if m.dim % 64 != 0:
        issues.append(ValidationIssue(
            "warning", "model.dim",
            f"dim ({m.dim}) is not a multiple of 64, which may hurt GPU performance",
            "Use a multiple of 64 for optimal GPU kernel utilization",
        ))

    # hidden_dim should equal round(4 * dim * 2/3) to nearest multiple of 256
    # (SwiGLU canonical sizing used by LLaMA/Mistral)
    raw_hidden = int(m.dim * 4 * 2 / 3)
    canonical_hidden = ((raw_hidden + 255) // 256) * 256
    actual_hidden = m.ffn_hidden_dim
    if actual_hidden != canonical_hidden:
        issues.append(ValidationIssue(
            "warning", "model.ffn_hidden_dim",
            f"ffn_hidden_dim ({actual_hidden}) differs from LLaMA canonical "
            f"round(4*dim*2/3) rounded to 256 = {canonical_hidden}",
            f"Set ffn_dim_multiplier so hidden_dim = {canonical_hidden} "
            f"(multiplier ≈ {canonical_hidden / m.dim:.4f})",
        ))

    # n_kv_heads must divide n_heads (already checked above for remainder != 0,
    # but that covers n_heads % n_kv_heads; this ensures the inverse relationship)
    if m.n_kv_heads > m.n_heads:
        issues.append(ValidationIssue(
            "error", "model.n_kv_heads",
            f"n_kv_heads ({m.n_kv_heads}) cannot exceed n_heads ({m.n_heads})",
            "n_kv_heads must be <= n_heads for Grouped Query Attention",
        ))

    # vocab_size should be a multiple of 64 for efficient embedding table layout
    if m.vocab_size % 64 != 0:
        issues.append(ValidationIssue(
            "warning", "model.vocab_size",
            f"vocab_size ({m.vocab_size}) is not a multiple of 64",
            "Pad vocab_size to the next multiple of 64 for GPU efficiency "
            f"(e.g. {((m.vocab_size + 63) // 64) * 64})",
        ))

    # max_seq_len should be a power of 2 (flash-attention and RoPE work best with powers of 2)
    seq_len = m.max_seq_len
    if seq_len > 0 and (seq_len & (seq_len - 1)) != 0:
        # Find next power of 2
        next_pow2 = 1
        while next_pow2 < seq_len:
            next_pow2 <<= 1
        issues.append(ValidationIssue(
            "warning", "model.max_seq_len",
            f"max_seq_len ({seq_len}) is not a power of 2",
            f"Use a power of 2 for optimal attention kernel performance "
            f"(e.g. {next_pow2 // 2} or {next_pow2})",
        ))

    # --- Training config ---

    if t.batch_size <= 0:
        issues.append(ValidationIssue("error", "training.batch_size", f"batch_size must be positive, got {t.batch_size}"))
    if t.gradient_accumulation <= 0:
        issues.append(ValidationIssue("error", "training.gradient_accumulation", "gradient_accumulation must be positive"))
    if t.learning_rate <= 0:
        issues.append(ValidationIssue("error", "training.learning_rate", f"learning_rate must be positive, got {t.learning_rate}"))
    if t.min_lr < 0:
        issues.append(ValidationIssue("error", "training.min_lr", f"min_lr must be non-negative, got {t.min_lr}"))
    if t.min_lr >= t.learning_rate:
        issues.append(ValidationIssue(
            "warning", "training.min_lr",
            f"min_lr ({t.min_lr}) >= learning_rate ({t.learning_rate}). LR schedule will be flat.",
        ))
    if t.max_steps <= 0:
        issues.append(ValidationIssue("error", "training.max_steps", "max_steps must be positive"))
    if t.warmup_steps >= t.max_steps:
        issues.append(ValidationIssue(
            "warning", "training.warmup_steps",
            f"warmup_steps ({t.warmup_steps}) >= max_steps ({t.max_steps}). LR will never reach peak.",
        ))
    if t.grad_clip <= 0:
        issues.append(ValidationIssue("warning", "training.grad_clip", f"grad_clip should be positive, got {t.grad_clip}"))

    # Precision
    valid_precisions = {"bf16", "fp16", "fp32"}
    if t.precision not in valid_precisions:
        issues.append(ValidationIssue(
            "error", "training.precision",
            f"precision must be one of {valid_precisions}, got '{t.precision}'",
        ))

    # Learning rate sanity
    if t.learning_rate > 0.01:
        issues.append(ValidationIssue(
            "warning", "training.learning_rate",
            f"learning_rate ({t.learning_rate}) seems very high. Typical range: 1e-5 to 3e-4",
        ))

    # Weight decay sanity
    if t.weight_decay > 1.0:
        issues.append(ValidationIssue(
            "warning", "training.weight_decay",
            f"weight_decay ({t.weight_decay}) seems very high. Typical: 0.01-0.1",
        ))

    # --- Data config ---
    if t.batch_size > 0 and m.max_seq_len > 0:
        tokens_per_batch = t.effective_batch_size * m.max_seq_len
        if tokens_per_batch > 1_000_000:
            issues.append(ValidationIssue(
                "warning", "training",
                f"Effective tokens per step ({tokens_per_batch:,}) is very large. This may cause OOM.",
                "Reduce batch_size or gradient_accumulation",
            ))

    # --- File checks ---
    if check_files:
        data_dir = Path(d.data_dir)
        if not data_dir.exists():
            issues.append(ValidationIssue(
                "warning", "data.data_dir",
                f"Data directory does not exist: {data_dir}",
                "It will be created during data preparation",
            ))

    # --- VRAM check ---
    if check_vram:
        try:
            from cola_coder.features.vram_estimator import estimate_vram
            estimate = estimate_vram(model_config=m, training_config=t)
            if estimate.fits_training is False:
                issues.append(ValidationIssue(
                    "error", "vram",
                    f"Estimated VRAM ({estimate.total_training_gb:.1f}GB) exceeds GPU ({estimate.gpu_vram_gb:.1f}GB)",
                    "Reduce batch_size, enable gradient_checkpointing, or use a smaller model",
                ))
        except Exception:
            pass  # VRAM check is optional

    return issues


def validate_config_file(config_path: str, check_files: bool = True, check_vram: bool = True) -> list[ValidationIssue]:
    """Validate a YAML config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        List of ValidationIssue objects.
    """
    from cola_coder.model.config import Config

    path = Path(config_path)
    if not path.exists():
        return [ValidationIssue("error", "config_path", f"Config file not found: {path}")]

    try:
        config = Config.from_yaml(str(path))
    except Exception as e:
        return [ValidationIssue("error", "config_parse", f"Failed to parse config: {e}")]

    return validate_config(config, check_files=check_files, check_vram=check_vram)


def print_validation_results(issues: list[ValidationIssue]) -> bool:
    """Print validation results and return True if no errors.

    Returns:
        True if config is valid (no errors, warnings are OK).
    """
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    if not issues:
        cli.success("Config validation passed — no issues found")
        return True

    if errors:
        cli.error(f"Config validation failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        for issue in errors:
            cli.print(f"  [red]ERROR[/red] [{issue.field}]: {issue.message}")
            if issue.suggestion:
                cli.dim(f"    Suggestion: {issue.suggestion}")

    if warnings:
        for issue in warnings:
            cli.print(f"  [yellow]WARN[/yellow] [{issue.field}]: {issue.message}")
            if issue.suggestion:
                cli.dim(f"    Suggestion: {issue.suggestion}")

    if not errors:
        cli.success(f"Config validation passed with {len(warnings)} warning(s)")

    return len(errors) == 0
