"""Inference Latency Optimizer.

Analyze the inference pipeline for latency bottlenecks and suggest
optimizations across:
  - KV-cache size and strategy
  - Batch size selection
  - Quantization level trade-offs
  - Attention computation strategy
  - Prefill vs decode bottlenecks

Works with profiling data dicts — no GPU required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Model / hardware configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Static model configuration for latency analysis."""

    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    hidden_dim: int | None = None  # inferred if not provided

    def __post_init__(self) -> None:
        if self.hidden_dim is None:
            self.hidden_dim = self.num_heads * self.head_dim

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Bytes needed for one token in the KV cache (fp16, both K and V)."""
        # 2 (K+V) * num_layers * num_heads * head_dim * 2 (fp16 bytes)
        return 2 * self.num_layers * self.num_heads * self.head_dim * 2

    @property
    def kv_cache_mb_per_token(self) -> float:
        return self.kv_cache_bytes_per_token / (1024 ** 2)


@dataclass
class HardwareConfig:
    """Hardware specification for latency modelling."""

    gpu_memory_gb: float = 24.0
    gpu_bandwidth_gbps: float = 900.0   # memory bandwidth
    gpu_tflops_fp16: float = 77.0       # theoretical throughput
    cpu_memory_gb: float = 64.0
    nvme_bandwidth_gbps: float = 7.0


# ---------------------------------------------------------------------------
# Profiling data
# ---------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    """Profiling data from an inference run."""

    # Timing (ms per forward pass or per batch)
    prefill_ms: float = 0.0          # time to process prompt tokens
    decode_ms_per_token: float = 0.0 # time to generate one token

    # Memory usage
    kv_cache_mb: float = 0.0
    model_weights_mb: float = 0.0
    activations_mb: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    batch_size: int = 1
    prompt_length: int = 0
    generated_length: int = 0

    # Component breakdowns (ms)
    attention_ms: float = 0.0
    ffn_ms: float = 0.0
    embedding_ms: float = 0.0
    sampling_ms: float = 0.0
    other_ms: float = 0.0

    @property
    def total_decode_ms(self) -> float:
        return self.decode_ms_per_token * max(self.generated_length, 1)

    @property
    def attention_fraction(self) -> float:
        total = self.attention_ms + self.ffn_ms + self.embedding_ms + self.sampling_ms + self.other_ms
        return self.attention_ms / max(total, 1e-9)


# ---------------------------------------------------------------------------
# Optimization suggestions
# ---------------------------------------------------------------------------


class Suggestion(NamedTuple):
    """A single optimization suggestion."""

    category: str            # "kv_cache" | "batch_size" | "quantization" | "attention" | "pipeline"
    priority: str            # "high" | "medium" | "low"
    title: str
    description: str
    estimated_speedup: float  # estimated multiplier (1.0 = no change, 2.0 = 2x faster)


@dataclass
class LatencyReport:
    """Complete latency analysis report."""

    bottleneck: str          # identified primary bottleneck
    bottleneck_fraction: float  # fraction of time in bottleneck
    suggestions: list[Suggestion] = field(default_factory=list)
    total_estimated_speedup: float = 1.0
    analysis_notes: list[str] = field(default_factory=list)

    def high_priority(self) -> list[Suggestion]:
        return [s for s in self.suggestions if s.priority == "high"]

    def summary(self) -> str:
        lines = [
            f"Bottleneck: {self.bottleneck} ({self.bottleneck_fraction:.1%})",
            f"Total estimated speedup (if all suggestions applied): {self.total_estimated_speedup:.1f}x",
            f"Suggestions ({len(self.suggestions)} total, {len(self.high_priority())} high priority):",
        ]
        for s in self.suggestions:
            lines.append(f"  [{s.priority.upper()}] {s.title}: {s.estimated_speedup:.1f}x")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KV-cache analysis
# ---------------------------------------------------------------------------


def analyze_kv_cache(
    profile: InferenceProfile,
    model_config: ModelConfig,
    hardware: HardwareConfig,
    target_utilization: float = 0.8,
) -> list[Suggestion]:
    """Generate KV-cache optimization suggestions."""
    suggestions: list[Suggestion] = []

    # Current KV cache usage
    current_kv_mb = profile.kv_cache_mb
    available_gpu_mb = hardware.gpu_memory_gb * 1024 - profile.model_weights_mb
    kv_fraction = current_kv_mb / max(available_gpu_mb, 1)

    if kv_fraction > 0.6:
        # KV cache too large — suggest quantization or reduction
        suggestions.append(Suggestion(
            category="kv_cache",
            priority="high",
            title="Quantize KV cache to INT8",
            description=(
                f"KV cache uses {kv_fraction:.1%} of available GPU memory. "
                "INT8 quantization can halve KV cache size with minimal quality impact."
            ),
            estimated_speedup=1.3,
        ))

    # Check if sliding window would help
    if profile.prompt_length > 512:
        suggestions.append(Suggestion(
            category="kv_cache",
            priority="medium",
            title="Use sliding window attention",
            description=(
                f"With prompt length {profile.prompt_length}, sliding window "
                "attention can reduce KV cache by ~50% for long-context generation."
            ),
            estimated_speedup=1.4,
        ))

    # Multi-query attention
    if model_config.num_heads > 4:
        suggestions.append(Suggestion(
            category="kv_cache",
            priority="medium",
            title="Consider Multi-Query Attention (MQA)",
            description=(
                f"Model has {model_config.num_heads} heads. MQA shares KV projections "
                "across heads, reducing KV cache by ~{model_config.num_heads}x."
            ),
            estimated_speedup=1.5,
        ))

    return suggestions


def analyze_batch_size(
    profile: InferenceProfile,
    hardware: HardwareConfig,
    *,
    max_latency_ms: float = 100.0,
) -> list[Suggestion]:
    """Generate batch size optimization suggestions."""
    suggestions: list[Suggestion] = []
    current_batch = profile.batch_size

    if current_batch == 1:
        suggestions.append(Suggestion(
            category="batch_size",
            priority="high",
            title="Increase batch size for throughput",
            description=(
                "Batch size 1 leaves GPU underutilised. Batching 4-8 requests "
                "can improve throughput 3-6x with minimal latency increase."
            ),
            estimated_speedup=3.0,
        ))
    elif current_batch > 32:
        latency_per_req = profile.decode_ms_per_token * profile.generated_length
        if latency_per_req > max_latency_ms:
            suggestions.append(Suggestion(
                category="batch_size",
                priority="medium",
                title="Reduce batch size to meet latency SLA",
                description=(
                    f"Large batch ({current_batch}) causes {latency_per_req:.0f}ms latency. "
                    f"Consider batch size 8-16 to stay under {max_latency_ms:.0f}ms."
                ),
                estimated_speedup=0.8,
            ))

    return suggestions


def analyze_quantization(
    profile: InferenceProfile,
    model_config: ModelConfig,
    hardware: HardwareConfig,
) -> list[Suggestion]:
    """Generate quantization suggestions."""
    suggestions: list[Suggestion] = []
    model_mb = profile.model_weights_mb
    total_gpu_mb = hardware.gpu_memory_gb * 1024

    # If model barely fits
    if model_mb > 0 and model_mb / total_gpu_mb > 0.7:
        suggestions.append(Suggestion(
            category="quantization",
            priority="high",
            title="Apply INT4 quantization (GPTQ/AWQ)",
            description=(
                f"Model uses {model_mb / total_gpu_mb:.1%} of GPU memory. "
                "INT4 quantization (4-bit) reduces memory by ~4x with <5% quality loss."
            ),
            estimated_speedup=2.0,
        ))
    elif model_mb > 0 and model_mb / total_gpu_mb > 0.4:
        suggestions.append(Suggestion(
            category="quantization",
            priority="medium",
            title="Apply INT8 quantization",
            description=(
                "INT8 quantization halves model memory usage with minimal quality impact. "
                "Compatible with most GPU architectures."
            ),
            estimated_speedup=1.5,
        ))

    # Activation quantization
    if profile.ffn_ms > profile.attention_ms * 2:
        suggestions.append(Suggestion(
            category="quantization",
            priority="low",
            title="Quantize FFN activations",
            description=(
                "FFN layers dominate compute. Activation quantization can speed up "
                "FFN by 20-40% on supported hardware."
            ),
            estimated_speedup=1.2,
        ))

    return suggestions


def analyze_attention(
    profile: InferenceProfile,
    model_config: ModelConfig,
) -> list[Suggestion]:
    """Generate attention optimization suggestions."""
    suggestions: list[Suggestion] = []

    if profile.attention_fraction > 0.4:
        suggestions.append(Suggestion(
            category="attention",
            priority="high",
            title="Enable Flash Attention",
            description=(
                f"Attention uses {profile.attention_fraction:.1%} of compute. "
                "Flash Attention reduces memory bandwidth usage by ~10x for long sequences."
            ),
            estimated_speedup=2.5,
        ))

    if profile.prompt_length > 1024:
        suggestions.append(Suggestion(
            category="attention",
            priority="medium",
            title="Use chunked prefill",
            description=(
                f"Long prompt ({profile.prompt_length} tokens) causes high prefill latency. "
                "Process in chunks of 512 to overlap compute with decode."
            ),
            estimated_speedup=1.4,
        ))

    return suggestions


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LatencyOptimizer:
    """Analyze inference latency and suggest optimizations.

    Parameters
    ----------
    model_config:
        Static model architecture configuration.
    hardware:
        Hardware specification.
    max_latency_ms:
        Target maximum latency per request in milliseconds.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hardware: HardwareConfig | None = None,
        max_latency_ms: float = 100.0,
    ) -> None:
        self.model_config = model_config
        self.hardware = hardware or HardwareConfig()
        self.max_latency_ms = max_latency_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, profile: InferenceProfile) -> LatencyReport:
        """Analyze *profile* and return a LatencyReport with suggestions."""
        bottleneck, bottleneck_fraction = self._identify_bottleneck(profile)

        suggestions: list[Suggestion] = []
        suggestions.extend(analyze_kv_cache(profile, self.model_config, self.hardware))
        suggestions.extend(analyze_batch_size(profile, self.hardware, max_latency_ms=self.max_latency_ms))
        suggestions.extend(analyze_quantization(profile, self.model_config, self.hardware))
        suggestions.extend(analyze_attention(profile, self.model_config))
        suggestions.extend(self._pipeline_suggestions(profile))

        # Sort by priority: high > medium > low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (priority_order.get(s.priority, 3), -s.estimated_speedup))

        # Compute combined speedup (optimistic — assume independent)
        combined = 1.0
        for s in suggestions:
            combined *= s.estimated_speedup

        notes = self._build_notes(profile)

        return LatencyReport(
            bottleneck=bottleneck,
            bottleneck_fraction=bottleneck_fraction,
            suggestions=suggestions,
            total_estimated_speedup=round(combined, 2),
            analysis_notes=notes,
        )

    def estimate_kv_cache_capacity(self, available_memory_mb: float | None = None) -> int:
        """Estimate how many tokens fit in the KV cache given memory."""
        available = (
            available_memory_mb
            if available_memory_mb is not None
            else self.hardware.gpu_memory_gb * 1024 * 0.4  # 40% for KV cache
        )
        bytes_per_token = self.model_config.kv_cache_bytes_per_token
        mb_per_token = bytes_per_token / (1024 ** 2)
        return int(available / max(mb_per_token, 1e-9))

    def optimal_batch_size(
        self,
        available_memory_mb: float,
        prompt_len: int,
        generation_len: int,
    ) -> int:
        """Estimate optimal batch size given memory and sequence lengths."""
        tokens_needed = prompt_len + generation_len
        kv_mb = tokens_needed * self.model_config.kv_cache_mb_per_token
        if kv_mb <= 0:
            return 1
        batches = int(available_memory_mb / kv_mb)
        return max(1, batches)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _identify_bottleneck(self, profile: InferenceProfile) -> tuple[str, float]:
        """Identify the primary latency bottleneck."""
        components = {
            "attention": profile.attention_ms,
            "ffn": profile.ffn_ms,
            "embedding": profile.embedding_ms,
            "sampling": profile.sampling_ms,
            "other": profile.other_ms,
        }
        total = sum(components.values())
        if total < 1e-9:
            # Fall back to prefill/decode
            if profile.prefill_ms > profile.decode_ms_per_token * max(profile.generated_length, 1):
                return "prefill", 1.0
            return "decode", 1.0

        bottleneck = max(components, key=lambda k: components[k])
        fraction = components[bottleneck] / total
        return bottleneck, round(fraction, 4)

    def _pipeline_suggestions(self, profile: InferenceProfile) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        if profile.prefill_ms > 0 and profile.decode_ms_per_token > 0:
            # Speculative decoding suggestion
            if profile.decode_ms_per_token > 5.0:
                suggestions.append(Suggestion(
                    category="pipeline",
                    priority="medium",
                    title="Use speculative decoding",
                    description=(
                        "Slow decode phase benefits from speculative decoding: "
                        "use a small draft model to generate candidate tokens, "
                        "verify with large model in parallel. 2-4x decode speedup."
                    ),
                    estimated_speedup=2.0,
                ))
        return suggestions

    def _build_notes(self, profile: InferenceProfile) -> list[str]:
        notes: list[str] = []
        if profile.tokens_per_second > 0:
            notes.append(f"Current throughput: {profile.tokens_per_second:.1f} tokens/sec")
        if profile.kv_cache_mb > 0:
            notes.append(
                f"KV cache: {profile.kv_cache_mb:.1f} MB "
                f"({profile.kv_cache_mb / (self.hardware.gpu_memory_gb * 1024):.1%} of GPU memory)"
            )
        return notes
