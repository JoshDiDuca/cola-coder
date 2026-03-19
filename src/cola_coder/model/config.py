"""Model and training configuration.

Think of this like a TypeScript interface/type that defines the shape of your config,
except in Python we use dataclasses (similar to TS classes with readonly fields).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Defines the transformer architecture shape.

    Every number here controls a dimension of the model.
    Bigger numbers = more parameters = smarter but slower and more VRAM.
    """

    vocab_size: int = 32768  # How many unique tokens the model knows
    dim: int = 512  # Width of the model (embedding dimension)
    n_layers: int = 8  # Depth of the model (number of transformer blocks)
    n_heads: int = 8  # Number of attention "perspectives" (query heads)
    n_kv_heads: int = 4  # Number of key/value heads (GQA). Less = less VRAM at inference
    ffn_dim_multiplier: float = 2.667  # Controls feed-forward hidden size
    max_seq_len: int = 1024  # Maximum sequence length in tokens
    dropout: float = 0.1  # Randomly zero out this fraction during training (prevents overfitting)
    rope_theta: float = 10000.0  # RoPE base frequency (controls position encoding wavelength)

    @property
    def head_dim(self) -> int:
        """Size of each attention head. dim must be divisible by n_heads."""
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """Hidden dimension of the SwiGLU feed-forward network.

        SwiGLU uses 2/3 of the "natural" hidden dim because it has 3 projections
        instead of 2 (gate + up + down), so this keeps param count similar.
        """
        hidden = int(self.dim * self.ffn_dim_multiplier)
        # Round to nearest multiple of 64 for GPU efficiency
        return ((hidden + 63) // 64) * 64

    @property
    def total_params(self) -> int:
        """Approximate total parameter count."""
        # Embedding: vocab_size * dim (shared with output head via weight tying)
        embedding = self.vocab_size * self.dim

        # Per layer:
        # Attention: Q projection + KV projections + output projection
        q_proj = self.dim * self.dim  # dim -> dim
        kv_proj = 2 * self.dim * (self.n_kv_heads * self.head_dim)  # K and V
        out_proj = self.dim * self.dim  # dim -> dim
        attn_per_layer = q_proj + kv_proj + out_proj

        # FFN: gate + up + down projections
        ffn_per_layer = 3 * self.dim * self.ffn_hidden_dim

        # Norms: 2 RMSNorm per layer (just a weight vector each) + 1 final
        norms = (2 * self.n_layers + 1) * self.dim

        total = embedding + self.n_layers * (attn_per_layer + ffn_per_layer) + norms
        return total

    @property
    def total_params_human(self) -> str:
        """Human-readable parameter count (e.g., '125M')."""
        p = self.total_params
        if p >= 1e9:
            return f"{p / 1e9:.1f}B"
        return f"{p / 1e6:.0f}M"


@dataclass
class TrainingConfig:
    """Controls how the model learns."""

    batch_size: int = 32  # Samples processed per GPU at once
    gradient_accumulation: int = 1  # Accumulate N micro-batches before updating weights
    learning_rate: float = 3e-4  # How big of a step to take (will be scheduled)
    min_lr: float = 3e-5  # Minimum learning rate (floor of cosine schedule)
    warmup_steps: int = 500  # Gradually increase LR for this many steps
    max_steps: int = 20000  # Total training steps
    weight_decay: float = 0.1  # Regularization (penalize large weights)
    grad_clip: float = 1.0  # Clip gradients to prevent explosions
    precision: str = "bf16"  # "bf16" for RTX 4080+, "fp16" for RTX 3080
    gradient_checkpointing: bool = False  # Trade compute for VRAM (needed for 350M+)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation


@dataclass
class DataConfig:
    """Controls what data the model trains on."""

    dataset: str = "bigcode/starcoderdata"  # HuggingFace dataset name
    languages: list[str] = field(default_factory=lambda: ["python", "typescript", "javascript"])
    max_tokens_per_file: int = 2048  # Truncate files longer than this
    data_dir: str = "./data"  # Where to store processed data
    num_workers: int = 4  # Parallel data loading workers


@dataclass
class CheckpointConfig:
    """Controls model saving."""

    save_every: int = 1000  # Save checkpoint every N steps
    output_dir: str = "./checkpoints"  # Where to save
    max_checkpoints: int = 5  # Keep only the N most recent checkpoints


@dataclass
class Config:
    """Top-level config combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file.

        In TypeScript terms, this is like a static factory method:
        const config = Config.fromYaml("configs/tiny.yaml")
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        data_cfg = DataConfig(**raw.get("data", {}))
        checkpoint_cfg = CheckpointConfig(**raw.get("checkpoint", {}))

        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            checkpoint=checkpoint_cfg,
        )

    def summary(self) -> str:
        """Print a human-readable summary of the config."""
        lines = [
            f"Model: {self.model.total_params_human} parameters",
            f"  dim={self.model.dim}, layers={self.model.n_layers}, "
            f"heads={self.model.n_heads} (KV: {self.model.n_kv_heads})",
            f"  ffn_hidden={self.model.ffn_hidden_dim}, max_seq={self.model.max_seq_len}",
            f"Training: {self.training.precision}, batch={self.training.effective_batch_size} "
            f"(micro={self.training.batch_size} x accum={self.training.gradient_accumulation})",
            f"  lr={self.training.learning_rate} -> {self.training.min_lr}, "
            f"steps={self.training.max_steps}",
            f"Data: {self.data.dataset} [{', '.join(self.data.languages)}]",
        ]
        return "\n".join(lines)
