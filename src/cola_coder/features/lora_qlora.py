"""LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Instead of updating all model weights during fine-tuning, LoRA injects small
trainable rank-decomposition matrices into specific layers, leaving the original
weights frozen. This reduces trainable parameters by ~100-1000x.

For a TS dev: like monkey-patching a class — you wrap the original method and
add your own delta on top, without touching the original implementation.

Math: W' = W + delta_W = W + (B @ A) * scaling
where A: [in, rank], B: [rank, out], scaling = alpha / rank
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    rank: bottleneck dimension (4-64 typical; higher = more capacity)
    alpha: scaling factor (often set to rank or 2*rank)
    dropout: applied to lora_A output before lora_B
    target_modules: names of linear layers to replace with LoRA versions
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])


class LoRALinear(nn.Module):
    """A linear layer wrapped with LoRA adaptation.

    The original weight is frozen; only lora_A and lora_B are trainable.
    Output = original(x) + scaling * lora_B(lora_A(dropout(x)))
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self._merged = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Store original layer; freeze its parameters
        self.original = original_linear
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA down-projection: in_features -> rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        # LoRA up-projection: rank -> out_features
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialise: A with kaiming uniform, B with zeros (so delta starts at 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            # Weights already folded in — just run the original linear
            return self.original(x)
        base = self.original(x)
        lora_delta = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + lora_delta

    def merge(self) -> None:
        """Fold LoRA weights into the original weight matrix (inference mode)."""
        if self._merged:
            return
        # delta_W = lora_B.weight @ lora_A.weight  shape: [out, in]
        delta_W = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.original.weight.data += delta_W
        self._merged = True

    def unmerge(self) -> None:
        """Remove the LoRA delta from the original weight matrix."""
        if not self._merged:
            return
        delta_W = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.original.weight.data -= delta_W
        self._merged = False


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target linear layers in *model* with LoRA-wrapped versions.

    Only layers whose name (final segment) matches a name in
    config.target_modules are replaced. All other parameters stay frozen
    if you call model.requires_grad_(False) first — but this function
    doesn't freeze unrelated layers automatically (caller's choice).
    """
    target_set = set(config.target_modules)

    # Collect replacements first to avoid mutating while iterating
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Match on the last segment of the dotted name
        short_name = name.split('.')[-1]
        if short_name not in target_set:
            continue

        lora_layer = LoRALinear(
            original_linear=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )

        # Resolve parent module and attribute name
        parts = name.split('.')
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

        replacements.append((parent, attr, lora_layer))

    for parent, attr, lora_layer in replacements:
        setattr(parent, attr, lora_layer)

    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA-specific parameters to *path* (a .pt file).

    Saves lora_A, lora_B weights for every LoRALinear in the model.
    The full model weights are not saved — only the deltas.
    """
    lora_state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.clone()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.clone()

    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA parameters from *path* into the model's LoRALinear layers."""
    lora_state: dict[str, torch.Tensor] = torch.load(path, map_location='cpu', weights_only=True)

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        key_A = f"{name}.lora_A.weight"
        key_B = f"{name}.lora_B.weight"
        if key_A in lora_state:
            module.lora_A.weight.data.copy_(lora_state[key_A])
        if key_B in lora_state:
            module.lora_B.weight.data.copy_(lora_state[key_B])
