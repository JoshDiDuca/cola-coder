"""Router Model: lightweight classifier for domain routing.

A small model (MLP or shallow transformer) that reads a code prompt
and outputs a probability distribution over specialist domains.

Architecture options:
1. MLP Router: Bag-of-words embedding -> MLP -> softmax (fastest, ~100us)
2. Transformer Router: Shared BPE embedding -> 2 transformer layers -> CLS -> softmax (best quality)

The router is intentionally tiny (<5M params) for instant routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED

# Default domain labels
DEFAULT_DOMAINS = ["react", "nextjs", "graphql", "prisma", "zod", "testing", "general"]


@dataclass
class RouterConfig:
    """Configuration for the router model."""
    vocab_size: int = 32768
    embed_dim: int = 128  # Much smaller than main model
    hidden_dim: int = 256
    num_domains: int = 7  # Number of specialist domains
    max_seq_len: int = 256  # Only need first ~256 tokens to determine domain
    dropout: float = 0.1
    architecture: str = "mlp"  # "mlp" or "transformer"
    # Transformer-specific
    num_layers: int = 2
    num_heads: int = 4


class MLPRouter(nn.Module):
    """MLP-based router: bag-of-embeddings -> MLP -> domain logits.

    Fast and simple. Treats input as bag-of-words (order doesn't matter much
    for domain classification since imports are the main signal).
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_domains),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)

        Returns:
            Logits over domains (batch_size, num_domains)
        """
        # Embed and mean-pool
        embeds = self.embedding(input_ids)  # (B, S, D)
        pooled = embeds.mean(dim=1)  # (B, D) - bag of embeddings
        logits = self.classifier(pooled)  # (B, num_domains)
        return logits

    def predict(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict domain with confidence.

        Returns:
            (domain_indices, confidence_scores)
        """
        logits = self.forward(input_ids)
        probs = F.softmax(logits, dim=-1)
        confidence, domain_idx = probs.max(dim=-1)
        return domain_idx, confidence

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerRouter(nn.Module):
    """Transformer-based router: embed -> transformer layers -> classify.

    Higher quality than MLP but slightly slower (~1ms vs ~0.1ms).
    Uses mean pooling of the last hidden state for classification.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.classifier = nn.Linear(config.embed_dim, config.num_domains)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)

        Returns:
            Logits over domains (batch_size, num_domains)
        """
        B, S = input_ids.shape
        S = min(S, self.config.max_seq_len)
        input_ids = input_ids[:, :S]

        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        embeds = self.embedding(input_ids) + self.pos_embedding(positions)
        embeds = self.dropout(embeds)

        hidden = self.transformer(embeds)

        # Use mean pooling (more robust than CLS token)
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def predict(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict domain with confidence."""
        logits = self.forward(input_ids)
        probs = F.softmax(logits, dim=-1)
        confidence, domain_idx = probs.max(dim=-1)
        return domain_idx, confidence

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_router(config: RouterConfig | None = None, architecture: str = "mlp") -> nn.Module:
    """Create a router model.

    Args:
        config: Router configuration. Uses defaults if None.
        architecture: "mlp" or "transformer"

    Returns:
        Router model (MLPRouter or TransformerRouter)
    """
    if config is None:
        config = RouterConfig(architecture=architecture)

    if config.architecture == "transformer":
        model = TransformerRouter(config)
    else:
        model = MLPRouter(config)

    cli.info("Router architecture", config.architecture)
    cli.info("Router parameters", f"{model.num_parameters:,}")

    return model


class DomainRouter:
    """High-level router that combines model prediction with heuristics.

    Uses the router model for classification, falling back to heuristic
    detection when confidence is low.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        domains: list[str] | None = None,
        confidence_threshold: float = 0.5,
        use_heuristic_fallback: bool = True,
    ):
        self.model = model
        self.domains = domains or DEFAULT_DOMAINS
        self.confidence_threshold = confidence_threshold
        self.use_heuristic_fallback = use_heuristic_fallback

    def route(self, code: str, tokenizer=None) -> tuple[str, float]:
        """Route a code snippet to a domain.

        Args:
            code: Source code to classify.
            tokenizer: Tokenizer for encoding (needed if model is used).

        Returns:
            (domain_name, confidence) tuple.
        """
        # Try model-based routing first
        if self.model is not None and tokenizer is not None:
            try:
                tokens = tokenizer.encode(code, add_bos=False)[:256]  # Truncate
                input_ids = torch.tensor([tokens], dtype=torch.long)

                if next(self.model.parameters()).is_cuda:
                    input_ids = input_ids.cuda()

                with torch.no_grad():
                    domain_idx, confidence = self.model.predict(input_ids)

                domain = self.domains[domain_idx.item()]
                conf = confidence.item()

                if conf >= self.confidence_threshold:
                    return domain, conf
            except Exception:
                pass

        # Fall back to heuristic detection
        if self.use_heuristic_fallback:
            from cola_coder.features.domain_detector import detect_domain
            scores = detect_domain(code)
            if scores and scores[0].confidence > 0.1:
                return scores[0].domain, scores[0].confidence

        return "general", 1.0

    def route_batch(self, code_samples: list[str], tokenizer=None) -> list[tuple[str, float]]:
        """Route multiple code samples."""
        return [self.route(code, tokenizer) for code in code_samples]
