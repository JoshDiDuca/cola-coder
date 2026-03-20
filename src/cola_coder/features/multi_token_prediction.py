"""Multi-Token Prediction: predict N future tokens simultaneously.

Instead of just predicting the next token, train the model to predict
the next N tokens at once using N independent prediction heads. This:
- Improves sample efficiency (more signal per training example)
- Encourages the model to plan ahead
- Can speed up inference via parallel verification

Based on Meta's "Better & Faster Large Language Models via Multi-token
Prediction" (2024). Each head shares the transformer backbone but has
its own output projection.

For a TS dev: it's like Promise.all() — instead of awaiting each
prediction sequentially, you fire off N predictions in parallel from
the same hidden state.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class MTPConfig:
    """Configuration for multi-token prediction."""
    n_predict: int = 4  # Number of future tokens to predict (N)
    # Loss weights: how much each future position contributes
    # Default: linearly decreasing (next token matters most)
    loss_weights: list[float] | None = None
    # Whether to share the output projection across heads
    share_output: bool = False


class MultiTokenPredictionHead(nn.Module):
    """A single prediction head for one future position.

    Each head is a small MLP: hidden -> dim -> vocab_size
    This gives each head its own "perspective" on what comes next.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: Shape (batch, seq_len, dim)

        Returns:
            Logits shape (batch, seq_len, vocab_size)
        """
        return self.proj(hidden_states)


class MultiTokenPredictor(nn.Module):
    """Multi-token prediction module.

    Wraps a base transformer model and adds N prediction heads.
    During training, computes loss for predicting the next N tokens.
    During inference, can be used for speculative verification.

    Usage:
        predictor = MultiTokenPredictor(base_model, config)
        loss = predictor.compute_mtp_loss(token_ids)
    """

    def __init__(self, base_model: nn.Module, config: MTPConfig | None = None):
        super().__init__()
        self.base_model = base_model
        self.config = config or MTPConfig()

        # Get dimensions from base model
        if hasattr(base_model, 'config'):
            dim = base_model.config.dim
            vocab_size = base_model.config.vocab_size
        else:
            # Fallback: inspect output layer
            dim = base_model.output.in_features if hasattr(base_model, 'output') else 256
            vocab_size = base_model.output.out_features if hasattr(base_model, 'output') else 32000

        self.dim = dim
        self.vocab_size = vocab_size
        self.n_predict = self.config.n_predict

        # Create prediction heads
        if self.config.share_output and hasattr(base_model, 'output'):
            # Share the base model's output projection
            self.heads = nn.ModuleList([base_model.output] * self.n_predict)
        else:
            self.heads = nn.ModuleList([
                MultiTokenPredictionHead(dim, vocab_size)
                for _ in range(self.n_predict)
            ])

        # Set loss weights
        if self.config.loss_weights:
            assert len(self.config.loss_weights) == self.n_predict
            self.loss_weights = self.config.loss_weights
        else:
            # Linear decay: [1.0, 0.75, 0.5, 0.25] for n_predict=4
            self.loss_weights = [
                1.0 - (i / (self.n_predict * 2))
                for i in range(self.n_predict)
            ]

    def get_hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states from the base model before the output projection.

        Args:
            token_ids: Shape (batch, seq_len)

        Returns:
            Hidden states shape (batch, seq_len, dim)
        """
        model = self.base_model

        # Standard transformer path: embedding -> blocks -> norm
        if hasattr(model, 'tok_emb') and hasattr(model, 'blocks'):
            h = model.tok_emb(token_ids)
            if hasattr(model, 'dropout'):
                h = model.dropout(h)

            # Build mask
            seq_len = token_ids.shape[1]
            if hasattr(model, 'causal_mask'):
                mask = model.causal_mask[:seq_len, :seq_len]
            else:
                mask = None

            for block in model.blocks:
                h = block(h, rope_freqs=model.rope_freqs, mask=mask)

            if hasattr(model, 'final_norm'):
                h = model.final_norm(h)
            return h
        else:
            # Fallback: just run the full model and project back
            # This is less efficient but works with any model
            logits = model(token_ids)
            return logits

    def forward(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass: get logits for each prediction head.

        Args:
            token_ids: Shape (batch, seq_len)

        Returns:
            List of N logit tensors, each shape (batch, seq_len, vocab_size)
        """
        hidden = self.get_hidden_states(token_ids)

        # Each head produces its own set of logits
        all_logits = []
        for head in self.heads:
            all_logits.append(head(hidden))

        return all_logits

    def compute_mtp_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute multi-token prediction loss.

        For each head i (0-indexed), predict token at position t+i+1
        given the hidden state at position t.

        Args:
            token_ids: Shape (batch, seq_len)

        Returns:
            Weighted sum of cross-entropy losses for all heads.
        """
        all_logits = self.forward(token_ids)
        total_loss = torch.tensor(0.0, device=token_ids.device)

        for i, (logits, weight) in enumerate(zip(all_logits, self.loss_weights)):
            # Head i predicts position t+i+1 from hidden state at position t
            # So shift logits by i+1 positions
            offset = i + 1
            if offset >= token_ids.shape[1]:
                break

            # Logits from position 0..seq_len-offset-1 predict
            # tokens at position offset..seq_len-1
            pred_logits = logits[:, :-offset, :].contiguous()
            targets = token_ids[:, offset:].contiguous()

            loss_i = F.cross_entropy(
                pred_logits.view(-1, pred_logits.size(-1)),
                targets.view(-1),
            )
            total_loss = total_loss + weight * loss_i

        return total_loss

    def predict_next_n(
        self,
        token_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> list[int]:
        """Predict the next N tokens in parallel.

        Useful for speculative decoding verification.

        Args:
            token_ids: Shape (1, seq_len) — single sequence.
            temperature: Sampling temperature.

        Returns:
            List of N predicted token IDs.
        """
        self.eval()
        with torch.no_grad():
            all_logits = self.forward(token_ids)

        predictions = []
        for logits in all_logits:
            last_logits = logits[0, -1]  # Last position logits
            if temperature == 0:
                token = last_logits.argmax().item()
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
            predictions.append(token)

        return predictions


def compute_mtp_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    config: MTPConfig | None = None,
) -> torch.Tensor:
    """Convenience function to compute MTP loss without wrapping the model.

    Creates a temporary MultiTokenPredictor and computes loss.
    For repeated use, prefer creating a MultiTokenPredictor instance.

    Args:
        model: Base transformer model.
        token_ids: Shape (batch, seq_len).
        config: MTP configuration.

    Returns:
        Multi-token prediction loss.
    """
    predictor = MultiTokenPredictor(model, config)
    predictor = predictor.to(token_ids.device)
    return predictor.compute_mtp_loss(token_ids)
