"""ROUTER-001: router pooling must ignore padding.

train_router.py pads short snippets to max_seq_len with pad_id=0, but route()
(inference) does NOT pad. The old mean(dim=1) averaged the pad embeddings in —
diluting a short snippet's signal and mismatching train vs inference. Both
routers now masked-mean-pool over real tokens only, so appending padding leaves
the output unchanged.
"""

import torch

from cola_coder.features.router_model import (
    MLPRouter,
    RouterConfig,
    TransformerRouter,
)


def _cfg(**kw):
    base = dict(
        vocab_size=50, embed_dim=16, hidden_dim=32, num_domains=3,
        max_seq_len=32, dropout=0.0, num_layers=1, num_heads=2,
    )
    base.update(kw)
    return RouterConfig(**base)


class TestMLPRouterPooling:
    @torch.no_grad()
    def test_padding_does_not_change_logits(self):
        torch.manual_seed(0)
        model = MLPRouter(_cfg()).eval()
        real = torch.tensor([[1, 2, 3]])
        padded = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])  # pad_id=0 appended
        torch.testing.assert_close(model(real), model(padded))

    @torch.no_grad()
    def test_all_pad_input_is_finite(self):
        model = MLPRouter(_cfg()).eval()
        out = model(torch.zeros((1, 8), dtype=torch.long))  # all padding
        assert torch.isfinite(out).all()


class TestTransformerRouterPooling:
    @torch.no_grad()
    def test_padding_does_not_change_logits(self):
        torch.manual_seed(0)
        model = TransformerRouter(_cfg()).eval()
        real = torch.tensor([[1, 2, 3, 4]])
        padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
        torch.testing.assert_close(model(real), model(padded), rtol=1e-4, atol=1e-5)

    @torch.no_grad()
    def test_all_pad_input_does_not_nan(self):
        # An all-pad row would NaN the attention softmax without the guard.
        model = TransformerRouter(_cfg()).eval()
        out = model(torch.zeros((1, 8), dtype=torch.long))
        assert torch.isfinite(out).all()

    @torch.no_grad()
    def test_mixed_batch_padding_invariant(self):
        # A padded row in a batch must match that row run on its own.
        torch.manual_seed(1)
        model = TransformerRouter(_cfg()).eval()
        single = model(torch.tensor([[5, 6, 7]]))
        batched = model(torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]]))
        torch.testing.assert_close(single[0], batched[0], rtol=1e-4, atol=1e-5)
