"""BUG-107: the reasoning thinking-token-expanded tokenizer must be persisted.

Sibling of BUG-106 (SFT). train_reasoning.py calls add_thinking_tokens() to add
<think>/</think> (vocab +2) and trains the model on those ids, then saves the
checkpoint — but never persisted the expanded tokenizer. Inference then reloads
the BASE tokenizer.json (no thinking tokens): the reasoning markers fragment and
the trained ids can't be decoded, breaking extract_thinking()/strip_thinking().

The fix saves the expanded tokenizer into the checkpoint dir and records its
path in metadata.json (read back by resolve_tokenizer_path). These lock the
persistence round-trip + the resolution wiring.
"""

import json
from pathlib import Path

from cola_coder.tokenizer import train_tokenizer as tt
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
from cola_coder.reasoning.thinking_tokens import (
    add_thinking_tokens,
    THINK_OPEN,
    THINK_CLOSE,
)
from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer
from cola_coder.inference.loading import resolve_tokenizer_path


def _base_tokenizer(tmp_path: Path) -> str:
    samples = ["def f():\n    return 1\n", "const x = 1;\n"] * 100
    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(iter(samples), vocab_size=300, output_path=out)
    return out


def _tiny_model(vocab_size: int) -> Transformer:
    cfg = ModelConfig(
        vocab_size=vocab_size, dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_dim_multiplier=1.0, max_seq_len=32, dropout=0.0, rope_theta=10000.0,
    )
    return Transformer(cfg)


class TestThinkingTokenPersistenceRoundTrip:
    def test_base_tokenizer_lacks_thinking_tokens(self, tmp_path):
        tok = CodeTokenizer(_base_tokenizer(tmp_path))
        assert tok.tokenizer.token_to_id(THINK_OPEN) is None
        assert tok.tokenizer.token_to_id(THINK_CLOSE) is None

    def test_expanded_tokenizer_survives_save_reload(self, tmp_path):
        tok = CodeTokenizer(_base_tokenizer(tmp_path))
        model = _tiny_model(tok.vocab_size)

        open_id, close_id = add_thinking_tokens(tok, model)
        assert tok.tokenizer.token_to_id(THINK_OPEN) == open_id

        ckpt_dir = tmp_path / "reasoning_ckpt"
        ckpt_dir.mkdir()
        saved = str(ckpt_dir / "tokenizer.json")
        tok.tokenizer.save(saved)

        reloaded = CodeTokenizer(saved)
        assert reloaded.tokenizer.token_to_id(THINK_OPEN) == open_id
        assert reloaded.tokenizer.token_to_id(THINK_CLOSE) == close_id
        # The marker must encode to the single trained id, not fragmented pieces.
        assert reloaded.tokenizer.encode(THINK_OPEN).ids == [open_id]


class TestResolveReasoningTokenizer:
    def test_metadata_tokenizer_path_resolves(self, tmp_path):
        tok = CodeTokenizer(_base_tokenizer(tmp_path))
        model = _tiny_model(tok.vocab_size)
        add_thinking_tokens(tok, model)

        ckpt_dir = tmp_path / "step_00000000"
        ckpt_dir.mkdir()
        saved = str(tmp_path / "tokenizer.json")
        tok.tokenizer.save(saved)
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"tokenizer_path": saved}), encoding="utf-8",
        )

        resolved = resolve_tokenizer_path(ckpt_dir)
        assert resolved == saved
        assert CodeTokenizer(resolved).tokenizer.token_to_id(THINK_OPEN) is not None
