"""BUG-106: the SFT chat-token-expanded tokenizer must be persisted + resolved.

train_sft.py calls add_chat_tokens() to add <|im_start|>/<|im_end|> to the
tokenizer (and resize the model), trains the model on those new ids, then saves
the checkpoint. But the expanded tokenizer was NEVER written to disk, so
inference reloaded the BASE tokenizer.json (no chat tokens) — fragmenting the
ChatML role markers and breaking instruction following (a train/inference
mismatch). <|im_start|>/<|im_end|> are NOT in the base SPECIAL_TOKENS.

The fix saves the expanded tokenizer into the checkpoint dir and records its
path in metadata.json, which resolve_tokenizer_path() reads back first. These
tests lock the persistence round-trip and the resolution wiring.
"""

import json
from pathlib import Path

from cola_coder.tokenizer import train_tokenizer as tt
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
from cola_coder.tokenizer.chat_template import add_chat_tokens, has_chat_tokens, IM_START
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


class TestChatTokenPersistenceRoundTrip:
    def test_base_tokenizer_lacks_chat_tokens(self, tmp_path):
        tok = CodeTokenizer(_base_tokenizer(tmp_path))
        # The whole premise: ChatML tokens are NOT in the base vocab.
        assert has_chat_tokens(tok) is False

    def test_expanded_tokenizer_survives_save_reload(self, tmp_path):
        base = _base_tokenizer(tmp_path)
        tok = CodeTokenizer(base)
        model = _tiny_model(tok.vocab_size)

        im_start_id, im_end_id = add_chat_tokens(tok, model)
        assert has_chat_tokens(tok)

        # Persist exactly as train_sft now does.
        ckpt_dir = tmp_path / "sft_ckpt"
        ckpt_dir.mkdir()
        saved = str(ckpt_dir / "tokenizer.json")
        tok.tokenizer.save(saved)

        # Reload: the chat tokens must still be present and map to the SAME ids
        # the model was trained on (otherwise inference feeds the wrong ids).
        reloaded = CodeTokenizer(saved)
        assert has_chat_tokens(reloaded)
        assert reloaded.tokenizer.token_to_id(IM_START) == im_start_id
        # And encoding the marker yields exactly that single id (not fragmented).
        assert reloaded.tokenizer.encode(IM_START).ids == [im_start_id]


class TestResolveTokenizerFromMetadata:
    def test_metadata_tokenizer_path_takes_priority(self, tmp_path):
        # Simulate the SFT checkpoint: metadata.json points at the expanded
        # tokenizer saved next to it. resolve_tokenizer_path must return it.
        base = _base_tokenizer(tmp_path)
        tok = CodeTokenizer(base)
        model = _tiny_model(tok.vocab_size)
        add_chat_tokens(tok, model)

        ckpt_dir = tmp_path / "step_00000010"
        ckpt_dir.mkdir()
        sft_tok = str(tmp_path / "tokenizer.json")
        tok.tokenizer.save(sft_tok)
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"tokenizer_path": sft_tok}), encoding="utf-8",
        )

        resolved = resolve_tokenizer_path(ckpt_dir)
        assert resolved == sft_tok
        assert has_chat_tokens(CodeTokenizer(resolved))

    def test_missing_metadata_path_falls_through(self, tmp_path):
        # A nonexistent tokenizer_path must NOT be returned (falls through to
        # the DatasetResolver/storage fallback rather than a dead path).
        ckpt_dir = tmp_path / "step_00000020"
        ckpt_dir.mkdir()
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"tokenizer_path": str(tmp_path / "does_not_exist.json")}),
            encoding="utf-8",
        )
        resolved = resolve_tokenizer_path(ckpt_dir)
        assert resolved != str(tmp_path / "does_not_exist.json")
