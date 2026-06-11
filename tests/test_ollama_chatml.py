"""TOOL-006: the Ollama Modelfile must use cola-coder's ChatML format.

The Modelfile previously used a LLaMA-3 template (<|start_header_id|> /
<|eot_id|> / <|end_of_text|>) — tokens that are NOT in cola-coder's vocabulary.
An exported Ollama model would fragment that template and never see its trained
<|im_start|>/<|im_end|> chat tokens, and the stop tokens would never fire —
broken instruction following (the BUG-106 family, on the Ollama-export side).
"""

from pathlib import Path

from cola_coder.export.ollama_export import OllamaExporter, _IM_START, _IM_END, _EOS


def _modelfile(tmp_path) -> str:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    mf = OllamaExporter().create_modelfile(str(gguf), str(tmp_path), "cola-coder")
    return Path(mf).read_text(encoding="utf-8")


class TestChatMLTemplate:
    def test_template_uses_chatml_tokens(self, tmp_path):
        content = _modelfile(tmp_path)
        assert _IM_START in content
        assert _IM_END in content
        # Role + assistant generation prompt present.
        assert f"{_IM_START}assistant" in content

    def test_no_llama3_tokens(self, tmp_path):
        content = _modelfile(tmp_path)
        for stale in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>"):
            assert stale not in content, f"stale LLaMA-3 token {stale!r} still in Modelfile"

    def test_stop_params_match_chat_tokens(self, tmp_path):
        content = _modelfile(tmp_path)
        assert f"PARAMETER stop {_IM_END}" in content
        assert f"PARAMETER stop {_EOS}" in content


class TestTokensInSyncWithCanonical:
    def test_tokens_match_chat_template_module(self):
        # chat_template imports torch; only the test does, keeping ollama_export
        # torch-free. The literals must match the canonical SFT constants.
        from cola_coder.tokenizer.chat_template import IM_START, IM_END

        assert _IM_START == IM_START
        assert _IM_END == IM_END

    def test_eos_matches_base_special_token(self):
        from cola_coder.tokenizer.train_tokenizer import SPECIAL_TOKENS

        assert _EOS in SPECIAL_TOKENS
