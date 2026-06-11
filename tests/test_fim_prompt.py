"""INFER-007: /v1/fim must feed the model FIM markers, not a stripped prompt.

The server built its FIM prompt as decode(encode_fim(prefix, suffix)). decode()
skips special tokens, so the <|fim_prefix|>/<|fim_suffix|>/<|fim_middle|> markers
were STRIPPED — the model received a plain "prefix+suffix" with no
fill-in-the-middle structure, silently breaking inline completions (the VS Code
ghost-text feature). CodeTokenizer.fim_prompt() keeps the markers so generate()'s
re-encode recovers the FIM ids.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from cola_coder.tokenizer import train_tokenizer as tt
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
from cola_coder.inference.server import create_app
from cola_coder.model.config import ModelConfig


def _tokenizer(tmp_path: Path) -> CodeTokenizer:
    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 200),
        vocab_size=320, output_path=out,
    )
    return CodeTokenizer(out)


class TestFimPromptKeepsMarkers:
    def test_markers_present_in_string(self, tmp_path):
        tok = _tokenizer(tmp_path)
        s = tok.fim_prompt("PRE", "SUF")
        assert "<|fim_prefix|>" in s
        assert "<|fim_suffix|>" in s
        assert "<|fim_middle|>" in s
        assert s.index("<|fim_prefix|>") < s.index("<|fim_suffix|>") < s.index("<|fim_middle|>")

    def test_reencode_recovers_fim_ids(self, tmp_path):
        # The whole point: feeding fim_prompt() to encode() yields the SAME ids
        # encode_fim() produced — so the model sees the intended FIM layout.
        tok = _tokenizer(tmp_path)
        assert tok.encode(tok.fim_prompt("PRE", "SUF"), add_bos=False) == \
            tok.encode_fim("PRE", "SUF")

    def test_decode_path_strips_markers_documents_bug(self, tmp_path):
        # The old approach: decode(encode_fim(...)) loses the markers entirely.
        tok = _tokenizer(tmp_path)
        decoded = tok.decode(tok.encode_fim("PRE", "SUF"))
        assert "<|fim_prefix|>" not in decoded
        assert "<|fim_middle|>" not in decoded


# --------------------------------------------------------------------------
# Server-level: the prompt actually handed to generate() carries the markers
# --------------------------------------------------------------------------

class _CapturingTokenizer:
    def encode(self, text, add_bos=False):
        return list(range(len(text.split())))

    def fim_prompt(self, prefix, suffix):
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

    def decode(self, ids):
        # If the server ever regresses to decode(encode_fim(...)), markers vanish.
        return "stripped prefix suffix"

    def encode_fim(self, prefix, suffix):
        return [1, 2, 3]


class _CapturingGenerator:
    def __init__(self):
        self.tokenizer = _CapturingTokenizer()
        self.device = "cpu"
        self.captured_prompt = None

        class _M:
            num_parameters = 1000
            config = ModelConfig(vocab_size=32768, max_seq_len=128)
        self.model = _M()

    def generate(self, prompt, **kwargs):
        self.captured_prompt = prompt
        return prompt + " infilled"


class TestServerFimUsesMarkers:
    def test_generate_receives_fim_markers(self):
        gen = _CapturingGenerator()
        client = TestClient(create_app(gen))
        resp = client.post("/v1/fim", json={"prefix": "const a =", "suffix": ";"})
        assert resp.status_code == 200
        assert gen.captured_prompt is not None
        assert "<|fim_prefix|>" in gen.captured_prompt
        assert "<|fim_middle|>" in gen.captured_prompt
