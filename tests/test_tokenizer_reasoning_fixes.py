"""BUG-003 (extract_thinking multi-block) + BUG-001/002 (tokenizer token guard).

Found in a fresh scan of the tokenizer + reasoning pipelines.
"""

import pytest

from cola_coder.reasoning.thinking_tokens import extract_thinking, strip_thinking


# ---------------------------------------------------------------------------
# BUG-003 — extract_thinking must strip ALL thinking blocks from `code`
# ---------------------------------------------------------------------------


class TestExtractThinkingMultiBlock:
    def test_single_block_unchanged(self):
        thinking, code = extract_thinking("<think>reason</think>print(1)")
        assert thinking == "reason"
        assert code == "print(1)"

    def test_no_thinking_returns_full_text(self):
        thinking, code = extract_thinking("print(1)")
        assert thinking == ""
        assert code == "print(1)"

    def test_multiple_blocks_code_is_clean(self):
        # Before the fix, `code` kept the 2nd block and broke execution.
        text = "<think>step 1</think>x = 1\n<think>step 2</think>print(x)"
        thinking, code = extract_thinking(text)
        assert thinking == "step 1"               # first block is the thinking
        assert "<think>" not in code              # NO leftover thinking markers
        assert "</think>" not in code
        assert "x = 1" in code and "print(x)" in code

    def test_code_matches_strip_thinking(self):
        text = "<think>a</think>code1<think>b</think>code2"
        _, code = extract_thinking(text)
        assert code == strip_thinking(text)

    def test_extracted_code_is_executable_python(self):
        # The whole point: the cleaned code parses (so the reward can run it).
        import ast

        text = "<think>plan</think>def f():\n    return 1\n<think>check</think>"
        _, code = extract_thinking(text)
        ast.parse(code)  # raises if leftover <think> tags remain


# ---------------------------------------------------------------------------
# BUG-001/002 — CodeTokenizer fails loud on a tokenizer missing core tokens
# ---------------------------------------------------------------------------


def _train_tokenizer(tmp_path, special_tokens=None):
    from cola_coder.tokenizer import train_tokenizer as tt

    samples = ["def f():\n    return 1\n", "print('hi')\n"] * 100
    out = str(tmp_path / "tok.json")
    if special_tokens is None:
        tt.train_from_iterator(iter(samples), vocab_size=300, output_path=out)
    else:
        # Build a tokenizer with a custom (incomplete) special-token set.
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=300, special_tokens=special_tokens,
        )
        tok.train_from_iterator(iter(samples), trainer)
        tok.save(out)
    return out


class TestTokenizerCoreTokenGuard:
    def test_valid_tokenizer_loads(self, tmp_path):
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

        path = _train_tokenizer(tmp_path)
        tok = CodeTokenizer(path)  # has all SPECIAL_TOKENS → no error
        assert tok.bos_id is not None
        assert tok.eos_id is not None

    def test_missing_core_token_raises(self, tmp_path):
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

        # Missing <|eos|> and <|unk|> → must fail loud, not silently corrupt.
        path = _train_tokenizer(tmp_path, special_tokens=["<|pad|>", "<|bos|>"])
        with pytest.raises(ValueError, match="missing required special tokens"):
            CodeTokenizer(path)
