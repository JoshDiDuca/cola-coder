"""DATA-034: SFTDataset must drop examples with NO labeled (assistant) tokens.

A conversation with no assistant turn — or one whose assistant content is
truncated away by max_seq_len — tokenizes to labels that are ALL -100. Such an
example carries zero SFT signal and, if it fills a batch, makes
CrossEntropyLoss(ignore_index=-100, mean) return NaN → NaN gradients → corrupted
checkpoint (bf16, the primary precision, has no GradScaler to skip the step).
_load now skips these.
"""

import json

import pytest

from cola_coder.data.sft_dataset import SFTDataset
from cola_coder.tokenizer.chat_template import CHAT_TOKENS


@pytest.fixture()
def chat_tokenizer(tmp_path):
    from cola_coder.tokenizer import train_tokenizer as tt
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "hello world how are you\n"] * 100),
        vocab_size=340, output_path=out,
    )
    tok = CodeTokenizer(out)
    tok.add_special_tokens(CHAT_TOKENS)  # <|im_start|>/<|im_end|>, no model needed
    return tok


def _write_jsonl(tmp_path, rows) -> str:
    p = tmp_path / "sft.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _labeled(labels) -> int:
    return sum(1 for x in labels if x != -100)


class TestNoLabelExamplesDropped:
    def test_assistant_example_kept_with_labels(self, chat_tokenizer, tmp_path):
        data = _write_jsonl(tmp_path, [
            {"messages": [
                {"role": "user", "content": "write a function"},
                {"role": "assistant", "content": "def f():\n    return 1"},
            ]},
        ])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=128)
        assert len(ds) == 1
        assert _labeled(ds._labels[0]) > 0  # assistant tokens are labeled

    def test_user_only_conversation_dropped(self, chat_tokenizer, tmp_path):
        data = _write_jsonl(tmp_path, [
            {"messages": [{"role": "user", "content": "just a question, no answer"}]},
        ])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=128)
        assert len(ds) == 0  # no assistant tokens → no signal → dropped

    def test_system_only_conversation_dropped(self, chat_tokenizer, tmp_path):
        data = _write_jsonl(tmp_path, [
            {"messages": [{"role": "system", "content": "you are helpful"}]},
        ])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=128)
        assert len(ds) == 0

    def test_mixed_file_keeps_only_signal_examples(self, chat_tokenizer, tmp_path):
        data = _write_jsonl(tmp_path, [
            {"messages": [{"role": "user", "content": "no answer here"}]},
            {"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there friend"},
            ]},
            {"messages": [{"role": "system", "content": "system only"}]},
        ])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=128)
        assert len(ds) == 1
        assert all(_labeled(lbls) > 0 for lbls in ds._labels)

    def test_every_loaded_example_has_a_label(self, chat_tokenizer, tmp_path):
        # The invariant that prevents the NaN-loss path: nothing with all -100
        # ever reaches the trainer.
        data = _write_jsonl(tmp_path, [
            {"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"answer number {i}"},
            ]}
            for i in range(5)
        ] + [{"messages": [{"role": "user", "content": "dangling"}]}])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=128)
        assert len(ds) == 5
        assert all(_labeled(lbls) > 0 for lbls in ds._labels)
