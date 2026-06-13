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


class TestTruncationKeepsResponse:
    """SFT-001: over-long examples must truncate the PROMPT, not the response.

    The supervised assistant response sits at the end of the ChatML sequence.
    Right-truncation (`token_ids[:max_seq_len]`) keeps the prompt and drops the
    response, so an example whose prompt alone exceeds max_seq_len loses every
    labeled token and is then silently discarded by the no-label drop. The fix
    truncates from the left (keeping BOS + the tail) so the response survives.
    """

    def _long_prompt_short_answer(self):
        return {"messages": [
            # A prompt long enough that, with a small max_seq_len, the prompt
            # alone would overflow under right-truncation.
            {"role": "user", "content": "explain this in great detail " * 12},
            {"role": "assistant", "content": "the answer is yes"},
        ]}

    def test_long_prompt_short_response_is_kept_not_dropped(
        self, chat_tokenizer, tmp_path,
    ):
        data = _write_jsonl(tmp_path, [self._long_prompt_short_answer()])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=32)
        # Under the old right-truncation this example was dropped (len == 0).
        assert len(ds) == 1
        assert _labeled(ds._labels[0]) > 0  # response tokens survived

    def test_truncation_respects_max_seq_len_and_keeps_bos(
        self, chat_tokenizer, tmp_path,
    ):
        max_len = 32
        data = _write_jsonl(tmp_path, [self._long_prompt_short_answer()])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=max_len)
        ids, labels = ds._input_ids[0], ds._labels[0]
        assert len(ids) <= max_len
        assert len(ids) == len(labels)
        # BOS stays at position 0 and is never supervised.
        assert ids[0] == chat_tokenizer.bos_id
        assert labels[0] == -100

    def test_truncation_keeps_final_assistant_tokens(
        self, chat_tokenizer, tmp_path,
    ):
        # The last labeled token should be part of the (left-preserved) tail,
        # i.e. the response's own tokens — not lost to truncation.
        max_len = 32
        data = _write_jsonl(tmp_path, [self._long_prompt_short_answer()])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=max_len)
        ids, labels = ds._input_ids[0], ds._labels[0]
        labeled_positions = [i for i, lbl in enumerate(labels) if lbl != -100]
        assert labeled_positions  # at least one supervised token
        # Labeled tokens echo the corresponding input ids (token-aligned).
        for i in labeled_positions:
            assert labels[i] == ids[i]

    def test_short_example_is_untouched_by_truncation(
        self, chat_tokenizer, tmp_path,
    ):
        # An example that already fits must pass through unchanged.
        row = {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]}
        data = _write_jsonl(tmp_path, [row])
        ds = SFTDataset(data, chat_tokenizer, max_seq_len=256)
        assert len(ds) == 1
        assert _labeled(ds._labels[0]) > 0
