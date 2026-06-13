"""SFT (Supervised Fine-Tuning) dataset for ChatML instruction data.

Reads JSONL files where each line contains a ``"messages"`` key with a list
of ChatML-format messages::

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Tokenizes each conversation using ChatML delimiters
(``<|im_start|>``/``<|im_end|>``) and builds ``input_ids`` / ``labels``
tensors where user turns are masked with ``-100`` so only assistant tokens
contribute to the cross-entropy loss.

Usage::

    from cola_coder.data.sft_dataset import SFTDataset, SFTCollator

    dataset = SFTDataset("data/sft_train.jsonl", tokenizer, max_seq_len=2048)
    collator = SFTCollator(pad_id=0)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
from cola_coder.tokenizer.chat_template import (
    IM_END,
    format_chat_training,
)

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Tokenized ChatML dataset for supervised fine-tuning.

    Each JSONL line must have a ``"messages"`` key containing a list of
    ``{"role": str, "content": str}`` dicts.  The dataset tokenizes the
    full conversation and masks non-assistant tokens in the labels with
    ``-100`` so the loss only applies to assistant responses.

    Args:
        data_path: Path to the JSONL file.
        tokenizer: A ``CodeTokenizer`` instance (must already have ChatML
            tokens added if they are not part of the base vocab).
        max_seq_len: Maximum sequence length.  Conversations longer than
            this are truncated from the right.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_seq_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self._input_ids: list[list[int]] = []
        self._labels: list[list[int]] = []

        self._load(data_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, data_path: str) -> None:
        """Read the JSONL file and tokenize every conversation."""
        path = Path(data_path)
        skipped = 0
        skipped_no_labels = 0

        with open(path, encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                messages = record.get("messages")
                if not messages or not isinstance(messages, list):
                    skipped += 1
                    continue

                input_ids, labels = self._tokenize_conversation(messages)
                if len(input_ids) < 2:
                    skipped += 1
                    continue

                # An example with NO labeled tokens (every label is -100) carries
                # zero SFT signal — it happens when a conversation has no assistant
                # turn, or when truncation to max_seq_len cuts off the assistant
                # content. Worse than useless: if such examples fill a batch,
                # CrossEntropyLoss(ignore_index=-100, mean) returns NaN, which
                # backprops to NaN weights and CORRUPTS the checkpoint (bf16 — the
                # primary precision — has no GradScaler to skip the step). Drop it
                # (DATA-034; the fail-loud-not-poison class).
                if not any(lbl != -100 for lbl in labels):
                    skipped_no_labels += 1
                    continue

                self._input_ids.append(input_ids)
                self._labels.append(labels)

        if skipped:
            logger.warning(
                "SFTDataset: skipped %d malformed lines in %s", skipped, path,
            )
        if skipped_no_labels:
            logger.warning(
                "SFTDataset: skipped %d examples with no assistant tokens "
                "(no SFT signal / NaN-loss risk) in %s",
                skipped_no_labels, path,
            )
        logger.info(
            "SFTDataset: loaded %d examples from %s", len(self._input_ids), path,
        )

    def _tokenize_conversation(
        self, messages: list[dict[str, str]],
    ) -> tuple[list[int], list[int]]:
        """Tokenize a ChatML conversation and build a label mask.

        The full conversation is formatted via ``format_chat_training`` which
        returns character-level spans for each assistant response.  We
        tokenize the whole string, then map the character spans to token
        indices to build the labels tensor where only assistant tokens are
        kept and everything else is ``-100``.
        """
        formatted_text, assistant_spans = format_chat_training(messages)

        # Tokenize the entire formatted conversation
        token_ids = self.tokenizer.encode(
            formatted_text, add_bos=True, add_eos=True,
        )

        # Build a character → is_assistant boolean map
        char_is_assistant = [False] * len(formatted_text)
        for start, end in assistant_spans:
            for i in range(start, min(end, len(formatted_text))):
                char_is_assistant[i] = True

        # Map tokens to the formatted text to figure out which tokens are
        # part of assistant content.  We use the tokenizer's encoding
        # offsets for accurate mapping.
        encoding = self.tokenizer.tokenizer.encode(formatted_text)
        offsets = encoding.offsets  # list of (char_start, char_end)

        # Build labels: -100 for non-assistant tokens, token_id for assistant
        labels = [-100] * len(token_ids)

        # The first token is BOS (added by encode()), so offset by 1
        bos_offset = 1
        for tok_idx in range(bos_offset, min(len(token_ids), len(offsets) + bos_offset)):
            offset_idx = tok_idx - bos_offset
            if offset_idx >= len(offsets):
                break
            char_start, char_end = offsets[offset_idx]
            # A token is labeled if any of its characters fall in an assistant span
            if char_end > char_start and any(
                char_is_assistant[c]
                for c in range(char_start, min(char_end, len(char_is_assistant)))
            ):
                labels[tok_idx] = token_ids[tok_idx]

        # Also include the <|im_end|> token after assistant content and EOS
        # so the model learns to stop generating.
        im_end_id = self.tokenizer.tokenizer.token_to_id(IM_END)
        eos_id = self.tokenizer.eos_id
        for tok_idx in range(1, len(token_ids)):
            if token_ids[tok_idx] in (im_end_id, eos_id):
                # Check if the preceding labeled region was assistant content
                if tok_idx > 0 and labels[tok_idx - 1] != -100:
                    labels[tok_idx] = token_ids[tok_idx]

        # Truncate to max_seq_len. The assistant response — the ONLY supervised
        # content — sits at the END of the sequence, so right-truncation
        # (token_ids[:max_seq_len]) drops exactly the tokens we train on:
        # any example whose prompt alone exceeds max_seq_len would lose its
        # whole response and then get silently discarded by the no-label drop
        # in _load. Instead truncate from the LEFT of the prompt, keeping BOS
        # (position 0, where get_rope_freqs/the model expect it) plus the
        # max_seq_len-1 most recent tokens so the response survives intact.
        # Labels are sliced identically to stay token-aligned.
        if len(token_ids) > self.max_seq_len:
            tail = self.max_seq_len - 1  # reserve slot 0 for BOS
            if tail > 0:
                token_ids = [token_ids[0]] + token_ids[-tail:]
                labels = [-100] + labels[-tail:]
            else:  # max_seq_len <= 1: nothing trainable survives — drop in _load
                token_ids = token_ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]

        return token_ids, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self._input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self._labels[idx], dtype=torch.long),
        }


class SFTCollator:
    """Collator that pads ``input_ids`` and ``labels`` to the same length.

    Pads ``input_ids`` with ``pad_id`` and ``labels`` with ``-100`` so that
    padding tokens never contribute to the cross-entropy loss.

    Args:
        pad_id: Token ID used for padding ``input_ids``.
    """

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(
        self, batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids_list.append(
                    torch.cat([
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_id, dtype=torch.long),
                    ])
                )
                labels_list.append(
                    torch.cat([
                        item["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ])
                )
            else:
                input_ids_list.append(item["input_ids"])
                labels_list.append(item["labels"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
        }
