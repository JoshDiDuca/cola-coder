"""DATA-012: dynamic (train-time) FIM wired into create_dataloader + trainer.

A fraction of each batch is rearranged into FIM format on the fly (different
splits each epoch). Wired end-to-end: DataConfig.fim_rate → Trainer resolves
the <|fim_*|> ids from the tokenizer → create_dataloader wraps the collator
with FIMTrainingCollator. These tests prove it's NOT a silent no-op and that it
composes with quality weights, without running training.
"""

from types import SimpleNamespace

import numpy as np
import torch

from cola_coder.data.dataset import FIMTrainingCollator, create_dataloader
from cola_coder.training.trainer import Trainer
from cola_coder.tokenizer import train_tokenizer as tt

# Distinct FIM ids that never collide with the payload content (100..)
_FIM_IDS = {"fim_prefix_id": 4, "fim_suffix_id": 6, "fim_middle_id": 5}
_FIM_SET = set(_FIM_IDS.values())


def _examples(n=4, seq=40):
    return [{"input_ids": torch.arange(100 + i * seq, 100 + i * seq + seq)} for i in range(n)]


class TestFIMTrainingCollator:
    def test_applies_fim(self):
        c = FIMTrainingCollator(fim_rate=1.0, fim_ids=_FIM_IDS, seed=0)
        batch = c(_examples())
        assert batch["input_ids"].shape == (4, 40)
        # Every row was FIM-transformed → contains the marker ids.
        for row in batch["input_ids"].tolist():
            assert _FIM_SET & set(row), "no FIM markers in a transformed row"

    def test_rate_zero_no_markers(self):
        c = FIMTrainingCollator(fim_rate=0.0, fim_ids=_FIM_IDS, seed=0)
        batch = c(_examples())
        for row in batch["input_ids"].tolist():
            assert not (_FIM_SET & set(row))

    def test_preserves_weights(self):
        c = FIMTrainingCollator(fim_rate=1.0, fim_ids=_FIM_IDS, seed=0)
        exs = _examples()
        for i, ex in enumerate(exs):
            ex["weight"] = torch.tensor(1.0 + i)
        batch = c(exs)
        assert "weights" in batch
        assert batch["weights"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_empty_batch_does_not_crash(self):
        # An empty example list must not raise IndexError (examples[0]) or
        # RuntimeError (torch.stack([])). drop_last=True means the live
        # training path never produces this, so the guard is zero-numeric-impact.
        c = FIMTrainingCollator(fim_rate=1.0, fim_ids=_FIM_IDS, seed=0)
        batch = c([])
        assert batch["input_ids"].numel() == 0
        assert "weights" not in batch


class TestCreateDataloaderFIM:
    def _npy(self, tmp_path):
        arr = np.tile(np.arange(100, 140, dtype=np.uint16), (8, 1))  # (8, 40)
        p = tmp_path / "data.npy"
        np.save(str(p), arr)
        return str(p)

    def test_fim_applied_in_batches(self, tmp_path):
        dl = create_dataloader(
            self._npy(tmp_path), batch_size=4, shuffle=False, num_workers=0,
            fim_rate=1.0, fim_ids=_FIM_IDS,
        )
        batch = next(iter(dl))
        rows = batch["input_ids"].tolist()
        assert any(_FIM_SET & set(r) for r in rows)

    def test_no_fim_when_rate_zero(self, tmp_path):
        dl = create_dataloader(
            self._npy(tmp_path), batch_size=4, shuffle=False, num_workers=0,
            fim_rate=0.0, fim_ids=_FIM_IDS,
        )
        batch = next(iter(dl))
        for r in batch["input_ids"].tolist():
            assert not (_FIM_SET & set(r))


class TestTrainerResolveFimIds:
    def _tok(self, tmp_path, special=None):
        out = str(tmp_path / "tokenizer.json")
        samples = ["def f():\n  return 1\n", "const x = 1;\n"] * 100
        if special is None:
            tt.train_from_iterator(iter(samples), vocab_size=320, output_path=out)
        else:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            tok = Tokenizer(models.BPE(unk_token=None))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.train_from_iterator(iter(samples), trainers.BpeTrainer(
                vocab_size=320, special_tokens=special))
            tok.save(out)
        return out

    def _stub(self, fim_rate=0.5):
        return SimpleNamespace(config=SimpleNamespace(data=SimpleNamespace(fim_rate=fim_rate)))

    def test_resolves_ids_when_present(self, tmp_path):
        path = self._tok(tmp_path)  # base SPECIAL_TOKENS include <|fim_*|>
        ids = Trainer._resolve_fim_ids(self._stub(), path)
        assert ids is not None
        assert set(ids) == {"fim_prefix_id", "fim_suffix_id", "fim_middle_id"}
        assert all(isinstance(v, int) for v in ids.values())

    def test_none_when_tokenizer_missing(self, tmp_path):
        assert Trainer._resolve_fim_ids(self._stub(), str(tmp_path / "nope.json")) is None

    def test_none_when_no_fim_tokens(self, tmp_path):
        # Tokenizer without FIM tokens (only core) → FIM disabled, not crashed.
        path = self._tok(tmp_path, special=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"])
        assert Trainer._resolve_fim_ids(self._stub(), path) is None
