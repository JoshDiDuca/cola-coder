"""Tests for Fill-in-the-Middle (FIM) training support.

All tests run without a GPU and without an actual tokenizer.json file —
a lightweight stub tokenizer is used instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from cola_coder.data.fim import (
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMTransform,
    setup_fim_tokenizer,
)
from cola_coder.data.fim_dataset import FIMDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Stable token IDs that never collide with ordinary payload content
_FIM_PREFIX_ID = 9991
_FIM_SUFFIX_ID = 9992
_FIM_MIDDLE_ID = 9993
_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_UNK_ID = 3


def _make_tokenizer() -> MagicMock:
    """Return a lightweight mock that satisfies FIMTransform's interface."""
    tok = MagicMock()
    tok.fim_prefix_id = _FIM_PREFIX_ID
    tok.fim_suffix_id = _FIM_SUFFIX_ID
    tok.fim_middle_id = _FIM_MIDDLE_ID
    tok.pad_id = _PAD_ID
    tok.bos_id = _BOS_ID
    tok.eos_id = _EOS_ID
    tok.unk_id = _UNK_ID

    # Inner .tokenizer object used by setup_fim_tokenizer
    inner = MagicMock()
    inner.token_to_id.side_effect = lambda t: {
        "<|fim_prefix|>": _FIM_PREFIX_ID,
        "<|fim_suffix|>": _FIM_SUFFIX_ID,
        "<|fim_middle|>": _FIM_MIDDLE_ID,
    }.get(t)
    tok.tokenizer = inner
    tok.add_special_tokens = MagicMock(return_value=32770)
    tok.vocab_size = 32770

    return tok


def _make_token_ids(length: int = 50, offset: int = 100) -> list[int]:
    """Return a predictable list of token IDs (values 100..149 for length=50)."""
    return list(range(offset, offset + length))


def _make_base_dataset(n: int = 8, seq_len: int = 50) -> torch.utils.data.Dataset:
    """Return a trivial in-memory dataset of constant token tensors."""

    class _TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, idx):
            # Each item has a distinct offset so we can tell them apart
            ids = torch.tensor(
                list(range(100 + idx * seq_len, 100 + idx * seq_len + seq_len)),
                dtype=torch.int64,
            )
            return {"input_ids": ids}

    return _TinyDS()


# ===========================================================================
# 1. FIM special token strings
# ===========================================================================


class TestFIMTokenStrings:
    def test_psm_prefix_constant(self):
        assert FIM_PREFIX == "<fim_prefix>"

    def test_psm_suffix_constant(self):
        assert FIM_SUFFIX == "<fim_suffix>"

    def test_psm_middle_constant(self):
        assert FIM_MIDDLE == "<fim_middle>"


# ===========================================================================
# 2. FIMTransform — constructor validation
# ===========================================================================


class TestFIMTransformInit:
    def test_valid_rates(self):
        t = FIMTransform(fim_rate=0.5, psm_rate=0.5)
        assert t.fim_rate == 0.5
        assert t.psm_rate == 0.5

    def test_invalid_fim_rate_raises(self):
        with pytest.raises(ValueError, match="fim_rate"):
            FIMTransform(fim_rate=1.5)

    def test_invalid_psm_rate_raises(self):
        with pytest.raises(ValueError, match="psm_rate"):
            FIMTransform(psm_rate=-0.1)

    def test_fim_rate_zero_never_transforms(self):
        """fim_rate=0 must never modify a sequence."""
        tokenizer = _make_tokenizer()
        transform = FIMTransform(fim_rate=0.0, seed=0)
        ids = _make_token_ids(50)
        for _ in range(100):
            result = transform.apply(ids, tokenizer)
            assert result == ids, "fim_rate=0 should never transform"

    def test_fim_rate_one_always_transforms(self):
        """fim_rate=1 must always modify a sequence (assuming it is long enough)."""
        tokenizer = _make_tokenizer()
        transform = FIMTransform(fim_rate=1.0, seed=0)
        ids = _make_token_ids(50)
        transformed_count = 0
        for _ in range(20):
            result = transform.apply(ids, tokenizer)
            if result != ids:
                transformed_count += 1
        assert transformed_count == 20, "fim_rate=1 should always transform"


# ===========================================================================
# 3. FIMTransform.apply — PSM format
# ===========================================================================


class TestFIMTransformPSM:
    """fim_rate=1, psm_rate=1 → always produces PSM."""

    def _transform(self, ids: list[int]) -> list[int]:
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, truncate_or_pad=True, seed=7)
        return t.apply(ids, tokenizer)

    def test_psm_starts_with_fim_prefix_id(self):
        result = self._transform(_make_token_ids(50))
        assert result[0] == _FIM_PREFIX_ID, "PSM must start with fim_prefix token"

    def test_psm_contains_fim_suffix_id(self):
        result = self._transform(_make_token_ids(50))
        assert _FIM_SUFFIX_ID in result

    def test_psm_contains_fim_middle_id(self):
        result = self._transform(_make_token_ids(50))
        assert _FIM_MIDDLE_ID in result

    def test_psm_ordering_prefix_before_suffix_before_middle(self):
        """In PSM the special tokens must appear in order: prefix → suffix → middle."""
        result = self._transform(_make_token_ids(60))
        prefix_pos = result.index(_FIM_PREFIX_ID)
        suffix_pos = result.index(_FIM_SUFFIX_ID)
        middle_pos = result.index(_FIM_MIDDLE_ID)
        assert prefix_pos < suffix_pos < middle_pos

    def test_psm_length_preserved_with_truncate(self):
        ids = _make_token_ids(50)
        result = self._transform(ids)
        assert len(result) == len(ids), "Output length must equal input length"

    def test_psm_contains_all_special_tokens(self):
        result = self._transform(_make_token_ids(50))
        specials = {_FIM_PREFIX_ID, _FIM_SUFFIX_ID, _FIM_MIDDLE_ID}
        found = {r for r in result if r in specials}
        assert found == specials


# ===========================================================================
# 4. FIMTransform.apply — SPM format
# ===========================================================================


class TestFIMTransformSPM:
    """fim_rate=1, psm_rate=0 → always produces SPM."""

    def _transform(self, ids: list[int]) -> list[int]:
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=0.0, truncate_or_pad=True, seed=7)
        return t.apply(ids, tokenizer)

    def test_spm_starts_with_fim_suffix_id(self):
        result = self._transform(_make_token_ids(50))
        assert result[0] == _FIM_SUFFIX_ID, "SPM must start with fim_suffix token"

    def test_spm_ordering_suffix_before_prefix_before_middle(self):
        """In SPM the order is: suffix → prefix → middle."""
        result = self._transform(_make_token_ids(60))
        suffix_pos = result.index(_FIM_SUFFIX_ID)
        prefix_pos = result.index(_FIM_PREFIX_ID)
        middle_pos = result.index(_FIM_MIDDLE_ID)
        assert suffix_pos < prefix_pos < middle_pos

    def test_spm_length_preserved_with_truncate(self):
        ids = _make_token_ids(50)
        result = self._transform(ids)
        assert len(result) == len(ids)


# ===========================================================================
# 5. FIMTransform.apply_to_text
# ===========================================================================


class TestFIMTransformText:
    def test_apply_to_text_psm_format(self):
        """PSM text result contains tokens in the right order."""
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, seed=3)
        text = "line1\nline2\nline3\nline4\nline5\n"
        result, was_transformed = t.apply_to_text(text)
        assert was_transformed
        assert FIM_PREFIX in result
        assert FIM_SUFFIX in result
        assert FIM_MIDDLE in result
        # PSM: prefix comes first
        assert result.index(FIM_PREFIX) < result.index(FIM_SUFFIX)
        assert result.index(FIM_SUFFIX) < result.index(FIM_MIDDLE)

    def test_apply_to_text_spm_format(self):
        """SPM text result starts with fim_suffix."""
        t = FIMTransform(fim_rate=1.0, psm_rate=0.0, seed=3)
        text = "line1\nline2\nline3\nline4\nline5\n"
        result, was_transformed = t.apply_to_text(text)
        assert was_transformed
        assert result.index(FIM_SUFFIX) < result.index(FIM_PREFIX)
        assert result.index(FIM_PREFIX) < result.index(FIM_MIDDLE)

    def test_apply_to_text_not_transformed_when_rate_zero(self):
        t = FIMTransform(fim_rate=0.0)
        text = "def foo():\n    return 42\n"
        result, was_transformed = t.apply_to_text(text)
        assert not was_transformed
        assert result == text

    def test_apply_to_text_prefers_line_boundaries(self):
        """Splits should occur between whole lines, not mid-identifier."""
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, seed=0)
        text = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
        result, was_transformed = t.apply_to_text(text)
        if was_transformed:
            # Strip the special tokens; remaining text should only have whole lines
            content = result
            for token in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE):
                content = content.replace(token, "")
            for word in ("alpha", "beta", "gamma", "delta", "epsilon"):
                assert word in content, f"Word {word!r} disappeared after FIM split"

    def test_apply_to_text_roundtrip_content_preserved(self):
        """All characters from the original are present after transformation."""
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, seed=0)
        text = "def add(a, b):\n    return a + b\n\nresult = add(1, 2)\n"
        result, was_transformed = t.apply_to_text(text)
        assert was_transformed
        # Remove FIM markers and compare sorted character sets
        clean = result.replace(FIM_PREFIX, "").replace(FIM_SUFFIX, "").replace(FIM_MIDDLE, "")
        assert sorted(clean) == sorted(text), "No characters should be lost or added"


# ===========================================================================
# 6. setup_fim_tokenizer
# ===========================================================================


class TestSetupFIMTokenizer:
    def test_returns_three_ids(self):
        tokenizer = _make_tokenizer()
        ids = setup_fim_tokenizer(tokenizer)
        assert set(ids.keys()) == {"fim_prefix", "fim_suffix", "fim_middle"}

    def test_ids_are_integers(self):
        tokenizer = _make_tokenizer()
        ids = setup_fim_tokenizer(tokenizer)
        for key, val in ids.items():
            assert isinstance(val, int), f"{key} must be an int, got {type(val)}"

    def test_ids_are_unique(self):
        tokenizer = _make_tokenizer()
        ids = setup_fim_tokenizer(tokenizer)
        values = list(ids.values())
        assert len(values) == len(set(values)), "FIM token IDs must be unique"

    def test_sets_attributes_on_tokenizer(self):
        tokenizer = _make_tokenizer()
        ids = setup_fim_tokenizer(tokenizer)
        assert tokenizer.fim_prefix_id == ids["fim_prefix"]
        assert tokenizer.fim_suffix_id == ids["fim_suffix"]
        assert tokenizer.fim_middle_id == ids["fim_middle"]

    def test_add_tokens_called_when_missing(self):
        """When a token is absent from the vocab, add_special_tokens is called."""
        tokenizer = _make_tokenizer()
        # Simulate one token not being in the vocab initially
        tokenizer.tokenizer.token_to_id.side_effect = lambda t: (
            None if t == "<|fim_middle|>" else {
                "<|fim_prefix|>": _FIM_PREFIX_ID,
                "<|fim_suffix|>": _FIM_SUFFIX_ID,
            }.get(t)
        )

        # After add_special_tokens is called, make the inner mock return a new ID
        added_call_count = 0

        def _mock_token_to_id(t):
            nonlocal added_call_count
            if t == "<|fim_middle|>" and added_call_count > 0:
                return 9999
            if t == "<|fim_middle|>":
                return None
            return {
                "<|fim_prefix|>": _FIM_PREFIX_ID,
                "<|fim_suffix|>": _FIM_SUFFIX_ID,
            }.get(t)

        tokenizer.tokenizer.token_to_id.side_effect = _mock_token_to_id

        def _mock_add(tokens):
            nonlocal added_call_count
            added_call_count += 1
            return 32771

        tokenizer.add_special_tokens = MagicMock(side_effect=_mock_add)

        setup_fim_tokenizer(tokenizer)
        assert tokenizer.add_special_tokens.call_count >= 1


# ===========================================================================
# 7. FIMDataset
# ===========================================================================


class TestFIMDataset:
    def _make_ds(self, fim_rate: float = 1.0, seed: int = 7) -> FIMDataset:
        base = _make_base_dataset(n=10, seq_len=50)
        tokenizer = _make_tokenizer()
        return FIMDataset(base, tokenizer, fim_rate=fim_rate, psm_rate=1.0, seed=seed)

    def test_length_matches_base(self):
        base = _make_base_dataset(n=10)
        ds = FIMDataset(base, _make_tokenizer(), fim_rate=0.5)
        assert len(ds) == 10

    def test_getitem_returns_dict_with_input_ids(self):
        ds = self._make_ds()
        item = ds[0]
        assert "input_ids" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_getitem_preserves_sequence_length(self):
        ds = self._make_ds(fim_rate=1.0)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape[0] == 50, \
                f"Sequence length changed at index {i}"

    def test_fim_rate_zero_leaves_sequence_unchanged(self):
        base = _make_base_dataset(n=5, seq_len=50)
        tokenizer = _make_tokenizer()
        ds = FIMDataset(base, tokenizer, fim_rate=0.0)
        for i in range(len(ds)):
            base_item = base[i]
            fim_item = ds[i]
            assert torch.equal(base_item["input_ids"], fim_item["input_ids"]), \
                f"fim_rate=0 should never change input_ids (index {i})"

    def test_fim_rate_one_inserts_special_tokens(self):
        """With fim_rate=1 every item should contain at least one FIM special token."""
        ds = self._make_ds(fim_rate=1.0)
        fim_ids = {_FIM_PREFIX_ID, _FIM_SUFFIX_ID, _FIM_MIDDLE_ID}
        for i in range(len(ds)):
            ids = ds[i]["input_ids"].tolist()
            assert any(t in fim_ids for t in ids), \
                f"Index {i} was not transformed despite fim_rate=1"

    def test_extra_keys_preserved(self):
        """Any extra keys from the base dataset (e.g. 'weight') survive wrapping."""

        class _WeightedDS(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(list(range(100, 150)), dtype=torch.int64),
                    "weight": torch.tensor(1.5),
                }

        ds = FIMDataset(_WeightedDS(), _make_tokenizer(), fim_rate=0.0)
        item = ds[0]
        assert "weight" in item
        assert item["weight"].item() == pytest.approx(1.5)


# ===========================================================================
# 8. Round-trip: extract prefix + middle + suffix from FIM tokens
# ===========================================================================


class TestFIMRoundTrip:
    """Given a FIM-formatted token sequence, we can recover the three parts."""

    def _roundtrip(self, ids: list[int]) -> tuple[list[int], list[int], list[int]]:
        """Parse a PSM token list back into (prefix, middle, suffix)."""
        p = ids.index(_FIM_PREFIX_ID)
        s = ids.index(_FIM_SUFFIX_ID)
        m = ids.index(_FIM_MIDDLE_ID)
        prefix = ids[p + 1: s]
        suffix = ids[s + 1: m]
        middle = ids[m + 1:]
        return prefix, suffix, middle

    def test_roundtrip_recovers_all_content(self):
        """After FIM transform the three parts reassemble to the original content (minus 3 slots).

        truncate_or_pad=True keeps the output the same length as the input by
        using only the first (n - 3) content tokens.  The 3 saved positions are
        filled by the special tokens.  So the recovered content length is n - 3.
        """
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, truncate_or_pad=True, seed=42)
        original = _make_token_ids(50)

        result = t.apply(original, tokenizer)

        prefix, suffix, middle = self._roundtrip(result)
        recovered = prefix + middle + suffix

        # With truncate_or_pad=True the transform works on (n-3) content tokens,
        # so we expect exactly n-3 non-special tokens in the result.
        expected_content_len = len(original) - 3
        assert len(recovered) == expected_content_len, (
            f"Recovered {len(recovered)} content tokens, expected {expected_content_len} "
            f"(original {len(original)} - 3 special token slots)"
        )

        # The recovered tokens should be a subsequence of the original content
        # (drawn from the first n-3 tokens, reordered by the FIM split)
        original_content_set = set(original[:expected_content_len])
        assert set(recovered).issubset(original_content_set), \
            "Recovered tokens must all come from the original sequence"

    def test_middle_is_non_empty(self):
        """The middle segment should have at least one token for non-trivial splits."""
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, truncate_or_pad=True, seed=0)
        ids = _make_token_ids(60)
        result = t.apply(ids, tokenizer)
        _, _, middle = self._roundtrip(result)
        assert len(middle) > 0, "Middle portion must be non-empty"

    def test_prefix_is_non_empty(self):
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, truncate_or_pad=True, seed=5)
        ids = _make_token_ids(60)
        result = t.apply(ids, tokenizer)
        prefix, _, _ = self._roundtrip(result)
        assert len(prefix) > 0, "Prefix portion must be non-empty"

    def test_suffix_is_non_empty(self):
        tokenizer = _make_tokenizer()
        t = FIMTransform(fim_rate=1.0, psm_rate=1.0, truncate_or_pad=True, seed=5)
        ids = _make_token_ids(60)
        result = t.apply(ids, tokenizer)
        _, suffix, _ = self._roundtrip(result)
        assert len(suffix) > 0, "Suffix portion must be non-empty"
