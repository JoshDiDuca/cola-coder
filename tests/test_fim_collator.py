"""DATA-011: FIMCollator must not corrupt the FIM prediction target.

The orphaned FIMCollator (data/collator.py) hand-rolled its own FIM split that
built ``seq_len + 3`` tokens then truncated back to ``seq_len`` — silently
chopping the last 3 tokens, which are the END of ``middle`` (the segment the
model is trained to predict). For short middles it could even drop the
``<fim_middle>`` marker. It was also unreachable (only referenced in docs, never
wired into create_dataloader) and duplicated the correct, tested ``FIMTransform``.

The fix delegates to ``FIMTransform`` (which reserves 3 content slots up front,
so the output length equals the input and the target is never truncated). These
tests lock that behavior and add the module's first coverage.
"""

import torch

from cola_coder.data.collator import FIMCollator

# IDs chosen so they never collide with the payload content (100..)
_PREFIX_ID = 9991
_SUFFIX_ID = 9992
_MIDDLE_ID = 9993


def _collator(fim_rate=1.0, psm_rate=1.0, seed=7):
    return FIMCollator(
        fim_rate=fim_rate,
        fim_prefix_id=_PREFIX_ID,
        fim_middle_id=_MIDDLE_ID,
        fim_suffix_id=_SUFFIX_ID,
        psm_rate=psm_rate,
        seed=seed,
    )


def _example(seq_len=50, offset=100, dtype=torch.int64):
    return {"input_ids": torch.arange(offset, offset + seq_len, dtype=dtype)}


def _parse_psm(ids):
    """Split a PSM token list into (prefix, suffix, middle)."""
    p = ids.index(_PREFIX_ID)
    s = ids.index(_SUFFIX_ID)
    m = ids.index(_MIDDLE_ID)
    return ids[p + 1:s], ids[s + 1:m], ids[m + 1:]


class TestLengthPreserved:
    def test_batch_stacks_to_constant_length(self):
        c = _collator(fim_rate=1.0)
        batch = c([_example(50, offset=100 + i * 50) for i in range(4)])
        # Every row FIM-transformed yet still length 50 → stackable.
        assert batch["input_ids"].shape == (4, 50)

    def test_each_row_equals_input_length(self):
        c = _collator(fim_rate=1.0)
        out = c([_example(50)])["input_ids"]
        assert out.shape[1] == 50


class TestTargetNotTruncated:
    """The headline regression: middle (the prediction target) survives intact."""

    def test_middle_marker_and_target_present(self):
        # Across many seeds, fim_rate=1 must always leave a non-empty middle
        # AFTER the <fim_middle> marker — the old truncate-to-seq_len chopped it.
        for seed in range(40):
            c = _collator(fim_rate=1.0, psm_rate=1.0, seed=seed)
            ids = c([_example(60)])["input_ids"][0].tolist()
            assert _MIDDLE_ID in ids, f"missing <fim_middle> (seed {seed})"
            _, _, middle = _parse_psm(ids)
            assert len(middle) > 0, f"middle/target truncated away (seed {seed})"

    def test_no_content_lost_beyond_three_reserved_slots(self):
        # recovered content == n - 3 (the 3 slots taken by the special tokens),
        # all drawn from the original — i.e. nothing extra was dropped.
        c = _collator(fim_rate=1.0, psm_rate=1.0, seed=42)
        original = _example(50)["input_ids"].tolist()
        ids = c([{"input_ids": torch.tensor(original)}])["input_ids"][0].tolist()
        prefix, suffix, middle = _parse_psm(ids)
        recovered = prefix + middle + suffix
        assert len(recovered) == len(original) - 3
        assert set(recovered).issubset(set(original))


class TestRateGating:
    def test_rate_zero_leaves_sequence_unchanged(self):
        c = _collator(fim_rate=0.0)
        ex = _example(50)
        out = c([ex])["input_ids"][0]
        assert torch.equal(out, ex["input_ids"])
        # No FIM markers injected.
        assert _PREFIX_ID not in out.tolist()

    def test_rate_one_transforms_every_row(self):
        c = _collator(fim_rate=1.0, seed=3)
        out = c([_example(50, offset=100 + i * 50) for i in range(6)])["input_ids"]
        fim_ids = {_PREFIX_ID, _SUFFIX_ID, _MIDDLE_ID}
        for row in out.tolist():
            assert any(t in fim_ids for t in row)


class TestOrdering:
    def test_psm_ordering(self):
        c = _collator(fim_rate=1.0, psm_rate=1.0, seed=7)
        ids = c([_example(60)])["input_ids"][0].tolist()
        assert ids.index(_PREFIX_ID) < ids.index(_SUFFIX_ID) < ids.index(_MIDDLE_ID)

    def test_spm_ordering(self):
        c = _collator(fim_rate=1.0, psm_rate=0.0, seed=7)
        ids = c([_example(60)])["input_ids"][0].tolist()
        # SPM: suffix → prefix → middle. (The old collator could not do SPM.)
        assert ids.index(_SUFFIX_ID) < ids.index(_PREFIX_ID) < ids.index(_MIDDLE_ID)


class TestDtypePreserved:
    def test_dtype_matches_input(self):
        c = _collator(fim_rate=1.0)
        for dtype in (torch.int64, torch.int32, torch.int16):
            out = c([_example(50, dtype=dtype)])["input_ids"]
            assert out.dtype == dtype
