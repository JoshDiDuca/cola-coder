"""Tests for prompt-lookup (PLD) drafting + offline acceptance analysis (INFER-035).

Pure-Python, deterministic — no torch, no GPU, no network. Covers config
validation, the drafter's match/locality/ladder behaviour, and the offline
acceptance analyzer (including the lossless never-overshoot invariant).
"""

from __future__ import annotations

import pytest

from cola_coder.inference.prompt_lookup import (
    AcceptanceReport,
    AdaptiveDraftLength,
    PromptLookupConfig,
    PromptLookupDrafter,
    StepAcceptance,
    _common_prefix_len,
    analyze_acceptance,
)


class TestAdaptiveDraftLength:
    """The γ controller (IDEA-006): pure, deterministic, exact-verification-safe."""

    def test_starts_at_initial_then_min(self) -> None:
        assert AdaptiveDraftLength(min_pred_tokens=2, max_pred_tokens=8).draft_length == 2
        ctl = AdaptiveDraftLength(min_pred_tokens=1, max_pred_tokens=8, initial_pred_tokens=4)
        assert ctl.draft_length == 4
        assert ctl.acceptance_ema is None

    def test_initial_is_clamped(self) -> None:
        assert AdaptiveDraftLength(max_pred_tokens=5, initial_pred_tokens=99).draft_length == 5

    def test_full_acceptance_grows_gamma_to_cap(self) -> None:
        ctl = AdaptiveDraftLength(min_pred_tokens=1, max_pred_tokens=4, initial_pred_tokens=1)
        for _ in range(10):
            ctl.update(accepted=5, drafted=5)  # ratio 1.0 → EMA climbs past grow_threshold
        assert ctl.draft_length == 4  # clamped at max
        assert ctl.acceptance_ema is not None and ctl.acceptance_ema > 0.9

    def test_zero_acceptance_shrinks_gamma_to_floor(self) -> None:
        ctl = AdaptiveDraftLength(min_pred_tokens=2, max_pred_tokens=10, initial_pred_tokens=9)
        for _ in range(20):
            ctl.update(accepted=0, drafted=5)  # ratio 0.0 → EMA below shrink_threshold
        assert ctl.draft_length == 2  # clamped at min

    def test_no_match_step_is_neutral(self) -> None:
        ctl = AdaptiveDraftLength(initial_pred_tokens=3)
        before = ctl.draft_length
        assert ctl.update(accepted=0, drafted=0) == before  # no signal
        assert ctl.acceptance_ema is None  # EMA untouched

    def test_update_returns_current_length(self) -> None:
        ctl = AdaptiveDraftLength(min_pred_tokens=1, max_pred_tokens=4, initial_pred_tokens=1)
        assert ctl.update(accepted=3, drafted=3) == ctl.draft_length

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_pred_tokens": 0},
            {"min_pred_tokens": 5, "max_pred_tokens": 3},
            {"ema_alpha": 0.0},
            {"ema_alpha": 1.5},
            {"shrink_threshold": 0.8, "grow_threshold": 0.5},  # shrink > grow
        ],
    )
    def test_invalid_config_raises(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            AdaptiveDraftLength(**kwargs)

    def test_invalid_update_counts_raise(self) -> None:
        ctl = AdaptiveDraftLength()
        with pytest.raises(ValueError):
            ctl.update(accepted=6, drafted=5)  # accepted > drafted
        with pytest.raises(ValueError):
            ctl.update(accepted=-1, drafted=5)


# ── PromptLookupConfig validation ─────────────────────────────────────────────


def test_config_defaults_are_valid() -> None:
    cfg = PromptLookupConfig()
    assert cfg.max_ngram_size == 3
    assert cfg.num_pred_tokens == 10
    assert cfg.min_ngram_size == 1


def test_config_custom_valid() -> None:
    cfg = PromptLookupConfig(max_ngram_size=5, num_pred_tokens=4, min_ngram_size=2)
    assert cfg.max_ngram_size == 5
    assert cfg.num_pred_tokens == 4
    assert cfg.min_ngram_size == 2


def test_config_max_ngram_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_ngram_size"):
        PromptLookupConfig(max_ngram_size=0)


def test_config_min_ngram_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_ngram_size"):
        PromptLookupConfig(min_ngram_size=0)


def test_config_min_greater_than_max_raises() -> None:
    with pytest.raises(ValueError, match="must be <="):
        PromptLookupConfig(max_ngram_size=2, min_ngram_size=3)


def test_config_num_pred_below_one_raises() -> None:
    with pytest.raises(ValueError, match="num_pred_tokens"):
        PromptLookupConfig(num_pred_tokens=0)


# ── PromptLookupDrafter.draft ─────────────────────────────────────────────────


def test_draft_copies_continuation_on_recurring_ngram() -> None:
    # Pattern: "import X from Y" appears earlier; the suffix "import X from"
    # recurs, so the drafter should copy what followed it earlier ("Y" = 99).
    # tokens: [10, 11, 12, 99, 50, 51, 10, 11, 12]
    #          import X  from  Y  ...        import X from  (suffix)
    context = [10, 11, 12, 99, 50, 51, 10, 11, 12]
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3, num_pred_tokens=2))
    draft = drafter.draft(context)
    # Earlier "10,11,12" ends at index 3; copy the next num_pred_tokens (2) tokens.
    assert draft == [99, 50]


def test_draft_prefers_latest_earlier_match() -> None:
    # The bigram (1, 2) occurs twice earlier; the LATER one (followed by 8)
    # must win over the earlier one (followed by 7).
    context = [1, 2, 7, 0, 1, 2, 8, 0, 1, 2]
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=2, num_pred_tokens=1))
    draft = drafter.draft(context)
    assert draft == [8]


def test_draft_returns_empty_when_no_match() -> None:
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3))
    assert drafter.draft([1, 2, 3, 4, 5, 6]) == []


def test_draft_returns_empty_when_context_too_short() -> None:
    drafter = PromptLookupDrafter()
    assert drafter.draft([]) == []
    assert drafter.draft([7]) == []  # need >= 1 token before the needle


def test_draft_respects_num_pred_cap() -> None:
    # A long run after the earlier match, but num_pred_tokens caps the draft.
    context = [5, 5, 0, 1, 2, 3, 4, 9, 9, 9, 5, 5]
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=2, num_pred_tokens=3))
    draft = drafter.draft(context)
    assert len(draft) == 3
    assert draft == [0, 1, 2]


def test_draft_ngram_ladder_falls_back_to_shorter() -> None:
    # No 3-gram suffix match, but a 1-gram (the token 7) does recur earlier.
    # max=3, min=1 -> ladder tries 3, 2, then 1 and succeeds at 1.
    context = [7, 42, 43, 44, 45, 7]
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3, num_pred_tokens=2, min_ngram_size=1))
    draft = drafter.draft(context)
    assert draft == [42, 43]  # tokens following the earlier lone 7


def test_draft_min_ngram_suppresses_short_matches() -> None:
    # The only earlier recurrence is a single token (7); with min_ngram_size=2
    # the lone-token match is disallowed, so no draft.
    context = [7, 42, 43, 44, 45, 7]
    drafter = PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3, num_pred_tokens=2, min_ngram_size=2))
    assert drafter.draft(context) == []


def test_drafter_exposes_config() -> None:
    cfg = PromptLookupConfig(max_ngram_size=4)
    assert PromptLookupDrafter(cfg).config is cfg
    assert isinstance(PromptLookupDrafter().config, PromptLookupConfig)


# ── _common_prefix_len ────────────────────────────────────────────────────────


def test_common_prefix_len() -> None:
    assert _common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert _common_prefix_len([1, 2, 3], [9, 2, 3]) == 0
    assert _common_prefix_len([], [1, 2]) == 0
    assert _common_prefix_len([1, 2], []) == 0
    assert _common_prefix_len([1, 2, 3, 4], [1, 2]) == 2  # shorter list bounds it


# ── analyze_acceptance ────────────────────────────────────────────────────────


def test_analyze_no_repeats_zero_acceptance() -> None:
    # Strictly increasing -> no n-gram ever recurs -> nothing ever accepted.
    tokens = list(range(20))
    report = analyze_acceptance(tokens, seed_len=1)
    assert isinstance(report, AcceptanceReport)
    assert report.total_tokens == len(tokens) - 1
    assert report.total_accepted == 0
    assert report.acceptance_rate == 0.0
    # No acceptance => one token committed per step => steps == tokens predicted.
    assert report.decode_steps == report.total_tokens
    assert report.baseline_steps == report.total_tokens
    assert report.speedup_estimate == pytest.approx(1.0)
    assert report.mean_accepted_length == 0.0


def test_analyze_repetitive_high_acceptance_and_speedup() -> None:
    # A short motif repeated many times: once seen, the drafter predicts the
    # whole next repetition, so acceptance is high and decode steps collapse.
    motif = [11, 22, 33, 44]
    tokens = motif * 12
    report = analyze_acceptance(
        tokens,
        PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3, num_pred_tokens=8)),
        seed_len=4,
    )
    assert report.total_accepted > 0
    assert report.acceptance_rate > 0.5
    assert report.decode_steps < report.total_tokens
    assert report.speedup_estimate > 1.0
    assert report.draft_hit_rate > 0.0


def test_analyze_seed_len_too_large_raises() -> None:
    with pytest.raises(ValueError, match="must exceed seed_len"):
        analyze_acceptance([1, 2, 3], seed_len=3)
    with pytest.raises(ValueError, match="must exceed seed_len"):
        analyze_acceptance([1, 2, 3], seed_len=10)


def test_analyze_seed_len_below_one_raises() -> None:
    with pytest.raises(ValueError, match="seed_len must be >= 1"):
        analyze_acceptance([1, 2, 3], seed_len=0)


def test_analyze_step_accounting_sums() -> None:
    motif = [1, 2, 3]
    tokens = motif * 8
    report = analyze_acceptance(tokens, seed_len=3)
    # total_accepted / total_drafted must equal the per-step sums.
    assert report.total_accepted == sum(s.accepted for s in report.steps)
    assert report.total_drafted == sum(s.drafted for s in report.steps)
    assert report.decode_steps == len(report.steps)
    steps_with_draft = sum(1 for s in report.steps if s.drafted > 0)
    expected_hit = steps_with_draft / report.decode_steps
    assert report.draft_hit_rate == pytest.approx(expected_hit)
    if report.total_drafted:
        assert report.acceptance_rate == pytest.approx(
            report.total_accepted / report.total_drafted
        )
    assert report.mean_accepted_length == pytest.approx(
        report.total_accepted / report.decode_steps
    )


def test_analyze_step_is_stepacceptance() -> None:
    report = analyze_acceptance([1, 2, 1, 2, 1, 2], seed_len=2)
    assert all(isinstance(s, StepAcceptance) for s in report.steps)
    first = report.steps[0]
    assert first.position == 2  # prediction begins right after the seed


# ── Lossless invariant: advance always, never overshoot ───────────────────────


@pytest.mark.parametrize(
    "tokens",
    [
        list(range(15)),  # no repeats
        [1, 2, 3] * 7,  # tight repetition
        [9, 9, 9, 9, 9, 9, 9, 9],  # degenerate single-token repetition
        [4, 5, 6, 4, 5, 6, 99, 4, 5, 6, 7, 8],  # partial overlap
        [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],  # growing prefix
    ],
)
def test_lossless_advance_and_no_overshoot(tokens: list[int]) -> None:
    for seed_len in (1, 2):
        if len(tokens) <= seed_len:
            continue
        report = analyze_acceptance(
            tokens,
            PromptLookupDrafter(PromptLookupConfig(max_ngram_size=3, num_pred_tokens=6)),
            seed_len=seed_len,
        )
        # Every step accepts no more than it drafted.
        for s in report.steps:
            assert 0 <= s.accepted <= s.drafted
            assert s.drafted >= 0

        # The simulated decode advances strictly: reconstruct final position from
        # the committed (accepted + 1) tokens per step and confirm it lands
        # exactly at the end of the trace (never overshoots, never stalls short).
        pos = seed_len
        for s in report.steps:
            assert s.position == pos  # steps are contiguous in trace order
            advance = s.accepted + 1
            assert advance >= 1  # always moves forward
            pos = min(pos + advance, len(tokens))
        assert pos == len(tokens)

        # decode_steps can never exceed a one-token-per-step baseline.
        assert report.decode_steps <= report.total_tokens
