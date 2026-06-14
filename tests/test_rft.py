"""MODEL-045: RFT / self-verified distillation harness.

Generates self-solved candidates via best-of-N, keeps only verifier-passed +
secure ones as SFT records. Uses the best-of-N fakes (no GPU/tsc/sandbox).
"""

import sys

sys.path.insert(0, "tests")
from test_best_of_n import FakeGroupGenerator, FakeTscRunner  # noqa: E402

from cola_coder.distillation.rft import generate_rft_dataset  # noqa: E402


def test_keeps_verified_secure_completions():
    gen = FakeGroupGenerator(["const x: number = 1;", "const y: number = 2;"])
    runner = FakeTscRunner({})  # both candidates verify clean
    records, stats = generate_rft_dataset(
        gen, ["// solve A", "// solve B"], num_candidates=2,
        language="typescript", tsc_runner=runner,
    )
    assert stats["kept"] == 2
    assert stats["verified"] == 2
    # ChatML shape: user prompt + assistant completion.
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][-1]["role"] == "assistant"


def test_drops_unverified_when_keep_only_verified():
    gen = FakeGroupGenerator(["const a: number = 1;", "const b: number = 2;"])
    runner = FakeTscRunner({0: ["TS2322: e"], 1: ["TS2322: e"]})  # none verify
    records, stats = generate_rft_dataset(
        gen, ["// p"], num_candidates=2, language="typescript",
        tsc_runner=runner, keep_only_verified=True,
    )
    assert records == []
    assert stats["rejected_unverified"] == 1
    assert stats["kept"] == 0


def test_keep_unverified_when_flag_off():
    gen = FakeGroupGenerator(["const a: number = 1;", "const b: number = 2;"])
    runner = FakeTscRunner({0: ["e"], 1: ["e"]})
    records, stats = generate_rft_dataset(
        gen, ["// p"], num_candidates=2, language="typescript",
        tsc_runner=runner, keep_only_verified=False, require_secure=False,
    )
    assert len(records) == 1
    assert stats["rejected_unverified"] == 0  # not rejected; kept despite failing


def test_drops_insecure_completion():
    # The only candidate verifies but contains a dangerous pattern → rejected.
    gen = FakeGroupGenerator(["const r = eval(userInput);"])
    runner = FakeTscRunner({})  # tsc passes
    records, stats = generate_rft_dataset(
        gen, ["// p"], num_candidates=1, language="typescript",
        tsc_runner=runner, require_secure=True,
    )
    assert records == []
    assert stats["rejected_insecure"] == 1


def test_tests_length_validation():
    gen = FakeGroupGenerator(["x"])
    import pytest
    with pytest.raises(ValueError, match="same length"):
        generate_rft_dataset(gen, ["a", "b"], tests=["only one"])
