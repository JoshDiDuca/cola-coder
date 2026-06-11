"""DATA-026: stream_code_data must balance max_samples across languages.

The language loop yielded the ENTIRE first language before moving on, stopping
at max_samples total — so a multi-language request with a sample cap
(collect_data --config <multi-lang> --max-samples N) returned N of the first
language and ZERO of the rest (starvation), producing a language-imbalanced
training set. Each language now gets an even share of the remaining budget.

These tests stub _iter_hf_streaming (streaming=True path) so no network/cache is
touched, and assert per-language balance.
"""

from collections import Counter

import cola_coder.data.download as dl


def _install_fake(monkeypatch, counts: dict[str, int]):
    """Make each language's source yield `counts[lang]` distinct items."""
    def fake(dataset_name, lang, split):
        n = counts.get(lang, 0)
        for i in range(n):
            yield f"{lang}__{i}" + "x" * 50  # >= 50 chars (filter passes)

    monkeypatch.setattr(dl, "_iter_hf_streaming", fake)


def _dist(items):
    return Counter(s.split("__")[0] for s in items)


def test_two_languages_split_evenly(monkeypatch):
    _install_fake(monkeypatch, {"typescript": 100, "python": 100})
    out = list(dl.stream_code_data(
        languages=["typescript", "python"], max_samples=10, streaming=True,
    ))
    assert len(out) == 10
    assert _dist(out) == {"typescript": 5, "python": 5}


def test_three_languages_balanced(monkeypatch):
    _install_fake(monkeypatch, {"a": 100, "b": 100, "c": 100})
    out = list(dl.stream_code_data(languages=["a", "b", "c"], max_samples=10, streaming=True))
    assert len(out) == 10
    d = _dist(out)
    # Even-ish split; no language starved.
    assert all(d[k] >= 3 for k in ("a", "b", "c")), d


def test_underfull_language_rolls_budget_forward(monkeypatch):
    # Language 'a' has only 2 samples available; with a cap of 10 over 2
    # languages, 'b' must absorb the deficit so the total still reaches 10.
    _install_fake(monkeypatch, {"a": 2, "b": 100})
    out = list(dl.stream_code_data(languages=["a", "b"], max_samples=10, streaming=True))
    d = _dist(out)
    assert d["a"] == 2
    assert d["b"] == 8
    assert len(out) == 10


def test_single_language_unchanged(monkeypatch):
    _install_fake(monkeypatch, {"python": 100})
    out = list(dl.stream_code_data(languages=["python"], max_samples=7, streaming=True))
    assert len(out) == 7
    assert _dist(out) == {"python": 7}


def test_no_cap_yields_everything(monkeypatch):
    _install_fake(monkeypatch, {"a": 30, "b": 40})
    out = list(dl.stream_code_data(languages=["a", "b"], max_samples=None, streaming=True))
    assert len(out) == 70
    assert _dist(out) == {"a": 30, "b": 40}
