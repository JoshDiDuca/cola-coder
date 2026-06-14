"""Tests for the inspect-tools endpoints (tokenizer health + data stats) and
their shared compute libraries. All CPU-only — no model/GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cola_coder.data.stats import compute_data_stats
from cola_coder.tokenizer.health import HealthCheckResult, run_health_checks
from cola_coder.tokenizer.train_tokenizer import SPECIAL_TOKENS
from cola_coder.ui.app import create_app
from cola_coder.ui.jobs import JobManager
from cola_coder.ui.schemas import DataStats


# ── Fake tokenizer: reversible ~4-char chunk codec (exact roundtrip, realistic
#    chars/token) so the full health battery exercises without the heavy dep ───

class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Minimal stand-in exercising the health battery without the heavy dep."""

    def __init__(self, vocab_size: int = 32768, include_optional: bool = True) -> None:
        self._size = vocab_size
        self._vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        if include_optional:
            self._vocab["<think>"] = len(self._vocab)
            self._vocab["</think>"] = len(self._vocab)
        self._chunk_to_id: dict[str, int] = {}
        self._id_to_chunk: dict[int, str] = {}
        self._next_id = 1000

    def _id_for(self, chunk: str) -> int:
        if chunk not in self._chunk_to_id:
            self._chunk_to_id[chunk] = self._next_id
            self._id_to_chunk[self._next_id] = chunk
            self._next_id += 1
        return self._chunk_to_id[chunk]

    def encode(self, text: str) -> _FakeEncoding:
        # ~4 chars/token → realistic avg-length, exact roundtrip via reverse map.
        return _FakeEncoding([self._id_for(text[i:i + 4]) for i in range(0, len(text), 4)])

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_chunk[i] for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_vocab_size(self) -> int:
        return self._size


def test_health_checks_all_pass_on_good_tokenizer() -> None:
    results = run_health_checks(_FakeTokenizer(), expected_vocab=32768)
    assert all(isinstance(r, HealthCheckResult) for r in results)
    assert len(results) == 5
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_health_vocab_mismatch_fails() -> None:
    results = {r.name: r for r in run_health_checks(_FakeTokenizer(vocab_size=100), expected_vocab=32768)}
    assert results["Vocab size"].ok is False


def test_health_missing_special_tokens_fails() -> None:
    tok = _FakeTokenizer()
    tok._vocab.pop(SPECIAL_TOKENS[0])  # drop a required token
    results = {r.name: r for r in run_health_checks(tok)}
    assert results["Special tokens"].ok is False


def test_data_stats_compute(tmp_path: Path) -> None:
    arr = np.random.randint(0, 32768, size=(50, 64), dtype=np.uint16)
    data = tmp_path / "train_data.npy"
    np.save(data, arr)
    np.save(tmp_path / "train_data.weights.npy", np.clip(np.random.rand(50), 0, 1).astype(np.float32))

    stats = compute_data_stats(str(data))
    payload = {**stats.__dict__, "weight_tiers": [t.__dict__ for t in stats.weight_tiers]}
    model = DataStats.model_validate(payload)
    assert model.total_tokens == 50 * 64
    assert model.num_chunks == 50 and model.seq_len == 64
    assert model.has_weights is True
    assert len(model.weight_tiers) == 5
    # Tier counts must sum to the number of sequences.
    assert sum(t.count for t in model.weight_tiers) == 50


def test_data_stats_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        compute_data_stats("/definitely/not/here.npy")


# ── HTTP endpoints ────────────────────────────────────────────────────────────

def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(job_manager=JobManager(), project_root=str(tmp_path)))


def test_data_stats_endpoint_ok_and_validates(tmp_path: Path) -> None:
    arr = np.random.randint(0, 32768, size=(20, 32), dtype=np.uint16)
    data = tmp_path / "train_data.npy"
    np.save(data, arr)
    c = _client(tmp_path)
    r = c.get("/api/data-stats", params={"data_path": str(data)})
    assert r.status_code == 200, r.text
    DataStats.model_validate(r.json())  # forbid-extra: response matches schema


def test_data_stats_endpoint_missing_is_error_union(tmp_path: Path) -> None:
    c = _client(tmp_path)
    r = c.get("/api/data-stats", params={"data_path": str(tmp_path / "nope.npy")})
    assert r.status_code == 200 and "error" in r.json()  # 200 + {error}, not 500


def test_tokenizer_health_endpoint_missing_is_error_union(tmp_path: Path) -> None:
    # No tokenizer under the temp project root → graceful {error}, never 500.
    c = _client(tmp_path)
    r = c.get("/api/tokenizer-health")
    assert r.status_code == 200 and "error" in r.json()
