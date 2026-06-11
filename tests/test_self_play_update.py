"""BUG-108: self-play must fine-tune on the best solution exactly once.

train_episode called _update_on_solution at the >0.9 break AND again in the
post-loop "final update" block (since best_reward > 0.9 > 0.3) — double
fine-tuning on the same passing solution, over-weighting it. The fix guards the
post-loop update with an `updated` flag.
"""

import torch
import torch.nn as nn

from cola_coder.reasoning.self_play import SelfPlayTrainer, SelfPlayConfig


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))


def _trainer(reward_fn):
    cfg = SelfPlayConfig(
        max_iterations=3, num_candidates=2, include_error_context=False,
    )
    return SelfPlayTrainer(
        _TinyModel(), tokenizer=object(), reward_fn=reward_fn, config=cfg, device="cpu",
    )


def _run(monkeypatch, rewards_value):
    calls = []
    t = _trainer(reward_fn=lambda cands, tc: ([rewards_value] * len(cands), [{}] * len(cands)))
    monkeypatch.setattr(
        t, "_generate_candidates",
        lambda prompt, num_candidates, temperature: ["sol"] * num_candidates,
    )
    monkeypatch.setattr(t, "_update_on_solution", lambda prompt, sol: calls.append(sol))
    result = t.train_episode("prompt", "tests", problem_id="p1")
    return calls, result


class TestUpdateExactlyOnce:
    def test_passing_solution_updates_once(self, monkeypatch):
        # reward 0.95 → breaks at >0.9; must NOT also update post-loop.
        calls, _ = _run(monkeypatch, 0.95)
        assert len(calls) == 1

    def test_partial_solution_updates_once(self, monkeypatch):
        # reward 0.5 → never breaks; one post-loop update (>0.3).
        calls, result = _run(monkeypatch, 0.5)
        assert len(calls) == 1
        assert result.iterations_used == 3  # ran all iterations (no early stop)

    def test_no_solution_no_update(self, monkeypatch):
        # reward 0.0 → best_solution stays "" → no update at all.
        calls, _ = _run(monkeypatch, 0.0)
        assert len(calls) == 0
