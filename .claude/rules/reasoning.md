---
match: "**/reasoning/**,scripts/train_reasoning.py,configs/reasoning.yaml"
---

# Reasoning Module Rules

- Pipeline: SFT warmup (optional) → GRPO fine-tuning with test-based rewards
- Thinking tokens: `<think>` / `</think>` — vocabulary expansion via embedding resize
- SFT warmup: supervised fine-tuning on curated CoT examples before RL (DeepSeek-R1 approach)
- GRPO: generate G solutions per problem, run tests, reinforce correct ones (PPO-clipped objective)
- Reward registry: pluggable — `python_exec`, `typescript` (tsc --noEmit --strict), `combined` (multi-signal)
- Parallel generation: batched same-prompt forward pass with KV-cache expansion
- Curriculum learning: easy → medium → hard with per-difficulty temperature scaling
- Problem set: 62 built-in + JSONL custom problems
- Config: `configs/reasoning.yaml` — EVERY key in its `reasoning`/`problem_set`/`sft_warmup`
  sections must be read by train_reasoning.py (CLI flag > config > default);
  enforced by tests/test_reasoning_config_wiring.py. No phantom knobs: kl_coeff
  was removed because no KL term exists (DAPO/Dr. GRPO drop it deliberately).
- GRPO defaults in reasoning.yaml: advantage_norm "mean" (Dr. GRPO),
  clip_epsilon 0.2 / clip_epsilon_high 0.28 (DAPO clip-higher),
  parallel_generation + parallel_rewards enabled
- CLI flags: `--sft-warmup`, `--reward {python_exec,typescript,combined}`, `--problems {builtin,extended,all,curriculum}`
- Reasoning behavior is controlled by configs/reasoning.yaml + CLI flags, NOT features.yaml
  (the old sft_warmup/typescript_rewards/expanded_problems/parallel_generation feature keys
  were phantoms — no module backed them — and were removed 2026-06-10)
