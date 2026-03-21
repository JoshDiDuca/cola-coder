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
- Config: `configs/reasoning.yaml`
- CLI flags: `--sft-warmup`, `--reward {python_exec,typescript,combined}`, `--problems {builtin,extended,all,curriculum}`
- Feature toggles: `sft_warmup`, `typescript_rewards`, `expanded_problems`, `parallel_generation` in features.yaml
