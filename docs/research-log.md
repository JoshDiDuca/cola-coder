# Research Log — 2025/2026 AI techniques tracked for cola-coder

Living log of external research (frontier-lab techniques, papers, standards) that
the autonomous improvement loop consults and turns into backlog items. **Each loop
cycle should add or update an entry here from a fresh web search**, then file
concrete backlog items referencing it. Newest first.

---

## 2026-06-13 — Post-training & optimizer landscape (initial sweep)

Searched: small-model training recipes (Qwen3/DeepSeek), Muon vs AdamW, on-policy
distillation, RLVR/GRPO/DAPO.

### 1. On-policy distillation (OPD) — now the dominant post-training paradigm
- OPD is "the indispensable post-training paradigm for scaling reasoning," shipped
  in **DeepSeek-V4, Qwen3, Gemma-2, Nemotron, MiMo, GLM-5**.
- At fixed small deployment size, a strong teacher distilled through a **dense OPD
  bridge beats direct GRPO on the student** (e.g. Qwen3-1.7B: 79.3 vs 75.9 MATH,
  25.2 vs 19.8 AIME'24).
- Recipe: GRPO on teacher (capability discovery) → forward-KL warmup + OPD dense
  bridge → optional student-side sparse RL.
- Teacher-access decision: logits available → white-box (GKD, Entropy-Aware OPD);
  no logits → black-box (GAD, OVD) or **self-distillation (OPSD, SDFT)**.
- Relevance: cola-coder does SFT + GRPO but **no distillation** — a strong lever for
  a ~100M model. Black-box / self-distillation fits a no-local-teacher setup. → MODEL-024.
- Sources: Thinking Machines Lab "On-Policy Distillation"
  (https://thinkingmachines.ai/blog/on-policy-distillation/); "A Survey of On-Policy
  Distillation for LLMs" (https://arxiv.org/html/2604.00626v3); "Rethinking On-Policy
  Distillation: Phenomenology, Mechanism, and Recipe" (https://arxiv.org/html/2604.13016v1).

### 2. Muon optimizer — ~2× compute efficiency vs AdamW, scalable
- Muon expands the Pareto frontier over AdamW on the compute–time tradeoff; ~2×
  compute efficiency under compute-optimal training, ~33% memory savings, retains
  data efficiency past the critical batch size; validated to multi-billion scale.
- Relevance: **cola-coder already implements Muon** (`training/optimizer.py`,
  audited correct) but the live run uses AdamW. Strong case to validate + default to
  Muon for the project's "best quality on limited compute" goal. → MODEL-025.
- Sources: "Muon is Scalable for LLM Training" (Moonshot/Moonlight,
  https://arxiv.org/pdf/2502.16982); "Practical Efficiency of Muon for Pretraining"
  (https://arxiv.org/html/2505.02222v1); Nubank "Muon for Improved Foundation Model
  Pretraining Data Efficiency".

### 3. RLVR / GRPO / DAPO — cola-coder is already mostly aligned
- 2026 stack: SFT (instruction) → preference optimization → RLVR (GRPO/DAPO) for
  reasoning, reward from verifiers (unit tests for code — exactly cola-coder's
  test-based rewards).
- DAPO = decoupled clip-higher + **dynamic sampling** + drop KL + token-level loss.
  cola-coder's reasoning.yaml ALREADY has clip-higher (0.28), Dr.GRPO mean-norm, and
  drops KL — the one remaining DAPO ingredient is **dynamic sampling** (drop prompt
  groups whose G samples are all-correct/all-incorrect → zero advantage → wasted
  compute). → MODEL-026.
- Sources: "Post-Training in 2026: GRPO, DAPO, RLVR & Beyond"
  (https://llm-stats.com/blog/research/post-training-techniques-2026); "RLVR
  Implicitly Incentivizes Correct Reasoning in Base LLMs"
  (https://arxiv.org/html/2506.14245v2).

### 4. Qwen3 multi-stage data recipe — reasoning-stage curriculum
- Qwen3: 30T tokens @4k → reasoning stage 5T **higher-quality** STEM/coding tokens →
  long-context @32k. Small models still trained on huge data (Qwen3-0.6B on 36T).
- Relevance: validates cola-coder's staged pipeline; the "higher-quality reasoning
  stage" maps to a data curriculum + model-based quality scoring. → DATA-056 (with DATA-055).
- Sources: ICLR'26 frontier-training notes; HuggingFace "Best Open-Source LLMs 2026".
