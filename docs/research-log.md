# Research Log — 2025/2026 AI techniques tracked for cola-coder

Living log of external research (frontier-lab techniques, papers, standards) that
the autonomous improvement loop consults and turns into backlog items. **Each loop
cycle should add or update an entry here from a fresh web search**, then file
concrete backlog items referencing it. Newest first.

---

## 2026-06-13 — Post-training RL: DAPO dynamic sampling (rotate: post-training)

- **DAPO = clip-higher + dynamic sampling + token-level loss + overlong reward
  shaping.** cola-coder's reasoning.yaml already has clip-higher (0.28), Dr.GRPO
  mean-norm, token-level/length-norm options, and a zero-variance COLLAPSE GUARD
  (skips groups where all rewards are equal — BUG-121). The remaining DAPO piece is
  **dynamic sampling's RESAMPLING**: when a group collapses (all-correct/all-incorrect
  → zero gradient), don't just skip — oversample/redraw prompts until the batch is
  full of informative (std>0) groups, so no step is wasted. → MODEL-026 (still open;
  needs a resample_fn + rollout refactor).
- This cycle I implemented the related IDEA-008 GRPO half instead (smaller, completes
  a partial item): a security penalty in the reward so RL avoids insecure code.
- Sources: DAPO site (https://dapo-sia.github.io/); NVIDIA NeMo-RL DAPO walkthrough
  (https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html); "DAPO: GRPO on steroids"
  (https://medium.com/@syed_hasan/dapo-decoupled-clip-and-dynamic-sampling-policy-optimization-grpo-on-steroids-9c571a0536f3);
  TRL dynamic-sampling issue (https://github.com/huggingface/trl/issues/4764).
- ORIGINAL idea → **MODEL-032**: difficulty-aware dynamic sampling using cola-coder's
  best-of-N verifier as a *cheap pre-filter* — before a full GRPO rollout, run a
  small best-of-N probe per prompt; skip prompts that are trivially all-pass or
  all-fail (predicted zero-variance) and spend the rollout budget on
  medium-difficulty prompts (predicted std>0). Combines best-of-N + GRPO + curriculum
  to get DAPO's dynamic-sampling benefit without wasting full rollouts on collapsed groups.

## 2026-06-13 — Architecture (rotate: architecture)

- **QK-Norm is now standard** (Qwen3, Gemma3, DeepSeek-V3, Gemini 2.0): per-head
  RMSNorm of Q/K before RoPE — reduces perplexity, improves convergence, and prevents
  attention-logit explosion at depth > ~12 layers. **Validates cola-coder's existing
  `qk_norm=true`** (small_react_best = 12 layers, 4080_max) → reinforces MODEL-023
  (enable it on tiny/small too for CI coverage of the path).
- **muP / muTransfer**: maximal-update parametrization lets you tune hyperparameters
  (LR, init) on a SMALL proxy model and transfer them zero-shot to larger models —
  directly relevant to cola-coder's size ladder (tiny→small→medium→4080_max). → MODEL-031.
- Other 2026 arch advances noted (for later, non-urgent): nGPT (everything on the
  unit hypersphere), hybrid attention (sliding-window + full), DAPE V2 position
  encoding, MoE routing refinements (the project already has MoE upcycling).
- Sources: QK-Norm overview (https://www.emergentmind.com/topics/query-key-normalization-qk-norm);
  "Methods of improving LLM training stability" (https://arxiv.org/pdf/2410.16682);
  "Transformer Architecture in 2026" (https://dev.to/jintukumardas/transformer-architecture-in-2026-from-attention-to-mixture-of-experts-moe-3d46);
  Stanford "Rethinking the Primitives" (https://web.stanford.edu/~jksun/blog/llm-architecture.html).
- ORIGINAL idea → already covered: the project's qk_norm+z_loss combo is exactly the
  2026 stability stack; the actionable new item is muP hyperparameter transfer (MODEL-031)
  to tune cheaply on tiny and transfer up the size ladder.

## 2026-06-13 — Safety / guardrails (rotate: safety)

- **Functional ≠ secure:** 2026 studies find secure-pass@1 stays <12% even when
  functional pass@1 >50% — generated code that compiles / passes tests is often
  INSECURE. So functional verification (tsc/tests) is NOT enough; add an OUTPUT
  guardrail that screens generated code for dangerous patterns and prefers/keeps the
  secure candidate.
- Guardrail layers (LLM-Guard, LlamaFirewall, Azure Content Safety): input
  (prompt-injection/jailbreak detection — fine-tuned BERT) + output (vulnerability/
  dangerous-pattern scan, retry on fail). Anthropic: proactive PRETRAINING DATA
  FILTERING (fix at the source) + Constitutional AI + lightweight pre-screening model.
- Relevance: cola-coder already has malware scanners (yara/clamav/defender), a
  CredentialScanner, and safety_eval's dangerous-pattern probes — but the latter was
  Python-only and not reused. This cycle extracted a canonical TS/JS-extended
  dangerous-code scanner and wired it as a security screen on DISTILLED data (don't
  distill insecure code — the Anthropic "filter at source" philosophy). → SEC-017.
- Sources: Confident AI LLM Guardrails guide
  (https://www.confident-ai.com/blog/llm-guardrails-the-ultimate-guide-to-safeguard-llm-systems);
  LlamaFirewall (https://arxiv.org/pdf/2505.03574); "How Secure is Secure Code
  Generation?" (https://arxiv.org/pdf/2601.07084); OWASP LLM Prompt Injection Cheat
  Sheet (https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).
- ORIGINAL idea → **IDEA-008**: security-aware best-of-N + GRPO reward. cola-coder's
  best-of-N picks the functionally-best candidate (tsc/tests); add the dangerous-code
  screen as a SECONDARY signal so among functionally-correct candidates it prefers the
  SECURE one ("secure-pass best-of-N"), and add a small security penalty to the GRPO
  reward (reward = test/tsc pass − λ·dangerous) so the RL policy learns to avoid
  insecure patterns. Reuses the new scanner + the existing verifier/reward.

## 2026-06-13 — Evaluation (rotate: evaluation)

- **HumanEval is saturated + contaminated** (frontier models ~93%, contamination
  well-documented) — now just a ~90% qualification bar, not a selection signal.
- 2026 real signals: **LiveCodeBench** (continuous, CONTAMINATION-FREE — sources
  problems that POSTDATE training, so gains reflect real capability not memorization)
  and **SWE-bench Verified** (300 curated real GitHub-issue tasks — agentic). Plus
  HumanEval+ (more tests). Guidance: use only 2-3 benchmarks (avoid benchmark
  overfitting) + build a **proprietary 100-500 example test set from production data**.
- Relevance: refines EVAL-023. For cola-coder the contamination-free angle is the key
  one — the model trains on scraped code, so a held-out eval must postdate/limit-out
  the training set. The project's own GitHub scraper can build this. → IDEA-007.
- Sources: LiveCodeBench (https://arxiv.org/pdf/2403.07974); SWE-Compass
  (https://arxiv.org/pdf/2511.05459); DevBench (https://arxiv.org/pdf/2601.11895);
  TokenMix "LLM Leaderboard 2026" (https://tokenmix.ai/blog/llm-leaderboard-2026);
  lxt.ai LLM Benchmarks 2026 (https://www.lxt.ai/blog/llm-benchmarks/).
- ORIGINAL idea → **IDEA-007**: self-refreshing CONTAMINATION-FREE eval. Use the
  project's GitHub scraper to pull RECENT, license-clean TS/React files that postdate
  the training cutoff, hold out the middle of each as a FIM task, and score the model's
  infill with the SANDBOXED tsc/test verifier (LiveCodeBench philosophy, built from the
  project's own scraper + FIM + sandbox). Refreshable each eval → gains can't be
  memorization. Combines scraper + FIM + sandbox-verifier assets.

## 2026-06-13 — Data curation (rotate: data quality)

- **Quality > quantity** is the dominant lesson (Phi-1 on 7B curated tokens beat
  models trained on 100× more; FineWeb's value is its filtering pipeline). Standard
  pipeline: collect → heuristic filter → dedup → model-based quality filter → mixture
  + curriculum. cola-coder already does most of this (filters, MinHash/exact dedup,
  scorers, quality weights).
- **SoftDedup (2024+)**: instead of HARD-deleting near-duplicates, REWEIGHT recurring
  data down — preserves information while reducing over-representation. This maps
  exactly onto cola-coder's existing `.weights.npy` per-sample weights + dedup.py
  MinHash. → original idea **DATA-057**.
- **Synthetic data = Generate→Critique→Filter** (3-stage). Frontier labs rewrite
  math/code at scale (2.3-16B tokens) as synthetic pretraining data. This is exactly
  the shape of the distillation harness shipped this cycle (MODEL-028): teacher
  generates → sandbox verifies → keep verified. Validates the design.
- Relevance: reinforces DATA-055 (model-based perplexity scoring) and IDEA-003
  (online perplexity curriculum); adds DATA-057 (soft-dedup reweighting).
- Sources: NVIDIA "Mastering LLM Techniques: Data Preprocessing"
  (https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/);
  YuLan-Mini (https://arxiv.org/pdf/2412.17743); "Recycling the Web"
  (https://arxiv.org/pdf/2506.04689); "Every Sample Matters: MoE + High-Quality Data
  for Code LLMs" (https://arxiv.org/pdf/2503.17793).
- ORIGINAL idea → **DATA-057**: soft-dedup reweighting — when dedup.py finds a
  near-duplicate (MinHash), instead of dropping it, DOWN-weight it in `.weights.npy`
  (weight ∝ 1/cluster_size). Preserves rare info in the cluster while cutting
  over-representation; reuses the project's two existing mechanisms (MinHash + quality
  weights) with no new infra. Combine with the loss's per-sample weighting.

## 2026-06-13 — Inference: speculative decoding (rotate: inference/decoding)

- Speculative decoding (draft-and-verify) is now THE default inference-acceleration
  layer of every serious 2026 LLM stack; 2-4× speedup at >80% draft acceptance, and
  it's **mathematically lossless** (accepted tokens follow the exact target
  distribution) so it does NOT change eval results — safe to add to serve/REPL/FIM.
- SOTA: **EAGLE-3** (a small auto-regressive draft head conditioning on the target's
  EARLY+MIDDLE+LATE hidden states — tri-layer fusion, +20-40% over EAGLE-2, reuses
  the target's own representations so low memory) and **Medusa** (multiple parallel
  decoding heads on a frozen backbone — simplest to bolt on, no separate draft model).
  DeepSeek MTP (multi-token prediction) is the same family.
- Relevance: refines MODEL-022. For a single ~100M model with a spare GPU, a
  **Medusa-style head** (no separate draft model, GPU-light) is the most tractable;
  EAGLE-3 is higher-acceptance but more involved. Lossless → no quality risk.
- ORIGINAL idea → **IDEA-006**: train Medusa/EAGLE draft heads on the project's OWN
  FIM data so the speculative *drafts* are FIM-aware (prefix+suffix conditioned),
  and combine with IDEA-004 (the known suffix as an extra accept/verify signal) —
  fast, suffix-consistent ghost-text. Self-speculative (heads on the same model),
  so no second model and minimal extra VRAM.
- Sources: PremAI "Speculative Decoding 2026"
  (https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/);
  EAGLE (https://arxiv.org/pdf/2401.15077); SyncSoft "EAGLE3/Medusa/DeepSeek-MTP 2026"
  (https://www.syncsoft.ai/en/blog/speculative-decoding-eagle3-medusa-deepseek-mtp-chinese-chuhai-2026).

## 2026-06-13 — Untrusted-code sandboxing best practices (SEC-012/013 epic)

Searched frontier-lab / platform practices for executing untrusted (scraped +
LLM-generated) code.

### Hard reality: plain Docker on Windows is NOT "bulletproof"
- 2026 consensus: a shared-kernel container is NOT sufficient isolation for
  untrusted code — one kernel CVE = host escape. The production tier is
  **kernel isolation**: gVisor (user-space kernel; Modal/Daytona) or Firecracker/
  Kata microVMs (hardware VM; E2B). Isolation ladder: V8 isolate < Docker <
  gVisor < Firecracker/Kata.
- **Project-critical caveat:** gVisor / Firecracker / Kata are **NOT available on
  Windows Docker Desktop / WSL2** (gVisor needs bare metal or nested-virt Linux).
  Docker's stronger-isolation option there (Enhanced Container Isolation / Sysbox)
  is **Business-license only**. And WSL2's own VM boundary has had documented
  escapes (Trend Micro WSL2 Docker-Desktop VM-escape; Aug-2025 Docker Desktop
  container-escape bug). So "completely bulletproof on this Windows host" is not
  achievable with Docker alone — be honest about it.
- **Verdict for cola-coder:** (1) maximally harden Docker (SEC-012) and treat it as
  "raises the bar, not bulletproof"; reduce surface via offline + `--network=none`
  + ephemeral; (2) for true kernel isolation, move untrusted execution to a
  disposable Linux VM or a cloud sandbox (E2B/Modal/Daytona) behind the same
  sandbox interface → **SEC-015**.

### Highest-impact hardening to ADD (beyond non-root/read-only/no-net/pids+mem)
1. Resource-exhaustion layer (most likely real failure for scraped/teacher code):
   container-level wall-clock timeout → SIGKILL the whole cgroup; **bounded/truncated
   stdout/stderr** (output-bomb); `ulimit fsize`; tmpfs size cap; `--memory-swap ==
   --memory` + `--memory-swappiness=0`.
2. **Custom deny-by-default seccomp** profile (Docker's default is a permissive
   allow-list) + keep `--cap-drop=ALL` + `--security-opt=no-new-privileges`; custom
   AppArmor if the WSL2 kernel supports it. → fold into SEC-012.
3. Disposable-VM / cloud-sandbox backend for real kernel isolation. → SEC-015.
4. Default-deny egress everywhere + verify ephemeral teardown (no state reuse).
5. Keep host kernel + runc patched (named escape CVEs: runc CVE-2024-21626, kernel
   CVE-2024-1086, Nov-2025 runc CVEs → runc ≥1.2.8/1.3.3/1.4.0-rc.3).
- Sources: Northflank (untrusted-code platforms; Firecracker-vs-gVisor; E2B/Modal/
  Daytona); Blaxel container-escape 2026; gVisor.dev (no-WSL2); Docker ECI docs;
  Trend Micro WSL2 escape; The Register Aug-2025 Docker Desktop bug; HackTricks
  container security.

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

### Original ideas / hypotheses (cola-coder-specific cross-technique combinations)
cola-coder has a rare asset combo most stacks lack: **dynamic FIM + sandbox test
runner + TscRunner (tsc --strict) + best-of-N verification + per-sample quality
weights + a Muon implementation**. These let us combine 2026 techniques in ways the
generic recipes don't. All are HYPOTHESES to validate (worktree, off the live run),
not committed approaches. → IDEA-001..005.

- **IDEA-001 — FIM-RLVR (infill GRPO with execution/tsc reward).** Most RLVR does
  full-program generation. Instead run GRPO where each prompt is a FIM task (real
  code with the middle held out) and the verifiable reward is "does the infill make
  the file type-check (tsc --strict) / pass its tests." This optimizes the *actual
  deployment objective* (IDE ghost-text infill at temp~0.2), not full generation.
  Reuses TscRunner + sandbox + GRPO + the FIM tokens. Prior art does FIM+RL
  (DeepSeek-Coder, aiXcoder, IFIM@ICSE'26) but verifiable-reward FIM-GRPO is sharp
  and directly fits this project. Hypothesis: beats full-gen GRPO for completion quality.
- **IDEA-002 — Verified self-distillation bridge (VSD).** The 2026 OPD recipe wants a
  "forward-KL warmup / dense bridge" before student RL, but cola-coder has no local
  teacher. Use its EXISTING best-of-N + sandbox verifier to harvest the model's OWN
  verified-correct completions, SFT on them (rejection-sampling RFT) as that bridge,
  THEN GRPO. Research caveat (arXiv 2505.14216): self-distillation alone overfits and
  underperforms RLVR — so use VSD strictly as a warmup INTO GRPO, plus the hybrid
  finding (self-distill for update magnitude, RLVR for direction; arXiv 2601.18734).
  No external teacher needed — a teacher-free OPD bridge from the project's own verifier.
- **IDEA-003 — Online perplexity curriculum (live-loss reweighting).** Combine
  MODEL-020 (online reweighting) + DATA-055 (model-based scoring): skip the separate
  scoring pass and use the RUNNING model's per-chunk loss as a real-time
  difficulty/quality signal — down-weight already-mastered low-loss chunks, focus
  compute on high-loss-but-learnable ones (and flag near-zero-loss as likely dup/noise).
  Closes the eval→data loop *during* training at ~zero extra cost.
- **IDEA-004 — Suffix-guided speculative decoding for FIM.** For ghost-text the suffix
  is KNOWN. Generic speculative decoding ignores this. Use the known suffix to bias/
  verify draft tokens toward connecting cleanly to it — making spec-decoding both
  faster AND more suffix-consistent for infill. Combines MODEL-022 + FIM; a
  completion-specific decoding trick.
- **IDEA-005 — Quality-weight × Muon interaction (theory).** Muon orthogonalizes the
  gradient (Newton-Schulz); per-sample quality weights scale the loss. Open question:
  does loss-level quality-weighting survive orthogonalization, or does Muon wash out
  the per-sample magnitude? If the latter, weight at the data-SAMPLING level instead.
  A small AdamW-vs-Muon × weighted-vs-unweighted ablation answers it; matters because
  the project relies on .weights.npy AND wants to adopt Muon (MODEL-025).
