# Research Log — 2025/2026 AI techniques tracked for cola-coder

Living log of external research (frontier-lab techniques, papers, standards) that
the autonomous improvement loop consults and turns into backlog items. **Each loop
cycle should add or update an entry here from a fresh web search**, then file
concrete backlog items referencing it. Newest first.

---

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
