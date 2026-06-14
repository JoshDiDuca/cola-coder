# Research Log — 2025/2026 AI techniques tracked for cola-coder

Living log of external research (frontier-lab techniques, papers, standards) that
the autonomous improvement loop consults and turns into backlog items. **Each loop
cycle should add or update an entry here from a fresh web search**, then file
concrete backlog items referencing it. Newest first.

---

## 2026-06-14 — Beyond functional correctness: secure-pass@k (rotate: evaluation)

Sources:
- Rethinking the Evaluation of Secure Code Generation — https://arxiv.org/html/2503.15554v2
- Beyond Correctness: multi-dimensional codegen benchmarking (RACE) — https://arxiv.org/abs/2407.11470
- CodeAlignBench (developer-preferred adjustments) — https://arxiv.org/pdf/2510.27565
- Evaluating LLM-generated code: benchmark + developer study — https://arxiv.org/html/2605.09059v1

Findings:
- **Functional correctness over-credits insecure code.** The 2026 secure-codegen
  evaluation line (CWEval / CodeGuard+, and the "Rethinking..." paper) argues pass@k
  alone rewards working-but-vulnerable solutions. The fix: an EXPANDED pass@k that counts
  a sample only when it is correct AND secure — "secure-pass@k". The gap
  pass@k − secure-pass@k quantifies exactly how much of the model's "success" is insecure.
- **Multi-dimensional eval is the standard (RACE):** correctness + readability +
  maintainability + efficiency (ENAMEL's eff@k normalizes runtime vs human reference).
  Single-number pass@k is no longer enough.
- **Contamination control:** prefer tasks published after the training cutoff and
  evolution-aware historical context; relevant when we add fresh benchmarks later.

**Implemented this cycle (EVAL-024):** secure-pass@k. `metrics.ProblemResult` gains
`num_secure_correct` (passed AND clean; None = unassessed, back-compat) with a
`__post_init__` guard (secure-correct ⊆ correct) + `secure_pass_rate` property.
`compute_secure_pass_at_k` reuses the unbiased `pass_at_k` estimator (DRY) so it's directly
comparable to pass@k, mirroring its None/exclusion-warning semantics; `format_results`
prints it whenever any problem is assessed. `scripts/evaluate.py` populates it by running
the shared `scan_dangerous` scanner on each PASSING sample. Eval-only → committed on main.
+6 tests (32 inference-metric tests + ruff green).

**ORIGINAL cross-technique idea (IDEA-015): security-gap-driven data synthesis.** The
per-problem `pass@k − secure-pass@k` gap is a precise weakness map: problems the model
solves *but insecurely* are exactly where secure-coding training data is missing. Mine the
high-gap problems, take their insecure-but-passing completions, and synthesize paired
(insecure → secure) FIM/instruction examples (the dangerous span becomes the FIM middle to
infill securely); up-weight them via quality weights and feed the GRPO security penalty
(IDEA-008). Closes the eval→data loop using cola-coder's rare combo — scanner + sandbox
verifier + dynamic FIM + quality weights — turning an eval metric into targeted data. → IDEA-015.

---

## 2026-06-14 — Test-time scaling: verifiers + self-consistency (rotate: inference/decoding)

Sources:
- Multi-Agent Verification: scaling test-time compute with multiple verifiers — https://arxiv.org/pdf/2502.20379
- DynScaling: efficient verifier-free inference scaling via dynamic sampling — https://arxiv.org/pdf/2506.16043
- Seer Self-Consistency: advance budget estimation for adaptive TTS — researchgate 397556617
- Speculative decoding benchmark for efficient TTS — https://arxiv.org/abs/2509.04474

Findings:
- **Best-of-N scales two ways:** more *samplers* (diverse candidates) AND more/diverse
  *verifiers* (diverse verification signal). cola-coder already has the sampler axis
  (generate_group) + one hard verifier (tsc/tests); the cheap win is adding a SECOND,
  free verification signal that needs no extra model.
- **Self-consistency = the free second signal.** Sampling N reasoning paths and taking the
  most frequent answer beats single-sample; for code the standard cheap proxy is
  AlphaCode-style **clustering by (normalized) program** and voting by cluster size —
  correct solutions converge, wrong ones scatter. This composes with a verifier: vote
  *within* the verified set.
- **Efficiency is the 2026 theme:** redundant traces waste compute; adaptive-budget and
  advance budget-estimation (Seer) allocate samples only where needed. cola-coder's
  adaptive best-of-N (IDEA-009) already does the budget side.

**Implemented this cycle (INFER-026):** self-consistency tiebreaker in `best_of_n._rank`.
Ranking key went `(verified, secure, score)` → `(verified, secure, consistency, score)`,
where `consistency` = size of the candidate's normalized-completion cluster
(`_normalize_completion` collapses whitespace/blank lines, AlphaCode-style). So within the
verified+secure tier the model's most-repeated solution wins, with score as final tiebreak.
Backward-compatible (all-unique → all clusters size 1 → prior order). Pure inference → main.
+6 tests; 31 best-of-N tests + ruff green.

**ORIGINAL cross-technique idea (IDEA-014): cluster-gated verifier budget.** In adaptive
best-of-N, the sandbox verifier is the expensive step (a tsc/test run per candidate). Once
candidates are clustered for self-consistency, verify only ONE representative per cluster,
then propagate its verdict to the whole cluster — N candidates cost ~(#distinct clusters)
sandbox runs instead of N. Combine with quality-weights: when budget-limited, verify the
largest clusters first (most likely to be the consensus answer). Exploits cola-coder's rare
combo — clustering + sandbox verifier + adaptive budget + quality weights — to cut the
dominant test-time cost without losing the consensus signal. Builds on INFER-026 + IDEA-009.

---

## 2026-06-14 — RLVR entropy dynamics (rotate: post-training / RLVR)

Sources:
- Clip-Low raises / Clip-High lowers entropy in RL of LLMs — https://arxiv.org/pdf/2509.26114
- Hidden Costs & Measurement Gaps of RLVR — https://arxiv.org/html/2509.21882
- Rethinking Entropy Interventions in RLVR (entropy-change view) — https://arxiv.org/abs/2510.10150
- RLVR implicitly incentivizes correct reasoning in base LLMs — https://arxiv.org/abs/2506.14245

Findings:
- **Entropy collapse is THE dominant RLVR failure mode.** Under verifiable-reward RL
  the policy quickly converges to a near-deterministic argmax, killing exploration and
  stalling further gains. Most prior fixes (KL term, temperature tweaks) are heuristic.
- **The clip asymmetry is a mechanistic entropy knob (2509.26114, Thm 1-2):** the PPO/GRPO
  *lower* clip (on negative-advantage tokens) restrains probability *decrease* → preserves
  mass → **raises entropy**; the *upper* clip (on positive-advantage tokens) limits
  probability *increase* → **lowers entropy**. So `clip_low` and `clip_high` aren't just a
  trust region — they directly steer the entropy trajectory. The project already supports
  asymmetric clips (DAPO clip-higher 0.2/0.28); what it lacked was the *observable* to
  drive them.
- **Measurement gap (2509.21882):** RLVR pipelines routinely DON'T log policy entropy, so
  collapse is invisible until pass-rate flatlines. Cheap to fix — entropy is one extra
  reduction over the log-softmax you already compute.

**Implemented this cycle (MODEL-037):** `completion_entropy(log_probs_2d, prompt_len)` —
mean per-token Shannon entropy (nats) over completion positions only (prompt masked,
mirroring `_completion_logprobs`). Wired into `GRPOTrainer.train_step` (measured once at
PPO epoch 0, where weights == pi_old) → returned as `policy_entropy`, aggregated per epoch,
and printed alongside loss/reward/pass_rate. +6 tests. Off the live *pretraining* path
(GRPO is reasoning-only) → committed on main.

**ORIGINAL cross-technique idea (IDEA-013): entropy-gated closed-loop clip controller.**
Now that `policy_entropy` is observable every step, make the clip asymmetry self-tuning:
hold a target entropy floor; when measured entropy drops below it, nudge `clip_low` UP
(raises entropy per 2509.26114) and/or `clip_high` DOWN; when it's healthy, relax back
toward the DAPO defaults. A PI controller on the entropy error turns the static
0.2/0.28 into a self-stabilizing RLVR loop — and because cola-coder rewards are
*verifier-grounded* (sandbox tests/tsc), the controller can be pass-rate-aware: only
inject exploration (raise entropy) when entropy is low AND pass-rate hasn't saturated,
avoiding wasted exploration once the verifier is already satisfied. → IDEA-013.

---

## 2026-06-14 — Data efficacy & folding curriculum (rotate: data curation)

Sources:
- Data Efficacy for LM Training (DELT paradigm) — https://arxiv.org/html/2506.21545v1
- Rethinking proxy-model practice for data curation — https://arxiv.org/html/2512.24503
- MobileLLM-R1 (capability-aware curation, 11.7% of competitors' tokens) — https://arxiv.org/html/2509.24945
- Multilingual LM-based pretraining data filtering — https://arxiv.org/html/2505.22232v1

Findings:
- **Data EFFICACY ≠ data efficiency.** Efficiency picks the best *subset*; efficacy
  maximizes performance by optimizing the *organization* (ordering) of the SAME data,
  no content or architecture change. DELT = Score → (optional) Select → **Order**.
- **Folding Ordering (FO).** Pure easy→hard curriculum sorts once and suffers
  forgetting (early easy data forgotten by the end), distribution bias, and an effective
  duplication problem. FO repeats the sorted sweep **L times at fixed intervals** —
  the model revisits the full easy→hard progression L times. Measured: +1.7% abs on an
  8-benchmark avg at 160M/1B-tok; **+2.76% HumanEval (code)**, +2.53% math; gains hold
  across 160M/470M/1B and 10B/50B tokens, and unlike random shuffle, *improve* across
  epochs. This is a pure-prep, zero-train-cost reordering — ideal while a run is live.
- **Learnability-Quality Scoring (LQS).** DELT's scorer = learnability (how much a sample
  cuts loss over steps) × quality (gradient cosine-sim to the target direction). Needs
  gradient tracking + a proxy set — heavier, deferred (see DATA-059).
- **Proxy-model caution (2512.24503):** small-proxy-model data-selection rankings do NOT
  always transfer to the target scale — validate any proxy-scored curation at target size
  before trusting it. Tempering for our quality-classifier-driven curation.

**Implemented this cycle:** Folding Ordering as a new `CurriculumStrategy.FOLDING`
(`--curriculum folding --curriculum-folds L`, default L=4) reusing the existing
`CurriculumOrderer` + `.weights.npy` quality scores. Round-robin stride of the
score-sorted index into L folds → a true permutation (no drops/dups), each fold a
strided full-range easy→hard sweep. Prep-time only, off the live train path. → DATA-058.

**ORIGINAL cross-technique idea (DATA-060): verifier-grounded learnability folding.**
DELT's LQS needs gradient tracking we can't afford mid-run, BUT cola-coder already owns a
*verifier* (sandbox test/tsc rewards + best-of-N) and *quality weights*. Idea: derive a
cheap per-bucket "learnability" signal by sampling the base checkpoint's best-of-N
pass-rate on held-out FIM probes drawn from each quality bucket — buckets the model
*almost* solves (mid pass-rate) are the high-learnability frontier. Fold the curriculum so
those frontier buckets recur most often (weight fold frequency by 4·p·(1−p), peaked at
p=0.5), turning the verifier into a no-gradient LQS proxy. Closes the eval→data loop using
assets DELT papers don't have.

---

## 2026-06-13 — Mixture-of-Experts (rotate: architecture/MoE)

- **DeepSeek-V3 aux-loss-FREE load balancing**: instead of an auxiliary load-balancing
  loss (which fights task performance), use per-expert BIAS terms updated from running
  load stats — expert SELECTION uses biased scores, gating WEIGHTS use the original
  scores (preserves specialization). The 2026 standard; the project's MoE uses an
  aux-loss → MODEL-035.
- **Drop-Upcycling**: upcycle a dense model into MoE with PARTIAL re-initialization of
  expert weights (not plain copy) → better expert specialization + faster knowledge
  acquisition. Upgrade for the project's stage-7 upcycle. Also **softmax-then-topK**
  routing beats topK-then-softmax, and higher granularity (more, smaller experts)
  helps. → MODEL-036.
- Shared experts (always-on, capture common knowledge) are now OPTIONAL — DeepSeek/
  Llama-4 use them; Qwen3/OLMoE/GPT-oss skip them. The project's MoE has shared experts.
- Sources: Upcycling LLMs into MoE (https://arxiv.org/abs/2410.07524); Drop-Upcycling
  (https://arxiv.org/pdf/2502.19261); DeepSeekMoE (https://arxiv.org/html/2401.06066v1);
  DeepSeek-V3 load balancing (https://medium.com/yugen-ai-technology-blog/deepseek-v3-advances-in-moe-load-balancing-and-multi-token-prediction-training-f6d68c59749c);
  MoE survey (https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs).
- ORIGINAL idea → **IDEA-012**: domain-aligned expert init. The project trains a
  SEMANTIC ROUTER (stage 8) over its domains (React/Next/GraphQL/Prisma/Zod/Testing)
  and has MoE upcycling (stage 7). Warm-start / bias the upcycled experts toward those
  router domains (initialize expert k from data routed to domain k, or add a
  domain-prior to the router gate), so experts specialize along the project's actual
  specialist vision instead of emerging arbitrarily. Combines MoE upcycle + the domain
  router + Drop-Upcycling partial re-init.

## 2026-06-13 — Agentic coding / tool-use (rotate: agents)

- Agentic coding = a multi-step loop (plan → act: write code / run tests / read output
  → adapt → repeat to a stop condition). On SWE-bench, agentic systems with EXECUTION
  FEEDBACK beat zero-shot 3-5×; ~60% of open-source agents use the "Agent Loop" pattern.
- "The harness matters more than the model" — same model swings 30-50 pts on
  Terminal-Bench 2.0 depending on harness/toolset/retry policy. Key model capabilities:
  reliable tool-use (don't hallucinate tool output), long-context coherence, and
  adapting (not repeating) a failed subtask.
- Relevance: cola-coder has tools/executor + an agent loop + the sandbox. The
  execution-feedback loop is the highest-leverage agentic pattern and the project's
  sandbox/tsc/test verifier is exactly the feedback signal. → IDEA-011.
- Sources: "Agentic Systems Without the Hype"
  (https://medium.com/codetodeploy/agentic-systems-without-the-hype-when-multi-step-llm-workflows-actually-improve-software-e1492ebdfacf);
  awesome-harness-engineering (https://github.com/ai-boost/awesome-harness-engineering);
  awesome-ai-agent-papers 2026 (https://github.com/VoltAgent/awesome-ai-agent-papers).
- ORIGINAL idea → **IDEA-011**: self-repair execution loop for code generation — on a
  failed tsc/test verification, feed the sandbox's ERROR OUTPUT back to the model as
  context and regenerate (bounded retries), instead of just sampling more blind
  candidates (best-of-N). Combines the agent loop + sandbox verifier + FIM; turns the
  one-shot generator into a 2026-style execution-feedback agent. Also a GRPO signal:
  reward solving-after-feedback.

## 2026-06-13 — Optimizers: Muon practical details (rotate: optimizers)

- **muP scaling works for Muon too** — the same maximal-update parametrization that
  transfers AdamW hyperparameters small→large works for Muon. So MODEL-025 (adopt
  Muon) + MODEL-031 (muP) compose: tune LR + weight-decay on the TINY config and
  transfer up the size ladder. Muon needs tuning over essentially just max-LR +
  weight-decay.
- Muon has the **lightest memory footprint** of common optimizers (only the first
  moment — lighter than AdamW) — a real win on the 16 GB 4080.
- **MuonClip** (Kimi-2, 1T params): clips the Muon/attention update (qk-clip) for
  smoother loss curves + stable large-scale dynamics. The 2026 stability refinement
  on top of plain Muon. → MODEL-034.
- Hybrid convention (confirmed): Muon for 2-D matmul weights, AdamW for embeddings /
  1-D params / norms — which cola-coder's optimizer.py grouping ALREADY follows
  (verified in the earlier optimizer audit).
- Sources: "Practical Efficiency of Muon for Pretraining" (https://arxiv.org/html/2505.02222v1);
  MuLoCo (https://arxiv.org/abs/2505.23725); MuCon "Clipped Muon Updates"
  (https://arxiv.org/html/2605.26459); Tyler Romero Muon guide
  (https://www.tylerromero.com/translations/scientific-spaces/muon-optimizer-guide-quick-start-and-key-details/).
- ORIGINAL idea: the project's quality-weights × Muon question (IDEA-005) gains
  urgency — Muon orthogonalizes the gradient, so per-sample loss weighting may need
  to move to the data-sampling level; the new DATA-057 soft-dedup weights are part of
  that same `.weights.npy` signal, so validate them together under Muon.

## 2026-06-13 — Long context / context extension (rotate: long-context)

- **YaRN** (NTK-by-parts + attention temperature) extends to 128K+ with low perplexity
  loss — cola-coder already has a YaRN path (rope_scaling; MODEL-017 flagged reviewing
  its interpolation formula). The 2026 upgrade is **LongRoPE2**: near-lossless 128K via
  (1) the insight that high RoPE dims are under-trained, (2) evolutionary "needle-driven"
  RoPE rescaling, (3) **mixed context-window training** that adopts rescaled RoPE for
  long sequences while PRESERVING short-context accuracy (>98.5%). Llama.cpp v0.5 ships
  integrated YaRN for 2M+ tokens.
- Relevance: code models need repo-level context; cola-coder's Stage-4 context extension
  (YaRN) + repo_context module + memory module are the building blocks. LongRoPE2's
  short-context preservation matters (don't regress the 1024-ctx pretraining). → MODEL-033.
- Sources: YaRN (https://arxiv.org/pdf/2309.00071); LongRoPE2
  (https://arxiv.org/pdf/2502.20082); RoPE context-extension deep dive
  (https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html).
- ORIGINAL idea → **IDEA-010**: repo-level FIM for long-context training/eval. Use the
  repo_context module to assemble a large multi-file context, hold out a whole function
  in the middle, and train/eval FIM infill conditioned on the full repo (PSM with repo
  prefix + suffix). Verified by the sandbox tsc against the real repo. Combines
  long-context + FIM + repo_context + sandbox verifier — directly targets the
  IDE-ghost-text-in-a-real-codebase use case, and pairs with IDEA-007's
  contamination-free eval.

## 2026-06-13 — Test-time compute scaling (rotate: inference/reasoning)

- Test-time scaling = (1) aggregation (sample N → self-consistency / best-of-N with a
  verifier), (2) search (MCTS/beam), (3) iterative self-refine. Optimal allocation of
  inference compute can beat scaling parameters for reasoning. cola-coder already has
  best-of-N + a sandbox verifier — the missing piece is DYNAMIC budget allocation.
- 2026 trend: **dynamic / adaptive** test-time compute — confidence- or
  difficulty-aware budgets (e.g. "Deep Think with Confidence", DynScaling): spend MORE
  candidates on hard prompts, fewer on easy ones; early-stop when a candidate is
  confidently correct. Small models especially benefit from tool-integrated
  self-verification (T1) — which cola-coder has (sandbox tsc/tests).
- Relevance → IDEA-009: cola-coder's best-of-N uses a FIXED N. Make it adaptive.
- Sources: "Scaling LLM Test-Time Compute Optimally..." (https://openreview.net/forum?id=4FWAwZtd2n);
  "Deep Think with Confidence" (https://arxiv.org/pdf/2508.15260); DynScaling
  (https://arxiv.org/pdf/2506.16043); T1 tool-integrated self-verification for small
  LMs (https://arxiv.org/pdf/2504.04718).
- ORIGINAL idea → **IDEA-009**: adaptive best-of-N budget driven by the sandbox
  verifier. Generate a small initial batch (e.g. 2); if none VERIFY, expand the batch
  (×2) up to a max; if the first candidate verifies clean AND secure (SEC-017),
  early-stop. Trades compute for accuracy only when needed — cheap prompts cost 2
  candidates, hard ones get the full budget. Reuses best-of-N + sandbox verifier +
  the security screen; pure inference (non-train, safe on main).

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
