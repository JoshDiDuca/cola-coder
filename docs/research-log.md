# Research Log — 2025/2026 AI techniques tracked for cola-coder

Living log of external research (frontier-lab techniques, papers, standards) that
the autonomous improvement loop consults and turns into backlog items. **Each loop
cycle should add or update an entry here from a fresh web search**, then file
concrete backlog items referencing it. Newest first.

---

## 2026-06-15 — Optimizers: Muon (Newton-Schulz orthogonalized momentum) for 2D hidden layers (rotate: optimizers)

**Sources (2024–2026):** *Muon: An optimizer for hidden layers* — https://kellerjordan.github.io/posts/muon/
(SGD-momentum update → quintic Newton-Schulz orthogonalization of the 2D update → apply; scalars/vectors and the
input embedding + output head stay on AdamW; NS runs stably in bf16). *Muon Practical Guide* —
https://medium.com/@jenwei0312/going-beyond-adamw-a-practical-guide-to-the-muon-optimizer-93d90e91dbd3 (~2×
efficiency; each step costlier — NS is a cubic matmul). *Gram Newton-Schulz* (Tri Dao, 2026) —
https://tridao.me/blog/2026/gram-newton-schulz/ (faster hardware-aware NS). *Error Feedback for Muon* —
https://arxiv.org/pdf/2510.00643. *Preconditioning Benefits of Spectral Orthogonalization in Muon* —
https://arxiv.org/pdf/2601.13474. Adopted in Kimi K2, GLM-5.

**Summary:** Muon reaches a target loss in ~2× fewer steps than AdamW by orthogonalizing the momentum update for
matrix (2D) parameters — replacing the raw update `M` with `≈ (M Mᵀ)^{-1/2} M` via a few Newton-Schulz iterations
(no SVD). It is matrix-only: embeddings, the LM head, norms, and biases keep AdamW. **FINDING this cycle: cola-coder
ALREADY implements Muon** — `training/optimizer.py` has `_zeropower_via_newtonschulz5` + a `class Muon` (single
optimizer with a per-group `use_muon` flag doing the hybrid Muon-2D / AdamW-rest split), selectable via
`optimizer: "muon"` and already used by `configs/4080_max.yaml`, with ~18 tests in test_modern_techniques.py.
So the optimizer itself is DONE (MODEL-025/047). A subagent that re-implemented a standalone parallel Muon this
cycle was DISCARDED to avoid a DRY violation / two code paths — the existing implementation is kept. The open
contribution is therefore not the optimizer but how the project's quality-weighting interacts with it (below).

**Original idea (cross-technique, cola-coder-specific): match the quality-weighting LEVEL to the optimizer's
invariances.** The project scales the loss per sample by `.weights.npy` quality weights (training.md: a weighted
mean over per-sequence losses). But Muon ORTHOGONALIZES the 2D update — it largely discards per-coordinate
MAGNITUDE, keeping only direction. So loss-level magnitude weighting is partly washed out for Muon-optimized
matrices, while it still works for the AdamW-optimized embedding/head/scalars. Proposal: a hybrid weighting
strategy keyed to the optimizer split — keep loss-level quality weighting for the AdamW params, but realise quality
preference for the Muon params at the DATA-SAMPLING level (quality-weighted / curriculum sampling, which changes
WHICH directions are seen, surviving orthogonalization) — exactly the "weight at the sampling level instead"
fallback the MODEL-025 entry hypothesised. This makes the project's existing `.weights.npy` system compatible with
Muon. Measurable test: a small AdamW-vs-Muon × loss-weighted-vs-sampling-weighted ablation on tiny (the existing `"muon"` optimizer
makes this directly runnable). Filed as MODEL-052. (No new optimizer code shipped this cycle — the existing Muon
is complete; the contribution is this analysis + the filed weighting-interaction experiment.)

---

## 2026-06-15 — Architecture: attention/final logit soft-capping as a layered logit-magnitude stack (rotate: architecture)

**Sources (2024–2026):** *QK norm is probably a free lunch* — https://ishanjmukherjee.github.io/qk-norm
(RMSNorm on Q/K bounds attention-logit growth → stable at high LR; `QK_norm_cap` 3% lower ppl than baseline).
*Controlling changes to attention logits* — https://arxiv.org/pdf/2511.21377 (logit-capping family). *Gemma-2*
(attention + final-logit soft-cap `cap·tanh(x/cap)`). *Methods of improving LLM training stability* —
https://arxiv.org/pdf/2410.16682. *Scalable-Softmax Is Superior for Attention* — https://arxiv.org/pdf/2501.19399.
*When/Why Attention Sinks Emerge* — https://arxiv.org/pdf/2410.10781.

**Summary:** the 2026 stability stack for softmax transformers is a LAYERED control of logit magnitude:
QK-Norm (bound Q/K before the dot-product), attention-logit soft-cap (bound pre-softmax scores), and
final-logit soft-cap (bound the output logits) — each a cheap, training-stabilizing clamp that also tends to
improve perplexity slightly and permits higher learning rates. cola-coder ALREADY has QK-Norm (the live
small_react_best run trains with it). The missing, cheapest complementary piece is **soft-capping**:
`y = cap * tanh(x / cap)` — a smooth bounded clamp, applied to attention logits and/or final logits, fully
backward-compatible when disabled.

**Original idea (cross-technique, cola-coder-specific): final-logit soft-cap as a RELIABILITY lever, measured by
pass^k.** Beyond stability, a final-logit soft-cap softens an over-confident output distribution — and
over-confidence is a plausible driver of the low-`pass^k` (inconsistent) failure mode quantified by EVAL-037.
Hypothesis: for this small, data-bound model, enabling final-logit soft-cap raises `pass^k` (single-shot
reliability — the IDE-completion regime) more than it changes `pass@k` (capability), i.e. it shrinks the
capability-reliability gap. This makes soft-cap an architecture knob with a *measurable eval target* (run the
EVAL-037 pass^k/gap before vs after), not just a training-stability nicety — and it composes with the existing
QK-Norm to complete the layered logit-control stack, supporting the higher-LR Muon direction (MODEL-025/047).
Tractable MAIN-SAFE first step (this cycle): a tested `soft_cap_logits(logits, cap)` helper + an OPT-IN,
default-OFF config field wired into the attention/final path — DEFAULT-OFF so the live run is byte-identical;
gated behind the full test_checkpoint + test_transformer suite. Wiring it into a config sweep is the follow-up.

---

## 2026-06-15 — Safety: static CWE vulnerability screening as a shared data-filter + eval probe (rotate: safety)

**Sources (2024–2026):** *Rethinking the Evaluation of Secure Code Generation* (ICSE 2026) —
https://conf.researchr.org/details/icse-2026/icse-2026-research-track/175 (eval must measure security AND
functionality together; many mitigations degrade base performance >50%). *Spring 2026 GenAI Code Security Update*
(Veracode) — https://www.veracode.com/blog/spring-2026-genai-code-security/ (AI models still routinely emit
insecure code). *A Systematic Literature Review of LLMs in Code Security* — https://arxiv.org/pdf/2412.15004
(SVEN/SafeCoder fine-tune on curated secure data; ProSec preference-learns, ~35% vuln reduction; MA-CoT embeds
CWE mitigation guidance; vulnerabilities span ~18 CWE categories across C/Python/Go/JS). *Enhancing Reliability
in LLM-Based Secure Code Generation* — https://arxiv.org/html/2605.24300.

**Summary:** the consensus 2026 finding is that **the training-data problem is the root cause** — until models
learn from secure-code corpora, generated code keeps reproducing the same CWE patterns. cola-coder already has
secrets/PII/dangerous-pattern safety probes (safety_eval), a malware scanner, slopsquat import triage (SEC-023),
and an injection scorer — but NO broad static CWE screen covering the recurring high-frequency weaknesses:
CWE-78 (OS command injection, `os.system`/`subprocess(..., shell=True)`), CWE-502 (unsafe deserialization,
`pickle.loads`/`yaml.load`), CWE-89 (SQL injection via string-built queries), CWE-327/328 (weak crypto md5/sha1),
CWE-330 (insecure `random` for secrets), CWE-22 (path traversal), CWE-95 (`eval`/`exec`). These are detectable
purely statically (regex/AST on code as TEXT — never executing it), language-aware for Python + TS/JS.

**Original idea (cross-technique, cola-coder-specific): ONE static CWE scanner, used bidirectionally to close the
eval→data loop.** The same screen plugs into BOTH ends: (a) as a **data-quality scorer/filter** (down-weight or
drop training examples that themselves contain CWE patterns — directly attacking the "train on secure corpora"
root cause, complementing DATA-072 educational-value), AND (b) as an **eval safety probe** (measure the CWE rate
in the model's *generated* code). That makes the loop explicit: CWEs the model emits → the SAME detector finds
those patterns in the training data → filter/reweight them out → re-evaluate the generated-CWE rate. It reuses
the project's `language_detect` + `ScoreMapper` + `SandboxedRunner` discipline (static only, untrusted code is
never run), and composes with the existing injection scorer (this adds the non-injection CWE families). Tractable
MAIN-SAFE first step (this cycle): the static `CweSecurityScorer` (pattern set + per-CWE findings + a 0–1 score)
registered as a data scorer, with tests; wiring it into safety_eval as a probe is the filed follow-up.

---

## 2026-06-15 — Post-training: DAPO overlong reward shaping + length-bias fixes for GRPO (rotate: post-training)

**Sources (2025–2026):** *DAPO: An Open-Source LLM RL System at Scale* — https://arxiv.org/pdf/2503.14476
(four techniques: Clip-Higher, Dynamic Sampling, Token-level Policy-Gradient Loss, and **Soft Overlong
Reward Shaping** — a length-aware penalty that ramps reward down as a response approaches/exceeds the length
budget, cutting the reward NOISE from truncated overlong generations; 50 AIME pts, 50% fewer steps).
*Dr.GRPO* — mitigates the length bias that per-response advantage normalization introduces. *Geometric-Mean
Policy Optimization* — https://arxiv.org/pdf/2507.20673 (GMPO: geometric mean over tokens is robust to
outlier per-token ratios). *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond* —
https://llm-stats.com/blog/research/post-training-techniques-2026. *The Art of Scaling RL Compute for LLMs* —
https://arxiv.org/pdf/2510.13786.

**Summary:** cola-coder's reasoning GRPO already adopts the DAPO/Dr.GRPO core (clip-higher 0.2/0.28,
advantage_norm "mean", parallel gen). The remaining cheap, high-value DAPO piece it LACKS is **soft overlong
reward shaping**: a deterministic length-aware penalty applied to the reward before advantages, so a solution
that runs to (or past) the max-length budget is smoothly penalised rather than contributing a noisy, abruptly
truncated signal. It is a pure function of (length, max_len, soft_buffer) — testable, no model in the loop,
and composes with any existing reward (python_exec/typescript/combined). Length-normalization choice
(per-response vs group) is the related lever Dr.GRPO addresses.

**Original idea (cross-technique, cola-coder-specific): reliability-gap-weighted GRPO advantages — fuse last
cycle's pass^k (EVAL-037) into post-training.** GRPO weights every sampled problem's advantage equally. But the
**capability-reliability gap** `pass@k − pass^k` (EVAL-037) identifies exactly the problems the model CAN solve
but not RELIABLY — the highest-headroom targets for consistency training. Proposal: scale each problem's GRPO
advantage by a function of its measured gap (high gap → up-weight), so RL compute concentrates where it most
improves single-shot reliability (the IDE-completion use case), not on already-reliable or hopeless problems.
The gap is computed for free from the rollouts GRPO already generates (n samples, c correct per problem). This
turns the eval metric into a training signal — a closed eval→post-training loop. Tractable MAIN-SAFE first step
(this cycle): the DAPO soft overlong reward-shaping utility + tests (a pure function); the gap-weighting is the
filed follow-up. NEVER touch the live pretraining run — this is the separate reasoning/GRPO path only.

---

## 2026-06-15 — Reliability eval: pass^k / consistency as the counterpart to pass@k (rotate: eval)

**Sources (2024–2026):** *Statistics for AI/ML, Part 4: pass@k and the Unbiased Estimator* —
https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/ (the Chen et al. 2021 numerically-stable
unbiased pass@k estimator `1 - prod((n-c-i)/(n-i))`, and the symmetric "all-k-pass" estimator).
*Pass@k Metrics for LLM Evaluation* — https://www.emergentmind.com/topics/pass-k-metrics-2508a3b6
(pass@k = P(≥1 of k correct); the consistency/reliability family measures the opposite tail).
*Revisiting the Reliability of Language Models* — https://arxiv.org/pdf/2512.14754 ("Reliable@k":
how consistently a model satisfies a criterion across related prompts). *RobustPass@k* — prompt-perturbation
robustness. Theme: 2026 eval rigor has shifted from "can it ever solve this?" (pass@k, a CAPABILITY/best-case
metric) to "does it solve this RELIABLY?" (consistency/reliability, a worst-case metric).

**Summary:** pass@k rewards lucky one-in-k successes — exactly what best-of-N + a verifier exploits. But for
*single-shot* use (an IDE inline-completion, an un-verified agent step) what matters is **consistency**: of k
independent samples, how often do they ALL pass. The unbiased "all-k-pass" estimator is the clean mirror of
pass@k: with `c` of `n` samples correct, `pass^k = C(c,k)/C(n,k) = prod_{i=0}^{k-1} (c-i)/(n-i)` (0 when c<k,
1 when c=n) — same numerically-stable product form the project's `pass_at_k` already uses. cola-coder already
has pass@k with bootstrap CIs (EVAL-028) and best-of-N verification (the capability-exploiting consumer), but
NO reliability metric.

**Original idea (cross-technique, cola-coder-specific): the capability–reliability GAP as a routing + temperature
diagnostic.** Report `gap = pass@k − pass^k` per problem and aggregate. A large gap = "the model CAN solve it but
not RELIABLY" — the precise regime where (a) best-of-N + the sandbox verifier pays off most (worth the extra
compute), and (b) single-shot IDE completion will feel flaky. Two concrete uses for this project: (1) **temperature
selection** — sweep temperature and pick the point that maximises pass^k (reliability) for the IDE/base-model path,
while a separate higher-temperature setting maximises pass@k for the best-of-N path; the gap quantifies the
trade-off the existing `--best-of N` flag is implicitly making. (2) **tie to the PLD/speculative work** — a
low-pass^k (inconsistent) model is exactly where draft-free speculative decoding helps LEAST (the running buffer
is a poor predictor when the model itself is unstable), so the gap is a cheap pre-screen for where INFER-035/036
will and won't speed things up. Tractable MAIN-SAFE first step (this cycle): the `pass^k` estimator + aggregation
+ the gap, reported alongside pass@k in `evaluation/metrics.py`, with tests.

---

## 2026-06-15 — Educational-value data filtering: LLM-annotated classifiers (FineWeb-Edu / Stack-Edu) + a cheap static prior (rotate: data quality)

**Sources (2024–2026):** *FineWeb-Edu / The FineWeb Datasets* — https://arxiv.org/html/2406.17557v1 (LLM-as-annotator:
Llama-3-70B rates documents 0–5 for "educational value", a lightweight embedding regressor is trained on those
labels and applied at scale; educational filtering beat every prior heuristic on downstream benchmarks). *Stack-Edu*
(BigCode/SmolLM2 line) — Llama-3-70B-Instruct annotates 500k code fragments 0–5 on educational+structural quality,
then per-language StarEncoder classifiers (F1 > 0.7 most languages) filter The-Stack-v2. *The Stack v2 / StarCoder2* —
https://github.com/bigcode-project/starcoder2 (improved license+language detection + filtering heuristics). *Rewriting
Pre-Training Data Boosts LLM Performance in Math and Code* — https://arxiv.org/pdf/2505.02881 (model-rewritten data, a
synthesis-side complement to filtering). *Phi-1 "Textbooks Are All You Need"* — the canonical "quality beats quantity,
even by 100×" result that motivates educational filtering for code.

**Summary:** the 2026 canonical pre-training pipeline for code is dedup → quality/educational filtering → reweight toward
educational sources, at large token scale. The frontier quality signal is no longer a hand-tuned heuristic but an
**educational-value classifier**: an expensive LLM annotates a sample 0–5, a cheap model distills those labels, and the
cheap model scores the whole corpus. cola-coder already has the distillation HALF of this exactly: `train_judge_classifier.py`
distills LLM-judge scores into a local TF-IDF classifier, plus 11 static `data/filters/` and `data/scorers/`. What it
lacks is a code-specific *educational-value* target and a way to bound the (expensive) LLM-annotation budget.

**Original idea (cross-technique, cola-coder-specific): a CHEAP static educational-value prior as a cascade gate.**
Build a no-LLM, CPU-only `EducationalValueScorer` (a `data/scorers` plugin) that combines signals the project can compute
for free — comment/docstring density, presence of a usage example or test, identifier-naming quality, structural
completeness (defs/returns/imports balance), and the DATA-071 Gopher repetition score (penalise boilerplate/degenerate)
— into a 0–1 educational proxy. Use it two ways: (a) directly as a curriculum/reweighting signal (`.weights.npy`), and
(b) as a **cascade gate that slashes the LLM-judge annotation budget** — only documents whose static prior is in the
*uncertain* middle band get routed to the expensive judge classifier; confidently-good and confidently-bad docs skip it.
This fuses the project's existing judge-distillation pipeline + DATA-071 repetition machinery with the FineWeb-Edu/Stack-Edu
educational-value paradigm at a fraction of the LLM cost. Tractable MAIN-SAFE first step (this cycle): the static scorer
itself, registered + tested; the cascade routing and any LLM annotation are follow-ups (filed as DATA backlog items).

---

## 2026-06-15 — Speculative decoding for code: draft-free Prompt-Lookup + the acceptance-first discipline (rotate: inference)

**Sources (2023–2026):** *Prompt Lookup Decoding* (apoorvumang, 2023) — https://github.com/apoorvumang/prompt-lookup-decoding
(draft-free single-model speculative drafter: match the running suffix n-gram against earlier context, copy the
continuation; lossless — the target still verifies every token). *REST: Retrieval-Based Speculative Decoding*
(NAACL 2024) — https://arxiv.org/abs/2311.08252 (generalises PLD to an external (context→continuation) datastore).
*EAGLE-3 / feature-level drafting* and *Medusa* (extra MLP heads) — the trained-drafter frontier, ~0.75–0.85
acceptance on structured/code text. *Speculative Decoding 2-3x Faster LLM Inference (2026)* —
https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/ (now production-standard in
vLLM/SGLang/TRT-LLM; NVIDIA 3.6× on H200; real acceptance 0.6–0.85, code/structured at the top of that band).
*Training-Free Loosely Speculative Decoding* — https://arxiv.org/pdf/2511.22972 (accept SEMANTICALLY-equivalent
drafts beyond exact match — a lossy variant). *Speculative Speculative Decoding* (ICLR 2026) — https://arxiv.org/pdf/2603.03251.

**Summary:** speculative decoding (draft K tokens cheaply, verify them in ONE target forward pass, keep the
longest correct prefix + 1 bonus token) is the dominant 2026 inference speedup, 2–3.6×. The two axes are
(a) draft source — trained head (EAGLE/Medusa, highest acceptance, needs training) vs **draft-free** (PLD/REST,
zero training, zero extra params, just string-matching), and (b) acceptance rate α, the first-order predictor of
speedup. The literature's discipline: **measure α on YOUR corpus BEFORE wiring a drafter into the hot path** —
mean-accepted-length and α determine whether per-step overhead erodes the theoretical win. Code is the best-case
domain for draft-free PLD: identifiers, imports, type annotations, and boilerplate recur VERBATIM within a file,
so the running buffer is itself a high-yield draft datastore at zero cost.

**cola-coder fit:** this project is a code model whose flagship use is IDE inline-completion (FIM) — exactly where
PLD shines and where a 2× latency cut is directly user-visible. It already has the pieces: an `inference/`
KV-cache generator, `--repo` repo-context assembly, and a `retrieval/vector_store`. The offline half now exists
and is tested: `inference/prompt_lookup.py` (`PromptLookupDrafter` + `analyze_acceptance`) measures α / mean
accepted length / idealised speedup over a recorded trace with NO model in the loop — implementing the
"acceptance-first" discipline above. (This cycle: tested + wired into a `scripts/pld_analysis.py` CLI + eval menu.)

**Original idea (cross-technique, cola-coder-specific): _context-seeded PLD_ — fuse PLD with the project's repo
retrieval.** Vanilla PLD only mines the *running buffer*. But for IDE code completion the highest-yield verbatim
continuations live in the **open file + imported/retrieved neighbour files** the project already assembles for
`--repo`/vector_store context. Proposal: seed the PLD lookup datastore with the *retrieved repo context tokens*
(REST-style, but the datastore is the user's own session repo, not a global corpus) in ADDITION to the running
buffer — a tiered, draft-free, training-free, lossless drafter specialised for code. Tier 1: running-buffer
n-gram (locality). Tier 2: retrieved-repo n-gram (cross-file recurrence: re-typing an import, a known signature,
a sibling component). The existing `analyze_acceptance` harness can A/B buffer-only vs buffer+repo α on real
prepared `.npy` corpora to size the win before any hot-path change — and the project's DATA-071 n-gram machinery
is reusable for the datastore indexing. Open question: does adding the repo datastore raise α enough to beat the
extra lookup cost, given code's already-high buffer-only α? Measure first (that is the whole point of the harness).

---

## 2026-06-14 — Gopher / MassiveText repetition filters: char-fraction degeneracy screen (rotate: data quality/curation)

**Sources (2021–2026):** *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* —
https://arxiv.org/html/2406.17557v1 (FineWeb re-derives MassiveText/Gopher's repetition filters via
ablation, TIGHTENING the duplicated-line-character ratio from MassiveText's ≥0.2 to ≥0.1 — a single
repetition metric removed 12.47% of tokens, more than any other quality screen). *RedPajama-Data-v2: 30T
tokens with quality signals* — https://www.together.ai/blog/redpajama-data-v2 (ships the Gopher repetition
annotations as reusable per-document QUALITY SIGNALS — duplicate-line/-paragraph fractions + top/duplicate
n-gram char fractions — so curators threshold them as filters or feed them to selection). *data-prep-kit
gopher_repetition_annotator* — https://github.com/data-prep-kit/data-prep-kit/blob/dev/transforms/universal/gopher_repetition_annotator/README.md
(the canonical reference thresholds, from Gopher Table A1: dup line/para frac 0.30, dup line/para CHAR frac
0.20, top 2/3/4-gram char frac 0.20/0.18/0.16, duplicate 5..10-gram char frac 0.15→0.10; reject if ANY exceeds).

**Summary:** the established frontier pre-training quality screen for DEGENERATE documents (looping
generations, copy-pasted boilerplate, accidental content duplication) is the Gopher repetition family: 13
metrics measuring the *character mass* locked up in repeated lines, paragraphs, and word n-grams (2..10).
For 2-4 grams it's the char fraction of the single MOST FREQUENT n-gram ("top"); for 5-10 grams the fraction
covered by ALL repeated n-grams ("duplicate"). A doc is rejected if any metric exceeds its threshold. It is
purely statistical, language-agnostic, CPU-only, no model — and catches a failure mode that exact/MinHash/
semantic dedup (which compare ACROSS documents) structurally cannot: *within-document* repetition.

**Original idea (filed) → DATA-071/072/073:** cola-coder's curation stack dedups ACROSS documents (exact
SHA-256, MinHash near-dup in `combine.py`, semantic clustering in `semantic_dedup.py` per DATA-069) and scores
quality per-file (`CodeScorer`), but the only WITHIN-document repetition signal it has is
`CodeScorer._score_duplication` — exact duplicate *lines* only, folded into a soft 0.0-1.0 weight, blind to
paragraph- and n-gram-level looping. The cross-technique idea is to add the full Gopher repetition family as a
first-class composable `FilterPlugin` (the same `@register_filter` interface as `content`/`length`/`quality`),
so a hard "this document is degenerate" reject runs in the prep pipeline ALONGSIDE the existing soft score —
the two are complementary (score down-weights mediocre code; the repetition filter drops looping garbage
outright). Unique to this repo: the metrics are also a natural EXTRA SIGNAL for the per-chunk
`weight_scoring.py` path and a corpus-audit cross-tab against `semantic_dedup` cluster sizes (does
within-doc repetition correlate with the redundant clusters?). DATA-072 (surface metrics as `CodeScorer`
breakdown signals → quality weights) + DATA-073 (corpus repetition audit / threshold-ablation report,
FineWeb-style "how many tokens does each metric remove") filed for future cycles.

**Implemented this cycle (DATA-071 — data curation, main-safe):** Gopher/MassiveText repetition filter.
`data/filters/repetition.py`: `compute_repetition_metrics` (pure function → `RepetitionMetrics`: dup
line/para fractions + char fractions, top 2-4 gram char fractions, duplicate 5-10 gram char fractions, all
bounded [0,1] via non-overlapping char COVERAGE — a naive chars×count overcounts overlapping windows and can
exceed 1.0, a bug caught and fixed during test bring-up), `RepetitionThresholds` (dataclass of the published
Gopher reference values, not bare dicts), and `@register_filter("repetition") RepetitionFilter` (FilterPlugin;
"reject if ANY metric strictly exceeds" with the reason string naming the offending metric; line/para screens
always run, n-gram screens skip docs < `min_words=50` per datatrove's short-doc guard; `setup()` overrides any
threshold from YAML). Registered in `filters/__init__.py`; listed in the data-menu Advanced Filters table (no
new script → no orphan). +19 tests (metric algebra, boundedness, empties, accept/reject + reason, short-doc
skip, configurable min_words, default-thresholds-match-Gopher, setup overrides, registry wiring); existing
`test_filters` (42) + `test_filter_registry_and_pii` intact. Pure-Python, no model/GPU/network. Closes the
across-doc-dedup → within-doc-repetition gap in the curation stack.

## 2026-06-14 — Slopsquat triage: typosquatting detection by string distance (rotate: safety/robustness)

**Sources (2020–2026):** *ConfuGuard: Using Metadata to Detect Active and Stealthy Package Confusion
Attacks* — https://arxiv.org/pdf/2502.20528 (frames the goal as separating CONFUSION attacks from
LEGITIMATE packages; positions prior string-similarity defenses — SpellBound, keyboard-proximity — as the
cheap first line, then adds registry metadata to cut their false positives). *Training LLMs for Advanced
Typosquatting Detection* — https://arxiv.org/pdf/2503.22406 (a Damerau-Levenshtein-based name matcher hits
98.4% accuracy; edit distance alone misses homoglyphs / keyboard-adjacent / separator-reorder squats, so
detectors combine distance + normalization + homoglyph folding). *Typosquatting & Slopsquatting: detecting
AI-hallucinated malicious packages* (Cloudsmith, 2026) — https://cloudsmith.com/blog/slopsquatting-and-typosquatting-how-to-detect-ai-hallucinated-malicious-packages
(separator normalization `mysql-import`↔`mysql_import`, homoglyph `l0dash`↔`lodash`, and edit-distance
neighborhoods are the standard offline screens; IQTLabs/pypi-scan and rustfoundation/typomania implement them).

**Summary:** slopsquatting registers a name a model HALLUCINATES (USENIX 2025: ~20% of generations import a
non-existent package; ~205k unique fabricated names, ~45% persistent → reliably exploitable). The defensive
distinction that matters is NOT "known vs unknown" but "legit-niche vs confusion-of-a-popular-name": a name
one edit / one separator / one homoglyph away from a popular package is a high-risk typosquat, whereas a
genuinely novel niche name is merely unverified. The established cheap, OFFLINE screen is Damerau-Levenshtein
distance to the popular set + separator normalization + homoglyph folding (ConfuGuard later layers registry
metadata, which an offline trainer like this repo cannot see).

**Original idea (filed) → SEC-023/024/025:** cola-coder already owns the binary half — `import_scanner.py`
`scan_unknown_imports` flags every out-of-allowlist import identically, and best-of-N surfaces them as a flat
`unknown_imports` review signal. But it can't tell `requsts`/`l0dash`/`bs_4` (squats of requests/lodash/bs4)
from a legit niche import, so every unknown gets the same weight. The cross-technique idea is to REUSE the
existing curated popular-package allowlists (`_PY_POPULAR`/`_JS_KNOWN`) as the typosquat NEIGHBORHOOD and run
the standard offline distance screen over the scanner's own survivors — turning the binary "unknown" into a
TRIAGED risk verdict (typosquat vs unknown) with the nearest popular name and distance attached. Unique to
this repo: the allowlist that powers the slopsquat flag IS the popular-package set typosquat detection needs,
and the best-of-N verifier already carries a per-candidate `details` channel to surface the higher-risk
verdict — no new corpus, no network. SEC-024 (down-rank verified-but-typosquatting candidates in best-of-N
tie-breaks) + SEC-025 (RFT/distillation reject gate on typosquat imports, mirroring the secure-pass gate)
filed for future cycles.

**Implemented this cycle (SEC-023 — safety, main-safe):** typosquat/slopsquat triage of unknown imports.
`security/import_scanner.py`: `_damerau_levenshtein` (optimal-string-alignment edit distance — counts adjacent
transpositions as one edit, the dominant typo class), `_normalize_name` (lowercase + separator unification +
homoglyph folding `0→o 1→l rn→m vv→w` + separator stripping, so `mysql-import`/`mysql_import`/`l0dash` collapse
onto their real neighbors), and `classify_unknown_imports` → an `ImportTriageReport` (typed `SuspectImport`
dataclasses + `ImportRisk` enum) that REUSES `scan_unknown_imports` for the allowlist screen (DRY — a test
asserts the triage partitions EXACTLY the scanner's survivors) then sorts each into TYPOSQUAT (≤`max_distance`
normalized edits to a popular name, `min_length` guard against short-name chance collisions, distance-0
separator/homoglyph confusion) vs UNKNOWN (legit-niche, not over-flagged). Wired into best-of-N: when unknown
imports exist, a `typosquat_imports` review signal is added to candidate `details` alongside the existing
flat `unknown_imports` (signal only — no ranking change, back-compat preserved). +24 tests (distance algebra
incl. transposition/empties/symmetry, normalization, Python+JS triage, legit-niche-is-UNKNOWN, min_length /
max_distance thresholds, reuse-partition consistency, best-of-N wiring); import-scanner's 15 intact, best-of-N's
40 intact. Pure string logic — no model/GPU/network/execution. Closes the flat→triaged gap on the slopsquat signal.

## 2026-06-14 — Contamination-aware code evaluation: decontaminated-variant comparison (rotate: evaluation)

**Sources (2024–2026):** *LiveCodeBench: Holistic and Contamination-Free Evaluation* —
https://arxiv.org/abs/2403.07974 (date-annotated contest problems → "evaluation over time": score
only problems released AFTER a model's cutoff so memorisation can't help; also self-repair / test-output /
execution scenarios). *LLM Benchmark Methodology 2026 (contamination & leaderboard guide)* —
https://www.digitalapplied.com/blog/llm-benchmark-methodology-2026-contamination-leaderboard-guide
(documents the 2026 standard: OpenAI's *verbatim-reproduction audit* — frontier models recover gold
SWE-bench patches from just a task ID; SWE-ReBench's *decontaminated-variant comparison* — report how much
of a score SURVIVES decontamination, i.e. pass@k on a contamination-resistant subset vs the full set).
*Benchmark Data Contamination survey* — https://arxiv.org/abs/2502.14425 (n-gram/containment overlap at
≥0.8 is the established contamination screen).

**Summary:** a bare pass@k is unfalsifiable once the benchmark leaks into training — a model that memorised
three answers scores the same as one that reasoned about three. The 2026 fix is not just to *detect* leakage
but to *quantify how much of the score survives decontamination*: split problems by contamination/recoverability,
report pass@k on the clean (memorisation-impossible) subset alongside the contaminated subset, and treat the
clean-minus-contaminated GAP as the memorisation signature. All computable offline from already-collected
results + a containment screen — no model load.

**Original idea (filed) → EVAL-036/037/038:** cola-coder already owns the binary half (DataLeakageDetector
MinHash+containment; `check_contamination.py`; DATA-065 DecontaminationFilter) but only emits a yes/no verdict
at one threshold and never feeds it back into the SCORE. The cross-technique idea is a *contamination-trust-
stratified pass@k*: reuse the detector's exact `_containment`/`_shingles` to assign each benchmark problem a
CONTINUOUS recoverability score (max containment of its prompt OR canonical solution in the corpus), bucket
into clean/suspect/contaminated, then join with the eval harness's own `ProblemResult` records to report
pass@k per tier + the clean-vs-contaminated `trust_delta` and a quotable decontaminated `trusted_pass_at_k`.
Unique to this repo: the verifier-effort difficulty tiers (EVAL-026) and the containment scorer are both
model-relative and free — so the same machinery can later cross-tabulate contamination × difficulty (is the
"hard" tier just the memorised tier?). EVAL-037 (corpus-sampling CLI binding the existing train-corpus loader)
+ EVAL-038 (contamination × verifier-effort cross-tab) filed for future cycles.

**Implemented this cycle (EVAL-036 — eval, main-safe):** contamination-trust-stratified pass@k.
`evaluation/contamination_stratified.py`: `score_problem_contamination` (continuous per-problem max-containment
of prompt OR canonical_solution, REUSING DataLeakageDetector `_shingles`/`_containment` — not a fork),
`contamination_tier` (clean<0.50≤suspect<0.80≤contaminated), `stratified_pass_at_k` (joins tiers with
`ProblemResult` via `compute_pass_at_k`; `StratifiedPassAtKReport.trusted_pass_at_k` = decontaminated score,
`trust_delta` = clean−contaminated, None-safe when a tier is empty or pass@k not estimable for n<k;
`unmatched_task_ids` surfaced), and `build_contamination_detector` (one indexed detector for the binary path).
Library-only (consumes existing eval results → no new script → no orphan menu entry). +20 tests, CPU-only,
no model/GPU/sandbox. Closes the detect→quantify gap: turns the yes/no leakage check into a falsifiable score.

## 2026-06-14 — Draft-free speculative decoding: prompt-lookup / REST (rotate: inference)

**Sources (2023–2026):** *Prompt Lookup Decoding* — https://github.com/apoorvumang/prompt-lookup-decoding
(match last-n generated tokens against the prompt, draft the following span, verify in one forward;
lossless, training-free, no datastore; large gains on code copying). *REST: Retrieval-Based Speculative
Decoding* — https://arxiv.org/abs/2311.08252 (datastore retrieval, 1.62–2.36× on code/text). *AdaPLD*
— https://arxiv.org/html/2606.05742 (exact→semantic fallback + branched drafts, 3.10× on CodeEditorBench).
*Prompt Multi-Lookup* (ACL Findings 2025) — https://aclanthology.org/2025.findings-acl.355.pdf.

**Summary:** get draft tokens by COPYING from text the model already has (prompt / generated suffix /
code datastore) instead of a second neural draft model; the target verifies the span in one forward.
Mathematically lossless (eval scores unchanged), training-free, single-model — acceptance is high for code
(generations copy identifiers/imports/structure from the prompt).

**Original idea (filed) → INFER-035/036/037:** repo-context-anchored prompt-lookup — build the draft
source from cola-coder's `repo_context` assembler + the FIM suffix, so inline completions draft from the
USER'S codebase (where a code model copies symbols), pushing acceptance above generic PLD. Lossless ⇒ zero
eval-quality risk, composes with best-of-N. NOTE: existing `features/speculative_decoding.py` is the
two-model variant; this draft-free family is orthogonal. Scoped as a pure drafter + offline acceptance
analyzer (off the KV-cache hot path) for main-safety; live decode is INFER-036 (worktree).

**Implemented this cycle (EVAL-035 — eval, main-safe):** Spectral-Alignment divergence-risk diagnostic.
`evaluation/spectral_health.py`: `principal_left_singular_vector` (plain power iteration, NOT Muon's
Newton-Schulz), `spectral_alignment` (reuses depth_profile's block iteration; per-token cosine of each
probed weight's response — q_proj / ffn down_proj — with u₁), `sign_collapse_stat` (majority-sign fraction,
0.5 healthy → 1.0 collapsed), `profile_spectral_health` (per-layer + worst-layer + optional per-tier).
`scripts/spectral_health.py` (`--layers/--by-difficulty/--json`, HEALTHY/WATCH/DIVERGENCE-RISK verdict) +
eval-menu "Spectral Health / Divergence Risk" entry. +22 tests (rank-1 u₁ recovery, SA aligned≈1/orth≈0,
collapse direction). Main-safe checkpoint diagnostic. MODEL-047 (online ZClip clamp, worktree) + INFER-034
(RFT acceptance gate) open.

## 2026-06-14 — Spectral-alignment / training-stability diagnostics (rotate: optimizers/stability)

**Sources (2025):**
- *Spectral Alignment as Predictor of Loss Explosion in DNN Training* — https://arxiv.org/html/2510.04202.
  Per-layer Spectral Alignment (SA) = cosine between layer input activations and the principal left
  singular vector u₁(W) (power iteration, no full SVD). Healthy = sign-balanced ~0; impending failure =
  sign-collapse to one side. Detects nascent instability ~4,400 steps (Flash-Attn) / ~570 (FFN) BEFORE a
  loss explosion — mechanistically upstream of grad-norm/loss spikes.
- *ZClip: Adaptive Spike Mitigation for LLM Pre-Training* — https://arxiv.org/pdf/2504.02507. z-score
  anomaly detection on grad norm vs an EMA; clip only statistical outliers.
- *SPAM: Spike-Aware Adam with Momentum Reset* — https://arxiv.org/pdf/2501.06842. Conditional spike
  clipping + momentum reset so accumulated momentum doesn't compound a spike.
- *AdaGC: Adaptive Gradient Clipping* — https://arxiv.org/pdf/2502.11034. Per-param EMA-driven thresholds.

**Summary:** divergence is predictable from per-layer weight+activation stats before the loss moves. SA
(sign-collapse of the principal-singular-vector alignment) is the earliest, upstream signal; spike-aware
optimizers react to the downstream grad-norm z-score. All are computable from a saved checkpoint's weights
+ one forward pass — no train-loop edit needed.

**Original idea / hypothesis (cola-coder-specific cross-technique) → EVAL-035 (filed):**
**Verifier-stratified spectral-alignment health map.** Combine SA with `depth_profile.py`'s block-iteration
(already captures per-layer residuals over a real forward) and the verifier-effort difficulty tiers /
correctness-critical token regions. Compute per-layer SA over a batch, stratify the sign-collapse statistic
by token region (boilerplate vs assert-critical). Hypothesis: layers whose SA collapses are the layers whose
hidden states DECIDE functional correctness — so a single checkpoint's spectral health predicts both
divergence risk AND where instability lands. Also a checkpoint-acceptance gate for the RFT flywheel (reject a
self-distilled checkpoint whose spectral health regressed before spending verifier-graded eval budget).
Filed for a future cycle (main-safe checkpoint diagnostic).

**Implemented this cycle (EVAL-034 — eval, main-safe):** verifier-anchored function-step process-credit
profiler ("poor-man's PRM"). `evaluation/process_credit.py`: `decompose_functions` (Python AST, TS regex
fallback via language_detect), `function_step_scores` (reuses `partial_credit.split_test_cases`; attributes
each assert to the function(s) it references, per-function score = fraction of attributed asserts passing via
an injected sandbox `execute_fn`; untested functions get an executability probe; `process_score` =
LENGTH-NORMALIZED mean so a verbose vacuous function can't inflate it — resists FunPRM's verbosity hack),
and `fragile_functions` (passes overall but a dead/non-executable function). `scripts/process_credit.py`
(runs best-of-N with the real sandbox runner) + eval-menu "Process / Function-Step Credit" entry. +22 tests.
INFER-033 (process_score as best-of-N tie-break) + SEC-022 (per-function scan_dangerous) open.

## 2026-06-14 — Process Reward Models for code (rotate: post-training/reward design)

**Sources (2025–2026):**
- *A Survey of Process Reward Models: From Outcome Signals to Process Supervisions* (2026) —
  https://arxiv.org/html/2510.08049v3. PRMs score partial solutions → scalar; CodePRM uses tree-search +
  execution feedback for automatic step labels. Key failure: step-level signals noisier than
  trajectory-level → length/verbosity hacking. Best-of-N reranking with PRM scores consistently helps.
- *FunPRM: Function-as-Step Process Reward Model with Meta Reward Correction for Code* (2026) —
  https://chatpaper.com/chatpaper/paper/231379. Treats FUNCTIONS as steps + a meta-learning correction
  that uses clean unit-test final rewards to purify noisy partial rewards. Beats TTS baselines on
  LiveCodeBench/BigCodeBench across 5 base LLMs.
- *Process Reward Models That Think (ThinkPRM)* — https://arxiv.org/pdf/2504.16828. Verbalized PRM
  verifies each step via a CoT; far fewer process labels than discriminative PRMs.

**Summary:** a PRM scores PARTIAL code for dense step-aligned credit (best-of-N reranking, RL). The hard
problems for code: step decomposition (FunPRM: function = step) and label noise (Monte-Carlo partial
correctness is high-variance + length-hackable; anchor to clean unit-test outcomes).

**Original idea / hypothesis (cola-coder-specific cross-technique) → EVAL-034 (filed):**
**Verifier-anchored function-step credit ("poor-man's PRM").** FunPRM needs a trained PRM + meta-correction
to purify noisy step rewards; cola-coder owns the clean signal directly — the sandbox unit-test verifier +
AST assert-split (`partial_credit.split_test_cases`) + best-of-N. Decompose each candidate into
function-as-steps via AST, score each function by the test subset that exercises it (or an executability
probe), aggregate into a length-normalized process_score that reranks best-of-N ABOVE the heuristic score
but BELOW the hard verdict, and emit a fragility map (passes overall but contains a dead/non-executable
function). FunPRM's clean-reward signal WITHOUT training a PRM — no PRM paper owns the execution verifier
FunPRM only uses as a correction oracle. Filed for a future cycle (main-safe analysis module).

**Implemented this cycle (DATA-069 — data curation, main-safe):** semantic (embedding) dedup.
`data/semantic_dedup.py`: dependency-free numpy TF-IDF embedder (pluggable `embed_fn`/precomputed),
`cluster` (sklearn KMeans if present else a seeded numpy Lloyd's fallback — fallback is the CI path),
`find_semantic_duplicates` (within-cluster cosine; keep the HIGHEST-quality member when `quality_weights`
given — the original idea — else the SemDeDup centroid-distant default), and `semantic_dedup_array`
returning `(kept_data, removed_count, kept_indices)` — a superset of `ExactDeduplicator`'s contract.
`scripts/prepare_data.py --dedup semantic` (+ `--semantic-threshold`/`--semantic-clusters`, runs after the
exact pass) + data-menu "Semantic (SemDeDup)" choice (no orphan). +14 tests, numpy-only. Catches reordered/
renamed near-dups that exact+MinHash miss. DATA-070 (D4 diversification) + EVAL-033 (semantic-dedup audit) open.

## 2026-06-14 — Semantic (embedding) deduplication: SemDeDup / D4 (rotate: data curation)

**Sources (2023–2026):**
- *SemDeDup: Data-efficient learning at web-scale through semantic deduplication* —
  https://arxiv.org/abs/2303.09540 (FAIR). Embed → k-means cluster → within-cluster cosine sim →
  drop near-duplicate pairs, keeping the point FARTHEST from the centroid (representative-yet-distinct).
  Removes ~50% of data with minimal performance loss (halves training time), improves OOD.
- *D4: Improving LLM Pretraining via Document De-Duplication and Diversification* —
  https://arxiv.org/abs/2308.12284. SemDeDup + SSL-prototype diversification; ~20% training-efficiency
  gain and up to ~2% downstream accuracy with pre-trained-model embeddings.
- *Beyond the Black Box: Survey on Theory & Mechanism of LLMs (2026)* — https://arxiv.org/html/2601.02907v2.
  Positions D4/SemDeDup as the evolution beyond syntactic (hash/MinHash) to semantic matching.

**Summary:** SemDeDup catches SEMANTIC duplicates (close in embedding space — renamed vars,
reformatted whitespace, paraphrased comments, structurally-equivalent code) that byte/MinHash dedup
provably misses: embed → cluster → within-cluster cosine → keep one representative per near-dup set.
cola-coder's three dedup layers (`data/dedup.py`: exact SHA-256, MinHash-Jaccard, SoftDedup reweight)
are all surface/lexical — no semantic layer exists.

**Original idea / hypothesis (cola-coder-specific cross-technique) → DATA-069 (filed):**
**Verifier/quality-anchored semantic dedup.** Vanilla SemDeDup keeps the centroid-distant point (pure
geometry). cola-coder owns (1) its OWN trained model as a free code-embedder (`get_hidden_states` +
mean-pool — already used by `depth_profile.py`) and (2) quality scorers / SoftDedup mass. Within each
near-dup semantic cluster, keep the HIGHEST-QUALITY member (rolling dropped members' SoftDedup weight
into it) — a quality-aware semantic coreset that fuses SemDeDup with the project's quality signal and
DATA-057's reweight-over-drop philosophy. Composes with `DecontaminationFilter` (clusters straddling an
eval problem flag rephrased leakage). Filed for a future cycle (prep-time, main-safe, `--dedup semantic`).

**Implemented this cycle (INFER-031 — eval/inference, main-safe):** logit-lens per-token DEPTH profiler.
`evaluation/depth_profile.py`: `logit_lens(model, input_ids)` projects EVERY block's hidden state
through the model's final norm + TIED output head (no new weights, reuses the `get_hidden_states`
block-iteration), `convergence_depth` (argmax-match or entropy≤tau), and `profile_depth` (mean/median
exit depth, frac-converged-by-depth curve, optional per-difficulty-tier `by_tier`). `scripts/depth_profile.py`
(`--mode {argmax,entropy} --tau --by-difficulty --json`) + an eval-menu "Depth / Early-Exit Profile"
entry. +17 tests (last-layer lens == real forward logits anchor; entropy monotonicity; determinism).
Sets up EVAL-032 (verifier-stratified depth map) and INFER-032 (opt-in early-exit decode, worktree).

## 2026-06-14 — Adaptive-depth / early-exit inference (rotate: architecture/inference)

**Sources (2025–2026):**
- *TIDE: Token-Informed Depth Execution for Per-Token Early Exit* — https://arxiv.org/html/2603.21365.
  Lightweight binary routers at checkpoint layers; exit at the first layer whose hidden state has
  converged (cosine ≥ 0.98 to final). NO retraining (calibrated on 2k samples, ~4 MB). 5.5–8.1%
  latency gains; 98–99% of decode tokens exit at intermediate layers, reasoning preserved.
- *Adaptive Layer-skipping in Pre-trained LLMs (FlexiDepth)* — https://arxiv.org/pdf/2503.23798.
  Per-token skip decisions; token difficulty is heterogeneous (boilerplate skips deep layers,
  rare/reasoning tokens need full depth). ~20–30% speedup at comparable HumanEval/MMLU/GSM8K.
- *Think Just Enough: Sequence-Level Entropy as a Confidence Signal* — https://arxiv.org/pdf/2510.08146.
  Training-free entropy gate to halt computation when confident.

**Summary:** LLMs spend uniform compute (all N layers) per token, but most next-token predictions
stabilize before the last layer. Early-exit measures per-token convergence depth via logit-lens
agreement, hidden-state cosine convergence, or prediction entropy — and depth-need is token-TYPE
dependent.

**Original idea / hypothesis (cola-coder-specific cross-technique) → INFER-031 / EVAL-032 (filed):**
**Verifier-stratified depth profiling.** No early-exit paper has an execution verifier; cola-coder
does. Cross a logit-lens depth probe (reusing `Transformer.get_hidden_states` + the TIED output head
— zero new weights) with the sandbox verifier + EVAL-026 difficulty tiers: record per-token
convergence depth, stratify by token region (boilerplate vs the asserted/correctness-critical line)
and by verifier-effort tier. Hypothesis no public harness can test: the tokens that DECIDE functional
correctness converge later / at higher entropy than boilerplate — a correctness-grounded depth map
that could later justify a safe per-token early-exit floor (INFER-032, worktree). Filed for a future
cycle (main-safe analysis module `evaluation/depth_profile.py` + script + eval-menu entry).

**Implemented this cycle (EVAL-031 — eval, main-safe):** stratified robustness. `robustness_eval.py`
now accepts an injected `difficulty_tiers` map and reports `by_tier` (per verifier-effort tier:
n, robust_pass@1, consistency, bootstrap CI) plus an overall robust_pass@1 CI — all DRY on
`bootstrap_pass_at_k` (single-sample `ProblemResult`s, k=1) and `difficulty_profile.TIERS`.
`scripts/robustness_eval.py` gains `--by-difficulty` (per-tier table via `cli.kv_table`). +11 tests;
EVAL-030's 8 tests intact (back-compat). Closes the robustness→difficulty loop (answers "is the model
more fragile on harder problems, and is the drop credible?").

## 2026-06-14 — Robustness to input perturbation for code gen (rotate: safety/robustness)

**Sources (2025–2026):**
- *ReCode: Robustness Evaluation of Code Generation Models* — https://arxiv.org/abs/2212.10264.
  30+ semantically-preserving natural transforms over 4 families (docstrings, function names,
  code syntax, code format), >90% human-verified meaning-preserving; reports WORST-CASE
  robustness. Code models degrade markedly under benign rewording/formatting.
- *Evaluating Robustness of LLMs in Enterprise Applications: Perturbation Consistency* (2026) —
  https://arxiv.org/html/2601.06341. Five perturbation families + a consistency metric;
  positional reordering hurts most (−18 pts); training methodology beats parameter count for
  robustness (an 8B out-robusts a 120B).
- *PromptRobust: Evaluating Robustness of LLMs on Adversarial Prompts* —
  https://arxiv.org/abs/2306.04528. 4-level attacks (char/word/sentence/semantic), 4,788
  prompts; small benign perturbations cause large systematic accuracy drops → robustness must
  be measured, not assumed.

**Summary:** apply meaning-preserving transforms to a benchmark's INPUTS (for code: docstring
paraphrase/typo/whitespace/casing, reorder doctest examples) while keeping the task identical,
then measure quality drop. Headline = worst-case robust accuracy + a consistency rate. Public
harnesses grade "still correct" with surface proxies (CodeBLEU/embedding similarity) because
they lack ground truth.

**Original idea / hypothesis (cola-coder-specific cross-technique) → realized as EVAL-030:**
**Verifier-graded functional robustness ("robust pass@k").** Instead of CodeBLEU/similarity
proxies, grade robustness with cola-coder's SANDBOX VERIFIER (`runner.evaluate_solution`):
generate against the original docstring AND K meaning-preserving perturbations, then report
robust_pass@1 (worst perturbation, execution-graded), consistency (verdict invariance), and a
FRAGILITY map (problems solved clean but failing when merely reworded). Composes with EVAL-028
(bootstrap CI on robust_pass@1) and EVAL-026 difficulty tiers (is the model more fragile on
hard problems?) — a metric no public code-robustness harness can produce (none has an execution
verifier on the perturbed inputs).

**Implemented this cycle (EVAL-030 — eval, main-safe):** `evaluation/perturbations.py`
(`PerturbedProblem`, `perturb_docstring`, `perturb_problem_set`; kinds typo/whitespace/casing/
reorder_examples/paraphrase). HARD INVARIANT: only the docstring prose is mutated — the `def`
signature, `entry_point`, and `test_code` stay byte-identical, and every variant is re-AST-parsed
to confirm it still defines `entry_point` (task-changing perturbations are skipped). Plus
`evaluation/robustness_eval.py` (`RobustnessReport`, `evaluate_robustness(generate_fn, ...)`
reusing the verifier + `bootstrap_pass_at_k` for the CI), `scripts/robustness_eval.py`
(`--checkpoint/--config/--kinds/--problems/--ci`, loads via `load_generator`), and an eval-menu
"Robustness Evaluation" entry (no orphan). +25 tests, ruff clean. EVAL-031 (stratify robustness
by difficulty tier + CI) and DATA-068 (robustness-driven paraphrase augmentation — worktree, train
data) filed for follow-up.

## 2026-06-14 — pass@k uncertainty quantification: bootstrap CIs (rotate: evaluation)

**Sources (2025–2026):**
- *Adding Error Bars to Evals: A Statistical Approach to LM Evaluations* — https://arxiv.org/abs/2411.00640.
  Treat each QUESTION as the random unit (independent Bernoulli), report CLT standard
  errors / CIs, use clustered SEs for templated questions, and PAIRED difference tests
  (variance of the difference) when comparing two models on the same set.
- *Towards More Rigorous Evaluations of Language Models* (ICLR 2025 Blogposts) —
  https://iclr-blogposts.github.io/2025/blog/towards-more-rigorous-llm-evals/.
  Build binomial CIs (Wilson/Clopper-Pearson), use Fisher/Wilcoxon for comparisons,
  account for template-level correlation.
- *Don't Pass@k: A Bayesian Framework for LLM Evaluation* — https://arxiv.org/abs/2510.04265.
  pass@k is a bare point estimate; few samples mislead pass@k RANKINGS. Posterior +
  credible intervals separate credible differences from noise.
- *Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred* —
  https://arxiv.org/pdf/2503.01747. Below a few hundred items the normal approximation
  is unreliable → prefer BOOTSTRAP percentile intervals (Efron) or exact binomial.

**Summary:** a pass@k number without an interval is unfalsifiable — you can't tell a
real gain from sampling noise, and rankings flip with few samples. The random unit is
the PROBLEM (pass@k is a mean of per-problem unbiased estimates), so report the spread
across problems. On cola-coder's 62-problem HumanEval set the CLT is shaky, so a
bootstrap over problems (resample problems w/ replacement, recompute aggregate pass@k,
take percentiles) is the right, assumption-free recipe. For comparing two checkpoints
on the SAME set, a PAIRED bootstrap on per-problem differences cancels difficulty and
is far tighter than two independent CIs.

**Original idea / hypothesis (cola-coder-specific cross-technique) → IDEA-028:**
**Verifier-effort-STRATIFIED pass@k with per-tier bootstrap CIs.** cola-coder uniquely
owns a *model-relative* difficulty signal — the verifier-effort tier from
`evaluation/difficulty_profile.py` (easy/medium/hard/unsolved, from how much adaptive
best-of-N budget a verified solve consumed). Bootstrap WITHIN strata then combine: a
stratified/Rao-Blackwellized variance-reduction estimator that is tighter than a flat
bootstrap (most pass@k variance comes from mixing easy+hard problems) AND gives per-tier
error bars no public harness can build (none has the sandbox verifier's difficulty signal).

**Implemented this cycle (EVAL-028 — eval, main-safe):** uncertainty quantification for
pass@k in `evaluation/metrics.py`. Three stdlib-only functions (DRY on the existing
unbiased `pass_at_k`): `pass_at_k_stderr` (cross-problem SE), `bootstrap_pass_at_k`
(`(point, lo, hi)` percentile CI, deterministic by seed, `None` when n<k for every
problem — mirrors `compute_pass_at_k`), and `paired_bootstrap_delta` (B−A paired
bootstrap; CI excluding 0 ⇒ credible change). `format_results` now renders
`pass@1: 42.0% [95% CI 35.1–49.2]`; `scripts/evaluate.py` gains `--no-bootstrap` / `--ci`.
+10 tests (SE=0 on identical problems, CI brackets point, narrower CI with more problems,
seed determinism, not-estimable→None, paired spans-0 for identical models, paired
excludes-0 when B dominates, format renders/omits CI). EVAL-029 (wire paired bootstrap
into compare_models.py + regression gating) and IDEA-028 (stratified) filed for follow-up.

## 2026-06-14 — Secure code generation (rotate: safety) + BUG-128 fix

Sources:
- On Fixing Insecure AI-Generated Code (fine-tuning + prompting) — https://arxiv.org/html/2605.05867v1
- Security Vulnerabilities in AI-Generated Code: large-scale GitHub analysis — https://arxiv.org/abs/2510.26103
- SecureCode: multi-turn dataset for security-aware codegen — https://arxiv.org/html/2512.18542
- Constitutional Spec-Driven Development (security by construction) — https://arxiv.org/html/2602.02584

Findings:
- **~45% of AI-generated code fails security tests** (86% vulnerable to XSS, 88% to log injection)
  across GPT-5/Gemini/DeepSeek etc. on CWE-Top-25 scenarios — AI code is measurably less secure.
  Reinforces the project's screen-everything stance (scan_dangerous, secure-pass@k, RFT security gate).
- **Mitigations that work:** negative-example prompting, CoT, and FINE-TUNING on secure exemplars
  (LoRA). cola-coder's RFT (MODEL-045) already enforces a security gate on kept samples — so its
  self-distilled SFT data is secure-by-construction, a fine-tuning mitigation by default.
- **Constitutional / spec-driven** security embeds CWE constraints into the spec layer — but those
  specs are themselves a prompt-injection surface (26% of agent skills had exploitable vulns), tying
  back to SEC-019 (the project's injection scanner on retrieved content).

**Implemented this cycle (BUG-128 — user-reported crash, main-safe):** the user tried interactive
generation from the live `small_react_best` checkpoint (dim=768) but the menu passed
`configs/tiny.yaml` (dim=512) → `load_state_dict` size-mismatch crash; they couldn't generate from
their own checkpoint. Fix: a checkpoint is GROUND TRUTH for its architecture. New
`apply_model_config_from_checkpoint(config, checkpoint)` (inference/loading.py) reads the saved
`metadata.json` config and overrides `config.model`'s scalar arch fields (dim/layers/heads/kv/ffn/
vocab/max_seq_len/rope_theta/qk_norm/...) before the model is built — so a wrong `--config` can't
build a mismatched model. Wired into `generate.py` AND the central `load_generator` (so serve /
smoke_test / RFT are all robust). +5 tests incl. one that reproduces the exact crash against the
REAL checkpoint metadata (tiny dim 512 → corrected 768). Reads checkpoint metadata only (no
checkpoint.py change) → main-safe. ruff green.

**ORIGINAL cross-technique idea (no new id needed — folds into SEC backlog):** since RFT already
gates on the security scanner, the REJECTED-for-insecurity completions are a free corpus of the
model's OWN insecure patterns — pair each with the verifier-passed secure sibling for the same prompt
to build (insecure→secure) contrastive SFT pairs (the fine-tuning mitigation the research validates),
exactly IDEA-016's adversarial secure-FIM but sourced from real self-generated failures. Strengthens
IDEA-016 with RFT-sourced data.

---

## 2026-06-14 — Iterative self-improvement (STaR/ReST) (rotate: data curation)

Sources:
- Think, Prune, Train, Improve: scaling reasoning without scaling models — https://arxiv.org/pdf/2504.18116
- B-STaR: balancing exploration/exploitation in self-taught reasoners — https://arxiv.org/pdf/2412.17256
- STaR: bootstrapping reasoning with reasoning — https://openreview.net/pdf?id=_3ELRdg2sgI
- AdaSTaR (adaptive sampling for self-taught reasoners) — referenced in the STaR survey

Findings:
- **The STaR/ReST loop = a data FLYWHEEL:** (i) sample multiple solutions, (ii) FILTER to the
  ones a verifier/ground-truth accepts, (iii) SFT/RFT on the filtered set, repeat. The filtered
  self-generated data is the curation engine — this cycle wired its single-round form (MODEL-046).
- **AdaSTaR (2026): don't sample uniformly.** Prioritize STALE and HARD instances via a
  MinHeap + curriculum weighting so harder samples dominate as accuracy rises → ~58.6% FLOPs
  reduction. This independently confirms last cycle's IDEA-027 (verifier-effort-curriculum RFT) —
  the project's verifier-effort (EVAL-026) is exactly the difficulty signal AdaSTaR needs.
- **B-STaR:** monitor + balance exploration vs exploitation across rounds — ties to the project's
  entropy controller (IDEA-013) and verifier-calibrated escalation (INFER-029).

**Implemented this cycle (MODEL-046 — tooling, main-safe):** the runnable RFT front-end. New
`scripts/generate_rft_data.py` loads a checkpoint via `load_generator` (DRY), pulls prompts (built-in
problems OR a `--jsonl` of `{prompt, test_code}`), runs the MODEL-045 `generate_rft_dataset` (best-of-N
→ keep verified+secure), and writes ChatML SFT JSONL with a stats summary. Wired into the Training
menu → Post-Training group (`_generate_rft_menu`). The self-improvement engine is now usable end-to-end:
generate RFT data → `train_sft.py` → repeat. Tests: +2 (JSONL prompt loading incl. missing-test→None,
built-in max_prompts); 7 RFT tests + ruff green. Loading needs a checkpoint (GPU) so it runs between
training runs or on the spare GPU.

**ORIGINAL cross-technique idea (DATA-067): verified-RFT data as a quality-classifier teacher.** The
RFT pipeline produces a growing pool of VERIFIER-PASSED (objectively correct, secure) completions and,
implicitly, the REJECTED ones (failed tsc/tests or insecure). That's a free, perfectly-labelled
binary corpus — feed BOTH classes into `train_judge_classifier` / the quality classifier so it learns
"what verifier-passing TS looks like" from ground truth, not LLM-judge opinion (DATA-062/IDEA-025).
Each RFT round both improves the model AND sharpens the data-quality scorer that curates the NEXT
pretraining batch — a compounding flywheel where the verifier's labels propagate into the quality-
weights pipeline. Combines RFT + verifier + quality classifier + quality weights. Builds on MODEL-045/046
+ DATA-062 + train_judge_classifier. Data → main-safe. → DATA-067.

---

## 2026-06-14 — Rejection-sampling fine-tuning / self-verified distillation (rotate: post-training)

Sources:
- Step Rejection Fine-Tuning: a practical distillation recipe — https://arxiv.org/abs/2605.10674
- Self-Verified Distillation: your LM is its own synthetic data pipeline — https://arxiv.org/abs/2605.26132
- RL on Incorrect Synthetic Data scales LLM efficiency — https://arxiv.org/pdf/2406.14532
- Inference-Aware Fine-Tuning for Best-of-N — https://arxiv.org/pdf/2412.15287

Findings:
- **RFT = the model's own VERIFIED output is the training data.** Sample N candidates, keep only the
  ones that pass an EXTERNAL verifier, SFT on those, repeat. No teacher, no human labels — a
  self-improvement loop. Larger verification budget (more candidates) → higher-quality self-curated
  data. cola-coder owns every piece (best-of-N + sandbox verifier + SFT) but had no harness wiring
  them together — the obvious missing self-improvement engine.
- **Self-Verified Distillation (2605.26132):** "your LM is secretly its own synthetic data pipeline"
  — exactly the project's vision (best-of-N verified generation feeding SFT).
- **Step-RFT (2605.10674):** instead of discarding failed trajectories whole, keep the correct
  PREFIX — a future refinement once partial-credit (MODEL-041) data is available.

**Implemented this cycle (MODEL-045 — distillation, main-safe):** the RFT / self-verified
distillation harness `distillation/rft.py::generate_rft_dataset`. For each prompt it generates
``num_candidates`` completions, verifies + ranks them with the existing `generate_best_of_n` (sandbox
tsc/tests + security screen + self-consistency), and keeps the best ONLY if it passed the hard
verifier (``keep_only_verified``) AND is secure (``require_secure``) — emitting ChatML SFT records
(reusing `_to_messages`, DRY). Self-distillation (student's own verified output), the complement to
the teacher-based `generate_distillation_dataset`; never executes model output itself (all execution
in the sandboxed verifier). Exported + +5 tests (keep verified+secure, drop unverified, keep-when-off,
drop insecure, length guard); distillation regression (11) + ruff green. FOLLOW-UP: a CLI script +
menu entry (a `--self-rft` mode) → MODEL-046.

**ORIGINAL cross-technique idea (IDEA-027): verifier-effort-curriculum RFT.** Plain RFT samples every
prompt uniformly, wasting budget on already-trivial and currently-impossible prompts. Use EVAL-026's
verifier-EFFORT to self-curate RFT's curriculum: after a round, classify each prompt by how much
best-of-N budget its verified solve took (or whether it stayed unsolved), and in the NEXT round
CONCENTRATE sampling on the FRONTIER prompts (solved-but-hard / just-barely-unsolved) while retiring
trivially-solved ones (E2H, MODEL-042) and shelving the impossible — a self-curating RFT that spends
its verification budget where learning actually happens. Combines RFT + best-of-N + verifier-effort
(EVAL-026) + E2H curriculum (MODEL-042) — a self-improvement loop no RFT paper (no verifier-effort
difficulty signal) can build. Builds on MODEL-045 + EVAL-026 + MODEL-042. Distillation → main-safe. → IDEA-027.

---

## 2026-06-14 — Inference serving efficiency (rotate: inference/decoding)

Sources:
- PagedAttention / vLLM (efficient KV-cache memory) — https://arxiv.org/pdf/2309.06180
- KV Cache Optimization Strategies survey (2026) — https://arxiv.org/abs/2603.20397
- Persistent Q4 KV cache for multi-agent edge inference — https://arxiv.org/pdf/2603.04428
- vLLM vs TGI performance study — https://arxiv.org/html/2511.17593

Findings:
- **KV-cache is the serving bottleneck.** PagedAttention (non-contiguous KV blocks) + continuous
  batching give vLLM 10-24× throughput over static batching/TGI; quantization (GPTQ/AWQ + Q4 KV)
  shrinks weights/cache 4-8× at minimal quality loss. The project's KV-cache is simple/contiguous —
  fine for single-user 4080 serving, not datacenter scale (out of scope now).
- **No single KV technique dominates** — the survey maps techniques to SCENARIOS (long-context,
  high-throughput, edge, multi-turn, accuracy-critical). cola-coder serves TWO regimes: latency-
  critical inline FIM and throughput-oriented best-of-N — which want different trade-offs.
- **Q4 KV cache** is viable on edge with minimal accuracy loss — relevant to fitting more best-of-N
  candidates in the 4080's 16 GB.

**Implemented this cycle (INFER-030 — inference/server, main-safe):** completed the top-nσ
(INFER-028) exposure so the 2026 sampler is usable end-to-end from the IDE. Threaded `top_n_sigma`
through `generator.generate_stream` (the chat SSE path) and added a `top_n_sigma: float = 0.0` field
to all four server request models (Generate / ChatCompletion / Completion / FIM), forwarded at all 6
`generate`/`generate_stream` call sites. +3 regression tests (4 fields present, all 6 sites forward
it via a no-silent-no-op regex guard, generator methods accept it); 10 server tests + ruff green.
NOTE: best-of-N (`_best_of_generate`) uses its own batched sampling and does NOT yet thread top-nσ
(documented; → INFER-031 follow-up to thread it through generate_group).

**ORIGINAL cross-technique idea (IDEA-026): verifier-gated KV-cache precision.** The two serving
regimes want opposite KV trade-offs, and the VERIFIER tells you which is safe. For best-of-N
(throughput, accuracy-TOLERANT because the sandbox verifier filters bad candidates anyway), use a
Q4-quantized KV-cache so MORE candidates fit in the 4080's 16 GB → a larger best-of-N budget for the
same VRAM; for latency/accuracy-critical inline FIM (single shot, no verifier downstream), keep
full-precision KV. The verifier makes Q4's slight quality loss free for best-of-N (it discards the
losers), converting saved KV memory into more verified samples — a precision/budget trade no
KV-quant paper makes because none has a downstream verifier. Builds on the best-of-N verifier +
generate_group KV-cache. Inference → main-safe (quantization is opt-in per request path). → IDEA-026.

---

## 2026-06-14 — LLM-as-judge reliability & calibration (rotate: evaluation)

Sources:
- Noisy but Valid: robust statistical eval of LLMs with imperfect judges (ICLR 2026) — https://arxiv.org/html/2601.20913v1
- Bias in the Loop: auditing LLM-as-a-Judge for software engineering — https://arxiv.org/html/2604.16790v1
- Benchmarking LLM-as-a-Judge for long-form output — https://arxiv.org/html/2606.01629v1

Findings:
- **LLM-judges are biased + noisy** (verbosity, position, authority cues; poor test-retest
  reliability) — and that noise can INVALIDATE statistical guarantees. For CODE specifically the
  consensus is that EXECUTION-BASED verification beats LLM-judging where available.
- **Noisy-but-Valid calibration:** use a small ground-truth-labelled set to estimate the judge's
  TPR/FPR, then variance-correct the reported rate (Rogan-Gladen prevalence correction) so the
  judge's bias doesn't inflate measured quality — with finite-sample Type-I error guarantees.
- **cola-coder's edge:** it OWNS bias-free ground truth for code — the sandbox verifier (tsc/tests).
  So it can calibrate its LLM-judge against the verifier instead of needing human labels, which
  most projects can't.

**Implemented this cycle (EVAL-027 — eval, main-safe):** verifier-anchored judge calibration
`evaluation/judge_calibration.py`. `agreement_stats(judge_pass, verifier_pass)` → the judge's
TPR/FPR/accuracy/Cohen's κ using the verifier as the oracle; `corrected_prevalence(observed, tpr,
fpr)` → Rogan-Gladen recovery of the TRUE pass-rate from the judge's noisy rate (so a verbosity-
biased judge can't inflate corpus quality); `best_score_threshold(judge_scores, verifier_pass,
metric)` → the judge-score cut-point that best matches the verifier (accuracy or Youden's J). Pure
logic → runs/tests with no GPU. +12 tests (confusion, perfect/biased judge, TPR-undefined, Rogan-
Gladen recovery + clamping + no-signal, threshold by accuracy/Youden, empties); ruff clean.

**ORIGINAL cross-technique idea (IDEA-025): verifier-recalibrated judge distillation.**
`train_judge_classifier` currently distills the LLM-judge's (biased) scores into a local TF-IDF
classifier — baking the judge's verbosity/position bias into the project's permanent quality signal.
Instead, on a calibration set scored by BOTH the judge and the sandbox verifier (EVAL-027), use
`best_score_threshold` + `corrected_prevalence` to RECALIBRATE the judge's labels to verifier
ground truth BEFORE distilling — and where the verifier can run, distill the VERIFIER's label
directly (DATA-062). The distilled classifier then learns verifier-grounded quality, not an LLM's
biased opinion — debiasing the project's whole data-scoring stack with assets the eval papers
(human-label calibration only) lack. Builds on EVAL-027 + train_judge_classifier + the verifier +
DATA-062. Data/eval → main-safe. → IDEA-025.

---

## 2026-06-14 — GRPO entropy-control recipes (rotate: post-training/RLVR)

Sources:
- Rethinking Exploration in RLVR (bidirectional entropy modulation) — https://arxiv.org/html/2604.04894v1
- Compress the Easy, Explore the Hard: difficulty-aware entropy regularization — https://arxiv.org/pdf/2602.22642
- SCOPE-RL: stable quantitative control of policy entropy — https://arxiv.org/html/2510.08141
- Rethinking Entropy Regularization in Large Reasoning Models — https://arxiv.org/pdf/2509.25133

Findings:
- **Entropy collapse is THE GRPO failure mode** (entropy decreases monotonically → premature loss
  of exploration). 2026 work splits into: objective-level regulators (SCOPE-RL temperature-adaptive
  control), recipe-level heuristics (advantage shaping, Pass@k), and SELECTIVE regularization (SIREN:
  restrict entropy to the top-p nucleus + peak-entropy tokens; self-anchor to the initial level).
- **Difficulty-aware entropy is a named frontier** ("Compress the Easy, Explore the Hard",
  2602.22642): regularize entropy MORE on hard problems, LESS on easy ones — independently confirming
  the project's IDEA-020 per-difficulty entropy floors built two cycles ago.
- **Verifier pass-rate gating is sound:** coupling exploration to the verifier (don't explore solved
  problems) is exactly the project's controller design; the literature is converging on it.

**Implemented this cycle (MODEL-044 — wiring, main-safe):** the project's entropy/curriculum stack
(MODEL-037 metric → IDEA-013 controller → IDEA-020 per-difficulty floors → MODEL-042 E2H scheduler)
was built with opt-in constructor params but was UNREACHABLE from the CLI — implemented-but-dead.
Wired into `scripts/train_reasoning.py`: `--entropy-control` (+ `--entropy-target`) constructs an
`EntropyClipController` (per-difficulty floors auto-enabled with `--curriculum`) passed to the
trainer; `--e2h` constructs a `VerifierEffortCurriculum` passed to `.train()`. CLI-only (no
phantom-config risk). +4 regression tests assert the trainer accepts the kwargs AND the script
forwards them (AST check) — guarding against the features silently un-wiring. config-wiring + ruff green.

**ORIGINAL cross-technique idea (IDEA-024): verifier-localized entropy injection.** SIREN restricts
entropy regulation to the nucleus / peak-entropy tokens; cola-coder can localize it by CORRECTNESS
instead. The MODEL-037 entropy metric currently averages over ALL completion tokens; combine it with
IDEA-023's execution-trace token map (which tokens are in the failing assert's region) so the entropy
FLOOR is enforced selectively on the tokens that produced the FAILURE — inject exploration exactly
where the verifier says the model is wrong, and exploit (low entropy) where it's already passing.
Difficulty-aware (IDEA-020) × token-localized (SIREN) × verifier-grounded (IDEA-023) entropy control —
a combination no entropy paper (no execution verifier) can do. Builds on MODEL-037 + IDEA-013 +
IDEA-023. Reasoning-only → main-safe. → IDEA-024.

---

## 2026-06-14 — Benchmark decontamination (rotate: data curation)

Sources:
- A Survey on Data Contamination for LLMs — https://arxiv.org/html/2502.14425v2
- Rethinking Benchmark and Contamination with Rephrased Samples — https://arxiv.org/abs/2311.04850
- Benchmarks Should Be Contamination-Resistant — https://arxiv.org/html/2605.19999v1
- lm-sys llm-decontaminator — https://github.com/lm-sys/llm-decontaminator

Findings:
- **N-gram / containment overlap is the standard decontamination screen** — index the eval set,
  drop training docs that contain a benchmark problem. Cheap, high-precision for VERBATIM and
  near-verbatim leakage. The project already had a `DataLeakageDetector` (MinHash + containment)
  but only as an OFFLINE REPORT (scripts/check_contamination.py) — it never DROPPED contaminated
  samples during prep. The mitigation half was missing.
- **N-gram MISSES rephrased leakage** (2311.04850): a paraphrased/reformatted benchmark evades
  exact overlap → needs embedding/LLM detection (the llm-decontaminator approach). So n-gram is a
  necessary first layer, not sufficient.
- **Future benchmarks should be contamination-RESISTANT by construction** (time-segmented /
  generated-fresh) — aligns with the project's EVAL-023/IDEA-007 contamination-free eval plans.

**Implemented this cycle (DATA-065 — data filter, main-safe):** `DecontaminationFilter`
(data/filters/decontamination.py), a registered FilterPlugin that REUSES the leakage detector's
`_shingles`/`_containment` (DRY) to DROP training records overlapping eval benchmarks at prep time.
Indexes the eval/benchmark texts (explicit `eval_texts` and/or the built-in problem prompts via
`benchmark: true`), then drops any record whose containment of an eval doc's shingles ≥ threshold
(0.8 default) — the right metric for a short eval problem embedded in a larger scraped file. Opt-in;
no-op when no refs configured. Closes the report→mitigation gap. +6 tests (drops embedded benchmark,
keeps unrelated, no-op, threshold sensitivity, empty); filter registry intact, ruff clean.

**ORIGINAL cross-technique idea (DATA-066): verifier-confirmed semantic decontamination.** N-gram
misses REPHRASED leakage; the embedding/LLM approach is heavy. cola-coder can catch the most
DANGEROUS rephrased leakage — a functionally-equivalent rewrite of an eval problem — with its
verifier instead: for each training sample that imports/defines a function whose NAME matches an
eval problem's entry point (cheap candidate filter), run the EVAL's own tests against the training
sample's function in the sandbox; if they PASS, the sample is a functional copy of the benchmark
(even if reworded) → drop it. Catches the rephrased leakage n-gram misses, using the sandbox
verifier + the benchmark's test suite — which no n-gram or embedding decontaminator has. Builds on
DATA-065 + the sandbox verifier + the benchmark problem/test sets. Prep-time → main-safe. → DATA-066.

---

## 2026-06-14 — Attention-logit & output stability (rotate: architecture)

Sources:
- Controlling changes to attention logits (param-dependent Q/K LRs) — https://arxiv.org/html/2511.21377
- Output Embedding Centering for Stable LLM Pretraining — https://arxiv.org/pdf/2601.02031
- Small-scale proxies for large-scale Transformer training instabilities — https://arxiv.org/pdf/2309.14322
- MERIT: max-normalized ratio for large-batch training — https://arxiv.org/pdf/2508.20577

Findings:
- **Two instability modes, both reproducible in SMALL models at high LR** (so cola-coder-relevant):
  (1) attention-LOGIT growth — uncontrolled max attention logit → divergence; QK-Norm mitigates
  (project has qk_norm=true). (2) output-LOGIT divergence from log-probs — z-loss mitigates
  (project has z_loss). Both project defaults are now externally corroborated as the right calls.
- **New 2026 levers:** (a) parameter-dependent LRs for Q/K weights control logit *changes* and let
  you raise the base LR — competitive with QK-Norm, composes with it; a future optimizer-side knob.
  (b) **Output Embedding Centering** — subtract the running mean from the output embedding to stop
  logit drift — a cheap alternative/complement to z-loss. Both are model/optimizer (train-path).
- **LR-sensitivity at small scale predicts large-scale stability** — small proxies are a valid
  cheap testbed (relevant to the project's tiny-config validation runs).

**Implemented this cycle (MODEL-042 — reasoning-only, main-safe):** verifier-effort E2H curriculum
scheduler, closing the curriculum thread (EVAL-026 → MODEL-042 → IDEA-020). New
`reasoning/curriculum_scheduler.VerifierEffortCurriculum` (pure logic): tracks each problem's
verified pass-rate across epochs; `tier_for` re-tags difficulty from the LATEST measured rate;
`is_mastered` (rate ≥ threshold for a streak); `active` fades mastered problems Easy→Hard while
keeping ≥ min_active (re-including the least-mastered to fill the floor). Wired opt-in into
`GRPOTrainer.train(e2h_scheduler=…)`: records per-step pass-rate, and between epochs re-tags
`problem["difficulty"]` (so the per-difficulty temperature + IDEA-020 entropy floors use MEASURED
difficulty) and drops mastered problems. Default None → unchanged. +10 tests; GRPO + reasoning
suites green, ruff clean. Difficulty is now measured, evolving, model-relative — what E2H/SEC require.

**ORIGINAL cross-technique idea (MODEL-043): output-embedding centering, verifier-A/B'd.** Bring
the 2026 output-logit-stability lever (2601.02031) to cola-coder, but VALIDATE it the way only this
project can. Add opt-in output-embedding centering (subtract the running mean of `output.weight`'s
column means from logits, or center the tied embedding) as a checkpoint-safe flag (no new params),
then A/B it on a tiny-config Muon run measuring BOTH loss AND downstream secure-pass@k (EVAL-024) /
verifier-effort difficulty (EVAL-026) — not just perplexity, which the papers stop at. Composes with
z_loss + qk_norm. Train-path (model/) → implement in a WORKTREE, A/B before merge; pairs with the
MODEL-025 Muon A/B. Builds on the existing z_loss/qk_norm stability stack + the verifier. → MODEL-043.

---

## 2026-06-14 — Curriculum RL & difficulty estimation (rotate: post-training)

Sources:
- Curriculum RL from Easy to Hard improves LLM reasoning (E2H) — https://arxiv.org/abs/2506.06632
- Self-Evolving Curriculum for LLM Reasoning (bandit) — https://arxiv.org/abs/2505.14970
- Beyond Random Sampling: curriculum LM pretraining — https://arxiv.org/abs/2506.11300

Findings:
- **Easy→Hard scheduling helps, but FADE OUT easy tasks.** E2H gives convergence guarantees and
  shows that keeping easy tasks too long causes overfitting — the schedule must retire mastered
  tiers. The project's GRPO curriculum is static (fixed difficulty tags); it has no measured,
  evolving difficulty.
- **Curriculum needs a difficulty METRIC.** SEC learns the curriculum as a non-stationary
  multi-armed bandit; pretraining curricula use 6 linguistic/info-theoretic difficulty metrics.
  All require a difficulty signal — and cola-coder has a unique one: the sandbox VERIFIER's effort.
- **Difficulty is model-relative + non-stationary:** what's hard changes as the model learns, so a
  static label is wrong; measure it from the model's own verified-solve effort.

**Implemented this cycle (EVAL-026 — eval/inference, main-safe):** verifier-effort difficulty
profiling. `BestOfNResult` now carries `candidates_used / rounds / final_temperature / solved`
(populated by both fixed-N and adaptive best-of-N). New `evaluation/difficulty_profile.py`:
`verifier_effort_tier(candidates_used, max_candidates, solved)` → easy/medium/hard/unsolved by how
much of the adaptive budget a verified solve consumed; `profile_difficulty(results, max_candidates)`
→ a difficulty-distribution report (per-tier counts, solve-rate, mean candidates). A free,
objective, MODEL-RELATIVE difficulty label from the verifier — no human annotation. Pure logic →
runs/tests with no GPU/sandbox. +10 tests; 40 best-of-N green, ruff clean. Off the train path → main.

**ORIGINAL cross-technique idea (MODEL-042): verifier-effort E2H scheduler.** Close eval→curriculum
with measured difficulty. Each curriculum epoch, RE-CLASSIFY every problem's tier from its current
verifier-effort (EVAL-026) — not a static tag — then apply an E2H schedule: train easy→hard and
FADE OUT problems that have become "easy" (the model mastered them, E2H's anti-overfitting rule),
promoting freed budget to the frontier (mid pass-rate) problems. The per-tier mix feeds IDEA-020's
per-difficulty entropy floors. Difficulty is thus measured, evolving, and model-relative — exactly
what E2H/SEC call for — using the project's verifier + adaptive best-of-N + curriculum + entropy
controller together, which no curriculum paper (no execution verifier) can. Builds on EVAL-026 +
IDEA-020 + the GRPO curriculum. Reasoning-only → main-safe. → MODEL-042.

---

## 2026-06-14 — Package hallucination / slopsquatting (rotate: safety)

Sources:
- We Have a Package for You! Analysis of package hallucinations (USENIX Security 2025) — https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-742-spracklen.pdf
- Importing Phantoms: measuring LLM package hallucination — https://arxiv.org/pdf/2501.19012
- When LLMs Invent Rust Crates (Internetware 2026) — https://arxiv.org/html/2606.08444v1
- AI-generated packages → slopsquatting (DevOps.com) — https://devops.com/ai-generated-code-packages-can-lead-to-slopsquatting-threat-2/

Findings:
- **Code models hallucinate package names at 5.2% (commercial) – 21.7% (open) rates** across
  576k samples / 16 models — 205k unique fabricated names, ~45% PERSISTENT across queries
  (so reliably exploitable). Attackers register the hallucinated name on PyPI/npm with malware:
  the model's invented `import foo` becomes an install-time compromise ("slopsquatting"). Real:
  a hallucinated `huggingface-cli` PyPI package got 30k+ installs.
- **AI code is measurably less safe:** ~15-18% more vulnerabilities than human code; ~1.7× more
  issues in AI-co-authored PRs — reinforces screening generated code, not trusting it.
- **Defense = validate imports against a known-good set** before surfacing/executing generated
  code; persistence makes an allowlist screen effective.

**Implemented this cycle (SEC-020 — security/inference, main-safe):** hallucinated-import scanner
`security/import_scanner.py`. `extract_imports`/`scan_unknown_imports`/`has_unknown_imports` pull
the imported package ROOTS from generated/scraped code (Python via AST — relative imports
excluded, submodules → root; JS/TS via regex — scoped `@scope/name` kept, relative paths excluded;
regex fallback for unparseable partials) and flag any NOT in a curated allowlist (Python stdlib via
`sys.stdlib_module_names` + popular PyPI by IMPORT name; node builtins + popular npm). Wired as a
NON-ranking review signal into best-of-N candidate `details["unknown_imports"]` (a niche-but-real
import is common, so it must not down-rank a verified candidate). Mirrors code_patterns.py. +15
tests; 40 best-of-N green, ruff clean. Off the train path → committed on main.

**ORIGINAL cross-technique idea (SEC-021): verifier-confirmed import quarantine.** An unknown
import is ambiguous (niche-real vs hallucinated). cola-coder can DISAMBIGUATE with assets the
scanners-alone lack: in best-of-N, when a candidate imports an unknown package, attempt to verify
it in the sandbox WITHOUT network (the project's default no-net sandbox) — a hallucinated package
fails to import (ModuleNotFoundError), a real-but-uninstalled one also fails but differently from a
typo of an installed one. Combine: down-rank candidates whose unknown imports fail sandbox import
resolution, and feed confirmed-hallucinated names into a deny-set that the GRPO security penalty
(IDEA-008) discourages — so the model is RL-trained away from inventing packages. Exploits the
no-net sandbox + best-of-N + GRPO penalty together. Builds on SEC-020 + the sandbox + IDEA-008. → SEC-021.

---

## 2026-06-14 — Compute-optimal test-time scaling (rotate: evaluation)

Sources:
- What If We Allocate Test-Time Compute Adaptively? — https://arxiv.org/pdf/2602.01070
- Budget-aware Test-time Scaling via Discriminative Verification — https://arxiv.org/pdf/2510.14913
- Sample, Scrutinize and Scale: scaling verification — https://arxiv.org/pdf/2502.01839
- Scaling test-time compute optimally > scaling params (OpenReview) — https://openreview.net/forum?id=4FWAwZtd2n

Findings:
- **Compute-optimal scaling is DIFFICULTY-dependent.** The best test-time strategy varies by
  prompt difficulty; "compute-optimal" allocation gives easy prompts few samples and hard ones
  many — exactly what the project's adaptive best-of-N (IDEA-009) does, gated by the verifier.
- **Don't just resample a FIXED distribution.** 2026 work moves to MODIFYING the sampling
  distribution (evolving conditioning, temperature/rectification) rather than drawing more from
  the same one — re-sampling rarely escapes a systematic failure.
- **Discriminative verifiers are budget-efficient:** under a fixed budget, a discriminative
  checker (does it pass?) beats costly generative verification — cola-coder's sandbox tsc/tests
  IS a discriminative verifier, so its best-of-N is already on the efficient side.

**Implemented this cycle (INFER-029 — inference, main-safe):** verifier-calibrated temperature
escalation in adaptive best-of-N. New `temperature_growth` (default 1.0 = off) + `max_temperature`:
each round the sandbox verifier rejects the WHOLE batch, the next round's temperature is multiplied
by `temperature_growth` (capped) so retries WIDEN the distribution instead of resampling the same
failure — the inference-time analogue of IDEA-013's RL entropy controller, and the "modify the
distribution" lesson above. Backward-compatible (growth 1.0 → fixed temp). +5 tests (escalates on
reject, capped, default-off, no-escalation-after-early-stop, invalid-config); 40 best-of-N green, ruff clean.

**ORIGINAL cross-technique idea (EVAL-026): verifier-effort difficulty profiling.** The adaptive
best-of-N already records HOW MUCH compute each prompt needed (candidates used + escalated
temperature) before the verifier was satisfied. That "verifier effort" is a continuous,
MODEL-RELATIVE difficulty label — free, objective, no human annotation. Use it two ways: (a) an
eval report stratified by verifier-effort tier (pass@k for easy/medium/hard-as-the-model-sees-it),
surfacing where the model is weak; (b) feed those labels back as the GRPO curriculum's difficulty
tiers (which IDEA-020's per-difficulty entropy floors already consume) — closing eval→curriculum
with the model's own measured difficulty rather than static heuristics. Exploits verifier + adaptive
best-of-N + curriculum. Eval/curriculum → main-safe. Builds on IDEA-009 + INFER-029 + IDEA-020. → EVAL-026.

---

## 2026-06-14 — Sampling for code generation (rotate: inference/decoding)

Sources:
- Top-nσ: not all logits are you need — https://arxiv.org/pdf/2411.07641
- Min-p sampling (creative + coherent outputs) — https://arxiv.org/pdf/2407.01082
- Hot or Cold? Adaptive temperature sampling for code — https://arxiv.org/pdf/2309.02772
- Non-determinism of "deterministic" LLM settings — https://arxiv.org/html/2408.04667v5

Findings:
- **Top-nσ is the 2026 truncation sampler** (now in the llama.cpp chain): keep tokens with
  `logit >= max_logit − n·σ` over the RAW pre-softmax logits. Because the keep-region is defined
  by the logit distribution's own spread, it's TEMPERATURE-INVARIANT — raising temperature for
  diversity doesn't drag in the noisy tail the way top-p/min-p do. Strong fit for code, where you
  want exploration without syntactic garbage. The project had top-k/top-p/min-p but not top-nσ.
- **Adaptive temperature for code (2309.02772):** vary temperature by token uncertainty — low
  where syntax is forced, high where logic is open. Future knob (→ idea below).
- **Determinism caveat:** even temp=0 isn't bit-deterministic across kernels/batch — report
  sampler settings with eval numbers (ties to EVAL harness versioning).

**Implemented this cycle (INFER-028 — inference, main-safe):** top-nσ sampling. New
`_top_n_sigma_filter` / `_top_n_sigma_filter_batch` in inference/sampling.py (keep logit ≥ max −
n·σ; computed over FINITE logits so an upstream n-gram ban can't corrupt σ; σ=0 → no-op). Applied
on raw logits BEFORE temperature (its temperature-invariance is the whole point), wired as
`top_n_sigma` through `sample_next_token`, `sample_next_tokens_batch`, and `generator.generate`
(default 0.0 = off → unchanged). +7 tests (within-σ keep, larger-n keeps more, ignores -inf,
constant no-op, batch per-row, end-to-end, default-off); 46 inference/sampling tests green, ruff clean.

**ORIGINAL cross-technique idea (INFER-029): verifier-calibrated adaptive sampling.** Combine
adaptive-temperature (2309.02772) with cola-coder's verifier. In adaptive best-of-N, START low-
temperature / tight top-nσ (exploit) and, only when the sandbox verifier keeps REJECTING the batch,
RAISE temperature and LOOSEN n·σ to widen exploration — a verifier-driven sampling schedule, the
inference-time analogue of the IDEA-013 entropy controller (which does this for RL training). The
verifier closes the loop: spend diversity only where correctness is unmet. Pairs adaptive sampling
(top-nσ + temperature) with the sandbox verifier + adaptive best-of-N (IDEA-009). Inference → main-safe. → INFER-029.

---

## 2026-06-14 — RLVR reward design & credit assignment (rotate: post-training/RLVR)

Sources:
- Execution-Grounded Credit Assignment for GRPO in code (ICLR 2026 SPOT) — https://arxiv.org/html/2603.16158
- JustRL: scaling a 1.5B LLM with a simple RL recipe (ICLR 2026) — https://iclr-blogposts.github.io/2026/blog/2026/justrl/
- Group Relative Reward Rescaling (length without trade-offs) — https://arxiv.org/html/2603.10535
- GRPO++ tricks — https://cameronrwolfe.substack.com/p/grpo-tricks

Findings:
- **Binary pass/fail is coarse credit.** RLVR optimizes unit-test pass rate, but GRPO with an
  all-or-nothing reward suffers poor credit assignment (a 4/5-test solution looks identical to
  0/5). EGCA localizes credit using EXECUTION TRACES; the cheap first step is FRACTIONAL test
  reward (fraction of cases passed) — denser signal, fewer zero-variance groups.
- **Avoid additive penalties:** additive length penalties create reward-hacking shortcuts;
  multiplicative rescaling (GR3) is preferred. Response length tends to converge naturally
  without explicit penalties — so for a code model, don't bolt on a length penalty.
- **Simple recipes scale (JustRL):** a stable, minimal GRPO recipe beats over-engineered ones at
  1.5B — favors the project's lean GRPO over feature creep.

**Implemented this cycle (MODEL-041 — reasoning-only, main-safe):** fractional/partial-credit
Python reward. The default `python_exec` runs the whole test block as one unit (1.0 all-pass /
0.0 else). New `reasoning/rewards/partial_credit.py` splits the block into individual `assert`
cases via the AST (keeping non-assert top-level statements as shared SETUP prepended to every
case) and returns the FRACTION passed; falls back to binary when there are no top-level asserts.
Registered as a new opt-in reward `python_partial` (`--reward python_partial`, CLI choice + help
added). `info["correct"]` = full pass, so the trainer's pass_rate / collapse-guard accounting is
preserved. Denser GRPO signal + fewer zero-variance groups (complements DAPO dynamic sampling).
Additive new reward — `python_exec` untouched. +9 tests; reasoning-config-wiring green, ruff clean.

**ORIGINAL cross-technique idea (IDEA-023): execution-trace token-level credit.** Full EGCA with
cola-coder's assets. The sandbox already produces a TRACEBACK on failure (the failing assert, its
line number). Map that failing region back to the candidate's tokens and apply the negative
advantage MORE strongly to those tokens in GRPO's already-per-token clipped surrogate, while
sparing tokens in regions whose asserts passed (from MODEL-041's per-case results). Turns the
verifier's execution trace into TOKEN-LEVEL credit — the localized GRPO update EGCA proposes,
built on the per-token logprobs the trainer already computes + the per-assert pass map. Builds on
MODEL-041 + the GRPO token-level surrogate + sandbox tracebacks. Reasoning-only → main-safe. → IDEA-023.

---

## 2026-06-14 — Synthetic data & model collapse (rotate: data curation)

Sources:
- Escaping Model Collapse via Synthetic Data Verification — https://arxiv.org/abs/2510.16657
- Seed-Coder: let the code model curate data for itself — https://arxiv.org/html/2506.03524v2
- Embarrassingly Simple Self-Distillation Improves Code Generation — https://arxiv.org/html/2604.01193v1
- OpenCodeInstruct (large-scale code instruction tuning) — https://arxiv.org/html/2504.04030v1

Findings:
- **Verified synthetic data does NOT collapse.** Naively retraining on a model's own outputs
  amplifies errors (model collapse), but injecting external information through a VERIFIER
  (human or a stronger checker) provably prevents it (2510.16657). cola-coder owns a sandbox
  verifier (tsc/tests) → its synthetic/distilled data is collapse-resistant BY DESIGN, a rare
  structural advantage most synthetic-data pipelines lack.
- **Retain real data:** keeping even ~10% real data per fine-tuning cycle measurably reduces
  perplexity drift — a cheap collapse-mitigation knob for the SFT/distillation stages.
- **Self-curation (Seed-Coder):** a code model can curate its own pretraining data with quality
  filters, reducing human/large-teacher dependence — aligns with the project's scorer pipeline.

**Implemented this cycle (DATA-063 soft-weight variant — main-safe):** `InjectionScorer`
(data/scorers/injection_scorer.py), the SOFT counterpart to last cycle's hard-drop
`InjectionFilter`. Reuses the SEC-019 `scan_injection` scanner + the shared `ScoreMapper` to
assign injection-carrying samples a LOW quality score (graded: 0 hits→1.0, 1→0.4, 2→0.15,
floor 0.05) so the composite quality WEIGHT is reduced rather than the sample dropped — the
project's reweight-over-filter preference for borderline content. Registered as `injection_safety`
in the scorer registry + configs/scoring.yaml (opt-in, default off). Prep-time → committed on
main. +7 tests (clean→1.0, graded down-weight, floor, registry wiring); ruff clean.

**ORIGINAL cross-technique idea (DATA-064): verifier-anchored synthetic mixing with a real-data
floor.** Combine both 2026 collapse defenses using cola-coder's assets. (1) Every synthetic/
distilled example must pass the sandbox verifier before entering the training mix (already the
spirit of MODEL-040's screen) — the "external verifier" that 2510.16657 proves prevents collapse.
(2) Enforce a configurable REAL-DATA FLOOR (≥10-20% verified-real scraped code) in each SFT/
distillation round via the existing weighted-mixing path (combine_datasets), tracked so synthetic
share can't silently dominate. Result: unbounded verified-synthetic augmentation that provably
won't collapse, with a real-data anchor — exploiting verifier + quality weights + dataset mixing
together. Builds on MODEL-040 + combine_datasets + the verifier. Prep-time → main-safe. → DATA-064.

---

## 2026-06-14 — Long-context RoPE extension (rotate: long-context)

Sources:
- LongRoPE2: near-lossless context window scaling — https://arxiv.org/pdf/2502.20082
- MrRoPE (mixed-radix RoPE, ICLR 2026, training-free train-short-test-long) — https://arxiv.org/pdf/2601.22181
- YaRN: efficient context window extension — https://arxiv.org/pdf/2309.00071
- RoPE extensions, an attention perspective — https://arxiv.org/pdf/2406.13282

Findings:
- **YaRN degrades on small models for DEEP retrieval.** YaRN avoids long-seq perplexity blowup
  but drops sharply on downstream tasks when key info sits deep beyond the training context — and
  fails needle retrieval specifically on small models (Phi3-mini, LLaMA3-8B). Relevant: cola-coder
  is small and uses YaRN (RoPEScalingConfig) → MODEL-033 (LongRoPE2) is the upgrade.
- **LongRoPE2** rescales per-dimension RoPE frequencies guided by needle-style evals → near-lossless
  128K on a 3.8B model where YaRN fails. **MrRoPE** (ICLR 2026) unifies RoPE extensions under a
  radix view and gets TRAINING-FREE "train short, test long" (>85% recall @128K, ~2× YaRN).
- **Perplexity ≠ retrieval:** the field now validates long-context with needle/retrieval tasks,
  not just perplexity — extension that holds perplexity can still fail deep retrieval.

**Implemented this cycle (IDEA-020 — reasoning-only, main-safe):** per-difficulty entropy floors in
the IDEA-013 `EntropyClipController`. New optional `difficulty_floors={"easy":…,"medium":…,"hard":…}`
+ `floor_for(difficulty)`; `update(..., difficulty=…)` uses that tier's floor instead of the single
`target_entropy` (fallback preserved). Hard problems (low pass-rate, need search) get a higher floor
→ more DAPO clip-higher exploration; easy/solved tiers get a low floor → exploit. Wired through the
GRPO curriculum's per-step `difficulty` (staged easy→hard, so the next-step clip is tier-appropriate).
Backward-compatible (no floors → single target). +5 tests (15 controller pass); GRPO suite green.

**ORIGINAL cross-technique idea (IDEA-022): verifier-graded long-range FIM.** Long-context extension
usually just rescales RoPE frequencies and validates on synthetic needles. cola-coder can MANUFACTURE
genuine, VERIFIABLE long-range dependencies: build a dynamic-FIM example where the symbol the middle
must use (a type, an imported helper) lives in a prepended repo-context block thousands of tokens
away — so a correct infill REQUIRES attending across the full window — and grade it with the sandbox
verifier (does prefix+infill+suffix type-check / pass tests using that distant symbol?). This turns
"needle in a haystack" into a VERIFIABLE TRAINING + eval objective (functional long-range retrieval,
not just perplexity), exactly the deep-retrieval failure YaRN has. Pairs RoPE extension (MODEL-033)
with FIM + repo context + verifier — a long-context signal the RoPE papers (synthetic needles only)
lack. Worktree (touches FIM/data) when implemented; the eval half is main-safe. → IDEA-022.

---

## 2026-06-14 — Contamination-free & holistic code eval (rotate: evaluation)

Sources:
- LiveCodeBench: holistic, contamination-free code eval — https://arxiv.org/abs/2403.07974
- Static→Dynamic eval against data contamination — https://arxiv.org/pdf/2502.17521
- TREAT: code-LLM trustworthiness/reliability framework — https://arxiv.org/pdf/2510.17163
- Cross-Context Verification (session-isolated contamination detection) — https://arxiv.org/pdf/2603.21454

Findings:
- **Contamination is the central eval problem.** LiveCodeBench solves it by TIME-SEGMENTATION:
  problems carry release dates, and a model is scored only on problems published AFTER its
  training cutoff — measuring generalization, not memorization. The project already plans this
  (EVAL-023 + IDEA-007: scrape RECENT license-clean TS/React, hold out the middle as FIM).
- **Eval should be HOLISTIC, not just generation:** LiveCodeBench adds self-repair, code
  EXECUTION, and TEST-OUTPUT PREDICTION as distinct capabilities. cola-coder's sandbox verifier
  makes execution-grounded scoring cheap — an asset most eval harnesses lack.
- **Harness sensitivity is huge:** the same model swings 30-50pp by agent harness. Implication:
  fix and version the eval harness; report it alongside scores.

**Implemented this cycle (DATA-063 — data filter, main-safe):** `InjectionFilter`
(data/filters/injection.py), a registered FilterPlugin that reuses the SEC-019 scanner
(`scan_injection`) to DROP scraped pretraining samples carrying prompt-injection payloads — so
the corpus itself can't teach the model to emit/obey injections (data poisoning). Closes the loop
between input-time defense (SEC-019 doc fetcher) and training-time hygiene. Opt-in via the filter
chain; `min_hits` config to require corroborating signals. Prep-time → committed on main. +7 tests
(registered+constructible, drops payloads, keeps clean/benign-trigger-words, min_hits, hidden
chars, empty); ruff clean.

**ORIGINAL cross-technique idea (EVAL-025): execution-grounded self-repair eval.** A
contamination-free eval (IDEA-007) usually needs reference solutions. cola-coder doesn't — it has
a sandbox verifier. Idea: scrape RECENT (post-cutoff) TS/React files WITH their test files; score
the model by EXECUTING its completion against the real tests (functional, reference-free,
contamination-resistant), and add a **self-repair@k** metric: on a failing completion, feed the
tsc/test ERROR back and measure whether the model fixes it within k tries (LiveCodeBench's
self-repair dimension, graded by the verifier instead of a reference). Reuses the verifier +
best-of-N + dynamic FIM; turns the verifier into a holistic, contamination-free eval the public
harnesses approximate with curated reference sets. Builds on EVAL-023/IDEA-007 + the sandbox. → EVAL-025.

---

## 2026-06-14 — Indirect prompt injection via retrieved content (rotate: safety)

Sources:
- Agentic AI Security: Threats, Defenses, Evaluation — https://arxiv.org/abs/2510.23883
- Are AI-assisted Dev Tools Immune to Prompt Injection? — https://arxiv.org/pdf/2603.21642
- Defense Against Indirect Prompt Injection via Tool Result Parsing — https://arxiv.org/pdf/2601.04795
- OWASP LLM01 Prompt Injection — https://www.stackhawk.com/blog/owasp-llm01-prompt-injection/

Findings:
- **Prompt injection is OWASP LLM01** (top LLM risk, 3rd year running). The dominant vector is
  now INDIRECT: poisoned RETRIEVED content (5 crafted docs flip RAG answers 90% of the time;
  "3 lines of hidden markdown in a skill file" can exfiltrate SSH keys). Tool poisoning and
  credential theft via tool output are new agentic surfaces.
- **Hidden-content vectors:** zero-width / bidi control characters smuggle instructions that
  are invisible to a human reviewing the doc.
- **Defense-in-depth is the consensus architecture:** independent layers, each raising attack
  cost. Input/retrieval scanning is the cheap first layer; PromptArmor (ICLR 2026) reaches
  <1% FP/FN on AgentDojo as a heavier layer.
- **Container escape is real (sandbox layer):** Docker shares the host kernel; production-safe
  execution needs microVM/userspace-kernel isolation — reinforces the project's OPEN SEC-015
  (true VM isolation unavailable on Windows Docker → hardened-Docker "raises the bar" only).

**Implemented this cycle (SEC-019 — security/inference, main-safe):** canonical prompt-injection
scanner `security/injection_patterns.py` — `scan_injection(text)` / `has_injection(text)` flag
indirect-injection directives (ignore-previous-instructions, disregard-system-prompt, new-
instructions blocks, system-prompt + secret exfiltration, pipe-to-shell, fake ChatML/[INST]/
<system> role markers) AND hidden invisible/bidi control characters, in UNTRUSTED retrieved text
BEFORE it enters a prompt. Mirrors code_patterns.py (different threat class). Wired as a
non-blocking defense-in-depth WARN into the doc fetcher (poisoned fetched docs are now visible in
logs, not silently smuggled into context). High-precision (benign uses of "token"/"env"/"previous
batch" don't trip). +19 tests; ruff clean. Off the train path → committed on main.

**ORIGINAL cross-technique idea (DATA-063): injection-aware training-data filtering.** A
from-scratch model is trained on SCRAPED code whose comments/docstrings can carry prompt-injection
payloads — so the pretraining CORPUS itself can teach the model to emit/obey injections (data
poisoning). Run `scan_injection` (SEC-019) over scraped samples during data prep and DOWN-WEIGHT
(via the quality-weights path) or drop the ones carrying injection payloads, so the model never
learns them. Combines the injection scanner + quality weights + the data pipeline — closing the
loop between input-time defense (SEC-019) and training-time hygiene, which neither a pure runtime
guardrail nor a pure data filter does alone. Prep-time → main-safe. → DATA-063.

---

## 2026-06-14 — FIM decoding quality: stopping & suffix bleed (rotate: inference/decoding)

Sources:
- Instruction-Aware Fill-in-the-Middle paradigm — https://arxiv.org/html/2509.24637v1
- Memorization Dynamics of FIM Pretraining — https://arxiv.org/html/2605.22981
- Constrained Decoding for FIM via grammar quotienting — https://arxiv.org/abs/2402.17988

Findings:
- **FIM models over-generate and "bleed" the suffix.** A recurring failure mode: the infill
  fails to stop at the true boundary and instead re-emits the document SUFFIX it was already
  given as context (or unrelated code). In an editor that renders as DUPLICATED code after the
  accepted completion — a visible, common quality bug in inline completion.
- **Stopping criteria are the hard part.** Auto-eval leans on dataset-specific truncation /
  post-processing; production needs a content-based stop. Two families: (a) cheap string
  post-processing (trim the infill's tail that duplicates the suffix's head); (b) incremental
  PARSING / constrained decoding that stops when prefix+infill+suffix is a complete program.
- **Suffix context is weaker than prefix:** verbatim recall stays prefix-anchored, so the model
  leans on the prefix and tends to reconstruct the suffix rather than truly infill.

**Implemented this cycle (INFER-027 — inference/server, main-safe):** suffix-overlap trim for the
`/v1/fim` endpoint. New `text_utils.trim_suffix_overlap(infill, suffix, min_overlap=3)` removes
the LONGEST verbatim overlap between the infill's tail and the suffix's head (with a small
min-overlap so a lone coincidental `;`/`}` isn't trimmed). Wired into the FIM handler so inline
completions no longer duplicate code that already follows the cursor. The cheap, robust family-(a)
fix. +7 tests; 13 text_utils + 80 FIM tests green, ruff clean. Off the train path → committed on main.

**ORIGINAL cross-technique idea (IDEA-021): verifier-gated FIM completion boundary.** Family-(b)
done with cola-coder's assets instead of a bespoke grammar parser. Generate the FIM infill, then
use the sandbox tsc verifier as the "is this a complete program?" oracle: find the SHORTEST infill
prefix such that prefix+infill+suffix type-checks clean (tsc --noEmit), and stop there — a
best-of-N over candidate stop points scored by the verifier. Reuses the FIM path + TscRunner +
best-of-N; turns the verifier into a syntactic-completeness stop signal that the FIM decoding
papers approximate with hand-written incremental parsers. Pairs with INFER-027 (string trim as the
fast path, verifier gate as the precise path). Builds on /v1/fim + TscRunner + best-of-N. → IDEA-021.

---

## 2026-06-14 — GRPO/DAPO variants & exploration control (rotate: post-training/RLVR)

Sources:
- Post-Training in 2026: GRPO, DAPO, RLVR & Beyond — https://llm-stats.com/blog/research/post-training-techniques-2026
- GRPO's effective loss, dynamics & success amplification (OpenReview) — https://openreview.net/forum?id=y4y7fvcR8W
- Prompt Augmentation Scales up GRPO Training — https://arxiv.org/pdf/2602.03190
- RLVR implicitly incentivizes correct reasoning — https://arxiv.org/html/2506.14245v2

Findings:
- **The GRPO variant zoo converges on a few levers:** advantage normalization (mean-only
  Dr. GRPO vs mean+std), decoupled clipping (DAPO clip-higher to promote exploration),
  dynamic sampling (drop zero-variance groups), drop-KL, and token-level length norm.
  cola-coder already has ALL of these (advantage_norm, clip_epsilon_high, dynamic_sampling,
  length_norm, no-KL) — the gap was making the clip ADAPTIVE rather than a fixed knob.
- **DAPO clip-higher is the exploration lever:** raising the upper clip (e.g. 0.28>0.2) lets
  low-probability tokens grow, directly countering entropy collapse — but it's set statically.
- **Verifiable rewards already encode the stopping signal:** once pass-rate saturates, extra
  exploration is wasted; coupling exploration to the verifier is an obvious but unexploited win.

**Implemented this cycle (IDEA-013 — reasoning-only, main-safe):** `EntropyClipController`
(reasoning/entropy_controller.py) closes the loop on MODEL-037's entropy metric. Proportional
controller: when measured policy entropy falls below a target floor, RAISE clip_high (DAPO
clip-higher) proportionally to the deficit (capped) to inject exploration; relax to base when
healthy. VERIFIER-AWARE — suppresses exploration when the group pass-rate is at/above a ceiling
(don't explore away from working solutions; couples RL exploration to executable success, which
the RLVR papers — no verifier — can't). Wired opt-in into GRPOTrainer (default None → unchanged);
applied once per non-skipped step to the next step's clip. +10 controller tests, GRPO suite green.

**ORIGINAL cross-technique idea (IDEA-020): per-difficulty entropy floors.** The curriculum
already varies sampling temperature by difficulty (easy/medium/hard). Extend the entropy
controller to hold a DIFFERENT entropy floor per difficulty tier — hard problems (low pass-rate,
need search) get a higher floor → more clip-higher exploration; easy problems (already solved)
get a low floor → exploit. Drives the DAPO clip from BOTH the live entropy AND the verifier's
per-difficulty pass-rate, turning the existing curriculum + verifier + entropy controller into a
difficulty-adaptive exploration schedule. Builds on IDEA-013 + the curriculum + per-difficulty
pass-rate already tracked in GRPOTrainer.train(). → IDEA-020.

---

## 2026-06-14 — Model-based data curation (rotate: data curation)

Sources:
- FineWeb / FineWeb-Edu (model-based edu-quality filtering) — https://arxiv.org/pdf/2406.17557
- DCLM-Baseline vs FineWeb-Edu (Karpathy llm.c discussion) — https://github.com/karpathy/llm.c/discussions/664
- Ultra-FineWeb (efficient filtering + verification) — researchgate 391575046
- Datasets, Documents, and Repetitions: unequal data quality — https://arxiv.org/html/2503.07879v1

Findings:
- **Model-based quality classifiers are the 2026 default.** FineWeb-Edu trains a classifier
  on SYNTHETIC LLM annotations (Llama-3-70B rates educational quality 0–5), then keeps the
  high-scoring subset: 10% of tokens (38B) matched 350B unfiltered tokens. Stack-edu does the
  same for CODE (thresholds >3.75 → >4.1). cola-coder already aligns (ClassifierScorer +
  train_quality_classifier/train_judge_classifier distill judge scores into a local TF-IDF).
- **Two-stage thresholds:** raise the quality bar in later training (>2.75→>3.2 edu;
  >3.75→>4.1 code) — cheap, pairs with the project's curriculum/folding (DATA-058) → DATA-061.
- **Quality × repetition interaction (2503.07879):** how many epochs a document earns should
  scale with its quality — ties quality weights to repetition/epoching.

**Implemented this cycle (IDEA-019 validation — main-safe, tests-only):** validated the
quality-weights × Muon interaction from last cycle. New tests/test_muon_quality_weights.py
proves RELATIVE in-batch quality weights DO change the Muon update (not neutered by
orthogonalization), monotonically with skew, and optimizer-agnostically (AdamW too). EMPIRICAL
REFINEMENT of the IDEA-019 hypothesis: the theoretical "global loss scale is fully washed out
by orthogonalization" does NOT hold cleanly in practice because Newton-Schulz runs in bf16 —
a 1000× global scale perturbs the bf16 update by an amount comparable to a real reweight at
tiny scale. Practical guidance is unchanged and now test-backed: keep quality weights RELATIVE
(weighted mean, which `language_modeling_loss` does), never a global multiplier. +3 tests, ruff green.

**ORIGINAL cross-technique idea (DATA-062): verifier-distilled quality classifier.** FineWeb-Edu's
labels are an LLM's OPINION of quality. cola-coder has something better — a sandbox verifier
(tsc/tests) + best-of-N + the security scanner that produce OBJECTIVE, executable ground truth.
Idea: label a code corpus with verifier outcomes (compiles? tests pass? secure? best-of-N
pass-rate) and distill THOSE labels into the local TF-IDF quality classifier (reuse
train_judge_classifier's distillation, swap LLM-judge → verifier labels). Result: a fast static
quality scorer grounded in executable correctness, not LLM judgment — cheaper and more objective
than the FineWeb-Edu recipe, and uniquely possible because cola-coder owns the verifier. Feeds
the quality weights / folding curriculum. Builds on the verifier + best-of-N + scanner. → DATA-062.

---

## 2026-06-14 — Muon optimizer maturity (rotate: optimizers)

Sources:
- Muon is Scalable for LLM Training (Moonlight, 16B MoE, ~2× AdamW efficiency) — https://arxiv.org/abs/2502.16982
- Keller Jordan — Muon / nanoGPT speedrun formulation — https://kellerjordan.github.io/posts/muon/
- Moonshot AI / Kimi (MuonClip, qk-clip stability at scale) — https://en.wikipedia.org/wiki/Moonshot_AI

Findings:
- **Muon = momentum → Newton-Schulz orthogonalization → spectrally-normed update**
  (steepest descent under a spectral-norm trust region). Validated at 16B-MoE scale with
  ~2× token efficiency over tuned AdamW; only 2D hidden matrices use Muon, embeddings/norms
  stay on AdamW. cola-coder already implements exactly this hybrid (optimizer.py).
- **Two scale-at-scale additions matter:** (1) decoupled WEIGHT DECAY (original Muon lacked
  it; Moonlight added it — already in cola-coder's `_muon_step`), and (2) update-RMS matching
  (`scale = max(1, rows/cols)**0.5`) so updates across differently-shaped matrices have a
  consistent RMS, letting an AdamW-tuned LR transfer (already in cola-coder).
- **MuonClip (Kimi-K2):** add qk-clip to bound attention logits — the large-run stability
  layer; cola-coder already has qk_norm, a related logit-control mechanism → MODEL-034.

**Implemented this cycle (MODEL-025 validation — main-safe, tests-only):** closed the Muon
test-coverage gaps without touching optimizer.py (no train-path edit). The existing suite
tested orthogonalization (rows<cols), loss-decrease, param-split, and state_dict resume; the
TRANSPOSED Newton-Schulz branch (rows>cols — taken by every tall weight matrix) and decoupled
weight decay were UNTESTED. Added: transposed-branch orthogonality (svdvals≈1, shape
preserved), shape preservation across orientations, and a precise weight-decay validation
(`p_decayed == p_undecayed − lr·wd·p₀` to 1e-5, isolating the decay term from an identical
orthogonal update). +3 tests (10 TestMuon pass), ruff green. Empirical Muon-vs-AdamW A/B
still deferred (needs a training run; the live run is AdamW).

**ORIGINAL cross-technique insight (IDEA-019): quality-weights × Muon interaction.** Non-obvious
catch: Muon ORTHOGONALIZES the update, discarding gradient MAGNITUDE for the 2D matrices. The
project's per-sample quality weights scale the loss → scale the gradient. A *global* batch loss
scale is therefore WASHED OUT by Newton-Schulz (no effect on Muon's update direction) — but
*relative* in-batch per-sample weights still reshape the averaged-gradient DIRECTION and so
survive. Implication: quality weighting must be applied as a weighted MEAN over per-sample
losses (relative weights — which `language_modeling_loss` already does), never as a global loss
multiplier, or it silently no-ops under Muon. Actionable follow-up: a test asserting a
high-quality-weighted batch yields a DIFFERENT Muon update direction than uniform weights
(proves weights aren't neutered), and document the constraint where Muon + quality weights meet.
This is exactly the kind of interaction only cola-coder (Muon + quality weights together) hits. → IDEA-019.

---

## 2026-06-14 — On-policy distillation (rotate: post-training)

Sources:
- On-Policy Distillation — Thinking Machines Lab — https://thinkingmachines.ai/blog/on-policy-distillation/
- Rethinking On-Policy Distillation of LLMs (phenomenology/mechanism/recipe) — https://arxiv.org/html/2604.13016v1
- A Survey of On-Policy Distillation for LLMs — https://arxiv.org/html/2604.00626v3
- Learning from Self-Generated Mistakes (MiniLLM/GKD lineage) — https://arxiv.org/pdf/2306.13649
- Entropy-Aware On-Policy Distillation — https://arxiv.org/pdf/2603.07079

Findings:
- **OPD fixes off-policy distribution mismatch.** Classic SeqKD (the project's current
  pipeline: teacher writes solutions, student SFTs on them) trains the student on the
  TEACHER's distribution; at inference the student samples its OWN distribution → mismatch.
  OPD instead has the STUDENT generate, then the teacher grades each student token
  (reverse-KL, mode-seeking) — matching RL-quality reasoning at ~10× lower compute than GRPO.
- **Mechanism (Yang 2026):** OPD = a special case of dense KL-constrained RL; the teacher's
  per-token log-ratio is an IMPLICIT dense reward. So OPD ≈ GRPO with a free, dense,
  per-token reward signal instead of a sparse end-of-sequence verifier score.
- **Reverse-KL is mode-seeking** → concentrates mass on correct solutions (good for code/
  math); entropy-aware variants (EOPD) re-inject diversity to avoid collapse (ties to the
  MODEL-037 entropy metric — same failure mode).
- **Caveat:** full OPD needs token-level teacher logprobs + a training loop → train-path
  (worktree-only while live). MODEL-024 refined with these findings; the dense-reward framing
  also strengthens IDEA-018 below.

**Implemented this cycle (MODEL-040, distillation data-gen — main-safe):** always-on security
screen in `generate_distillation_dataset`. The pipeline rejection-sampled only on FUNCTIONAL
`verify` (tsc/tests), and with `verify=None` it KEPT dangerous teacher code verbatim (a real
hole: distilling `os.system('rm -rf /')`). Added `screen_security=True` (default) — every
teacher completion is statically screened by the canonical `scan_dangerous` (SEC-018) and
dropped BEFORE functional verify, regardless of keep_only_verified; new `rejected_insecure`
stat. Defence-in-depth that no longer depends on the caller wiring security into `verify`
(the CLI's redundant inline check was removed — DRY). Static, execution-free → main-safe.
+4 tests (11 distillation pass), ruff green.

**ORIGINAL cross-technique idea (IDEA-018): verifier-densified distillation reward.** OPD's
power is a DENSE per-token teacher reward vs the verifier's SPARSE end-of-sequence score.
cola-coder has BOTH a teacher (local/cloud) AND a sandbox verifier + best-of-N. Idea: blend
them — use the teacher's per-token log-ratio as the dense shaping reward, but GATE the
episode return by the sandbox verifier (only reinforce student rollouts that actually pass
tsc/tests), and up-weight by quality weights. Dense teacher signal for credit assignment +
hard ground-truth gate for correctness — neither the OPD papers (no verifier) nor pure GRPO
(sparse only) has both. Pairs the teacher with the verifier. Builds on MODEL-024 + the GRPO
loop + best-of-N. → IDEA-018.

---

## 2026-06-14 — Transformer stability frontier (rotate: architecture)

Sources:
- The Big LLM Architecture Comparison (2025/26) — https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- When Attention Sink Emerges (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf
- Architectural Trade-offs in Small LMs Under Compute Constraints — https://arxiv.org/html/2512.20877v1
- IMU-1: Sample-Efficient Pre-training of Small LMs — https://arxiv.org/pdf/2602.02522

Findings:
- **Gated attention** (elementwise gate on the SDPA output before the output projection):
  reduces attention SINKS and massive activations, improves long-sequence generalization,
  and adds negligible params. A 2026 standard add-on; cola-coder has none → MODEL-038.
- **QK-Norm** (already in cola-coder, qk_norm=true): normalizes Q/K via RMSNorm before
  attention so logits can't blow up the softmax. Confirmed to ENLARGE the LR basin (less
  LR sensitivity across widths) AND be compatible with muP — strengthens the case for
  MODEL-031 (muP) since the two compose.
- **Depth-scaled sandwich norm:** four RMSNorms per block with the 2nd init ~1/√L so the
  residual update starts small and grows — a cheap deep-stability trick → MODEL-039.
- **Attention-sink anatomy (2026):** the first token's keys act as a learned bias/sink;
  monitoring max attention logits + activation magnitude is the recommended stability gauge.

**Implemented this cycle (IDEA-014, inference — main-safe):** cluster-gated verifier budget
in best-of-N. Architecture changes touch the train path (worktree-only while the run is
live), so the implementation went to the highest-value main-safe item instead:
`_verify_deduplicated` groups candidates by normalized completion (the INFER-026 clusters),
runs the sandbox verifier on ONE representative per cluster, and propagates the verdict
(fresh `details` copy per candidate — no aliasing) to the rest. N candidates now cost
~(#distinct programs) tsc/test runs instead of N, with identical results. Default-on
(`cluster_verify=True`), backward-compatible (all-unique → no change; 31 existing tests
green). +4 tests (35 best-of-N pass), ruff green.

**ORIGINAL cross-technique idea (IDEA-017): one-shot HP transfer for the specialist fleet.**
cola-coder's vision is a router + many 50M domain specialists (React/Next/GraphQL/…). muP
(MODEL-031) lets you tune LR/init ONCE on a tiny proxy width and zero-shot transfer to any
width; QK-Norm (already present) widens that basin, and Muon also admits muP-style transfer.
Combine them: tune the HP set once on a ~10M proxy with Muon+QK-Norm+muP, then stamp the
SAME recipe across the entire specialist fleet with no per-model sweep — turning the
multi-specialist architecture (a cost in most stacks) into a one-sweep-amortized advantage.
Builds on MODEL-031 + the Muon optimizer + qk_norm. → IDEA-017.

---

## 2026-06-14 — Insecure-code generation: CWE coverage (rotate: safety)

Sources:
- OWASP LLM Top 10 applied to code generation (Sonar) — https://www.sonarsource.com/resources/library/owasp-llm-code-generation/
- Is Your Prompt Poisoning Code? Defect induction + mitigation — https://arxiv.org/pdf/2510.22944
- From Vulnerabilities to Remediation: SLR of LLMs in code security — https://arxiv.org/pdf/2412.15004
- LLMs for source code analysis (models + datasets) — https://arxiv.org/pdf/2503.17502

Findings:
- **12–65% of LLM-generated snippets violate basic secure-coding standards;** one study
  found ~40% of Copilot output carried a CWE-classified vulnerability. So insecure
  generation is the norm, not the exception — screening matters.
- **Top classes:** injection (SQLi CWE-89, XSS CWE-79) dominate, then weak crypto
  (CWE-327), insecure deserialization, command exec (CWE-78), poor input validation.
- **No single static analyzer covers all classes** — but high-precision pattern screens
  catch the common, high-severity ones cheaply (the rest need dynamic/fuzzing/LLM-judge).
  Safety instructions + few-shot raise security accuracy 20–25%; a "vulnerability detector"
  role helps — both relevant to later GRPO/distillation prompting.

**Implemented this cycle (SEC-018):** expanded the canonical `scan_dangerous` scanner
(`security/code_patterns.py`) — the SINGLE screen feeding best-of-N ranking, the GRPO
security penalty, safety_eval, AND secure-pass@k (EVAL-024) — with high-precision patterns
for the missing top-CWE classes: XSS DOM sinks (`innerHTML/outerHTML =`, `insertAdjacentHTML`;
`(?!=)` excludes `==`/`===` comparisons), code-as-string timers (`setTimeout('…')`), SQLi via
template interpolation and string concatenation, weak hashing (`createHash('md5'|'sha1')`,
`hashlib.md5/sha1`), and `os.popen`. Improving the one scanner upgrades all four consumers at
once. Precision held (reading innerHTML, `sha256`, function-arg timers, static SQL all clean).
+11 tests; 36 scanner + 40 downstream (best-of-N/distillation) green. Off the train path → main.

**ORIGINAL cross-technique idea (IDEA-016): adversarial secure-FIM hardening.** cola-coder
can turn its scanner into a *training signal* no eval-only pipeline can: take a clean code
sample, programmatically inject a known CWE (e.g. swap a parametrized query for a string-concat
one), and create a dynamic-FIM task whose middle is the vulnerable span — the model must
infill the SECURE version, graded by `scan_dangerous` (no danger) AND tsc/tests (still
correct). This is contrastive (secure vs insecure minimal pairs), self-labeling (the injector
knows ground truth), and verifier-graded — combining dynamic FIM + scanner + sandbox + quality
weights. Pairs naturally with IDEA-015 (mine real gaps) for synthetic coverage of rare CWEs. → IDEA-016.

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

---

## 2026-06-15 — Inference: prompt-lookup / n-gram speculative decoding (realizing the speedup)

**Area:** inference / decoding. **State of the project:** `inference/prompt_lookup.py`
ships a clean, tested `PromptLookupDrafter` (draft-by-matching-the-context's-own-n-grams)
+ an offline `analyze_acceptance` driver, exposed via `scripts/pld_analysis.py` and the eval
menu. BUT the drafter is **not wired into `CodeGenerator.generate()`** — only the offline
acceptance *analysis* exists, so the real 2-4x wall-clock speedup is unrealized.

**2026 literature.** Prompt-lookup / n-gram speculative decoding is training-free: build the
draft by string-matching recent context, then verify the drafted tokens in ONE model forward
pass — accepting the longest greedily-correct prefix, so output is bit-identical to normal
decoding (exact verification). Reported 2-4x on input-grounded tasks, ~1.5-3x general; code is
an especially good fit (highly repetitive: imports, boilerplate, closing brackets, repeated
identifiers). 2025-26 work pushes it further: **PROMTEC** (prompt multi-lookup — propose
several candidate continuations and tree-verify them, HumanEval gains); suffix-automaton /
suffix-tree draft caches for O(1) longest-match retrieval (EMNLP 2025); GRIFFIN/OWL on draft-
token alignment and long-context window-length dependence.

**Original idea — IDEA-006: acceptance-calibrated adaptive draft length (γ controller).**
Standard PLD uses a FIXED draft length γ (`num_pred_tokens`). On code, realized acceptance is
*bimodal* — near-1.0 in boilerplate runs, near-0 at genuine decision points — so a fixed γ
both wastes verification compute in low-acceptance regions and under-drafts in high-acceptance
ones. The project is uniquely positioned: it ALREADY has `analyze_acceptance` measuring realized
acceptance. Close the loop LIVE — maintain a running acceptance EMA and grow γ after full
acceptances, shrink it after early rejection (a closed-loop controller, mirroring the reasoning
module's entropy controller, IDEA-013). Cross-technique: PLD (MODEL-022) × the existing offline
acceptance analyzer × an online controller. Pure, model-free, exact-verification-preserving, so
it's a safe building block toward the generator integration. **Implemented this cycle** as a
pure `AdaptiveDraftLength` controller in `prompt_lookup.py` (unit-tested); wiring it + the
drafter into the generate loop is filed as MODEL-044.

**Sources:**
- [prompt-lookup-decoding (apoorvumang)](https://github.com/apoorvumang/prompt-lookup-decoding)
- [PROMTEC: Fast LLM Inference Decoding using Prompt Multi-Lookup (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.355.pdf)
- [Faster In-Context Learning via N-Gram Trie (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.911.pdf)
- [Scaling Up, Speeding Up: A Benchmark of Speculative Decoding (arXiv 2509.04474)](https://arxiv.org/pdf/2509.04474)
- [GRIFFIN: Effective Token Alignment for Faster Speculative Decoding (arXiv 2502.11018)](https://arxiv.org/pdf/2502.11018)
