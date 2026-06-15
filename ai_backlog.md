# Cola-Coder AI Improvement Backlog

Persistent issue/opportunity log for the autonomous improvement loop. Append new
items with a stable ID; only mark `done` after a validated fix. Severity:
critical / high / medium / low. Status: open / in-progress / done / dropped.

Ratings reflect VERIFICATION against the code, not just an agent's first guess —
e.g. BUG-004 was downgraded to not-a-bug after checking the math.

---

## Open

### UI polish — fold "More tools" card-grids (2026-06-15)
- **UI-077** [ui, medium] `done` (2026-06-15) — Replaced the last three "More tools" card-GRIDS
  (Checkpoints/Data/Pipeline) with single tabbed containers: `CheckpointToolsPanel` (Model Card/Manifest/
  Average/Router/Export), `DataToolsPanel` (8 data tools), `PipelineToolsPanel` (Configs/VRAM/LR/Diff/
  Pipeline/Manager). One coherent panel per section, no grid. Built by 3 parallel agents (disjoint new
  files) wrapping the existing panels unchanged; App.tsx now imports 3 panels instead of 19. tsc + build green.

### UI — streaming generation (2026-06-15)
- **UI-080** [ui, high] `done` (2026-06-15) — Token-by-token STREAMING for the Generate playground (R10):
  `POST /api/generate/stream` (StreamingResponse of SSE `GenStreamChunk {delta,done,error}` frames via
  `CodeGenerator.generate_stream`; same gated load-per-request/free lifecycle as /api/generate, 409 while
  training live) + `useStreamingGeneration` hook (fetch ReadableStream reader, partial-frame buffering,
  AbortController stop) + `StreamingConsole` (live text + blinking caret + Stop + error state). InferenceScreen
  rewired from one-shot to streaming. tsc + build green (93 modules). Stream gate verified 409 during training.

### Eval — generalized consistency metric (2026-06-15)
- **EVAL-039** [eval, medium] `done` (2026-06-15) — Added `g_pass_at_k(n, c, k, tau)` to
  evaluation/metrics.py: unbiased G-Pass@k_τ (arXiv:2412.13147), the hypergeometric right-tail at
  ⌈τ·k⌉ correct. Interpolates between the project's existing extremes — reduces to `pass_at_k` at
  τ=1/k and `pass_hat_k` at τ=1. Tests prove both reductions + monotonicity + explicit-hypergeometric
  match (38 pass). Pure, MAIN-SAFE.
- **EVAL-040 / IDEA-008** [eval/post-training, medium] `open` — Consistency-targeted GRPO: add a
  per-group bonus ∝ G-Pass@k_τ of the rollout group (or the reliability gap pass@k − pass^k) so the
  policy optimizes for *consistently* solvable problems, not lucky one-offs. Prototype offline on
  recorded rollouts first. Reuses GRPO group machinery + the new metric.

### UI — config editing + command palette (2026-06-15)
- **UI-078** [ui, high] `done` (2026-06-15) — Config EDITING (R11): `POST /api/config/write` +
  `configs.write_config` (YAML-validate + path-traversal guard + atomic temp/replace; refuses bad edits
  before touching disk; safe vs the running trainer which read its config at launch) + `ConfigEditorPanel`
  (pick → edit → Validate & Save, dirty/revert, truncation-guard). 5 backend tests + drift examples.
- **UI-079** [ui, medium] `done` (2026-06-15) — `CommandPalette` (⌘K/Ctrl-K): fuzzy quick-switcher over
  all 11 nav sections; self-contained global key listener, navigates via the same hash as Sidebar.

### Data quality — decontamination (2026-06-15)
- **DATA-072 / BUG-133** [data-quality/bug, high] `done` (2026-06-15) — `DecontaminationFilter` screened
  only the eval PROMPT, missing the canonical SOLUTION + hidden TESTS — the most damaging leakage and the
  exact target of the 2026 standard (prompt + canonical_solution reference). ALSO its benchmark loader
  imported `get_all_problems` from `evaluation.problem_loader` (no such symbol — it's in
  `evaluation.humaneval`), so `ImportError` → silent `[]` → `benchmark=True` was a NO-OP. Fixed both:
  correct import; emit prompt + canonical_solution + test_code as separate references. Tests: solution-only
  contamination now dropped; loader exposes all three fields (8 pass). 8-18% of HumanEval overlaps public
  pretraining corpora, so this directly protects pass@k honesty. See research-log 2026-06-15.
- **DATA-072b / IDEA-007** [data-quality, medium] `open` — Soft decontamination: fold a graded containment
  penalty into `.weights.npy` (weight ∝ 1−containment above a floor) instead of a hard drop, so
  near-threshold paraphrases are down-weighted not kept-at-full. Reuses quality-weighted training + the
  containment score already computed.

### Inference — speculative decoding (2026-06-15)
- **MODEL-044** [inference, high] `open` — Wire `PromptLookupDrafter` into `CodeGenerator.generate()`
  for REAL prompt-lookup speculative decoding (exact greedy verification → byte-identical output,
  ~1.5-3x wall-clock on repetitive code). The drafter + offline `analyze_acceptance` already exist
  (`inference/prompt_lookup.py`); only the live hot-path integration is missing. Use the new
  `AdaptiveDraftLength` controller (IDEA-006, shipped this cycle, tested) to size γ per-step from a
  running acceptance EMA. Verify the lossless invariant against a greedy baseline (same tokens out).
  MAIN-SAFE (inference only; never touches train.py). Optional follow-up: PROMTEC-style multi-lookup
  tree verification (ACL Findings 2025). See docs/research-log.md 2026-06-15 entry.

### UI rebuild — typed forms, pipeline, inference, screens (2026-06-15)
- **UI-070** [ui, high] `done` (2026-06-15) — Typed argument forms (R3): `ActionParam`/`ActionDef.params`
  + `action_params.py` 1:1 argparse spec for all actions; `ActionForm.tsx`; `/api/actions` merge.
- **UI-071** [ui, high] `done` (2026-06-15) — One-click full pipeline launcher (R4) + Configs & Pipeline
  master-detail screen (R2). `train` + `auto_pipeline` added to ACTIONS (trainer-guarded).
- **UI-072** [tooling, medium] `done` (2026-06-15) — Static CLI↔UI parity test locking the 1:1 invariant
  (64 checks across 31 actions, no script execution).
- **UI-073** [ui, high] `done` (2026-06-15) — Tokenizer master-detail screen (R2/R6): tabbed
  info/health/tokenize/vocab single screen replacing the 4-panel grid.
- **UI-074** [ui, high] `done` (2026-06-15) — Inference playground (R10): `/api/generate` (gated) +
  `InferenceScreen` (prompt + sampling controls, completion view, training-alive guard). Generation
  loads the model per request and frees it; refused while training is live.
- **UI-075** [ui, high] `done` (2026-06-15) — Chat playground (R10): `ChatScreen` + gated `POST /api/chat`
  (ChatML-or-plain via `use_chat_template`); multi-turn transcript, per-reply stats, system-prompt seed.
- **UI-076** [ui, high] `done` (2026-06-15) — FIM playground (R10): `FimScreen` + gated `POST /api/fim`
  (`tokenizer.fim_prompt`); infill + stitched preview; clear message when tokenizer lacks `<|fim_*|>`.
  Backend DRY: shared `_training_busy()` gate + `_run_generation()` across generate/chat/fim.
- **OPS-002** [bug/safety, critical] `done` (2026-06-15) — Training-active detection FAILED under
  OPS-001 (trainer launched at higher OS integrity → psutil can't read its cmdline → `alive`=False
  while training ran). This broke the dashboard `alive` flag AND every GPU guard; `/api/generate`
  initially returned 200 and ran a generation concurrent with the live train. FIX: `status.is_training_active()`
  now ORs the psutil scan with per-step `.err` tqdm mtime freshness (≤600s) — elevation-proof. Wired
  into `get_training_status().alive` (dashboard/frontend guard) and the `/api/generate` 409 gate. Tests:
  tests/test_ui_training_guard.py (6). The 100-step `.log` is deliberately NOT used (can be 22-30 min stale).

### Local Web UI — React+TS/Vite over FastAPI (2026-06-14)
- **UI-001..016** [ui, high] `done` (2026-06-14) — React+TS+Vite dashboard in `webui/`
  over the FastAPI backend `src/cola_coder/ui/` with SSE push (no polling). Shipped views:
  live training status, GPU/system, checkpoints, datasets+preview+score histograms, jobs
  +log tail, runnable actions (allow-listed), config browser, pipeline runs, eval/quality
  artifacts, generic log tail, features.yaml toggles, reasoning config, tokenizer info,
  checkpoint detail (safetensors header param count). 24 `/api` routes; "start training"
  REFUSES if a trainer is alive (no second trainer). Built by parallel agents on disjoint
  files. ~135 UI tests, tsc --noEmit + vite build green.
- **UI-017** [ui, high] `open` — Continue CLI→UI parity per docs/UI_SPEC.md (84-row map).
  Remaining: data collection/prepare wizards, eval_history monitoring, self-play / GRPO
  launch controls, per-action arg forms (currently allow-list defaults only).
- **UI-019..021** [ui, high] `done` (2026-06-14) — Batch 5 (parallel agents): router view
  (`/api/router`, domains + router checkpoints), export view (`/api/exports`, GGUF/Ollama/
  quant formats + existing artifacts), and a training-metrics CHART (`/api/metrics/history`
  + dependency-free SVG `Sparkline`: loss curve + tok/s). Runnable-actions allow-list
  expanded to 15 (added quality_report, safety_eval, completion_benchmark, benchmark,
  env_check). 27 `/api` routes. +37 UI tests, tsc+vite green, ruff clean.
- **UI-018** [ui, medium] `open` — Config EDITOR (write path): validate + save YAML configs
  from the browser (currently read-only). Guard against editing while a run uses the config.

### Eval rigor (2026-06-14)
- **EVAL-028** [eval, high] `done` (2026-06-14) — Bootstrap CIs + cross-problem SE for
  pass@k in `evaluation/metrics.py` (`pass_at_k_stderr`, `bootstrap_pass_at_k`,
  `paired_bootstrap_delta`); `format_results` renders `[95% CI lo–hi]`; `evaluate.py`
  `--no-bootstrap`/`--ci`. A bare pass@k on 62 problems was unfalsifiable. +10 tests.
- **EVAL-029** [eval/regression, medium] `done` (2026-06-14) — Wired `paired_bootstrap_delta`
  into significance verdicts: `regression.py` gains `SignificanceVerdict` +
  `assess_pass_at_k_significance`/`assess_aggregate_delta` + `RegressionSuite.compare_pass_at_k`;
  `model_comparison.py` `ComparisonResult` carries optional per-problem `ProblemResult` lists +
  `significance_report`; `compare_models.py` gains `--ci`/`--n-boot` and prints
  "+6.1pp [95% CI +1.2 to +11.0] — significant" vs "within noise", with raw-delta fallback when
  only aggregate scores exist. +15 tests (66 regression pass). Already CLI-menu-wired (eval_menu).
- **IDEA-028** [eval×difficulty, medium] `open` — Verifier-effort-STRATIFIED pass@k with
  per-tier bootstrap CIs (combine EVAL-028 + EVAL-026 `difficulty_profile` tiers): a
  stratified variance-reduction estimator + per-difficulty error bars unique to this repo.

- **UI-022..023** [ui, high] `done` (2026-06-14) — Batch 6 (parallel agents): data-sources
  view (`/api/data-sources`, weighted code/text/math mix + languages), eval-over-training
  history chart (`/api/eval-history`, pass@1/pass@5 vs step via Sparkline), and ActionsPanel
  EDITABLE per-action args (run any allow-listed script with custom args as a bg job).
  29 `/api` routes. +25 UI tests, tsc+vite green, ruff clean.

### Eval / robustness (2026-06-14)
- **EVAL-030** [eval, high] `done` (2026-06-14) — Verifier-graded functional robustness:
  `evaluation/perturbations.py` (meaning-preserving docstring perturbations; invariant: only
  prose mutated, `def`/entry_point/test_code byte-identical, AST-re-validated) +
  `robustness_eval.py` (`robust_pass@1` worst-case, consistency_rate, fragility map; reuses the
  sandbox verifier + `bootstrap_pass_at_k`) + `scripts/robustness_eval.py` + eval-menu entry.
  +25 tests. Main-safe.
- **EVAL-031** [eval, medium] `done` (2026-06-14) — Stratify robustness by verifier-effort tier
  + bootstrap CI on robust_pass@1. `robustness_eval.py` `by_tier` (n/robust_pass@1/consistency/CI per
  tier) + overall CI; `scripts/robustness_eval.py --by-difficulty`. DRY on bootstrap_pass_at_k +
  difficulty_profile.TIERS. +11 tests, EVAL-030's 8 intact.

### Inference / architecture (2026-06-14)
- **INFER-031** [inference, enhancement] `done` (2026-06-14) — Logit-lens per-token depth/convergence
  profiler: `evaluation/depth_profile.py` (`logit_lens` via tied head + final norm — no new weights;
  `convergence_depth` argmax/entropy; `profile_depth` w/ per-tier `by_tier`) + `scripts/depth_profile.py`
  (`--mode/--tau/--by-difficulty`) + eval-menu entry. +17 tests (lens last-layer == forward anchor). Main-safe.
- **DATA-069** [data curation, high] `done` (2026-06-14) — Semantic (embedding) dedup (SemDeDup/D4):
  `data/semantic_dedup.py` (numpy TF-IDF embedder + sklearn-or-numpy KMeans + within-cluster cosine;
  keep highest-quality member when weights given, else centroid-distant); `--dedup semantic`
  (+threshold/clusters, runs after exact) + data-menu choice. +14 tests, numpy-only. Catches reordered/renamed near-dups.

### Post-training / reward design (2026-06-14)
- **EVAL-034** [eval, medium] `done` (2026-06-14) — Verifier-anchored function-step process-credit profiler
  (`evaluation/process_credit.py`): AST function-as-steps, score each via sandbox verifier (assert subset
  / executability probe), length-normalized process_score (resists verbosity hack) + fragility map.
  "Poor-man's PRM". `scripts/process_credit.py` + eval-menu entry. +22 tests. Reuses split_test_cases + runner.
- **INFER-033** [inference, medium] `open` — Wire process_score as a best-of-N tie-break (above heuristic,
  below hard verified/secure verdict; back-compat: all-pass → no reorder).
- **SEC-022** [security, low] `open` — Per-function scan_dangerous in the process profiler (SecCodePRM-style):
  localize WHICH function carries a dangerous pattern, feeding the secure best-of-N tie-break.

### Tech-debt / standards (2026-06-14)
- **TYPE-001** [tooling/quality, high] `done` (2026-06-14) — Strict-typing overhaul per
  `.claude/rules/typing.md`. **55 Pydantic models** in `ui/schemas.py` (single source of truth) →
  `scripts/gen_ts_types.py` generates `webui/src/types.gen.ts` (1:1, drift-guarded by a regen-equals test,
  58 tests); `webui/src/types.ts` is now a thin re-export; shared `webui/src/format.ts` (formatBytes/
  Integer/Float/Percent/Duration/RelativeTime/Params + one exhaustive `formatJsonValue(JsonValue)`).
  ELIMINATED every `any`/`unknown`/`Record<string,unknown>` + catch-all `fmt(unknown)`/`typeof`/inline
  `JSON.stringify`-display + per-component `humanBytes`/`fmtInt` across all 18 panels (6 parallel agents).
  tsc --noEmit EXIT 0, vite build green. Open JSON typed via the one principled `JsonValue` union.
  Follow-up TYPE-002: add FastAPI `response_model=` to endpoints for runtime 1:1 enforcement.
- **TYPE-002** [tooling/quality, high] `done` (2026-06-14) — Wired `response_model=` on all 33 data
  routes in `ui/app.py` (single `from . import schemas as sch` import). Path-reading single-object
  endpoints use `Model | sch.ErrorResponse` unions (matching the TS `*OrError` aliases); list endpoints
  use `list[Model]`. `_UiModel`'s `extra="forbid"` makes any backend dict drift a hard validation error
  at the wire (not a silent mismatch) — verified the contract by validating each no-arg endpoint's real
  output against its schema, then a TestClient sweep of all 21 no-arg + config/config-diff endpoints
  (all 200, no response_model 500s). Surfaced & confirmed safe: `/api/tokenizer` returns `{error}` when
  no tokenizer exists (the union handles it); `/api/eval-history`'s `__init__` re-export shadows its
  submodule for late external imports, but app.py imports submodules first so the live route is fine.
  Endpoints intentionally NOT wired: `/api/pipeline/run` (genuinely-open run JSON → `dict[str, JsonValue]`),
  `/api/jobs/{id}/log` + `/api/jobs/{id}/stop` + `/api/run` + `/api/train/start` (ad-hoc control dicts).
  63 UI tests + ruff green.
- **TYPE-003** [tooling/quality, medium] `open` — Make `ui/*.py` functions RETURN the Pydantic models
  (not bare dicts), closing the last gap to "endpoints return the model" in `.claude/rules/typing.md`.
  BLOCKER: ~50+ existing unit tests subscript the returns (`get_training_status(...)["step"]`); Pydantic
  v2 models don't support `[]`, so this needs a coordinated per-module test migration to attribute access
  (`.step`). Do it module-by-module with parallel agents (disjoint files), each ending green or reverting.

### Contamination-aware eval rigor (2026-06-14, evaluation research)
- **EVAL-036** [eval, high] `done` (2026-06-14) — **Contamination-trust-stratified pass@k.** A bare pass@k
  is unfalsifiable once a benchmark leaks (LiveCodeBench arXiv:2403.07974; SWE-ReBench's
  decontaminated-variant comparison; OpenAI verbatim-reproduction audit). The repo owned only the BINARY
  half (DataLeakageDetector / check_contamination.py / DATA-065) — a yes/no verdict never fed back into the
  SCORE. New `evaluation/contamination_stratified.py`: `score_problem_contamination` (continuous per-problem
  max-containment of prompt OR canonical_solution, REUSING the detector's `_shingles`/`_containment` — DRY,
  not a fork), `contamination_tier` (clean<0.50≤suspect<0.80≤contaminated), `stratified_pass_at_k` (joins
  tiers with `ProblemResult` via `compute_pass_at_k`; `StratifiedPassAtKReport.trusted_pass_at_k` =
  decontaminated score, `trust_delta` = clean−contaminated memorisation signature, None-safe for empty
  tiers / non-estimable n<k; `unmatched_task_ids` surfaced) + `build_contamination_detector`. Library-only
  (consumes existing eval results → no new script → no orphan). +20 tests, CPU-only, no model/GPU/sandbox.
- **EVAL-037** [eval, medium] `open` (2026-06-14) — Bind EVAL-036 into a CLI/report path: reuse
  `check_contamination.py`'s existing `_load_train_docs` (JSONL or sampled `.npy`+tokenizer) to produce the
  `train_docs`, run `evaluate.py` to collect `ProblemResult`s, and emit the stratified report (headline vs
  decontaminated `trusted_pass@k` + `trust_delta`) — wire as an `evaluate.py --contamination-corpus PATH`
  flag or a small `stratified_eval.py` (menu-wired under eval_menu). Needs a checkpoint → GPU → defer/worktree.
- **EVAL-038** [eval, medium-potential] `open` (2026-06-14, ORIGINAL) — **Contamination × verifier-effort
  cross-tab.** Both the containment scorer (EVAL-036) and the difficulty tiers (EVAL-026
  `difficulty_profile`) are model-relative and free. Cross-tabulate them to test the confound "is the model's
  'hard' tier just its *un-memorised* tier?" — a 3×4 (contamination × effort) pass@k grid would expose whether
  apparent reasoning gains are decontaminated. Pure analysis over the two existing reports → main-safe.

### Optimizers / training-stability (2026-06-14)
- **EVAL-035** [eval, medium] `done` (2026-06-14) — Spectral-Alignment divergence-risk diagnostic
  (`evaluation/spectral_health.py` + script + eval-menu): per-layer SA sign-collapse via power iteration
  (u₁ recovery), reuses depth_profile block-iteration; HEALTHY/WATCH/DIVERGENCE-RISK verdict. +22 tests. Main-safe.
- **INFER-035** [inference, medium] `open` — Draft-free prompt-lookup speculative drafter + offline
  acceptance analyzer (PLD/REST, lossless, single-model). Repo-context/FIM-anchored draft source (INFER-037).
- **INFER-036** [inference, medium, WORKTREE] `open` — live lossless self-speculative decode in CodeGenerator (hot path).
- **MODEL-047** [model/training, low, WORKTREE] `open` — online SA-monitored ZClip grad-norm z-score clamp
  in the trainer (A/B-gated on a tiny Muon run; never merged while live).
- **INFER-034** [inference/tooling, low] `open` — use SpectralHealthReport as an RFT checkpoint-acceptance
  gate (reject a self-distilled candidate whose worst-layer sign-collapse regressed, before verifier eval).
- **DATA-071** [data curation, high] `done` (2026-06-14) — Gopher/MassiveText repetition filter
  (`data/filters/repetition.py`): `@register_filter("repetition") RepetitionFilter` + pure
  `compute_repetition_metrics` (`RepetitionMetrics` dataclass) computing dup line/para fractions + char
  fractions, top 2-4 gram char fractions, duplicate 5-10 gram char fractions (published Gopher Table A1
  thresholds in a `RepetitionThresholds` dataclass; reject if ANY strictly exceeds; n-gram screens skip docs
  < min_words=50). Char fractions bounded [0,1] via non-overlapping coverage. Closes the within-document
  repetition gap (across-doc dedup — exact/MinHash/semantic — can't see looping inside one file). Registered
  + listed in data-menu Advanced Filters. +19 tests, CPU-only, no model/GPU. See research-log 2026-06-14.
- **DATA-072** [data curation, medium] `open` — Surface DATA-071's repetition metrics as extra `CodeScorer`
  breakdown signals → per-chunk quality weights (`weight_scoring.py`), so looping/boilerplate down-weights
  the soft score in addition to the hard filter reject. Reuse `compute_repetition_metrics` (no recompute).
- **DATA-073** [data curation/eval, low] `open` — Corpus repetition AUDIT: FineWeb-style threshold-ablation
  report ("how many tokens does each Gopher metric remove") + cross-tab against `semantic_dedup` cluster
  sizes (does within-doc repetition correlate with the semantically-redundant clusters?). Pure offline.
- **DATA-070** [data curation, medium] `open` — D4 diversification pass after semantic dedup (prune
  over-dense clusters toward a target size), reusing DATA-069's clustering.
- **EVAL-033** [eval, low] `open` — Semantic-dedup audit: stratify a corpus by semantic-cluster size +
  quality/verifier tier to quantify redundancy that's semantic-only (caught by DATA-069, missed by exact+MinHash).
- **EVAL-032** [eval, enhancement] `open` — Verifier-stratified depth map: cross depth_profile with
  EVAL-026 tiers + assert-region token map — do correctness-critical tokens converge later than boilerplate?
- **INFER-032** [inference, research/WORKTREE] `open` — Opt-in per-token early-exit decode gated by a
  convergence threshold + exec-validated safe-layer floor, A/B'd on HumanEval pass@1 + CI (EVAL-028).

### UI (2026-06-14)
- **UI-036** [ui, medium] `done` (2026-06-14) — Navigation/IA: grouped the 30+ flat panels into 8
  COLLAPSIBLE sections (Overview, Run & Jobs, Checkpoints & Models, Data, Configs & Pipeline, Evaluation,
  Tokenizer, System & Tools). New reusable `CollapsibleSection.tsx` (a11y `<button>` header + chevron) +
  typed generic `useLocalStorage<T>` hook persisting each section's open/closed state across reloads.
  Pure frontend (no backend/endpoint, zero training risk); section body reuses the exact `.app-grid`
  template (`repeat(auto-fit, minmax(330px,1fr))`, gap 16px) so cards tile identically. tsc + vite green
  (220 KB). Overview + Run & Jobs default-open; the rest collapsed.
- **UI-037..039** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 3 read-only
  inspect views. UI-037 Checkpoint Health Inspector (`GET /api/checkpoint-health`, `CheckpointHealth`:
  size_mb, num_tensors from safetensors header, files, loss/config_stem from metadata.json; mirrors
  checkpoint_info.py) → Checkpoints & Models. UI-038 Project Memory Inspector (`GET /api/memory-stats`,
  `MemoryStats`/`MemoryEntry`; built against the REAL markdown store `memory/manager.py` under
  `.cola/memory/`) → System & Tools. UI-039 Vector Index Stats (`GET /api/retrieval/index-stats`,
  `IndexStats`; reads `retrieval/vector_store.py` JSON sidecar without loading embeddings) → Data.
  +4 Pydantic models (gen_ts_types regen, 67 ifaces), +4 test examples, 6 new files. pytest 70 + tsc +
  vite green (227 KB). Keystone files integrated by main; agents built only their 2 files each.
- **UI-040..041** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 views.
  UI-040 Security/Malware Scan (`GET /api/security/scan?path=&max_files=`, `MalwareScanResult`/`ThreatInfo`;
  bounded synchronous `CompositeMalwareScanner` over a path, fail-closed `is_clean and not had_errors`,
  exhaustive severity switch) → Data. UI-041 Environment Check (`GET /api/env-check`, `EnvCheckReport`/
  `EnvCheckItem`; python/torch/CUDA/GPU/VRAM/deps/HF_TOKEN, structured not just a log) → System & Tools.
  +4 Pydantic models (gen_ts_types regen, 71 ifaces, 31 OrError), +4 test examples, 4 new files. pytest 74
  + ruff + tsc + vite green (232 KB). Keystone integrated by main; agents built only their 2 files each.
- **UI-042..043** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 views.
  UI-042 VRAM Estimate (`GET /api/vram-estimate?config=`, `VramEstimate`/`VramComponent`; reuses the real
  `features/vram_estimator.estimate_vram` math, GPU-independent, per-component breakdown + fits-16GB badge)
  → Configs & Pipeline. UI-043 Project Health (`GET /api/project-health`, `ProjectHealthReport`/
  `HealthDimension`; CHEAP filesystem-derived dimensions — deliberately avoids the heavy pytest/ruff/--help
  subprocess sweep that project_health.py runs, proxies labelled in detail) → System & Tools. +4 models
  (gen_ts_types regen, 75 ifaces, 33 OrError), +4 test examples, 4 new files. pytest 78 + ruff + tsc + vite
  green (237 KB). Keystone integrated by main.
- **UI-046..047** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 read-only
  catalog views. UI-046 Data Filters Catalog (`GET /api/filters-catalog`, `FiltersCatalog`/`FilterInfo`;
  enumerates the 11 `@register_filter` plugins across 4 categories via the real `data/registry`) → Data.
  UI-047 Reasoning Problems browser (`GET /api/reasoning-problems?which=`, `ReasoningProblemSet`/
  `ReasoningProblem`; reads the real `evaluation/problem_loader.load_problem_set` — 67 problems) →
  System & Tools. +4 models (gen_ts_types regen, 84 ifaces, 37 OrError), +4 test examples, 4 new files.
  pytest 87 + ruff + tsc + vite green (246 KB). Keystone integrated by main.
- **INFER-035** [inference, high] `done` (2026-06-15) — Prompt-Lookup Decoding (PLD) OFFLINE acceptance
  analysis. The dangling untracked module `inference/prompt_lookup.py` (draft-free n-gram speculative
  drafter + `analyze_acceptance` measuring α / mean accepted length / idealised speedup with NO model in
  the loop — the "acceptance-first" discipline the 2026 spec-decoding literature mandates) is now brought
  under version control, TESTED (tests/test_prompt_lookup.py, 26 tests inc. a lossless-invariant property
  test) and WIRED end-to-end: `scripts/pld_analysis.py` CLI (loads a .npy token corpus, reports
  acceptance/speedup via cli.kv_table) + an eval-menu entry. MAIN-SAFE (pure-Python, no torch/GPU/model).
  Smoke: 83% acceptance / 2.56x speedup on a repetitive corpus. See research-log 2026-06-15. 138 tests pass.
- **INFER-036** [inference, medium] `open` — Context-seeded PLD (original cross-technique idea, research-log
  2026-06-15): seed the PLD drafter's lookup datastore with the RETRIEVED repo-context tokens (the project's
  `--repo`/`retrieval/vector_store` assembly), REST-style, in addition to the running buffer — a tiered,
  draft-free, training-free, lossless drafter specialised for IDE code completion. First step is OFFLINE:
  extend `analyze_acceptance` to A/B buffer-only vs buffer+repo α on real .npy corpora (no hot-path change)
  to size the win before any live-decoder integration. The DATA-071 n-gram machinery is reusable for indexing.
- **UI-044..045** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 read-only
  results viewers → Evaluation section. UI-044 Benchmark Results (`GET /api/benchmark-results`,
  `BenchmarkResults`/`BenchmarkRun`; scans conventional drop dirs for `inference_benchmark.py --json`
  reports, best throughput/latency per run). UI-045 Safety Eval Results (`GET /api/safety-eval-results`,
  `SafetyEvalResults`/`SafetyEvalRun`/`SafetyProbe`; scans for persisted safety_eval JSON). FINDING:
  benchmark.py/nano_benchmark.py and safety_eval.py currently PRINT only (no persisted artifacts), so both
  views correctly ship as empty-state scanners that light up once those scripts persist JSON — filed as
  DATA/OBS items below. +5 models (gen_ts_types regen, 80 ifaces, 35 OrError), +5 test examples, 4 new
  files. pytest 83 + ruff + tsc + vite green (242 KB). Keystone integrated by main.
- **DATA-072** [data-quality, high] `done` (2026-06-15) — Static educational-value scorer (FineWeb-Edu/Stack-Edu
  proxy, NO LLM, CPU-only). New `data/scorers/educational_value.py` (`EducationalValueScorer`, ScorerProtocol)
  combining 5 weighted signals: comment/docstring density (0.25), example/test presence (0.20), identifier-naming
  quality (0.20), structural completeness (0.20), non-degenerate (0.15, reuses DATA-071 `compute_repetition_metrics`).
  Reuses shared `is_js_ts`/`ScoreMapper` per DRY. Registered in scorers/registry + configs/scoring.yaml
  (enabled:false, weight 0.1) + data_menu Score & Filter option. +tests/test_educational_value_scorer.py (20 tests).
  132 tests pass, ruff clean. See research-log 2026-06-15. Score-cascade follow-up = DATA-073.
- **DATA-073** [data-quality, medium] `open` — Educational-value CASCADE gate (original idea, research-log
  2026-06-15): use the cheap DATA-072 static prior to BOUND the expensive LLM-judge annotation budget — route only
  documents whose static score lands in an *uncertain* middle band to `train_judge_classifier.py`'s judge; skip
  confidently-good/-bad docs. Fuses DATA-072 + the existing judge-distillation pipeline. First step: measure the
  static prior's score distribution on a real corpus to choose the uncertain band, then wire the routing in score_data.
- **UI-048..049** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files). UI-048 Action
  Launcher (`ActionLauncherPanel.tsx`; lets the user pick an allow-listed action, EDIT its args, and launch as
  a background job via the existing `runAction(action, args)` — the UI-017 per-action arg-form gap; trainer
  actions disabled while training is alive, gpu actions amber-badged) → Run & Jobs. NO backend change needed
  (`/api/run` already accepts an args override). UI-049 Tokenizer Vocab Explorer (`GET /api/vocab-search`,
  `VocabSearchResult`/`VocabToken`; substring search over the real `tokenizers` `get_vocab()` + special tokens)
  → Tokenizer. +2 models (gen_ts regen, 86 ifaces, 38 OrError), +2 test examples, 3 new files. pytest 89 +
  ruff + tsc + vite green (252 KB). Keystone integrated by main. (UI-017 partially addressed — arg editing now
  exists; remaining: structured per-arg forms + data-collection wizards.)
- **MODEL-050** [architecture, high] `done` (2026-06-15) — Gemma-2-style logit SOFT-CAPPING (`y = cap*tanh(x/cap)`),
  OPT-IN + DEFAULT-OFF. New `model/logit_cap.py` `soft_cap_logits(logits, cap)` (pure, returns input when cap<=0).
  `ModelConfig` gains `attn_logit_softcap: 0.0` + `final_logit_softcap: 0.0` (0.0 = disabled). FINAL-logit cap is
  WIRED in `Transformer.forward` after the LM head, strictly gated `if final_cap>0`. ATTN cap is NOT wired
  (attention is always fused F.scaled_dot_product_attention with no eager logit hook — forcing an eager branch
  would change the live run's numerics/perf; the field is reserved + the helper is tested). VERIFIED byte-identical
  when off (torch.equal baseline) and live small_react_best config has no softcap key → unaffected. +tests/
  test_logit_cap.py (11). CRITICAL gate passed: test_checkpoint (36+4skip) + test_model (19) green. See
  research-log 2026-06-15. Completes the layered logit-control stack with the existing QK-Norm.
- **MODEL-051** [architecture, medium] `open` — (a) Wire ATTENTION-logit soft-cap: add an eager attention branch
  (used only when attn_logit_softcap>0) so the pre-softmax scores can be capped — keep the fused SDPA path as the
  default (zero-cost when disabled). (b) Config sweep + measure: enable final_logit_softcap on a SMALL eval and
  measure pass^k/pass@k (EVAL-037) before vs after to test the research-log hypothesis (soft-cap raises single-shot
  reliability / shrinks the capability-reliability gap more than it changes capability).
- **MODEL-052** [optimizers, medium] `open` — Optimizer-aware quality weighting (original idea, research-log
  2026-06-15). The project's `.weights.npy` quality weighting scales the LOSS per sample (magnitude), but the
  EXISTING Muon optimizer (`training/optimizer.py`, used by 4080_max) ORTHOGONALIZES the 2D update — discarding
  per-coordinate magnitude, keeping direction — so loss-level weighting is partly washed out for Muon-optimized
  matrices (it still works for the AdamW-optimized embedding/head/scalars). Proposal: hybrid weighting keyed to the
  optimizer split — loss-level weighting for the AdamW params, but DATA-SAMPLING-level (quality-weighted/curriculum
  sampling) weighting for the Muon params (changes WHICH directions are seen → survives orthogonalization).
  Measurable: a small AdamW-vs-Muon × loss-weighted-vs-sampling-weighted ablation on tiny. NOTE: this cycle a
  duplicate standalone Muon impl was generated and DISCARDED (DRY) — the existing `optimizer.py` Muon is complete
  (MODEL-025/047) and kept; the contribution here is the analysis + this experiment.
- **UI-061..062** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 action launchers.
  UI-061 Prepare Data wizard (`PrepareDataPanel`; config select + score/filter-mode/dedup/workers options →
  launches prepare_data.py via runAction with its real non-interactive flags; reuses existing `prepare_data`
  ACTION, no backend change) → Data. UI-062 Export Model launcher (`ExportModelPanel`; checkpoint + format
  {gguf-f16/q8/q4, ollama, quantize, benchmark} → launches export_model.py, non-interactive when --action is
  passed; new `export_model` ACTIONS entry, CPU) → Checkpoints & Models. Both verified non-interactive (won't
  hang a job, unlike scrape_github/BUG-132). No schema change. tsc + vite green (289 KB).
- **UI-060** [ui, medium] `done` (2026-06-15) — Combine Datasets launcher (`CombineDatasetsPanel`): select 2+
  prepared `.npy` datasets, assign per-dataset weights, set output, and launch combine_datasets.py as a background
  job via runAction with `--datasets PATH:WEIGHT ... --output PATH` (the script's real non-interactive flags; no
  phantom MinHash knob — that path is TUI-only). New `combine_datasets` ACTIONS entry (CPU, default args `[]`).
  → Data. No schema change. tsc + vite green (281 KB).
- **BUG-132** [tooling, medium] `open` — `scripts/scrape_github.py` is INTERACTIVE (only `--tokenizer` is a CLI
  flag; mode/language/min-stars/license/max-repos/output are gathered via cli.choose/input prompts). Launching it
  as a UI background job (stdin=DEVNULL) would HANG on the first prompt, so the GitHub-scraper UI launcher was NOT
  shipped this cycle (panel discarded). FIX: add a non-interactive flag set (`--language --min-stars --license
  --max-repos --output-dir --non-interactive`/preset bypass) so it's UI-launchable; then ship the launcher.
  (Discovered building the scraper launcher.)
- **UI-058..059** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files). UI-058 Docs Browser
  (`GET /api/docs` + `GET /api/doc?path=`, `DocsList`/`DocFile`/`DocContent`; lists the 26 docs/ guides +
  deep-dives, lightweight no-dep markdown render; PATH-GUARDED read confined to docs/ — traversal/out-of-tree
  rejected) → System & Tools. UI-059 Data Collection launcher (`CollectDataPanel`; config select + read-only
  source-weights display, launches collect_data.py as a background job via the existing runAction; no backend
  change — collect_data already allow-listed) → Data. +3 models (gen_ts regen, 107 ifaces), +3 test examples,
  3 new files. pytest 110 + ruff + tsc + vite green (277 KB). Keystone integrated by main.
- **UI-056..057** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 read-only views
  that surface the autonomous loop's OWN state. UI-056 Backlog Viewer (`GET /api/backlog`, `BacklogView`/
  `BacklogItem`; parses ai_backlog.md into a filterable table — 288 items, 66 open/222 done — status+category
  filters, badges) → System & Tools. UI-057 Research Log viewer (`GET /api/research-log`, `ResearchLog`/
  `ResearchEntry`; parses docs/research-log.md — 56 entries, date/area/source-count/original-idea badges) →
  System & Tools. +4 models (gen_ts regen, 104 ifaces), +4 test examples, 4 new files. pytest 145 + ruff + tsc +
  vite green (272 KB). Keystone integrated by main.
- **SEC-026** [safety, high] `done` (2026-06-15) — Static CWE vulnerability scorer (`data/scorers/cwe_security.py`,
  `CweSecurityScorer`). No-execution regex/pattern screen for the recurring CWE families NOT covered by the
  injection scorer: CWE-78 (cmd injection), CWE-94/95 (eval/exec), CWE-502 (unsafe deserialization), CWE-89 (SQL
  injection), CWE-327/328 (weak crypto md5/sha1), CWE-330 (insecure random near secrets), CWE-22 (path traversal).
  Structured per-CWE findings {cwe,name,severity,line} + a graded 0-1 score via shared `ScoreMapper`; language-aware
  via `is_js_ts`; strips comments to cut false positives. Reuses shared utils (DRY); distinct from injection_scorer
  + security/code_patterns. Registered in scorers/registry + configs/scoring.yaml (enabled:false, w0.1) + data_menu.
  +tests/test_cwe_security_scorer.py (38 tests). See research-log 2026-06-15.
- **SEC-027** [safety, medium] `done` (2026-06-15, part b) — CWE eval probe DONE: added a `cwe` suite to
  `evaluation/safety_probes.py` (17 CWE-prone prompts, registered in SUITES + `all`) + `cwe_probe_result` that
  REUSES the SEC-026 `CweSecurityScorer` (reuse-guard tests assert no reimplemented patterns); wired into
  `safety_eval.py --suite cwe` (auto-exposed via choices=sorted(SUITES)) + a CWE option in the eval menu.
  +tests/test_cwe_safety_probe.py (12). 125 tests pass, ruff clean. REMAINING (part a, see SEC-028): the
  data-FILTER side — drop/down-weight training examples containing CWE patterns — to fully close the eval->data loop.
- **SEC-028** [safety, medium] `open` — CWE eval->data loop, part (a): use CweSecurityScorer (SEC-026) as a TRAINING
  DATA filter/down-weighter so the model learns from secure code (attacks the "train on secure corpora" root cause);
  it's already a registered data scorer (configs/scoring.yaml, enabled:false) — wire it into the prepare/score path
  as an opt-in CWE-vulnerability filter + measure the generated-CWE rate (SEC-027 probe) before/after to close the loop.
- **UI-067** [ui, high] `done` (2026-06-15) — Panel redesign batch 4 (3 parallel agents, disjoint files):
  MetricsChartPanel → polished dependency-free SVG loss/tok-s chart (legend toggle, gridlines, callout) +
  SystemInfoPanel → env tiles + per-GPU rows + disk meter (FINISHES the Dashboard); EvalsPanel → artifact
  list+metric-tile detail + EvalHistoryPanel → trend sparkline + delta table (Evaluation); PipelinePanel → run
  cards w/ progress + PipelineManagerPanel → 10-stage TIMELINE/stepper (numbered rail nodes colored by status,
  per-stage override/reset preserved) (Configs & Pipeline). All preserve prop signatures + every api call; reuse
  tokens; CSS merged centrally (28.5 KB). tsc + vite green. The Dashboard, Run & Jobs, Checkpoints, Data,
  Configs & Pipeline, Evaluation, Tokenizer pages are now redesigned. NEXT (UI-068+): the System & Tools page
  panels (Features/Reasoning/Storage/ScriptsCatalog/Router/ModelCard/Export/SftData/DataSources + the newer
  inspect panels' visual consistency), then resume the OVERDUE SECONDARY model item.
- **UI-066** [ui, high] `done` (2026-06-15) — Panel redesign batch 3 (3 parallel agents, disjoint files):
  Datasets → card grid + DataStats → stat tiles/histogram (Data); Configs → list+viewer (parsed table or raw)
  + ConfigDiff → colored add/del/chg rows (Configs & Pipeline); Tokenizer → vocab stat + special-token chips +
  Tokenize → token-chip visualizer (Tokenizer). All preserve prop signatures + api calls; reuse design tokens;
  +~50 namespaced CSS classes (incl. a shared .textarea) merged centrally. tsc + vite green (CSS 21 KB). NEXT
  (UI-067+): Pipeline/PipelineManager, Evals/EvalHistory, Features/Reasoning/Storage/ScriptsCatalog, Router/
  ModelCard/Export, SftData/DataSources/MetricsChart → consistent styling, then a SECONDARY model item (overdue).
- **UI-065** [ui, high] `done` (2026-06-15) — Panel redesign batch 2 (3 parallel agents, disjoint files): (a)
  SystemPanel → GPU gauges (util/mem/power level-colored meters) + HealthPanel → scannable check-list w/ score
  badge (Dashboard); (b) JobsPanel → jobs list w/ status chips + relative time + preserved SSE log-follow/stop +
  LogsPanel → two-pane log picker/viewer (Run & Jobs); (c) CheckpointsPanel → grouped-by-model list + Compare →
  readable side-by-side A/B + delta strip (preserves the verified getCheckpointCompare). All reuse design tokens;
  +~40 namespaced CSS classes merged centrally. tsc + vite green (CSS 16.6 KB). NEXT (UI-066+): remaining crude
  panels (Datasets/DataStats/Configs/Pipeline tables, Tokenizer, eval tables) → consistent component styling.
- **UI-063** [ui, high] `done` (2026-06-15) — App SHELL redesign: replaced the single book-length scroll of 62
  stacked panels with a real application — fixed Sidebar (grouped nav Monitor/Build/Inspect + line-icons + live
  training pill), hash router (8 bookmarkable PAGES, one shown at a time), topbar with section title/subtitle +
  live status, exhaustive SectionId switch. nav model in src/sections.ts; index.css restructured to an app-shell.
- **UI-064** [ui, high] `done` (2026-06-15) — Panel redesign batch 1 + a connection banner. (a) ActionsPanel:
  raw `script|args|run` TABLE → grouped ACTION GALLERY (category cards Data/Eval/Training/Export/Tools, human
  titles, de-emphasized script subtext, trainer/GPU badges, primary Run + collapsible editable args, per-card
  result chip, trainer-while-alive guard). (b) TrainingPanel → wide live HERO (big step + progress bar, loss/ppl/
  tok-s/s-it stat tiles, last-log strip, graceful idle). (c) Global CONNECTION BANNER when SSE drops — turns the
  cryptic per-panel "Failed to fetch" into ONE clear "backend unreachable, run ps/cola-ui.ps1" message. NOTE: the
  user's "Failed to fetch on everything" was a STALE/STOPPED server (verified: a fresh ui_server returns 200 on
  /, /api/status, /api/actions, /api/jobs AND /api/checkpoints/compare) — restart + hard-refresh fixes it. tsc +
  vite green. FOLLOW-UP UI-065+: redesign the remaining panels (Jobs/Logs, System gauges, eval/data tables, etc.).
  (OPT-IN, default OFF). New `reasoning/rewards/overlong.py`: `soft_overlong_penalty(length, max_length,
  soft_buffer)` (0 below buffer → linear ramp → -1 at/over max) + `apply_overlong_shaping(reward, ...)`. Wired
  into `grpo.py` as `apply_overlong_shaping_rewards` (mirrors the existing `apply_security_penalty` modifier,
  applied AFTER security penalty, BEFORE advantages; measures completion tokens vs max_new_tokens; `num_overlong_shaped`
  metric) + 3 opt-in GRPOTrainer params; `configs/reasoning.yaml` `reasoning.overlong_shaping{enabled:false,
  soft_buffer:64, scale:1.0}` read in train_reasoning.py (raw-YAML, since Config.from_yaml doesn't surface the
  reasoning subsection — matches existing pattern + the config-wiring test). Skipped a length_normalize helper
  (GRPO already has Dr.GRPO length_norm — DRY). +tests/test_overlong_shaping.py (29 tests). Verified 198 passed
  (1 skipped) across the reasoning/grpo/reward suite — no regression. NOT the live pretraining path. See
  research-log 2026-06-15.
- **MODEL-049** [post-training, medium] `open` — Reliability-gap-weighted GRPO advantages (original idea,
  research-log 2026-06-15): scale each problem's GRPO advantage by a function of its capability-reliability gap
  `pass@k − pass^k` (EVAL-037) — computed for free from the n-samples/c-correct GRPO already rolls out — so RL
  compute concentrates on capable-but-unreliable problems (max single-shot-reliability headroom). Closes the
  eval→post-training loop. First step: log the per-problem gap during GRPO rollouts (no advantage change) to
  confirm the signal, then add an opt-in advantage scaler.
- **UI-054..055** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files). UI-054 Training
  Manifest viewer (`GET /api/training-manifests`, `TrainingManifests`/`TrainingManifest`; per-checkpoint
  provenance — flattens the nested `training_manifest.yaml` model/training sections + scans newest step_* for
  latest_step; 3 real rows) → Checkpoints & Models. UI-055 Checkpoint Averaging (Model Soup) LAUNCHER
  (`CheckpointAveragePanel`; multi-select 2+ checkpoints + method/output → launches average_checkpoints.py as a
  background job via runAction; new `average_checkpoints` ACTIONS allow-list entry, CPU not trainer) →
  Checkpoints & Models. +2 models (gen_ts regen, 100 ifaces), +2 test examples, 3 new files. Fixed the endpoint
  to pass the checkpoints dir (not project root) to the manifest scanner. pytest 103 + ruff + tsc + vite green
  (267 KB).
- **UI-052..053** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 read-only
  viewers. UI-052 LR Finder Results (`GET /api/lr-finder-results`, `LrFinderResults`/`LrFinderRun`/`LrPoint`;
  loss-vs-LR curve via dependency-free SVG + suggested-LR badge; find_lr.py only saves a PNG today so it
  empty-states until a JSON is dropped) → Configs & Pipeline. UI-053 Repo Scoring (`GET /api/repo-scores`,
  `RepoScoresResult`/`RepoScore`; ranked repos from score_repos.py --json, empty-state until saved) → Data.
  +5 models (gen_ts regen, 98 ifaces), +5 test examples, 4 new files. pytest 101 + ruff + tsc + vite green
  (262 KB). Keystone integrated by main.
- **EVAL-037** [eval, high] `done` (2026-06-15) — pass^k (consistency/reliability) metric — the unbiased mirror
  of pass@k. `evaluation/metrics.py`: `pass_hat_k(n,c,k) = prod (c-i)/(n-i) = C(c,k)/C(n,k)` (0 if c<k, 1 if
  c==n, c/n if k==1), `compute_pass_hat_k` aggregate mirroring compute_pass_at_k, and
  `capability_reliability_gap = pass@k − pass^k`. `format_results` now prints pass^k + the gap under each
  pass@k line (k>1); evaluate.py surfaces it by default. +tests/test_pass_hat_k.py (19 tests: brute-force vs
  comb, monotonicity pass^k ≤ pass@k, ValueErrors, aggregation). See research-log 2026-06-15. 68 tests pass.
- **EVAL-038** [eval, medium] `open` — Capability–reliability gap as a temperature/routing diagnostic (original
  idea, research-log 2026-06-15): sweep temperature and report the pass@k/pass^k gap to pick (a) the
  reliability-max temperature for the single-shot IDE/base path vs (b) the capability-max temperature for the
  `--best-of N` verifier path; also use a low pass^k as a cheap pre-screen for where PLD/speculative decoding
  (INFER-035/036) helps least. First step: a small temperature-sweep harness reporting the gap per setting.
- **UI-050..051** [ui, medium] `done` (2026-06-15) — Batch (parallel agents, disjoint files): 2 read-only
  views. UI-050 Scoring Config (`GET /api/scoring-config`, `ScoringConfig`/`ScorerConfigEntry`; merges
  configs/scoring.yaml enabled/weight with `registry.list_available_scorers` availability + scorer-class
  purpose — surfaces the new DATA-072 educational_value scorer) → Data. UI-051 Regression History
  (`GET /api/regression-history`, `RegressionHistory`/`RegressionRun`/`RegressionMetric`; scans
  regression_test.py `--save` JSON artifacts, empty-state until saved) → Evaluation. +5 models (gen_ts regen,
  93 ifaces), +5 test examples, 4 new files. ALSO fixed a tsc break committed in TYPE-003: `postJson` is now
  generic `<T extends object>` so named request interfaces (RunRequest/TrainStartRequest) are assignable.
  pytest 96 + ruff + tsc + vite green (257 KB).
- **TYPE-003** [typing, medium] `done` (2026-06-15) — Typed the weakly-typed `/api/run` + `/api/train/start`.
  Added `RunRequest{action: str; args: list[str] | None}` and `TrainStartRequest{config: str; resume: str |
  None}` Pydantic models (registered in gen_ts_types `_MODEL_ORDER`); endpoints now take the typed body with
  `response_model=sch.Job | sch.ErrorResponse`. Fixed `jobs.start()` to include `returncode: None` so its
  return is a COMPLETE `Job` (was missing the required field — only worked before because validation was
  bypassed by bare `JSONResponse`). Typed `runAction`/`startTraining` bodies in api.ts. Verified: unknown
  action→400, extra key→422 (extra=forbid now enforced), valid run→200 returning a validated Job; 91 tests
  pass, ruff clean, frontend tsc clean.
- **OBS-001** [tooling/observability, low] `open` — benchmark.py, nano_benchmark.py and safety_eval.py print
  metrics to console but DON'T persist JSON, so the new UI-044/045 viewers have nothing to read until a run
  is saved. Add an opt-in `--json/--output` persist path (inference_benchmark.py already has one) writing to
  a conventional dir (e.g. reports/) so results are viewable in the UI + trackable over time. (Discovered
  building UI-044/045.)
- **OPS-002** [tooling/infra, high] `done` (2026-06-15) — CORRECTED a dangerous false-positive in the
  training hang-detector. The small_react_best run logs only every ~100 steps and is dataloader-bound, so
  ~22-30 min elapse between log lines and GPU power legitimately fluctuates ~70-160W (avg <130W). The old
  criteria (log-mtime stale >10min + power <130W) repeatedly diagnosed HEALTHY slow training as HUNG (the
  run was advancing the whole time: 8500->8600->8700->8800, loss ~1.31/ppl 3.7). Only the taskkill
  access-denied wall (OPS-001) prevented an erroneous kill of a healthy run. FIX: the ONLY hang signal is
  the STEP NUMBER not advancing for >45 min (AND no new checkpoint AND ideally a .err Traceback). Rewrote
  the watchdog Monitor to parse max step and alert only on >45min step-stall; rewrote the cron babysitting
  criteria to match; power/log-mtime are explicitly NOT hang signals for this run. LESSON: prefer waiting +
  re-verify over killing; a wrong kill of a healthy run is far worse than a late recovery.
- **OPS-001** [tooling/infra, high] `open` — Autonomous training babysitting is BLOCKED when the trainer
  runs at higher OS integrity than the agent shell: `taskkill /F /T` returns Access-Denied for the whole
  tree (verified 2026-06-15 on the live hung run, PIDs 38260/13336/workers). The babysitter correctly
  detects the hang (avg GPU power <130W) but cannot kill/relaunch. WORKAROUND: launch training from a
  NON-elevated shell (same integrity as the agent) so `ps/cola-train-resume.ps1` recovery works; otherwise
  the human must kill+relaunch from an elevated PowerShell. Consider a privileged helper or a UI/IPC
  shutdown channel the trainer honors (cooperative stop file) so recovery never needs taskkill.
- **BUG-130** [bug, medium] `done` (2026-06-15) — `features/menus/tools_menu.py` project-memory action
  called a NON-EXISTENT API (`MemoryManager.initialize()`, `list_recent()`, `MemoryUpdater`, a `project.db`).
  FIXED: rewrote the project-memory submenu to the real markdown store API (`MemoryManager(project_root)`,
  `is_initialized`/`init_project`/`export`/`compact`/`stats`, module-level `_iter_sections`); dropped the
  unbacked "Edit Memory Entry" option; friendly message when uninitialized. +tests/test_cli_menu_fixes.py.
- **BUG-131** [bug, medium] `done` (2026-06-15) — `master_menu/_vector_store_stats` constructed
  `VectorStore(str(path))` (path where `embed_dim: int` is expected) and read non-existent `stats()` keys.
  FIXED: rewrote to check for the persisted `{base}.json` at `data/vector_index`, `VectorStore()` +
  `store.load(base)`, and display the real `stats()` keys (`total_items`/`embed_dim`/`memory_mb`/
  `unique_sources`); friendly "no index built yet" when absent. +tests/test_cli_menu_fixes.py.
- **UI-034** [ui, high] `open` — Execute pipeline stages from the UI. FINDING (this cycle): the CLI's
  per-stage handlers in `pipeline_menu.py` are INTERACTIVE and multi-command (they call cli.confirm/
  cli.choose, prompt to train a tokenizer, pick datasets, choose scoring) — NOT a pure argv builder. So
  this needs a NON-INTERACTIVE stage-plan builder (`build_stage_plan(run, stage) -> list[StageCommand]`,
  defaults instead of prompts, reusing run_manager.resolve_input + _model_scale) extracted into
  `pipeline/`, consumed by BOTH the CLI and the UI, with stage execution launched via JobManager behind
  the existing trainer/GPU guard (stages 3/6/7/9/10 are GPU). Own focused effort — do not half-extract.
- **UI-035** [ui, high] `done` (2026-06-14) — Inspect-tools parity: tokenizer-health + data-stats views
  (parallel agents: backend keystone w/ DRY extraction + 2 disjoint frontend panels). Extracted the CLI
  tools' logic into SHARED libs so CLI and UI run identical code: `cola_coder.tokenizer.health`
  (run_health_checks battery) and `cola_coder.data.stats` (compute_data_stats incl. quality-tier
  distribution + dtype-bounded unique-token estimate); refactored `scripts/{tokenizer_health,data_stats}.py`
  to thin wrappers (deleted ~200 lines of dup) and repointed their existing tests at the libs. 2 new
  endpoints `/api/tokenizer-health` + `/api/data-stats` (response_model error-unions; CPU-only — tokenizer
  load + numpy mmap, no GPU). 2 new panels (TokenizerHealthPanel pass/fail table, DataStatsPanel token
  stats + quality-tier bars). +4 Pydantic models, types regenerated (62 interfaces). 214 affected tests
  green (incl. fixed regressions in the moved tests) + tsc + vite (218 KB).
  Follow-up UI-034: extract a shared StageRunner so the pipeline manager can EXECUTE stages (the CLI's
  ~600 lines of per-stage arg-building are blocking + coupled) behind the GPU/trainer guard.
- **UI-033** [ui, high] `done` (2026-06-14) — Live job-log streaming + GPU-action guard (parallel agents:
  backend keystone + 2 disjoint frontend panels). Backend: SSE `GET /api/jobs/{id}/stream` follows a job's
  log file (initial tail then byte-offset incremental reads → no multibyte corruption), pushing strict
  `JobLogChunk` ({text, done}) frames, terminating with a `done` frame when the process exits. Tagged 11
  GPU-heavy non-trainer actions with `gpu: true` (new `ActionDef.gpu` field). Frontend: `JobsPanel` now
  LIVE-FOLLOWS via EventSource (lifecycle in a useEffect keyed on job id, doneRef stops reconnect thrash,
  5000-line cap, auto-scroll); `ActionsPanel` shows a GPU badge and `window.confirm`s before launching a
  GPU action while training is live (fed `trainingAlive` from the event stream) — protects the live run's
  VRAM. +2 Pydantic models, types regenerated. 70 UI tests (stream capture+done, gpu tags) + tsc + vite green.
  Follow-up UI-034: actually EXECUTE pipeline stages from the UI behind this same GPU/trainer guard.
- **UI-032** [ui, high] `done` (2026-06-14) — Pipeline Run Manager (parallel agents: backend keystone +
  frontend panel). New `ui/pipeline_ops.py` wraps `PipelineRunManager` with PURE STATE ops (create / reset
  / set-override / delete / detail) — never executes a stage, loads a model, or touches the GPU, so it's
  safe alongside the live trainer. 5 new routes (`/api/pipeline/{detail,create,reset,override,delete}`)
  with `response_model` unions (TYPE-002 style); strict name whitelist (filename-safe, no traversal),
  refuses to clobber, validates config exists + stage numbers. New `PipelineManagerPanel.tsx` (create form
  w/ config dropdown + optional-stage skips, selectable run list, per-stage table w/ reset/override, delete
  w/ confirm). +3 Pydantic models (PipelineStageState/RunDetail/DeleteResult), types regenerated (58
  interfaces). 13 new pipeline-ops tests + drift examples; 77 UI tests green, tsc+vite green (213 KB).
  Follow-up UI-033: execute stages from the UI (needs a stage→script runner + a GPU-load guard like the
  trainer guard, since stages 3/6/7/9/10 load the model — defer while training is live).
- **UI-030..031** [ui, high] `done` (2026-06-14) — Batch 10 (parallel agents): model-card view
  (`/api/model-card` — arch/params/training-provenance/tokenizer + rendered markdown per checkpoint) +
  features WRITE path (`POST /api/features/set` — line-level toggle preserving comments/order, atomic,
  refuses unknown keys; FeaturesPanel gets toggle switches). First mutate-config endpoint. 38 `/api` routes.
  +27 UI tests, tsc+vite green (56 modules).
- **UI-028..029** [ui, high] `done` (2026-06-14) — Batch 9 (parallel agents): storage view
  (`/api/storage` — resolved tokenizer/data/checkpoint paths + capped disk footprint) + checkpoint
  COMPARE (`/api/checkpoints/compare` — param/tensor/metadata/dtype diff between two checkpoints, reuses
  checkpoint_detail, header-only). 36 `/api` routes. +29 UI tests, tsc+vite green (55 modules).
- **UI-026..027** [ui, high] `done` (2026-06-14) — Batch 8 (parallel agents): SFT/instruction-JSONL
  view (`/api/sft` + `/api/sft/preview` — list+preview instruction/reasoning datasets) + CLI parity
  coverage catalog (`/api/scripts` — all 62 scripts by category, exists-on-disk dots). Actions allow-list
  expanded to 21 incl. TRAINER-class (train_sft/train_reasoning/train_router/upcycle/find_lr/full_pipeline)
  which `/api/run` now REFUSES (409) while a trainer is alive — never a 2nd trainer; ActionsPanel shows a
  trainer badge + refusal message. 26 actions (6 trainer-guarded), 34 `/api` routes. +26 UI tests, tsc+vite green.
- **UI-024..025** [ui, high] `done` (2026-06-14) — Batch 7 (parallel agents): tokenizer playground
  (POST `/api/tokenize` — encode text → ids/token pieces, GPU-free) + project-health checklist
  (`/api/health`, filesystem checks → 0-100 score). 31 `/api` routes. +24 UI tests, tsc+vite green.
- **DATA-068** [data curation, medium] `open` — Robustness-driven augmentation: feed fragile
  (clean-pass/perturbed-fail) problems' reworded docstrings back as paraphrase-augmented SFT
  prompts paired with the verified solution (ties EVAL-030 → RFT/SFT flywheel; WORKTREE — train data).

### Ops / perf (2026-06-13)
- **PERF-001** [ops/training-throughput, high] `open` — The autonomous loop's own heavy
  CPU activity STARVES the live training. Diagnosed during a cycle: GPU at 99% util but
  only **111 W / 320 W** (temp 40°C, not thermal) = data-starved (waiting on the 8
  dataloader workers), with s/it degrading 10→48 under concurrent pytest/agent/command
  load and RECOVERING (48→24) the moment heavy work stopped. ACTION/policy: while a
  long CPU-bound pretraining run is live, keep cron cycles LIGHT — research (network) +
  docs/backlog edits + tiny commits; DEFER pytest suites, parallel worktree agents, and
  other heavy local compute (or run them in small bursts), since they 4× the training
  wall-clock. Consider: lower the cron frequency during training, nice/affinity-pin the
  trainer, or reduce num_workers if oversubscribed. The babysitter check itself is cheap.
- **MODEL-035** [model/moe, medium] `open` (MoE research 2026-06-13) — Aux-loss-FREE
  MoE load balancing (DeepSeek-V3): replace/augment the MoE router's auxiliary
  load-balancing loss with per-expert BIAS terms updated from running load stats —
  selection uses biased scores, gating weights use original scores. Stage-7 MoE only
  (off the live dense run); touches features/moe_layer.py router → worktree.
- **MODEL-036** [model/moe, medium] `open` (MoE research 2026-06-13) — Drop-Upcycling
  for stage-7: upcycle dense→MoE with PARTIAL re-init of expert weights (not plain copy)
  for better specialization; switch router to softmax-then-topK. Improves
  upcycle_to_moe.py. Off the live dense run.
- **IDEA-012** [research/moe, medium-potential] `open` (2026-06-13) — Domain-aligned
  expert init: warm-start/bias upcycled MoE experts toward the project's semantic-router
  domains (React/Next/GraphQL/Prisma/Zod/Testing) so experts specialize along the actual
  specialist vision. Combines MoE upcycle (stage 7) + the domain router (stage 8) +
  Drop-Upcycling partial re-init (MODEL-036).

### Training recovery (2026-06-13 — found during a real crash recovery)
- **BUG-127** [tooling/training-recovery, critical] `done` (2026-06-13) — CRITICAL
  resume-recovery bug: `ps/cola-train-resume.ps1` passed `--resume <path>` via
  `Start-Process -ArgumentList` as an ARRAY; the project path contains a space
  ("ai research"), which Start-Process did NOT re-quote, so argparse got a split path
  and crashed (`unrecognized arguments: research\cola-coder\...step_00002000`). Effect:
  EVERY reboot/crash recovery resuming from a checkpoint would have FAILED silently —
  the babysitter "relaunches" but the process dies on startup. FIX: build a single
  explicitly-quoted argument string (`--resume "<path>"`, config/data quoted too).
  Verified live: relaunch resumed from step_00002000 and is progressing.
- **BUG-128** [inference/loading, high] `done` (2026-06-14) — USER-REPORTED crash: interactive
  generation from the `small_react_best` checkpoint (dim=768) loaded with `configs/tiny.yaml`
  (dim=512) → `load_state_dict` size-mismatch crash on every tensor; the user couldn't generate from
  their own training checkpoint. ROOT CAUSE: generate.py built the model from the passed `--config`
  yaml, trusting it over the checkpoint's real architecture. FIX: a checkpoint is ground truth for
  its own architecture — new `apply_model_config_from_checkpoint(config, checkpoint)`
  (inference/loading.py) reads the saved `metadata.json` model config and overrides `config.model`'s
  scalar arch fields (dim/n_layers/n_heads/n_kv_heads/ffn_dim_multiplier/vocab_size/max_seq_len/
  rope_theta/qk_norm/...) BEFORE the model is built; nested dicts (rope_scaling/moe) skipped (handled
  elsewhere). Wired into generate.py + the central `load_generator` (serve/smoke_test/RFT all robust).
  Reads checkpoint metadata only (no checkpoint.py change) → main-safe. Tests:
  test_apply_model_config.py +5 incl. one reproducing the EXACT crash against the real checkpoint
  metadata (tiny 512 → corrected 768); ruff green. FOLLOW-UP: BUG-129 — also fix the menu's config
  selection so it doesn't pass tiny.yaml for non-tiny checkpoints (the loading fix already makes
  generation work regardless, but the menu should pick the right config for clarity).
- **BUG-129** [tooling/menu, medium] `open` (2026-06-14) — the interactive-generation menu passed
  `configs/tiny.yaml` for a `small_react_best` checkpoint (root cause of the BUG-128 crash, now
  mitigated in the loader). The menu should resolve the config from the checkpoint's run / metadata
  (or the `auto/<name>.yaml`) rather than defaulting to tiny.yaml. Menu/tooling → main-safe.
- **BUG-126** [training/robustness, high] `done` (2026-06-13) — Checkpoint save crashed
  the whole run on a transient Windows file lock: at step 2500 `save_checkpoint`'s
  `tmp_dir.rename(final_dir)` raised `PermissionError [WinError 5]` (Defender/indexer
  holding a handle on a freshly-written safetensors) → killed the unattended run (it
  had reached loss 1.58 / ppl 4.9). FIX: `_atomic_replace_dir()` retries the
  rmtree+rename with linear backoff (6×) + copytree-fallback, so a transient lock no
  longer crashes training. Tests: test_checkpoint_atomic_replace.py +4; 40 checkpoint
  tests + ruff green. Resume-compatible (load path unchanged) — applied before relaunch.
- **IDEA-011** [research/agents, medium-potential] `open` (2026-06-13) — Self-repair
  execution loop: on a failed tsc/test verification, feed the sandbox's ERROR OUTPUT
  back to the model and regenerate (bounded retries) instead of blind best-of-N
  resampling — the 2026 agentic execution-feedback pattern (3-5× over zero-shot on
  SWE-bench). Combines agent loop + sandbox verifier + FIM; also a GRPO signal (reward
  solving-after-feedback). Inference + reasoning (non-train).

### SECURITY — output guardrail + remaining items

- **SEC-017** [security/output-guardrail, medium] `done` (2026-06-13, safety research)
  — Functional≠secure (secure-pass@1 <12% even at functional pass@1 >50%). Extracted a
  CANONICAL dangerous-code static scanner `security/code_patterns.py`
  (`scan_dangerous`/`is_dangerous`) — the original Python-only safety_eval set
  EXTENDED with TS/JS patterns (new Function, child_process, dangerouslySetInnerHTML,
  document.write, vm.runInThisContext, pickle.loads, unsafe yaml.load), with
  negative-lookbehind so benign `regex.exec(` isn't flagged. safety_eval now imports
  it (DRY — single source of truth). Wired as a SECURITY SCREEN in the distillation
  verifier (scripts/generate_distillation_data.py): completions with dangerous
  patterns are rejected before being distilled (the Anthropic "filter at source"
  philosophy — don't train the student on insecure code). Tests: test_code_patterns.py
  +6 incl. precision (regex.exec not flagged) + DRY-shared assertion; 45 distillation/
  pattern tests + 39 safety tests + ruff green. Follow-up IDEA-008 (secure best-of-N +
  security-aware GRPO reward).
- **IDEA-008** [research/safety, medium-potential] `done` (2026-06-13) —
  Security-aware best-of-N + GRPO reward, BOTH halves landed. (inference) best_of_n
  ranks by `(verified, secure, score)` — among equally-verified candidates the SECURE
  one wins (insecure flagged in details); security only breaks ties. (RL) GRPOTrainer
  gained `security_penalty` (default 0.0): `apply_security_penalty` subtracts λ from a
  solution's reward when its completion has a dangerous pattern (SEC-017 scanner),
  creating advantage signal toward secure completions without changing functional
  correctness; `num_security_penalized` surfaced in step metrics. grpo.py is
  reasoning-only (NOT the live pretraining path → safe on main). Tests:
  test_best_of_n.py::TestSecurityAwareRanking +2 and
  test_modern_techniques.py::TestSecurityPenalty +3 (penalize-only-dangerous,
  zero-penalty no-op, negative-advantage-for-dangerous); 72 best_of_n + 69
  grpo/checkpoint + 57 surrogate/reward tests + ruff green.
- **MODEL-026** [reasoning, medium] `done` (2026-06-13) — DAPO dynamic sampling
  (resampling half). Added `GRPOTrainer.train_step_resampled(problem_sampler, …)`:
  when a group collapses (zero reward variance → the existing `skipped:True` from the
  BUG-121 guard), it redraws a FRESH problem and retries up to `max_resample_attempts`
  (config `dynamic_sampling`/`max_resample_attempts`, off by default), returning the
  first informative step with `resample_attempts` (or `resample_exhausted:True`). A
  thin WRAPPER over the existing train_step — zero changes to the PPO/old-logps update
  path (no numerics risk) and reasoning-only (off the live pretraining path). Completes
  the DAPO stack (clip-higher + Dr.GRPO mean-norm + no-KL + token/length-norm +
  collapse-skip were already present). Tests: test_modern_techniques.py::
  TestDynamicSamplingResample +3 (resamples-until-informative, disabled-single-attempt,
  exhausted-after-max); 36 modern + 17 surrogate tests + ruff green. Follow-up wiring:
  the train_reasoning loop should call train_step_resampled with a problem-set sampler
  when dynamic_sampling is on (small, → note in MODEL-032).
- **IDEA-009** [research/inference, medium-potential] `done` (2026-06-13) — Adaptive
  best-of-N budget. Added `generate_best_of_n_adaptive()`: generates an initial small
  batch (default 2), verifies it, and grows ×`growth` up to `max_candidates`, EARLY-
  STOPPING the moment a candidate both VERIFIES and is SECURE (SEC-017). Cheap prompts
  cost `initial_candidates`; hard ones get the full budget — trades compute for
  accuracy only when the verifier says it's needed (2026 test-time scaling). Extracted
  shared `_build_candidates`/`_rank` helpers (DRY with fixed-N best-of-N — same verify
  + secure-aware ranking). Pure inference (non-train, safe on main). Tests:
  test_best_of_n.py::TestAdaptiveBudget +3 (early-stop on first verify, expand-to-cap
  when none verify, secure-preferred on early-stop); 54 best_of_n/server tests + ruff
  green. Follow-up: have the FastAPI /best-of-N path optionally use the adaptive variant.
- **MODEL-034** [model/training, medium] `open` (Muon research 2026-06-13) — MuonClip
  (Kimi-2): clip the Muon update (qk-clip) for smoother loss + stable large-scale
  dynamics — the 2026 stability refinement on plain Muon. Layer onto MODEL-025 (adopt
  Muon) once Muon is the default; muP scaling (MODEL-031) composes with Muon (tune LR +
  weight-decay on tiny, transfer up). Train-path (optimizer) → worktree, validate
  before any merge.
- **MODEL-033** [model/long-context, medium] `open` (long-context research 2026-06-13)
  — LongRoPE2-style Stage-4 context extension: upgrade the YaRN path with mixed
  context-window training that PRESERVES short-context accuracy (>98.5%) while adopting
  rescaled RoPE for long sequences — so extending context doesn't regress the
  1024-ctx pretraining. Pairs with MODEL-017 (YaRN formula review). Train-path
  (model/rope + training) → worktree, validate before any merge.
- **IDEA-010** [research/long-context, medium-potential] `open` (2026-06-13) —
  Repo-level FIM: use the repo_context module to assemble a large multi-file context,
  hold out a whole function in the middle, and train/eval FIM infill conditioned on the
  full repo (PSM: repo-prefix + suffix), verified by the sandbox tsc against the real
  repo. Combines long-context + FIM + repo_context + sandbox verifier; targets
  IDE-ghost-text-in-a-real-codebase, and pairs with IDEA-007 (contamination-free eval).
- **MODEL-032** [reasoning, medium] `open` (research 2026-06-13) — Difficulty-aware
  dynamic sampling: before a full GRPO rollout, run a cheap best-of-N probe per prompt
  and skip prompts predicted zero-variance (trivially all-pass / all-fail), spending
  the rollout budget on medium-difficulty prompts (std>0). Gets DAPO's dynamic-sampling
  benefit (MODEL-026) without wasting full rollouts on collapsed groups; combines
  best-of-N + GRPO + curriculum. reasoning-only (safe on main).
- **MODEL-031** [model/training, medium] `open` (architecture research 2026-06-13) —
  muP / muTransfer: maximal-update parametrization so hyperparameters (LR, init scale)
  tuned on the TINY config transfer zero-shot up the size ladder
  (tiny→small→medium→4080_max), avoiding expensive per-size LR sweeps. Touches model
  init + optimizer LR scaling (train-path → worktree; validate via the tiny config).
  Reinforced by the 2026 arch consensus; pairs with the project's existing
  `_model_scale` size handling.

### SECURITY — bulletproof the untrusted-code sandbox (HIGH — user mandate 2026-06-13)
The sandbox that runs UNTRUSTED scraped/teacher-generated code (data/curation test
execution + scoring, and now distillation verification) must be bulletproof for
EVERY scenario. SEC-001/SEC-010 fixed timeout-kill + all-exit cleanup; the rest:

- **SEC-012** [security/sandbox, high] `done` (2026-06-13) — DONE for the Docker
  backend: `docker_sandbox.py` `_build_run_argv` now emits the full isolation profile
  on every untrusted run — `--user 65534:65534` (nobody), `--read-only` + one
  size-capped `--tmpfs=/tmp` (run_with_install copies there, HOME=/tmp), `--network=none`,
  `--cap-drop=ALL` + `--security-opt no-new-privileges` (never privileged/unconfined),
  `--pids-limit`, `--memory`==`--memory-swap` (swap off), `--cpus`, `--ulimit nofile/nproc`,
  no host namespaces, clean env (no host secrets), output-bomb truncation, and an outer
  wall-clock watchdog backing the subprocess timeout. SEC-001/010 cleanup preserved.
  Limits constructor-configurable. 21 new tests (TestDockerSandboxHardening/OutputCap/
  Watchdog); 72 curation tests + ruff green. Caveats (→ remain in SEC-015): `--storage-opt
  size` unsupported on Docker Desktop overlay2 (mitigated by tmpfs cap); gVisor/runsc
  unavailable on Windows; deny-by-default seccomp profile still TODO (→ SEC-015). Docker
  is now maximally hardened but per research is "raised bar, not bulletproof" — true
  kernel isolation needs SEC-015 (VM/cloud backend).
  (Original SEC-012 checklist items 1-14 all landed in the Docker backend except
  disk-quota/storage-opt (Docker Desktop overlay2 limitation), seccomp deny-by-default,
  and pinned-offline-image — those carry into SEC-015.)
- **SEC-013** [security/sandbox, high] `done` (2026-06-13) — Fixed a CRITICAL
  fail-open: `SandboxedRunner` (data/scorers/sandbox.py) silently dispatched to
  `_run_native` (HOST execution of untrusted code) when Docker isolation was
  REQUESTED but unavailable. Now FAILS CLOSED — returns `RC_SANDBOX_UNAVAILABLE=-3`
  ("not executed, fail closed"), never host-execs, unless the new explicit
  `allow_native_fallback` opt-in (off by default). Plain `native` mode (Docker not
  requested) unchanged = the documented trusted-code host path. Full audit of every
  other untrusted-exec call site: tsc_scorer/eslint_scorer/TscRunner/evaluation
  runner/best_of_n all route through SandboxedRunner (fix inherited); TestRunner
  already fails closed (raises if Docker down, host-exec needs explicit
  allow_host_execution); github clone is not code-exec. Tests:
  test_sandbox.py::TestFailClosedWhenDockerUnavailable +5 (Docker-absent → rc -3, no
  PWNED marker, _run_native never called); 97 sandbox+curation tests + ruff green.
- **SEC-016** [security/scoring, medium] `done` (2026-06-13) — Fixed the scoring-level
  fail-open. `TscRunner.check`/`check_batch` now detect a NEGATIVE sandbox returncode
  (tsc never ran — timeout -1 / error -2 / unavailable -3) and return a
  `SANDBOX_UNAVAILABLE` sentinel error (severity=error, uncached) instead of an empty
  list — so NO caller mistakes unverified code for "0 errors / clean / passing". This
  fixes ALL consumers at the source (TscScorer, the GRPO TypeCheckReward,
  batch_type_check, ts_benchmark, tools/executor — each sees a non-empty error list =
  not-clean, never perfect/rewarded). TscScorer additionally detects the sentinel and
  returns an explicit `not_verified` result with score 0.0 (never the old false-perfect
  1.0). A real tsc run that finds type errors exits POSITIVE, so `returncode < 0`
  unambiguously means "did not run". Tests: new test_tsc_failclosed.py +8 (sentinel
  on -3/-2, not cached, clean run still 0 errors, real errors still parsed, scorer
  0.0+not_verified vs clean 1.0); 145 tsc/sandbox/curation/reward tests + ruff green.
- **MODEL-022** [model/inference, medium] `open` (refined 2026-06-13 — see research-log)
  — Speculative decoding (draft-and-verify) for lower serve/REPL/FIM latency; now the
  DEFAULT 2026 inference layer, 2-4× speedup, MATHEMATICALLY LOSSLESS (no eval/quality
  risk). For a single ~100M model the tractable choice is a **Medusa-style** self-
  speculative head (no separate draft model, GPU-light) over EAGLE-3 (higher
  acceptance, more involved). Train on the spare GPU.
- **SEC-014** [security/distillation, high] `done` (2026-06-13) — Satisfied by the
  MODEL-028 design: `generate_distillation_dataset` NEVER executes teacher output —
  verification is an injected callable, and the CLI wires it to the SANDBOXED
  `TscScorer` (→ SandboxedRunner, hardened by SEC-012, fail-closed by SEC-013/016).
  Remote-teacher prompts are secret-redacted in `OpenAICompatibleTeacher`. A test
  (`test_loop_never_executes_completion`) asserts the loop stores a "malicious"
  completion as text without running it. Follow-up when other languages are added:
  ensure non-TS verifiers (python_exec) also route through the sandbox (they already
  do via SandboxedRunner per the SEC-013 audit).
- **DATA-057** [data-quality, medium] `done` (2026-06-13) — Soft-dedup reweighting.
  Added `CrossDatasetDeduplicator.compute_soft_weights(data) -> np.ndarray`: builds a
  MinHash per row, indexes them in an LSH, and assigns each row weight 1/cluster_size
  (cluster = near-dups within the Jaccard threshold). A cluster of k near-identical
  chunks collectively contributes ~1 sample's worth of weight, so over-represented
  content is down-weighted WITHOUT deleting rare info (SoftDedup). Multiply into the
  training `.weights.npy` (the loss already applies per-sample weights). No-op
  (all-ones) when datasketch is absent. Additive (new method, hard-dedup path
  untouched), prep-time (off the live pretraining path → safe on main). Tests:
  test_soft_dedup.py +4 (identical-trio→1/3 each, all-unique→1.0, empty, cluster-sum→1);
  27 dedup tests + ruff green. Follow-up: a flag in score_data/combine_datasets to
  fold these into the emitted weights. Validate alongside Muon (IDEA-005).
- **DATA-058** [data-quality, medium] `done` (2026-06-14) — **Folding Ordering**
  curriculum (DELT, arXiv:2506.21545). New `CurriculumStrategy.FOLDING` in
  `data/scorers/curriculum.py`: sorts by `.weights.npy` quality, then round-robin
  strides the sorted index into L folds and concatenates → a true permutation (no
  drops/dups) where each fold is a strided full-range easy→hard sweep, so the model
  revisits the whole curriculum L times. Fixes single-pass curriculum forgetting /
  distribution bias / duplication; paper reports +2.76% HumanEval. Wired through
  `score_data.py` (`--curriculum folding --curriculum-folds L`, default 4; also reads
  `curriculum.num_folds` from scoring.yaml). Prep-time only, off the live train path →
  committed on main. Tests: test_curriculum.py +5 (permutation, per-fold descending,
  data/weights alignment, L=1≡easy_to_hard, folds>n); 19 curriculum tests + ruff green.
- **DATA-059** [data-quality, low] `open` — **Learnability-Quality Scoring (LQS)** from
  DELT: score samples by learnability (loss-drop ratio across steps) × quality (grad
  cosine-sim to target direction). Needs gradient tracking + a proxy set → heavier and
  touches the train loop; defer until a worktree window (no live run) and pair with the
  cheaper verifier proxy in DATA-060. Caveat (arXiv:2512.24503): validate any
  proxy-model-scored curation at target scale — small-proxy rankings don't always transfer.
- **DATA-060** [data-quality, medium] `open` (ORIGINAL cross-technique) —
  **verifier-grounded learnability folding.** Replace DELT's gradient-based learnability
  with a no-gradient proxy built from cola-coder's own verifier: sample the base
  checkpoint's best-of-N pass-rate on held-out FIM probes per quality bucket; buckets at
  mid pass-rate (p≈0.5) are the high-learnability frontier. Weight each bucket's fold
  frequency by 4·p·(1−p) so frontier data recurs most often — a DELT-style efficacy gain
  using the sandbox verifier + best-of-N + quality weights that the source papers lack.
  Builds on DATA-058 (folding infra) + the best-of-N verifier. Prep-time → main-safe.
- **MODEL-037** [model/post-training, medium] `done` (2026-06-14) — **GRPO policy-entropy
  metric** (RLVR entropy-collapse instrumentation; arXiv:2509.26114, 2509.21882). Added
  `completion_entropy(log_probs_2d, prompt_len)` to grpo.py: mean per-token Shannon
  entropy (nats) over completion positions only (prompt masked, mirrors
  `_completion_logprobs`). `train_step` measures it once at PPO epoch 0 (weights==pi_old)
  → returns `policy_entropy`; the epoch loop aggregates + prints it next to
  loss/reward/pass_rate. Makes entropy collapse — the dominant RLVR failure mode —
  observable, so the existing asymmetric clip_low/clip_high knobs become actionable
  (clip-low raises entropy, clip-high lowers it). GRPO is reasoning-only = off the live
  PRETRAINING path → committed on main. Tests: test_grpo_entropy.py +6 (uniform→ln V,
  near-deterministic→~0, prompt-exclusion, no-completion→0, manual-match, non-negative);
  28 grpo tests + ruff green. Follow-up: IDEA-013 (close the loop on this metric).
- **IDEA-013** [research/post-training, high-potential] `done` (2026-06-14) —
  **entropy-gated closed-loop clip controller.** `EntropyClipController`
  (reasoning/entropy_controller.py): proportional controller on the MODEL-037 `policy_entropy`
  signal — entropy below a target floor → RAISE clip_high (DAPO clip-higher) proportionally to
  the deficit (capped) to inject exploration; healthy → relax to base. VERIFIER-AWARE: suppresses
  exploration when the group pass-rate is at/above `pass_rate_ceiling` (don't explore away from
  working solutions). Wired opt-in into GRPOTrainer (`entropy_clip_controller=None` default →
  unchanged); applied once per non-skipped step to the next step's clip_epsilon_high. Reasoning-
  only = off the live pretraining path → committed on main. Tests: test_entropy_controller.py +10
  (healthy stays at base, collapse raises + caps, proportional, verifier-gate both ways, relax
  after recovery, clip_low never modulated, invalid-config guards); GRPO suite green, ruff clean.
  Empirical entropy-collapse-prevention A/B deferred to an actual GRPO run.
- **IDEA-020** [research/post-training, medium-potential] `done` (2026-06-14) —
  **per-difficulty entropy floors.** Extended IDEA-013's `EntropyClipController` with optional
  `difficulty_floors={"easy":…,"medium":…,"hard":…}` + `floor_for(difficulty)`; `update(...,
  difficulty=…)` uses that tier's floor instead of the single `target_entropy` (fallback kept).
  Hard tiers get a higher floor → more clip-higher exploration; easy/solved tiers a low floor →
  exploit. Wired through the GRPO curriculum's per-step `difficulty`. Backward-compatible (no
  floors → single target). Reasoning-only → committed on main. Tests: test_entropy_controller.py
  +5 (floor lookup+fallback, hard>easy at same entropy, easy<medium<hard, no-floors fallback,
  negative-floor guard); 15 controller + GRPO suite green, ruff clean.
- **IDEA-022** [research/long-context, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-graded long-range FIM.** RoPE extension (MODEL-033) just rescales frequencies and
  validates on synthetic needles. cola-coder can MANUFACTURE verifiable long-range deps: build a
  dynamic-FIM example where the symbol the middle must use (a type / imported helper) lives in a
  prepended repo-context block thousands of tokens away, so a correct infill REQUIRES attending
  across the full window — graded by the sandbox verifier (does prefix+infill+suffix type-check /
  pass tests using that distant symbol?). Turns needle-in-a-haystack into a VERIFIABLE training +
  eval objective (functional deep retrieval, not perplexity) — the exact failure YaRN has on small
  models. Pairs RoPE extension + FIM + repo context + verifier. Training half = worktree (FIM/data
  numerics); eval half = main-safe. Builds on MODEL-033 + dynamic FIM + repo_context + TscRunner.
- **SEC-019** [security/injection, medium] `done` (2026-06-14) — **prompt-injection scanner**
  for untrusted retrieved content (OWASP LLM01 indirect injection; arXiv:2510.23883, 2601.04795).
  New `security/injection_patterns.py`: `scan_injection`/`has_injection` flag instruction-override
  (ignore-previous, disregard-system-prompt, new-instructions blocks), system-prompt + secret
  exfiltration, pipe-to-shell, fake chat/role markers (`<|im_start|>system`, `[INST]`, `<system>`),
  AND hidden invisible/bidi control characters — in retrieved text BEFORE it enters a prompt.
  Mirrors code_patterns.py (different threat class). Wired non-blocking WARN into doc_fetcher so a
  poisoned fetched doc shows in logs, not silently smuggled into context. High-precision (benign
  "token"/"env"/"previous batch" don't trip). Off the train path → committed on main. Tests:
  test_injection_patterns.py +19 (10 injection positives, invisible/bidi chars, 6 precision
  negatives, name lookup); ruff clean. Defense-in-depth layer, not a guarantee — heavier layers
  (PromptArmor-style) deferred.
- **SEC-020** [security/supply-chain, high] `done` (2026-06-14) — **hallucinated-import /
  slopsquatting scanner** (USENIX Security 2025; arXiv:2501.19012). Code models invent 5-21% of
  package names; attackers register them as malware → install-time compromise. New
  `security/import_scanner.py`: `extract_imports`/`scan_unknown_imports`/`has_unknown_imports` pull
  imported package roots (Python via AST: relative excluded, submodules→root; JS/TS via regex:
  scoped `@scope/name` kept, relative excluded; regex fallback for unparseable partials) and flag
  any NOT in a curated allowlist (Python stdlib via `sys.stdlib_module_names` + popular PyPI by
  import name; node builtins + popular npm). Wired as a NON-ranking review signal into best-of-N
  `details["unknown_imports"]` (niche-but-real imports common → must not down-rank a verified
  candidate). Mirrors code_patterns.py. Off the train path → committed on main. Tests:
  test_import_scanner.py +15 (stdlib/popular known, hallucinated flagged, submodule→root, relative
  skipped, regex fallback, JS scoped/subpath/relative, API, best-of-N wiring); 40 best-of-N green,
  ruff clean. FOLLOW-UP: refresh the popular allowlist periodically; consider a data-prep filter.
- **SEC-021** [security/supply-chain, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-confirmed import quarantine.** An unknown import (SEC-020) is ambiguous (niche-real vs
  hallucinated). Disambiguate with the no-net sandbox: in best-of-N, attempt to import the unknown
  package in the sandbox — a hallucinated one fails (ModuleNotFoundError). Down-rank candidates
  whose unknown imports fail sandbox resolution, and feed confirmed-hallucinated names into a
  deny-set the GRPO security penalty (IDEA-008) discourages → RL-train the model away from inventing
  packages. Exploits the no-net sandbox + best-of-N + GRPO penalty. Builds on SEC-020 + sandbox +
  IDEA-008. Inference/reasoning → main-safe.
- **SEC-023** [security/supply-chain, high] `done` (2026-06-14, safety research) — **typosquat/slopsquat
  triage of unknown imports.** SEC-020 flags every out-of-allowlist import identically — it can't tell a
  squat of a popular package (`requsts`/`l0dash`/`bs_4`) from a legit niche import. ConfuGuard
  (arXiv:2502.20528) frames the goal as separating CONFUSION attacks from legitimate packages; the standard
  offline screen (arXiv:2503.22406, 98.4% acc; Cloudsmith 2026) is Damerau-Levenshtein distance + separator
  normalization + homoglyph folding against the popular set. `import_scanner.py` gains `_damerau_levenshtein`,
  `_normalize_name` (separator unify + homoglyph `0→o 1→l rn→m vv→w` + strip), and `classify_unknown_imports`
  → `ImportTriageReport` (typed `SuspectImport`/`ImportRisk`) that REUSES `scan_unknown_imports` (DRY; a test
  asserts the triage partitions exactly its survivors) and sorts each into TYPOSQUAT (≤max_distance edits to
  a popular name + min_length guard + distance-0 separator/homoglyph confusion) vs UNKNOWN (legit-niche, not
  over-flagged), reusing the curated `_PY_POPULAR`/`_JS_KNOWN` allowlists as the neighborhood. Wired into
  best-of-N as a `typosquat_imports` review signal alongside the flat `unknown_imports` (no ranking change,
  back-compat). +24 tests (test_import_triage.py); 15 import-scanner + 40 best-of-N intact, ruff clean.
  Pure string logic — no model/GPU/network/execution → main-safe.
- **SEC-024** [security/supply-chain, medium] `open` (2026-06-14, ORIGINAL) — **down-rank
  verified-but-typosquatting candidates.** SEC-023 surfaces typosquats as a non-ranking signal only. Make a
  best-of-N candidate that imports a likely typosquat (high-confidence confusion of a popular package) lose
  ties to an equally-verified clean candidate — a slopsquat tie-break that mirrors the existing secure-pass
  tie-break, gated to only confusion-class names (not generic unknowns, which must stay neutral). Builds on
  SEC-023 + best-of-N. Inference → main-safe.
- **SEC-025** [security/supply-chain, medium] `open` (2026-06-14, ORIGINAL) — **typosquat reject gate in
  RFT/distillation.** The RFT/self-verified-distillation harness already rejects insecure (dangerous-pattern)
  completions; extend the reject gate so a verified completion importing a high-confidence typosquat
  (SEC-023) is also dropped from the self-curated SFT set — so the model never trains on (and thus learns to
  re-emit) slopsquat imports. Mirrors `require_secure`. Builds on SEC-023 + MODEL-045. Distillation → main-safe.
- **DATA-063** [data-quality/safety, high-potential] `done` (2026-06-14) —
  **injection-aware training-data filtering.** `InjectionFilter` (data/filters/injection.py): a
  registered FilterPlugin reusing the SEC-019 `scan_injection` scanner to DROP scraped pretraining
  samples carrying prompt-injection payloads (so the corpus can't teach the model to emit/obey
  injections — data poisoning). Closes input-time defense (SEC-019 doc fetcher) with training-time
  hygiene. Opt-in via the filter chain (like pii/content/license); `min_hits` config to require
  corroborating signals and cut false positives on security-related source. Prep-time → committed
  on main. Tests: test_injection_filter.py +7 (registered+constructible, drops payloads, keeps
  clean + benign-trigger-words, min_hits corroboration, hidden control chars, empty); ruff clean.
  FOLLOW-UP DONE (2026-06-14): SOFT-weight variant `InjectionScorer`
  (data/scorers/injection_scorer.py) — reuses scan_injection + ScoreMapper to grade injection
  severity (0 hits→1.0, 1→0.4, 2→0.15, floor 0.05) so the composite quality WEIGHT is reduced
  instead of dropping the sample (reweight-over-filter). Registered `injection_safety` in the
  scorer registry + scoring.yaml (opt-in). +7 tests, ruff clean. Filter (hard) + scorer (soft)
  now both available; pick per pipeline.
- **DATA-064** [data-quality, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-anchored synthetic mixing with a real-data floor.** Combine the two 2026 collapse
  defenses (arXiv:2510.16657) using cola-coder's assets: (1) every synthetic/distilled example
  must pass the sandbox verifier before entering the mix (the "external verifier" proven to prevent
  model collapse — MODEL-040 already screens security; extend to functional verify); (2) enforce a
  configurable REAL-DATA FLOOR (≥10-20% verified-real scraped code) per SFT/distillation round via
  the existing weighted-mixing (combine_datasets), tracked so synthetic share can't dominate.
  Unbounded verified-synthetic augmentation that provably won't collapse + a real anchor. Exploits
  verifier + quality weights + dataset mixing. Builds on MODEL-040 + combine_datasets. Prep-time → main-safe.
- **DATA-065** [data-quality, high] `done` (2026-06-14) — **benchmark-decontamination filter.**
  Contamination (eval problems leaking into training data) inflates eval scores; n-gram/containment
  overlap is the standard screen (arXiv:2502.14425). The project's `DataLeakageDetector` (MinHash +
  containment) only REPORTED (check_contamination.py) — never DROPPED during prep. New
  `data/filters/decontamination.py` `DecontaminationFilter`: a registered FilterPlugin REUSING the
  detector's `_shingles`/`_containment` (DRY) to index eval/benchmark texts (explicit `eval_texts`
  and/or built-in problem prompts via `benchmark: true`) and drop records whose containment of an
  eval doc ≥ threshold (0.8). Opt-in; no-op without refs. Closes the report→mitigation gap.
  Prep-time → committed on main. Tests: test_decontamination_filter.py +6 (drops embedded benchmark,
  keeps unrelated, no-op, threshold sensitivity, empty); filter registry intact (13), ruff clean.
  NOTE: n-gram misses REPHRASED leakage → DATA-066.
- **DATA-066** [data-quality, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-confirmed semantic decontamination.** N-gram (DATA-065) misses rephrased/reformatted
  benchmark leakage; embedding/LLM detection is heavy. Catch the most dangerous case — a
  FUNCTIONALLY-equivalent rewrite of an eval problem — with the verifier: for a training sample
  defining a function whose name matches an eval entry point (cheap candidate filter), run the
  EVAL's OWN tests against that function in the sandbox; if they PASS, it's a functional copy
  (even if reworded) → drop. Catches what n-gram/embedding miss, using the sandbox verifier + the
  benchmark test suites. Builds on DATA-065 + sandbox + problem/test sets. Prep-time → main-safe.
- **EVAL-025** [eval, high-potential] `open` (2026-06-14, ORIGINAL) —
  **execution-grounded self-repair eval.** A contamination-free eval (IDEA-007) usually needs
  reference solutions; cola-coder has a sandbox verifier instead. Scrape RECENT (post-cutoff)
  TS/React files WITH their tests; score by EXECUTING the model's completion against the real
  tests (functional, reference-free, contamination-resistant), and add a **self-repair@k** metric:
  feed the tsc/test ERROR back on a failure and measure whether the model fixes it within k tries
  (LiveCodeBench's self-repair dimension, graded by the verifier not a reference). Reuses verifier
  + best-of-N + dynamic FIM. Builds on EVAL-023/IDEA-007 + the sandbox. Defer until the model emits
  runnable code (~step 10k+). Eval → main-safe.
- **SEC-015** [security/sandbox, high] `open` (the REAL bulletproof path — research
  2026-06-13) — HONEST LIMITATION: plain Docker on this Windows Docker Desktop/WSL2
  host CANNOT be made truly bulletproof (shared kernel; gVisor/Firecracker/Kata
  microVMs — the 2026 isolation standard used by OpenAI Code Interpreter/E2B/Modal —
  are unavailable on Windows; WSL2 itself has had VM escapes). SEC-012 maximally
  hardens Docker ("raises the bar"), but for genuine kernel isolation the untrusted
  execution must move OFF the Windows host. Action: design the sandbox behind a
  clean `Sandbox` interface and add a pluggable backend for a **disposable Linux VM**
  (Hyper-V/WSL guest treated as expendable, snapshot-reverted per run) or a **cloud
  sandbox (E2B / Modal / Daytona)** that actually provides Firecracker/gVisor. Make
  the backend configurable; default to hardened-Docker, document the upgrade. Also
  add a **deny-by-default seccomp profile** to the Docker backend (Docker's default
  is a permissive allow-list) — folded into SEC-012's scope. See docs/research-log.md.
- **SEC-018** [security/static-scan, medium] `done` (2026-06-14) — **expanded CWE coverage**
  in the canonical `scan_dangerous` scanner (`security/code_patterns.py`). Research: 12–65%
  of LLM code carries CWE vulns (arXiv:2412.15004, 2510.22944); top classes are injection
  (SQLi/XSS), weak crypto, command exec. Added high-precision patterns: XSS DOM sinks
  (`innerHTML/outerHTML =` with `(?!=)` to skip `==`/`===`, `insertAdjacentHTML`), code-as-string
  timers (`setTimeout/Interval('…')`), SQLi (template `${}` + string `+` concat), weak hash
  (`createHash('md5'|'sha1')`, `hashlib.md5/sha1`), `os.popen`. Because this is the SINGLE
  screen behind best-of-N ranking + GRPO security penalty + safety_eval + secure-pass@k
  (EVAL-024), one change upgrades all four. Precision verified (reading innerHTML, sha256,
  function-arg timers, static SQL all clean). Off the train path → committed on main. Tests:
  test_code_patterns.py +11 (10 CWE positives, 5 added precision negatives); 36 scanner + 40
  downstream (best-of-N/distillation) green, ruff clean.
- **IDEA-016** [research/safety+data, high-potential] `open` (2026-06-14, ORIGINAL) —
  **adversarial secure-FIM hardening.** Turn the scanner into a TRAINING signal: take a clean
  sample, programmatically inject a known CWE (e.g. parametrized query → string-concat), and
  build a dynamic-FIM task whose middle is the vulnerable span; the model must infill the
  SECURE version, graded by `scan_dangerous` (clean) AND tsc/tests (still correct). Contrastive
  secure/insecure minimal pairs, self-labeling (injector knows ground truth), verifier-graded —
  dynamic FIM + scanner + sandbox + quality weights. Complements IDEA-015 (mine real gaps) with
  synthetic coverage of rare CWEs. Prep-time/reasoning → main-safe.
- **MODEL-029** [model/distillation, low] `open` — Reconcile + document the TWO
  distillation modes so they're not confused: (a) EXISTING white-box logit KD in
  `features/knowledge_distillation.py` + `features/distillation_helper.py` (KL on
  soft teacher logits — requires the teacher in-process as an nn.Module with the
  SAME tokenizer; GPU-heavy; can't use Qwen/DeepSeek whose tokenizer differs); (b)
  NEW black-box/SeqKD in `distillation/teacher.py` (collect teacher TEXT via API —
  tokenizer-agnostic, GPU-light, local-or-cloud Qwen/DeepSeek). They're
  COMPLEMENTARY, not duplicate. Add a short note in both modules cross-referencing
  the other + a docs section on when to use which. (Builds on the existing — likely
  Fable 5 — white-box code; do not rewrite it.)

### UI — control + observability dashboard (LAST PRIORITY — user request 2026-06-13)
Goal: a fast local UI to monitor/launch everything without always running. Background
training so the UI need not stay open; open it to see all runs, start a train with
options, and do most menu actions. MUST be fast, must NOT slow training, and must
REUSE existing scripts/managers (no logic duplication) — a thin control+observability
layer that launches DETACHED jobs (like cola-train-resume.ps1) and reads state from
manifests / pipeline_runs/ / checkpoints / logs. First objectives (in order):

- **UI-001** [ui/foundation, low] `open` — Scaffold a lightweight local web UI: a small
  FastAPI control-plane app (separate from the inference serve.py so it never competes
  with training/inference) + a static HTML/JS frontend (no heavy framework). Endpoints
  read state and launch detached background jobs. Decide: extend an existing app vs new
  `scripts/dashboard.py`. Background-job registry persisted to disk so the UI can close
  and reopen and still see running/finished jobs.
- **UI-002** [ui/training, low] `open` — See running + past training: read
  training manifests, `train_*.err/.log`, checkpoints/<run>/step_* and pipeline_runs/
  to show live step/loss/throughput/ETA and run history. Read-only; must not touch the
  run. (Reuse training_status.py logic.)
- **UI-003** [ui/training, low] `open` — Start a training run with options (config
  picker, resume/fresh, max_steps, etc.) launched DETACHED (the cola-train-resume.ps1
  pattern) so it survives the UI closing. Refuse to double-launch.
- **UI-004** [ui/data, low] `open` — Download datasets: trigger collect_data.py /
  prepare_data.py as background jobs with options; show progress + completion.
- **UI-005** [ui/data, low] `open` — Scoring + data browsing: run score_data.py, and
  browse prepared datasets + their .weights/.scores — view sample content and per-sample
  scores. (Reuse DatasetResolver + the scorers; read-only browsing.)
- **UI-006** [ui/parity, low] `open` — (Later) broaden toward full menu parity: pipeline
  runs, eval, distillation control, export, etc. — incrementally, each reusing the
  existing script/manager rather than reimplementing.

### Original ideas / hypotheses 2026-06-13 (cross-technique — see docs/research-log.md)
These are RESEARCH hypotheses (validate in a worktree, off the live run), exploiting
cola-coder's rare combo of dynamic FIM + sandbox test/tsc rewards + best-of-N verifier
+ quality weights + Muon. Not committed approaches until validated.

- **IDEA-001** [research/model, high-potential] `open` — FIM-RLVR: GRPO on FIM infill
  tasks with a tsc-/test-execution verifiable reward (optimizes the real ghost-text
  objective, not full generation). Reuses TscRunner + sandbox + GRPO + FIM tokens.
- **IDEA-002** [research/post-training, high-potential] `open` — Verified
  self-distillation bridge: harvest the model's own best-of-N sandbox-verified
  completions as a rejection-sampling SFT bridge (teacher-free OPD), used as a warmup
  INTO GRPO (self-distill alone overfits — pair with RLVR). 
- **IDEA-003** [research/data, medium-potential] `open` — Online perplexity curriculum:
  use the running model's per-chunk loss as a live difficulty/quality signal to
  reweight on the fly (merges MODEL-020 + DATA-055; closes eval→data loop during training).
- **IDEA-004** [research/inference, medium-potential] `open` — Suffix-guided
  speculative decoding for FIM: use the KNOWN suffix to bias/verify draft tokens for
  faster, more suffix-consistent infill (MODEL-022 + FIM).
- **IDEA-006** [research/inference, medium-potential] `open` (2026-06-13) — FIM-aware
  self-speculative decoding: train Medusa/EAGLE draft heads on the project's OWN FIM
  data so speculative drafts are prefix+suffix conditioned, and use the KNOWN suffix
  (IDEA-004) as an extra accept/verify signal → fast, suffix-consistent ghost text.
  Self-speculative (heads on the same model) so no second model + minimal VRAM;
  lossless. Combines MODEL-022 + IDEA-004 + the project's FIM asset.
- **IDEA-005** [research/training, low-potential] `open` — Quality-weight × Muon
  interaction ablation: does loss-level `.weights.npy` weighting survive Muon's
  orthogonalization, or must it move to the sampling level? Gates MODEL-025 adoption.

### Research-grounded 2026-06-13 (from web research — see docs/research-log.md)

- **MODEL-024** [model/post-training, high] `open` — **On-policy distillation (OPD)**,
  the dominant 2026 post-training paradigm (DeepSeek-V4/Qwen3/Gemma-2/MiMo/GLM-5). At
  small deployment size a teacher distilled through a dense OPD bridge BEATS direct
  GRPO on the student. cola-coder does SFT+GRPO but no distillation — a major quality
  lever for a ~100M model. Since there's no local teacher with logits, use a black-box
  / self-distillation variant (OPSD/SDFT/GAD), or distill from a strong API teacher
  with test-verification. Recipe: teacher GRPO → forward-KL warmup + OPD bridge →
  optional student sparse RL. Refs: Thinking Machines Lab OPD blog; arXiv 2604.00626
  (survey), 2604.13016 (recipe). Stage as a new pipeline step after SFT.
  PROGRESS (2026-06-13): teacher backend landed — `src/cola_coder/distillation/`
  with `OpenAICompatibleTeacher` (one backend covers LOCAL Qwen/DeepSeek via
  Ollama/llama.cpp/vLLM AND cloud DeepSeek/OpenAI/OpenRouter — black-box/SeqKD so
  it's tokenizer-agnostic + GPU-light; secret-redaction for remote endpoints) +
  `build_teacher()` factory + configs/distillation.yaml + 15 tests. Remaining
  sub-tasks: MODEL-027 (in-process HF teacher), MODEL-028 (generate-data harness),
  then the OPD/SFT bridge + pipeline stage.
  REFINED (2026-06-14, OPD research): formalize the bridge as dense KL-constrained RL —
  the teacher's per-token log-ratio is an IMPLICIT dense reward (Yang 2026, arXiv:2604.13016);
  use REVERSE-KL (mode-seeking, concentrates mass on correct code) with an entropy-aware
  floor (reuse the MODEL-037 policy_entropy metric to detect collapse). The OPD loss is a
  train-loop change → implement in a WORKTREE, never merge while small_react_best is live.
  See IDEA-018 for densifying it with the sandbox verifier.
- **MODEL-040** [model/distillation, medium] `done` (2026-06-14) — **always-on security
  screen in distillation data-gen.** `generate_distillation_dataset` rejection-sampled only
  on functional `verify`; with `verify=None` it KEPT dangerous teacher code verbatim (would
  distill e.g. `os.system('rm -rf /')`). Added `screen_security=True` (default): every teacher
  completion is statically screened by the canonical `scan_dangerous` (SEC-018) and dropped
  BEFORE functional verify, independent of keep_only_verified; new `rejected_insecure` stat
  surfaced in the CLI summary. Defence-in-depth not dependent on the caller wiring security
  into `verify`; removed the now-redundant inline `is_dangerous` from
  scripts/generate_distillation_data.py (DRY). Static/execution-free → committed on main.
  Tests: test_distillation_generate.py reworked +4 (default-on drops dangerous, screen
  precedes verify, off-switch keeps raw + never executes, verify path isolated); 11 pass, ruff green.
- **MODEL-041** [reasoning, medium] `done` (2026-06-14) — **fractional/partial-credit GRPO
  reward** (EGCA-style finer credit; arXiv:2603.16158). `python_exec` runs the whole test block
  all-or-nothing (1.0/0.0) → coarse credit + frequent zero-variance groups. New
  `reasoning/rewards/partial_credit.py`: AST-splits the test block into individual `assert` cases
  (non-assert top-level statements kept as shared SETUP prepended to each case) and returns the
  FRACTION passed; binary fallback when no top-level asserts. Registered opt-in `python_partial`
  (`--reward python_partial`, CLI choice + help). `info["correct"]` = full pass preserves the
  trainer's pass_rate/collapse-guard. Denser signal, complements DAPO dynamic sampling; additive
  (python_exec untouched). Reasoning-only → committed on main. Tests: test_partial_credit_reward.py
  +9 (split asserts/setup, no-assert + syntax-error empties, all/partial/none-pass fractions,
  shared-setup, binary fallback, registry wiring); reasoning-config-wiring green, ruff clean.
- **IDEA-023** [research/post-training, high-potential] `open` (2026-06-14, ORIGINAL) —
  **execution-trace token-level credit (full EGCA).** The sandbox already emits a TRACEBACK on
  failure (failing assert + line number). Map the failing region to the candidate's tokens and
  apply the negative advantage MORE strongly there in GRPO's per-token clipped surrogate, sparing
  tokens whose asserts passed (from MODEL-041's per-case map). Token-level credit from the
  verifier's execution trace — the localized GRPO update EGCA proposes, on the per-token logprobs
  the trainer already has. Builds on MODEL-041 + the GRPO token surrogate + sandbox tracebacks.
  Reasoning-only → main-safe.
- **IDEA-018** [research/post-training, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-densified distillation reward.** OPD's edge is a DENSE per-token teacher reward
  vs the verifier's SPARSE end-of-sequence score. cola-coder has BOTH a teacher and a sandbox
  verifier + best-of-N. Blend: teacher per-token log-ratio = dense shaping reward for credit
  assignment, but GATE the episode return by the sandbox verifier (only reinforce student
  rollouts that pass tsc/tests), up-weighted by quality weights. Dense teacher signal + hard
  ground-truth gate — neither the OPD papers (no verifier) nor pure GRPO (sparse only) has
  both. Builds on MODEL-024 + GRPO loop + best-of-N. Train-path → worktree when implemented.
- **MODEL-027** [model/distillation, medium] `open` — In-process HuggingFace teacher
  backend (`backend: hf_local`) that loads Qwen/DeepSeek weights via transformers on
  a CHOSEN device (e.g. the spare RTX 3080, never the training GPU), for users who'd
  rather not run a server. The OpenAICompatibleTeacher already covers local via
  Ollama/llama.cpp, so this is a convenience path; lazy-import transformers so it's
  not a hard dep. Wire into `build_teacher()` (the `hf_local` branch currently raises
  NotImplementedError with a pointer here).
- **MODEL-028** [model/distillation, high] `done` (2026-06-13) — Distillation
  data-gen harness shipped. Core `distillation/generate.py`
  `generate_distillation_dataset(teacher, prompts, verify=…, keep_only_verified=…)`
  is teacher-agnostic + verifier-INJECTABLE (the loop never executes anything —
  untrusted code runs only behind the injected sandboxed verifier, SEC-014) and
  emits ChatML `{"messages":[...]}` for train_sft.py (the Generate→Filter shape of
  2026 synthetic-data pipelines). `scripts/generate_distillation_data.py` wires it to
  build_teacher(configs/distillation.yaml) + a sandboxed tsc verifier (rejection
  sampling) + `ps/cola-distill-data.ps1`. Tests: test_distillation_generate.py +8
  (ChatML output, system handling, rejection-sampling, teacher-error skip, empty-skip,
  never-executes-completion); 23 distillation tests + ruff green; script --help smoke
  + ps parse OK. Remaining tiny follow-up: add a master-menu entry (→ MODEL-030;
  there's an existing distillation menu to extend).
- **MODEL-030** [tooling/menu, low] `done` (2026-06-13) — Wired
  `generate_distillation_data.py` into the master menu: TrainingMenu → Post-Training →
  "Generate Distillation Data" (`_generate_distillation_menu`) prompts for prompts/
  output/language/verify and runs the script. ruff clean; 96 menu-smoke tests green
  (the new leaf cancels cleanly under stubs). Project rule (every script needs a menu
  entry) satisfied for the distillation harness.
- **MODEL-025** [model/training, high] `open` — **Adopt/validate Muon optimizer** as
  the default for small+ configs. Muon gives ~2× compute efficiency + ~33% memory
  savings vs AdamW and is data-efficient past the critical batch size (validated to
  multi-billion scale). cola-coder ALREADY implements Muon (`optimizer.py`, audited
  correct) but runs AdamW. Highest-leverage change for "best quality on limited
  compute": run a tiny-config AdamW-vs-Muon head-to-head (loss@fixed-steps), then
  default to Muon. Refs: arXiv 2502.16982 (Muon is Scalable), 2505.02222 (Practical
  Efficiency). NOTE: changing the live run's optimizer is numerics-changing → worktree
  + validate; apply to the NEXT run, not the in-flight one.
  VALIDATION PROGRESS (2026-06-14): unit-level validation now complete — Newton-Schulz
  orthogonalization (BOTH rows<cols and the previously-untested rows>cols transposed
  branch), shape preservation, decoupled weight decay (exact algebraic check
  p_decayed == p_undecayed - lr*wd*p0), param-split, state_dict resume, and scheduler
  scaling all covered (tests/test_modern_techniques.py::TestMuon, 10 tests, main-safe —
  no optimizer.py edit). REMAINING: the empirical AdamW-vs-Muon A/B (needs a training
  run; do in a worktree on the NEXT run, not the live AdamW one). See IDEA-019 for a
  quality-weights interaction to verify before defaulting to Muon.
- **IDEA-019** [research/training, medium-potential] `done` (2026-06-14) —
  **quality-weights x Muon interaction.** VALIDATED in tests/test_muon_quality_weights.py:
  RELATIVE in-batch per-sample quality weights DO change the Muon update (not neutered by
  orthogonalization), monotonically with skew, and optimizer-agnostically (AdamW too) — so
  the project's quality weighting survives a future switch to Muon, PROVIDED weights stay a
  weighted MEAN over per-sample losses (which `language_modeling_loss` does), never a global
  multiplier. EMPIRICAL REFINEMENT of the original hypothesis: the "global scale fully washed
  out by orthogonalization" claim does NOT hold cleanly — Newton-Schulz runs in bf16, so a
  1000x global scale perturbs the bf16 update by an amount comparable to a real reweight at
  tiny scale (documented, not asserted). Practical guidance unchanged + now test-backed. +3
  tests, ruff green, main-safe (no optimizer.py/train-path edit).
- **DATA-061** [data-quality, medium] `open` (data-curation research 2026-06-14) —
  **two-stage quality thresholds** (FineWeb-Edu/Stack-edu): raise the quality bar in later
  training (edu >2.75->3.2; code >3.75->4.1) — model trained on the high-quality 10% can match
  350B unfiltered tokens. cola-coder has quality scores + curriculum/folding (DATA-058); add a
  staged-threshold mode to score_data/curriculum that tightens the kept/weighted set across
  phases. Prep-time → main-safe. Pairs with DATA-062.
- **DATA-062** [data-quality, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-distilled quality classifier.** FineWeb-Edu labels are an LLM's OPINION of quality;
  cola-coder owns OBJECTIVE executable ground truth (sandbox tsc/tests + best-of-N pass-rate +
  security scanner). Label a code corpus with verifier outcomes (compiles? tests pass? secure?
  best-of-N pass-rate) and distill THOSE into the local TF-IDF quality classifier (reuse
  train_judge_classifier's distillation, swap LLM-judge labels → verifier labels). Yields a fast
  static quality scorer grounded in correctness, not LLM judgment — cheaper + more objective than
  FineWeb-Edu, uniquely possible because cola-coder has the verifier. Feeds quality weights /
  folding curriculum (DATA-058). Builds on verifier + best-of-N + scanner. Prep-time → main-safe.
- **MODEL-026** [reasoning, medium] `open` — Add **DAPO dynamic sampling** to GRPO:
  drop prompt groups where all G samples are all-correct or all-incorrect (zero
  advantage → zero gradient → wasted rollout compute), resampling to keep the
  effective batch full. reasoning.yaml already has the other DAPO ingredients
  (clip-higher 0.28, Dr.GRPO mean-norm, no-KL); this is the missing one. Touches
  `reasoning/grpo.py` (training path → worktree). Ref: DAPO; llm-stats post-training-2026.
- **DATA-056** [data-quality, medium] `open` — Qwen3-style **reasoning-stage data
  curriculum**: after base pretraining, a higher-quality STEM/code stage with curated
  tokens (pairs with DATA-055 model-based perplexity scoring to select the high-value
  tail and order easy→hard). Ref: Qwen3 multi-stage recipe (30T base → 5T HQ reasoning
  → long-context). Closes the eval→data loop.
- **EVAL-023** [eval, low] `open` (refined 2026-06-13 — see research-log) — HumanEval
  is saturated/contaminated (a ~90% qualification bar, not a selection signal). Add
  2026 real-signal evals: a **LiveCodeBench-style contamination-free** set (problems
  postdating the training cutoff) and a **proprietary 100-500 held-out set** from the
  project's own (recent, license-clean) scraped code; optionally SWE-bench-style
  repo-level later. Use only 2-3 benchmarks (avoid overfitting). Defer until the model
  produces runnable code (~step 10k+). See IDEA-007 for the contamination-free approach.
- **EVAL-024** [eval, medium] `done` (2026-06-14) — **secure-pass@k** (CWEval/CodeGuard+
  2026 secure-codegen standard; arXiv:2503.15554). pass@k alone over-credits
  working-but-insecure code. Added to `evaluation/metrics.py`:
  `ProblemResult.num_secure_correct` (passed AND no dangerous patterns; None=unassessed,
  back-compat) with a `__post_init__` guard (secure-correct ⊆ correct) + `secure_pass_rate`
  property; `compute_secure_pass_at_k` reuses the unbiased `pass_at_k` estimator (DRY) so
  it's directly comparable to pass@k (gap = share of correct-but-insecure), mirroring the
  None/exclusion-warning semantics; `format_results` prints it when any problem is assessed.
  `scripts/evaluate.py` populates it via the shared `scan_dangerous` scanner on each PASSING
  sample (DRY — same scanner as best-of-N/GRPO). Eval-only → committed on main. Tests:
  test_inference.py::TestSecurePassAtK +6 (==pass when all secure, insecure lowers it,
  unassessed→None, secure⊆correct guard, rate property, format shows it); 32 pass, ruff green.
- **IDEA-015** [research/eval+data, high-potential] `open` (2026-06-14, ORIGINAL) —
  **security-gap-driven data synthesis.** Per-problem `pass@k − secure-pass@k` (EVAL-024) is
  a weakness map: problems solved but INSECURELY are exactly where secure-coding data is
  missing. Mine high-gap problems, take their insecure-but-passing completions, synthesize
  paired (insecure→secure) FIM/instruction examples (the dangerous span = FIM middle to
  infill securely), up-weight via quality weights, and feed the GRPO security penalty
  (IDEA-008). Closes the eval→data loop with cola-coder's scanner + sandbox verifier +
  dynamic FIM + quality weights. Builds on EVAL-024 + IDEA-008.
- **IDEA-007** [research/eval, medium-potential] `open` (2026-06-13) — Self-refreshing
  contamination-free eval: use the GitHub scraper to pull RECENT license-clean TS/React
  files postdating the training cutoff, hold out the middle as a FIM task, and score
  infill with the SANDBOXED tsc/test verifier (LiveCodeBench philosophy from the
  project's own scraper + FIM + sandbox). Refreshable each eval so gains can't be
  memorization. Combines scraper + FIM + sandbox-verifier assets.

### Backlog fill 2026-06-13 (discovery batch — model / application / code)

**Model & training quality**
- **MODEL-020** [model/training, high] `open` — Online/curriculum data reweighting
  (2025-26 data-centric). `data/mixing.py` has `MixingOptimizer` (the reweighting
  math) but it's NOT wired into the trainer. Track running loss per source/difficulty
  bucket and up-weight underfit buckets (reweight, don't hard-filter — preserve
  diversity). Needs per-sample bucket tags or a multi-source dataloader. Stage it.
- **MODEL-021** [model/training, medium] `open` — Weight EMA: maintain an exponential
  moving average of params (ema_decay≈0.999, update post-optimizer-step), save an EMA
  checkpoint alongside the raw one. EMA weights usually eval better — a cheap quality
  win. Wire ema state into checkpoint.py save/load.
- **MODEL-022** [model/inference, medium] `open` — Speculative decoding (draft-and-
  verify) for lower inference latency on serve/REPL/FIM. Self-speculative or n-gram
  draft fits a single-model setup. 2025 latency technique.
- **MODEL-023** [model/eval-coverage, medium] `open` — Enable qk_norm + z_loss on
  tiny/small configs (relates MODEL-012) so CI/smoke actually exercise those paths; a
  regression in qk_norm/z_loss currently passes small-config tests. z_loss is
  backward-compatible; qk_norm adds params (fresh runs only).

**Application (VS Code extension + FastAPI server)**
- **EXT-005** [extension/UX, medium] `open` — Surface training progress in the status
  bar: poll a metrics endpoint / read the manifest to show "training: step N / loss X",
  so an unattended run is monitorable from the editor. Fits the unattended workflow.
- **EXT-006** [extension/robustness, low] `open` — HealthMonitor uses a fixed 5s poll;
  add exponential backoff when disconnected (5→10→30s cap) + a "Cola-Coder: Reconnect
  now" command so the user isn't stuck on the fixed interval after starting the server.
- **EXT-007** [extension/perf, low] `open` — Cache inline FIM completions by
  (prefix,suffix) hash in a small LRU; invalidate on edit. Cuts latency + server load
  for the ghost-text hot path.
- **INFER-024** [inference/robustness, medium] `open` — Server has no max-concurrency /
  queue cap: requests pile up behind `_gen_lock` unboundedly. Add a bounded queue (429
  when saturated) + per-request server-side timeout so a burst can't wedge latency.
  Complements INFER-021.
- **INFER-027** [inference/decoding, medium] `done` (2026-06-14) — **FIM suffix-overlap trim.**
  FIM models over-generate, re-emitting the document SUFFIX they were given as context
  (arXiv:2509.24637, 2605.22981) → duplicated code in the editor after an accepted completion.
  Added `text_utils.trim_suffix_overlap(infill, suffix, min_overlap=3)`: removes the LONGEST
  verbatim overlap between the infill's tail and the suffix's head (small min-overlap so a lone
  coincidental `;`/`}` is kept). Wired into the `/v1/fim` handler. Inference/server → off the
  train path → committed on main. Tests: test_text_utils.py +7 (verbatim dup, longest-not-shortest,
  no-overlap, tiny-coincidence kept, full-dup→empty, empty inputs, min_overlap threshold); 13
  text_utils + 80 FIM tests green, ruff clean.
- **INFER-028** [inference/decoding, medium] `done` (2026-06-14) — **top-nσ sampling**
  (the 2026 truncation sampler, now in llama.cpp; arXiv:2411.07641). Keep tokens with
  `logit >= max − n·σ` over RAW pre-softmax logits → TEMPERATURE-INVARIANT truncation (raising
  temperature for diversity doesn't drag in the noisy tail like top-p/min-p). Added
  `_top_n_sigma_filter` / `_top_n_sigma_filter_batch` (σ/mean over FINITE logits so an upstream
  n-gram ban can't corrupt it; σ=0 → no-op), applied BEFORE temperature, wired as `top_n_sigma`
  through sample_next_token / sample_next_tokens_batch / generator.generate (default 0.0 = off →
  unchanged). Inference → committed on main. Tests: test_inference.py::TestTopNSigma +7; 46
  inference/sampling tests green, ruff clean. FOLLOW-UP: thread top_n_sigma through generate_stream
  / generate_group / the server request models for full exposure.
- **INFER-029** [research/inference, medium-potential] `done` (2026-06-14) —
  **verifier-calibrated adaptive sampling.** `generate_best_of_n_adaptive` gained
  `temperature_growth` (default 1.0 = off) + `max_temperature`: each round the sandbox verifier
  rejects the whole batch, the next round's temperature is multiplied by `temperature_growth`
  (capped) so retries WIDEN the distribution instead of resampling the same systematic failure
  (2026 TTS: modify the distribution, don't just resample it; arXiv:2602.01070). Inference-time
  analogue of IDEA-013's RL entropy controller. Backward-compatible (growth 1.0 → fixed temp).
  Inference → committed on main. Tests: test_best_of_n.py::TestTemperatureEscalation +5 (escalates
  on reject, capped, default-off, no-escalation-after-early-stop, invalid-config); 40 best-of-N
  green, ruff clean. FOLLOW-UP (INFER-028 follow-up): thread top_n_sigma through generate_group so
  the schedule can also LOOSEN n·σ, not just temperature.
- **INFER-030** [inference/server, medium] `done` (2026-06-14) — **expose top-nσ on the server.**
  INFER-028 added the top-nσ sampler to core sampling + `generate`; this completes the path so it's
  usable from the IDE. Threaded `top_n_sigma` through `generator.generate_stream` (chat SSE) and
  added `top_n_sigma: float = 0.0` to all 4 server request models (Generate/ChatCompletion/Completion/
  FIM), forwarded at all 6 generate/generate_stream call sites. Inference/server → committed on main.
  Tests: test_server_top_n_sigma.py +3 (4 fields, all 6 sites forward via no-silent-no-op regex,
  generator methods accept it); 10 server tests + ruff green. NOTE: best-of-N (`_best_of_generate`)
  uses its own batched sampling and does NOT yet thread top-nσ → INFER-031.
- **INFER-031** [inference, low] `open` (2026-06-14) — thread `top_n_sigma` through `generate_group`
  (and `_generate_candidates` / `_best_of_generate`) so best-of-N and the INFER-029 escalation can
  also widen via n·σ, not just temperature. Mechanical plumbing across the generate_group fallback
  chain (sample_next_tokens_batch already accepts it). Inference → main-safe.
- **IDEA-026** [research/inference, medium-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-gated KV-cache precision.** The two serving regimes want opposite KV trade-offs and the
  verifier says which is safe: for best-of-N (throughput, accuracy-TOLERANT since the sandbox verifier
  filters losers), use a Q4-quantized KV-cache so MORE candidates fit in 16 GB → larger best-of-N
  budget per VRAM; for latency/accuracy-critical inline FIM (no downstream verifier), keep
  full-precision KV. The verifier makes Q4's quality loss free for best-of-N (it discards the bad
  ones), turning saved KV memory into more verified samples — a trade no KV-quant paper makes (none
  has a downstream verifier). Builds on best-of-N verifier + generate_group KV-cache. Inference → main-safe.
- **EVAL-026** [research/eval+curriculum, high-potential] `done` (2026-06-14) —
  **verifier-effort difficulty profiling.** `BestOfNResult` now carries
  `candidates_used / rounds / final_temperature / solved` (populated by both fixed-N and adaptive
  best-of-N). New `evaluation/difficulty_profile.py`: `verifier_effort_tier(candidates_used,
  max_candidates, solved)` → easy/medium/hard/unsolved by budget fraction consumed for a verified
  solve; `profile_difficulty(results, max_candidates)` → per-tier counts + solve-rate + mean
  candidates. A free, objective, MODEL-RELATIVE difficulty label from the verifier (no human
  annotation). Pure logic → runs/tests with no GPU. Off the train path → committed on main. Tests:
  test_difficulty_profile.py +10 (tier boundaries, zero-max safe, distribution+rates, empty,
  fixed-N + adaptive effort fields); 50 (incl. 40 best-of-N) green, ruff clean. FOLLOW-UP: a CLI
  report + wiring tiers back into the GRPO curriculum (MODEL-042).
- **EVAL-027** [eval, medium] `done` (2026-06-14) — **verifier-anchored LLM-judge calibration.**
  LLM-judges are biased/noisy (verbosity, position; arXiv:2601.20913); the project owns bias-free
  ground truth for code (the sandbox verifier). New `evaluation/judge_calibration.py`:
  `agreement_stats` (judge TPR/FPR/accuracy/Cohen's κ vs the verifier oracle), `corrected_prevalence`
  (Rogan-Gladen recovery of the TRUE pass-rate from a noisy judge — so verbosity bias can't inflate
  corpus quality), `best_score_threshold` (calibrate the judge's score cut-point to the verifier by
  accuracy or Youden's J). Pure logic → no GPU. Eval → committed on main. Tests:
  test_judge_calibration.py +12 (confusion, perfect/biased judge, TPR-undefined, Rogan-Gladen
  recovery+clamp+no-signal, threshold by accuracy/Youden, empties); ruff clean.
- **IDEA-025** [research/data+eval, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-recalibrated judge distillation.** `train_judge_classifier` distills the LLM-judge's
  BIASED scores into the local TF-IDF classifier — baking verbosity/position bias into the project's
  permanent quality signal. Instead, on a set scored by BOTH judge and sandbox verifier (EVAL-027),
  use `best_score_threshold` + `corrected_prevalence` to recalibrate the judge's labels to verifier
  ground truth BEFORE distilling — and where the verifier runs, distill the VERIFIER's label directly
  (DATA-062). The classifier then learns verifier-grounded quality, not an LLM's opinion — debiasing
  the whole data-scoring stack with assets the eval papers (human-label calibration only) lack.
  Builds on EVAL-027 + train_judge_classifier + verifier + DATA-062. Data/eval → main-safe.
- **MODEL-042** [reasoning/curriculum, high-potential] `done` (2026-06-14) —
  **verifier-effort E2H scheduler.** New `reasoning/curriculum_scheduler.VerifierEffortCurriculum`
  (pure logic): tracks each problem's verified pass-rate per epoch; `tier_for` re-tags difficulty
  from the LATEST rate; `is_mastered` (≥ threshold for a streak); `active` fades mastered problems
  Easy→Hard (arXiv:2506.06632) keeping ≥ min_active (re-includes least-mastered to fill the floor).
  Wired opt-in into `GRPOTrainer.train(e2h_scheduler=…)`: records per-step pass-rate, and between
  epochs re-tags `problem["difficulty"]` (feeding the per-difficulty temperature + IDEA-020 entropy
  floors with MEASURED difficulty) and drops mastered problems. Default None → unchanged. Difficulty
  is now measured/evolving/model-relative — what E2H/SEC call for — via verifier + curriculum +
  entropy controller, which no curriculum paper (no execution verifier) can. Reasoning-only →
  committed on main. Tests: test_curriculum_scheduler.py +10 (tier from rate, epoch-mean, mastery
  streak + reset, fade-out, min_active floor keeps hardest, never-empties, invalid-config); GRPO +
  reasoning suites green, ruff clean. FOLLOW-UP: wire `--e2h` into train_reasoning.py + a CLI report.
- **MODEL-043** [model/architecture, medium] `open` (2026-06-14, ORIGINAL) —
  **output-embedding centering, verifier-A/B'd.** 2026 output-logit-stability lever (arXiv:2601.02031):
  subtract the running column-mean of the (tied) embedding from logits to stop logit drift — a cheap
  complement to z_loss. Add as a checkpoint-safe opt-in flag (no new params) and A/B on a tiny Muon
  run measuring loss AND downstream secure-pass@k (EVAL-024) + verifier-effort difficulty (EVAL-026),
  not just perplexity (where the papers stop). Composes with z_loss + qk_norm. Train-path (model/) →
  WORKTREE, A/B before merge; pair with the MODEL-025 Muon A/B. Builds on the z_loss/qk_norm stack.
- **MODEL-044** [reasoning/wiring, high] `done` (2026-06-14) — **wire the entropy/curriculum stack
  into the reasoning CLI.** The stack (MODEL-037 metric → IDEA-013 controller → IDEA-020 per-difficulty
  floors → MODEL-042 E2H scheduler) was implemented with opt-in constructor params but UNREACHABLE
  from train_reasoning.py — implemented-but-dead. Added `--entropy-control` (+ `--entropy-target`) →
  builds `EntropyClipController` (per-difficulty floors auto-on with `--curriculum`) passed to
  GRPOTrainer; `--e2h` → builds `VerifierEffortCurriculum` passed to `.train()`. CLI-only (no
  phantom-config risk; the wiring test enforces config keys are read). Script-only → committed on
  main. Tests: test_reasoning_entropy_e2h_wiring.py +4 (trainer accepts kwargs, flags defined, script
  forwards both via AST check, curriculum-floor preset valid); config-wiring + ruff green. FOLLOW-UP:
  add menu toggles in training_menu's Alignment & Reasoning group.
- **MODEL-045** [model/post-training, high] `done` (2026-06-14) — **RFT / self-verified distillation
  harness** (rejection-sampling fine-tuning; arXiv:2605.10674, 2605.26132). The 2026 self-improvement
  recipe — sample N, keep only verifier-passed, SFT on those — and the project owned every piece
  (best-of-N + sandbox verifier + SFT) but had no harness. New
  `distillation/rft.py::generate_rft_dataset`: per prompt, generate ``num_candidates`` and verify+rank
  with the existing `generate_best_of_n` (sandbox tsc/tests + security + self-consistency), keep the
  best only if verified (``keep_only_verified``) AND secure (``require_secure``); emit ChatML SFT
  records (reuses `_to_messages`, DRY). Self-distillation (student's own verified output), complement
  to the teacher-based `generate_distillation_dataset`; never executes model output itself. Exported
  from distillation/__init__. Distillation/offline → committed on main. Tests: test_rft.py +5 (keep
  verified+secure, drop unverified, keep-when-off, drop insecure, length guard); distillation regression
  (11) + ruff green. FOLLOW-UP: MODEL-046.
- **MODEL-046** [model/post-training, medium] `done` (2026-06-14) — **runnable RFT front-end.**
  New `scripts/generate_rft_data.py`: loads a checkpoint via `load_generator` (DRY), pulls prompts
  (built-in problems OR `--jsonl` of `{prompt, test_code}`), runs MODEL-045 `generate_rft_dataset`
  (best-of-N → keep verified+secure), writes ChatML SFT JSONL + stats. Wired into the Training menu →
  Post-Training (`_generate_rft_menu`). The self-improvement engine is usable end-to-end: generate
  RFT data → train_sft.py → repeat. Tooling → committed on main. Tests: test_generate_rft_script.py
  +2 (JSONL loading incl. missing-test→None, built-in max_prompts); 7 RFT tests, menu imports,
  --help, ruff green. Running needs a checkpoint (GPU) → between runs / spare GPU. FOLLOW-UP: an
  iterative wrapper (generate → SFT → regenerate) + AdaSTaR-style frontier prioritization (IDEA-027).
- **DATA-067** [data-quality, high-potential] `open` (2026-06-14, ORIGINAL) — **verified-RFT data as
  a quality-classifier teacher.** RFT (MODEL-045/046) yields a growing pool of VERIFIER-PASSED
  (objectively correct + secure) completions plus the REJECTED ones (failed tsc/tests or insecure) —
  a free, perfectly-labelled binary corpus. Feed BOTH classes into `train_judge_classifier` so the
  quality classifier learns "what verifier-passing TS looks like" from GROUND TRUTH, not LLM-judge
  opinion (DATA-062/IDEA-025). Each RFT round improves the model AND sharpens the scorer that curates
  the NEXT pretraining batch — a compounding flywheel where verifier labels propagate into the
  quality-weights pipeline. Combines RFT + verifier + quality classifier + quality weights. Builds on
  MODEL-045/046 + DATA-062 + train_judge_classifier. Data → main-safe.
- **IDEA-027** [research/post-training, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-effort-curriculum RFT.** Plain RFT samples every prompt uniformly, wasting budget on
  trivial + impossible prompts. Use EVAL-026's verifier-EFFORT to self-curate RFT's curriculum: after
  a round, classify each prompt by best-of-N budget-to-solve (or unsolved), and in the NEXT round
  concentrate sampling on FRONTIER prompts (solved-but-hard / just-barely-unsolved), retire trivially-
  solved (E2H, MODEL-042), shelve impossible — a self-curating RFT that spends verification budget
  where learning happens. Combines RFT + best-of-N + verifier-effort (EVAL-026) + E2H (MODEL-042) — no
  RFT paper (no verifier-effort signal) can. Builds on MODEL-045 + EVAL-026 + MODEL-042. Distillation → main-safe.
- **IDEA-024** [research/post-training, high-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-localized entropy injection.** SIREN restricts entropy regulation to the nucleus /
  peak-entropy tokens; cola-coder can localize by CORRECTNESS. The MODEL-037 entropy metric averages
  over ALL completion tokens; combine with IDEA-023's execution-trace token map (tokens in the failing
  assert's region) so the entropy FLOOR is enforced SELECTIVELY on the tokens that produced the
  failure — explore where the verifier says the model is wrong, exploit where it passes. Difficulty-
  aware (IDEA-020) × token-localized (SIREN) × verifier-grounded (IDEA-023) — no entropy paper (no
  execution verifier) can do this. Builds on MODEL-037 + IDEA-013 + IDEA-023. Reasoning-only → main-safe.
- **IDEA-021** [research/inference, medium-potential] `open` (2026-06-14, ORIGINAL) —
  **verifier-gated FIM completion boundary.** Precise stopping via cola-coder's assets instead of
  a bespoke incremental grammar parser: use the sandbox tsc verifier as the "complete program?"
  oracle — find the SHORTEST infill prefix s.t. prefix+infill+suffix type-checks clean
  (tsc --noEmit), stop there (best-of-N over candidate stop points scored by the verifier).
  Turns the verifier into a syntactic-completeness stop signal the FIM-decoding papers approximate
  with hand-written parsers. Fast path = INFER-027 string trim; precise path = this. Builds on
  /v1/fim + TscRunner + best-of-N. Inference → main-safe.
- **INFER-025** [inference/feature, low] `done` (2026-06-13) — `/v1/fim` (`FimRequest`)
  now exposes `repetition_penalty` (default 1.1) and the FIM handler forwards it to
  `base_gen.generate` — parity with chat/completions (it was the only direct call site
  that omitted it). Tests: test_server_no_repeat_ngram.py::TestFimExposesRepetitionPenalty
  +3 (field present+default, settable, all 6 direct call sites forward it); 7 pass, ruff
  clean. Server-only, does not touch training.
- **INFER-026** [inference/decoding, medium] `done` (2026-06-14) — **self-consistency
  tiebreaker** in best-of-N (AlphaCode-style program clustering / self-consistency voting;
  arXiv:2502.20379). `best_of_n._rank` key `(verified, secure, score)` →
  `(verified, secure, consistency, score)`: `_normalize_completion` collapses
  whitespace/blank lines, candidates cluster by normalized completion, and `consistency` =
  cluster size annotates `details["consistency"]`. Within the verified+secure tier the
  model's most-repeated solution wins (correct solutions converge), score is the final
  tiebreak. A free SECOND verification signal needing no extra model. Backward-compatible:
  all-unique → clusters size 1 → prior order preserved (30 existing best-of-N tests still
  green). Pure inference → committed on main. Tests: test_best_of_n.py +6 (normalize
  clusters/keeps-diffs, majority wins tie, all-unique→score, can't override verified or
  secure tier); 31 best-of-N tests + ruff green.
- **IDEA-014** [research/inference, medium-potential] `done` (2026-06-14) —
  **cluster-gated verifier budget.** The sandbox verifier (tsc/test run per candidate) is
  best-of-N's dominant cost. `best_of_n._verify_deduplicated` groups candidates by normalized
  completion (INFER-026 clusters), runs the verifier on ONE representative per cluster, and
  propagates the verdict (fresh `details` copy per candidate — no aliasing) to the rest → N
  candidates cost ~(#distinct programs) runs, not N, with identical results. Default-on
  (`cluster_verify=True`) on both `generate_best_of_n` and `_adaptive`; backward-compatible
  (all-unique → no change). Pure inference → committed on main. Tests: test_best_of_n.py +4
  (dedups runs, off-switch verifies all, verdict propagation, independent details); 35
  best-of-N pass, ruff green. Follow-up: largest-cluster-first under a hard budget cap.
- **MODEL-038** [model/architecture, medium] `open` (architecture research 2026-06-14) —
  **gated attention**: elementwise gate on the SDPA output before the output projection
  (reduces attention sinks + massive activations, improves long-seq generalization,
  negligible params; a 2026 standard). Add as an opt-in `config.model.gated_attention`
  flag DEFAULT OFF so existing checkpoints still load (no new params when off). Train-path
  (model/) → implement in an ISOLATED worktree, run tests/test_checkpoint.py there, DO NOT
  merge while small_react_best is live. Composes with qk_norm (already on).
- **MODEL-039** [model/architecture, low] `open` (architecture research 2026-06-14) —
  **depth-scaled sandwich norm**: 4 RMSNorms per block, 2nd init ~1/√L so the residual
  update starts small and grows — cheap deep-training stability. Opt-in flag, checkpoint-
  safe default off. Train-path → worktree-only while live. Lower priority than MODEL-038.
- **IDEA-017** [research/training, high-potential] `open` (2026-06-14, ORIGINAL) —
  **one-shot HP transfer for the specialist fleet.** The vision is a router + many 50M
  domain specialists. muP (MODEL-031) tunes LR/init ONCE on a tiny proxy width and zero-shot
  transfers to any width; qk_norm (present) widens that basin; Muon also admits muP-style
  transfer. Tune one HP recipe on a ~10M Muon+QK-Norm+muP proxy, then stamp it across the
  whole fleet with no per-model sweep — turning the multi-specialist design (usually a tuning
  cost) into a one-sweep-amortized win. Builds on MODEL-031 + Muon + qk_norm.

**Code / data / eval / docs**
- **DATA-054** [data-quality, medium] `open` — Semantic (embedding/SimHash) dedup:
  combine_datasets.py has MinHash near-dup; add an optional semantic pass to catch
  paraphrase/near-clone code MinHash misses ("exact, near, semantic" levels). Flag-gated;
  expensive.
- **DATA-055** [data-quality, medium] `open` — Model-based data scoring: once a usable
  checkpoint exists, use per-example perplexity/loss as a quality+difficulty signal to
  drop the noisy high-ppl tail and/or build a curriculum. Closes the eval→data loop.
- **EVAL-022** [eval/automation, medium] `open` — Wire periodic auto-eval into the
  training loop (training_eval_history.py exists but is manual): every N steps run a tiny
  held-out loss + a few completion probes → append to an eval-history JSONL, so
  quality-over-time is visible and regressions surface early without manual runs.
- **DOC-001** [docs, low] `open` — docs/deep-dives/fill-in-the-middle.md still documents
  the deleted `FIMCollator` (removed in DATA-053); update it to the canonical
  `dataset.FIMTrainingCollator` + `create_dataloader(fim_rate=...)` path and mention the
  per-worker RNG reseed (DATA-052).
- **TOOL-021** [tooling/UX, low] `open` — smoke_test.py uses a module-level
  `FEATURE_ENABLED` toggle ("set FEATURE_ENABLED=False in smoke_test.py to skip") —
  confusing leftover; replace with a documented `--skip-smoke` CLI flag / env var instead
  of requiring users to edit the script.

### Prior open/done items

- **EXT-001** [extension/UX, medium] `done` (2026-06-13, fresh VS Code extension
  audit; tsc baseline clean) — `cola-coder.generateVerified` ran best-of-N
  (verifiedCandidates full generations + sandboxed tsc/python verification, 30s+)
  via `withProgress` with NO `cancellable` flag and `client.complete()` forwarding
  no `AbortSignal` — so the Cancel button was absent and nothing could stop a slow
  verified generation: the GPU stayed pinned and the status bar stuck in
  "generating" until the orphaned request returned. FIX: `complete()` now
  accepts+forwards a signal (`post()` already supported it); the progress is
  `cancellable: true` and its token aborts an AbortController. `npx tsc --noEmit`
  clean + esbuild bundle builds (dist/extension.js 53KB).
- **EXT-002** [extension/bug, medium] `done` (2026-06-13) — `ServerManager.stop()`
  reset `this.stopping=false` in a `finally` that ran BEFORE the process' async
  `'exit'` fired (the 5s SIGKILL-timeout path called `resolve()` immediately after
  `kill`), so the exit handler saw `!stopping` + a non-zero code and called
  `_handleCrash()` → spurious auto-restart of a deliberately-stopped server
  (Windows timing). FIX: the `'exit'` handler now OWNS clearing `stopping`
  (reads `wasStopping`, resets, decides crash from that) + a process-identity
  guard (`this.process === proc`) so a late exit can't null a restart-spawned
  process; `stop()` no longer clears `stopping`/nulls `process`, waits on the real
  `'exit'` (SIGKILL at 5s, 6s zombie fallback that leaves `stopping` set so no
  spurious restart). `npx tsc --noEmit` clean + esbuild bundle builds.
- **EXT-003** [extension/bug, low] `done` (2026-06-13, listener-dispose part) —
  InlineCompletionProvider, ChatParticipant, and LanguageModelProvider registered
  `token.onCancellationRequested(() => controller.abort())` without disposing the
  subscription, leaking a listener + AbortController closure per request for the
  token's lifetime. FIX: capture the Disposable and `dispose()` it in a `finally`
  (InlineCompletion: also disposes on the early already-cancelled return; Chat:
  folded into its existing finally). `npx tsc --noEmit` + esbuild clean. Remaining
  sub-item (debounceMs=0 unbounded request pile-up) tracked as EXT-008 — the
  stale-ghost-text case is already guarded by the token re-check, so this is the
  lower-impact half.
- **EXT-008** [extension/perf, low] `open` (follow-up from EXT-003) — With
  `inline.debounceMs=0` the InlineCompletionProvider issues an unbounded FIM
  request per keystroke; stale results are already discarded (token re-check) so
  there's no wrong-output bug, but rapid typing piles up in-flight requests behind
  the server's `_gen_lock`. Enforce a small minimum effective debounce (or coalesce
  to the latest pending request) so a burst can't queue N fetches. Pairs with the
  server-side cap in INFER-024.
- **EXT-004** [extension/bug, low] `done` (2026-06-13) — `FimFormatter` truncated the
  prefix/suffix on raw char offsets, which could split a UTF-16 surrogate pair
  (emoji/CJK in code) and send a lone surrogate to the server, corrupting
  tokenization on non-ASCII files. FIX: `extractFimContext` now snaps the prefix
  start forward off a low surrogate and pulls the suffix end back off a dangling
  high surrogate (isHighSurrogate/isLowSurrogate helpers). `npx tsc --noEmit` +
  esbuild clean. (Completes the VS Code extension audit batch EXT-001..004.)
- **MODEL-015** [training/wiring, low-medium] `open` (optimizer report-only audit)
  — `trainer.py` does NOT pass `wsd_decay_fraction` to `create_scheduler`, so the
  WSD decay fraction is always the hardcoded default (0.2). A user adding
  `training.wsd_decay_fraction` to a YAML would have it silently ignored — the
  "phantom knob" class the rules warn about. NON-numerics-changing for the live
  run (default already in effect; no config sets it), so safe to wire whenever.
  The rest of the optimizer/scheduler audit found the LIVE path (bf16+AdamW+WSD)
  SOUND: step-0 LR>0, decay starts at the right step, min_lr_ratio applied, no
  param silently frozen, fp16 scaler-skip→no-scheduler-step invariant enforced.
- **MODEL-016** [model/latent, low] `open` (model report-only audit) — Cached
  multi-token decode has no causal mask: `attention.py` cached branch passes
  `attn_mask=mask` with `is_causal` unset and `mask=None` for `start_pos>0`.
  Correct ONLY because every decode step feeds exactly 1 token (verified). If any
  future caller feeds >1 token at `start_pos>0` (chunked prefill, speculative
  decode), queries would attend to future keys with no mask → silently wrong.
  Latent (no current caller), inference-only, NOT live. Fix: `is_causal=(seq_len>1
  and mask is None)` or build a rectangular causal mask in the cached branch.
- **MODEL-017** [model/non-live, low] `open` (model report-only audit) — YaRN
  `precompute_rope_freqs_yarn` (rope.py) uses a per-frequency Python loop with
  `.item()` (slow, breaks tracing) and an interpolation-band `smooth` formula based
  on the wavelength ratio rather than the standard YaRN linear ramp on the rotation
  index — verify against the paper before any Stage-4 context-extension run. NOT
  exercised by the live run (rope_scaling none) → safe to fix now.
- **MODEL-018** [model/latent, low] `open` (model report-only audit) —
  `attention.py` `expand_cache` (`cache_k[:1].expand(batch,...)`) is correct only
  under its documented contract (batch=1 prefill ran first); no guard if called
  with cache=None (silent no-op) or after a batch>1 prefill (wrongly collapses to
  row 0). Used only by GRPO parallel generation, not pretraining. Add a contract
  assertion. NOT live.
- **MODEL-019** [model/design-note, info] `open` (model report-only audit) —
  z_loss uses an unweighted token `.mean()` while CE uses the per-sample quality
  weights; the live run uses both z_loss=1e-4 and `.weights.npy`, so low-quality
  samples contribute equally to the (tiny, 1e-4) z-penalty. Defensible design
  (z-loss is a logit-scale regularizer, not an LM cost) — flagged so the choice is
  conscious. Changing it is NUMERICS-CHANGING for the live objective → do NOT
  change mid-run; decide deliberately later.
- **DATA-052** [data-quality/training, high] `done` (2026-06-13, fresh
  collator/FIM audit) — Dynamic FIM RNG was REPLICATED, not independent, across
  DataLoader workers. `FIMTrainingCollator` builds its `FIMTransform` in the main
  process with `seed=None`; with `persistent_workers=True` + `num_workers=8` (all
  configs incl. the live run) the collator is pickled to each worker carrying an
  IDENTICAL `random.Random` state, so all 8 workers made identical FIM decisions
  (which samples FIM'd, PSM/SPM, split points) in lockstep — collapsing 8
  independent augmentation streams into 1 and sharply reducing FIM diversity for
  any `fim_rate>0` run (a real training-quality regression, not a crash). FIX:
  `FIMTransform._ensure_worker_rng()` re-seeds once per worker on first use from
  torch's distinct per-worker seed (mixed with base seed + worker id); no-op in
  the main process / num_workers=0 so seeded reproducibility (tests) is preserved.
  FIM is stochastic augmentation → safe on resume (no weight desync), so this was
  landed despite the live run. Tests: test_fim.py::TestPerWorkerRngReseed +2;
  60 FIM tests + ruff green. NOTE: the CURRENTLY-RUNNING process loaded the old
  code, so it keeps replicated-RNG FIM until it restarts/reboots — see the
  "restart?" decision surfaced to the user (run is only ~0.3% in).
- **DATA-053** [data-quality/cleanup, low] `open` (from the DATA-052 audit,
  zero-numeric-impact, deferred to avoid touching the live path mid-run) — (a)
  `fim.py` `MIN/MAX_MIDDLE_FRAC` docstring claims the middle is bounded to 10-90%
  of seq length; the constants actually bound the split-point INDICES and the
  middle can be a single token — reword. (b) `collator.py` `FIMCollator` is
  orphaned dead code (only its own test/docs reference it; the live path uses
  `dataset.FIMTrainingCollator`) and its docstring still calls DATA-012
  "optional/not wired" though DATA-012 is done — delete `FIMCollator` +
  test_fim_collator.py, or mark deprecated.
- **BUG-125** [tooling/pipeline, high] `done` (2026-06-13, fresh pipeline-scripts
  audit) — `full_pipeline.py` Stage 6 (instruction-tune) and Stage 9
  (train-reasoning) resolved the pretrained checkpoint via `ckpt_dir/"latest"`
  directly. That pointer can be stale — it can reference a `step_*` dir later
  pruned by `max_checkpoints` cleanup (notably when training restarts from scratch
  in a dir already holding higher-step checkpoints) — so the stage would fine-tune
  a deleted/older checkpoint or crash on load. The Pipeline Manager already scans
  `step_*` dirs directly; full_pipeline had diverged. FIX: module-local
  `_resolve_latest_checkpoint()` scans `step_*` (highest wins), falls back to the
  pointer only when no step dir exists; wired into both stages. Tests:
  test_full_pipeline_stage_args.py::TestStalePointerResolution +5; 13 pass.
  (Audit class #5 stale references.)
- **TOOL-020** [tooling/UX, medium] `done` (2026-06-13, fresh monitoring audit) —
  `scripts/training_status.py` misreported a HEALTHY fresh run: `_describe_size`
  used truthiness checks (`if step`, `if best_step`, `max_steps and step`) that
  treat the valid value `0` as "missing". Very early in an unattended run (only
  `step_00000000` saved, best loss at step 0) the CPU-only readout showed
  `Latest step: ?`, `Progress: ?`, and dropped the `(step 0)` annotation — falsely
  signalling "no progress" on a run that was fine. FIX: test `step is not None` /
  `best_step is not None` (max_steps keeps its truthiness guard since 0/None means
  "not configured"). Tests: new test_training_status.py +4 (verified failing
  pre-fix). Sweep candidate: other monitoring code using `if value` for numeric
  fields where 0 is legitimate.
- **VALIDATION (2026-06-13):** First real checkpoint `step_00000500` validated
  end-to-end on CPU through the hardened load path (`load_generator`, device=cpu):
  weight-tying intact (`tok_emb.weight is output.weight`), no key mismatch, vocab
  32768, ~100M params, tiny greedy generate ran without error. Confirms the
  save→load→generate chain (incl. BUG-122/loading hardening) works on a genuine
  artifact, not just synthetic ones. Op note: `CUDA_VISIBLE_DEVICES=""` does NOT
  hide the GPU on this Windows/PyTorch build — use `-1`.
- **SEC-009** [security/sandbox, high] `done` (2026-06-13, fresh curation-sandbox
  audit) (de-collided from a provisional "SEC-001" label used in the commit msg —
  SEC-001 was already taken by the 2026-06-11 in-stream-malware entry) — The
  Docker sandbox did not actually enforce its timeout on untrusted
  code. `DockerSandbox.run` (data/curation/docker_sandbox.py) relied on
  `subprocess.run(timeout=…)`, which on `TimeoutExpired` kills only the
  `docker run` CLIENT process — the daemon-managed container (and the untrusted
  scraped code executing inside it) keeps running indefinitely; with `--rm` the
  container is removed only when it EXITS, which a hung/malicious test command may
  never do. So the sandbox's primary runtime control was defeated and cloned-repo
  code could run unbounded on the host — exactly the "not monitoring the PC" risk.
  FIX: launch each container with a unique `--name=cola-curation-<uuid4>` and, on
  `TimeoutExpired`, force-remove it via `docker rm -f <name>` (new
  `_force_remove_container`, best-effort/non-raising so cleanup failure never masks
  the timeout result). `--network=none`, `--cap-drop=ALL`, `no-new-privileges`,
  resource limits unchanged. Tests: test_curation.py +2 (mocked subprocess: run
  carries --name + timeout issues `docker rm -f`; helper swallows errors); 48
  pass. Follow-up hardening noted (force-remove on ALL exception paths + outer
  wall-clock watchdog + read-only rootfs) — see SEC-010.
- **SEC-010** [security/sandbox, medium] `done` (2026-06-13, de-collided from a
  provisional "SEC-002" label — SEC-002 was taken by the 2026-06-11 entry) —
  Defense-in-depth follow-up to SEC-009: force-remove the sandbox container on
  ALL exit paths, not just TimeoutExpired. FIX: `DockerSandbox.run` now wraps the
  run in try/except/finally with the unique `--name` generated once and reused
  everywhere — cleanup (`docker rm -f`) runs on timeout, any unexpected exception,
  KeyboardInterrupt (caught, cleaned up, re-raised), AND normal completion (in
  case `--rm` didn't fire). No container or `docker run` child can outlive
  `DockerSandbox.run`. Security flags + success-path return contract unchanged.
  Tests: test_curation.py TestDockerCleanupOnAllExitPaths +5 (normal/exception/
  pre-propagation/KeyboardInterrupt/name-reuse); 53 pass. Read-only rootfs
  DEFERRED → SEC-011 (sandbox runs as root with writable HOME for npm/pip; `run_
  with_install` copies into /tmp/workdir, so `--read-only` risks breaking it).
- **SEC-011** [security/sandbox, low] `open` (deferred from SEC-010) — Consider
  `--read-only` rootfs + an explicit writable `/tmp` tmpfs for the sandbox, plus
  an outer wall-clock watchdog independent of subprocess timeout. Needs care: the
  curation sandbox currently runs as root with a writable HOME for npm/pip and
  `run_with_install` copies code into /tmp/workdir.
- **MEM-002** [memory/correctness, medium] `done` (2026-06-13, fresh memory audit)
  — `MemoryManager._trim_session_log` (memory/manager.py) never actually trimmed
  the rolling window. It used a `re.DOTALL` regex to split the file preamble from
  the `## ` entries, but `.` matched newlines so the non-greedy capture swallowed
  the first section bodies INTO the "header"; old entries were never dropped and
  each trim re-appended the kept entries — DUPLICATING content and growing
  `session_log.md` unbounded (silently inflating the retrieval corpus and the
  on-disk log every interaction), the opposite of `session_log_max_entries`. FIX:
  split on the first `## ` line (multiline anchor), keep the preamble, re-emit
  exactly the last N sections. Tests: new test_memory_session_log.py +6 (window
  size, recency, no-duplication, preamble preserved); 11 pass.
- **BUG-122** [inference/correctness, high] `done` (2026-06-13, fresh load-path
  audit) — `load_generator` (inference/loading.py) inspected the checkpoint for
  vocab size and MoE config BEFORE resolving a `latest` pointer. Since
  `checkpoints/<size>/latest` is a text pointer FILE (not a dir), the inspections
  (`safe_open` on model.safetensors; `apply_moe_config_from_checkpoint` reading
  moe_config.json) silently found nothing. For an upcycled MoE checkpoint loaded
  via its DOCUMENTED `latest` path, MoE detection returned None → config stayed
  dense → `Transformer` built dense FFNs → `load_model_only` crashed in
  `_load_state_dict_tied` on unplaceable `experts.*` keys (even though
  load_model_only itself resolves latest). Reachable from chat.py / serve.py /
  smoke_test.py. FIX: resolve the pointer at the top of load_generator (same
  pattern load_model_only uses). Tests: new test_loading.py +3; 25 pass.
- **BUG-123** [inference/robustness, low] `done` (2026-06-13) — In repo_context.py,
  `_file_tokens`/`import_graph` keys used unresolved `str(self.root / rel_path)`
  while `_resolve_file_path`/`_resolve_import` returned `.resolve()`d paths —
  fragile under symlinks / Windows 8.3 short names (a key mismatch would silently
  drop context files). FIX: new `RepoScanner._canonical()` helper normalizes BOTH
  key-building and every lookup (resolve-paths, import-exclude set) through
  `.resolve()`. Behavior unchanged on already-canonical roots (no regression).
  Tests: test_repo_context.py +2 (keys canonical; resolved lookup finds the
  entry); 71 pass.
- **BUG-124** [tooling/robustness, low] `done` (2026-06-13, fresh features scan) —
  `estimate_vram()` (features/vram_estimator.py) guarded its `torch.cuda` probe
  with `except ImportError` only — not the realistic broken-CUDA case
  (`is_available()` True but `get_device_properties()` raises `RuntimeError` on a
  driver/version mismatch), so `scripts/vram_estimate.py` crashed with an uncaught
  traceback instead of printing a (GPU-independent) estimate. FIX: broadened to
  `except Exception` and degrade to the no-GPU path (gpu_/fits_ fields = None),
  matching the defense already in hardware_profiler.py. Tests: new
  test_vram_estimator.py +3 (broken probe, no-CUDA, GPU-independent breakdown);
  29 pass.
- **SFT-001** [data-quality/correctness, high] `done` (2026-06-13, fresh SFT
  audit) — `SFTDataset._tokenize_conversation` (data/sft_dataset.py)
  right-truncated over-long conversations (`token_ids[:max_seq_len]`), keeping the
  prompt and chopping the trailing assistant RESPONSE (which in ChatML sits at the
  end). Any example whose prompt alone exceeded max_seq_len lost ALL its labeled
  tokens → labels all -100 → `_load` then SILENTLY DISCARDED it as "no signal".
  Net: long-context instruction examples — exactly what a 4096-seq coder model
  most needs — vanished from the SFT set with no warning. FIX: build labels at
  full length, then LEFT-truncate the prompt (keep BOS at pos 0 + the
  max_seq_len-1 tail) so the response survives; labels sliced identically;
  max_seq_len<=1 guarded. Rest of the masking path (multi-turn, boundary
  alignment, EOS supervision, pad exclusion) verified correct. Tests:
  test_sft_dataset_labels.py +4; 41 SFT tests pass.
- **BUG-121** [reasoning/correctness, high] `done` (2026-06-13, fresh GRPO audit)
  — GRPO single-completion group produced NaN advantages that silently corrupted
  the policy. `compute_group_advantages(norm="std")` (grpo.py) divided
  mean-centered rewards by `torch.std()`, which returns NaN for a size-1 group
  (unbiased std divides by N-1 = 0) — the `+1e-8` epsilon only guarded the
  all-equal multi-element case (std==0), not std==NaN. The NaN flowed through the
  clipped surrogate and `backward()`, corrupting all weights; the `train_step`
  collapse guard (`std_reward < 1e-4`) missed it because `NaN < 1e-4` is `False`.
  `group_size` is an unguarded public `GRPOTrainer` param, so any self-play/custom
  caller passing 1 hits it. FIX: return all-zero (mean-centered) advantages for
  groups of size < 2 (a single completion has no group baseline → zero advantage
  is correct GRPO) + added a `torch.isnan` check to the collapse guard. Tests:
  test_modern_techniques.py::TestGroupAdvantages +2 (single-completion, empty);
  47 reasoning/GRPO tests + ruff green.
- **INFER-021** [inference/concurrency, high] `done` (2026-06-13, fresh serving
  audit) — SSE streaming held the GPU lock across the network yield.
  `_stream_chat`/`_stream_completion` (server.py) wrapped the whole token loop —
  INCLUDING the `yield` to the client — in `async with _gen_lock`. Under Starlette
  backpressure a `yield` suspends until the chunk flushes to the socket, so one
  slow/paused SSE client (e.g. a stalled VS Code tab) held the single-GPU lock for
  the stream's ENTIRE lifetime, serializing every other request behind it —
  inline FIM completions and the extension's 5s `/health` poll included. FIX:
  extracted `stream_chunks_locked(stream_iter, lock, is_disconnected)` that holds
  the lock only around each `next(stream_iter)` decode step and releases it before
  the caller's network yield (still one GPU decode at a time), and `close()`s the
  generator on disconnect so `clear_caches()` runs promptly. Tests:
  test_server_openai.py::TestStreamChunksLocked +2 (lock free while caller holds a
  chunk — verified it fails on the old code; disconnect stops pulling + closes
  generator); 54+ server/stream/best_of tests + ruff green.
- **TOOL-019** [tooling/bug, medium] `done` (2026-06-13, fresh tools audit) —
  `parse_tool_call` (tools/formatter.py) crashed the whole agent loop on a bare
  JSON scalar in a `<tool_call>` block. It only caught `json.JSONDecodeError` but
  ran `"name" in parsed` before type-checking — a block containing `123`/`true`/
  `null`/`12.5` parses fine yet is non-iterable, so the membership test raised an
  uncaught `TypeError` that propagated out and crashed `AgentLoop.run`. Since the
  model's tool-call output is UNTRUSTED, one stray scalar block took down the
  loop instead of being skipped. FIX: guard with `isinstance(parsed, dict)` so
  non-object JSON is skipped like other junk. Tests: new tests/test_tool_formatter.py
  (module had zero coverage) — parametrized scalars never raise + skip-but-continue
  + valid-call parsing; 39 tools tests + ruff green. (Note: agent proposed TOOL-018,
  renumbered to TOOL-019 — TOOL-018 is the tokenizer_health names bug.)
- **INFER-023** [inference/correctness, high] `done` (2026-06-13, de-collided from
  a provisional "INFER-014" label — INFER-013/014 were taken by 2026-06-12
  entries) — Follow-up to INFER-022 (the context-window clamp): the GRPO batched
  group path `_generate_group_single_batch`
  (`start_pos = prompt_len + step`) had the SAME unguarded KV-cache overflow — a
  prompt+budget exceeding `config.max_seq_len` drove start_pos past the bound, so
  the per-step `cache_k[:, start_pos:start_pos+1] = k` hit a zero-size slice
  (dropped K/V → garbage rollouts, corrupting GRPO rewards). FIX: reuse
  INFER-013's `_fit_context_window`/`_max_seq_len` to clamp the shared group
  prompt once after encode (generator.py `_generate_group_single_batch`).
  `generate_batch` is covered per-row via its delegation to `generate()`. Applied
  DIRECTLY on main, not via the agent branch: that worktree branched from a stale
  base lacking INFER-013, and a wholesale merge regressed the stop-token
  streaming tests — a reminder that worktree agents can branch pre-recent-work, so
  conflicted files need a 3-way resolution, never blind --theirs. Test:
  test_parallel_generation.py::TestGenerateGroup +1 (max_seq_len=8, over-length
  prompt runs cleanly); 85 inference/generator tests + ruff green.
- **EVAL-021** [eval/correctness, high] `done` (2026-06-13, fresh evaluation-path
  audit) — `CompletionBenchmark.score_single` (completion_benchmark.py) matched
  required/forbidden regexes against `prefix + completion`, so any required
  pattern ALREADY present in the prefix counted as satisfied regardless of the
  model's actual output — a false-PASS that inflated the prefix-completion
  benchmark (a prefix-echoing base model "passed" everything; the old
  `test_run_all_fail` even documented the symptom with a loose `<0.5` assertion).
  FIX: score the `completion` only (the benchmark's stated purpose — grading the
  continuation). Built-in problems keep required patterns out of the prefix so
  legit completions are unaffected; tightened `test_run_all_fail` to a precise
  `<0.1`. Tests: test_completion_benchmark.py +2 prefix-leak regressions (required
  + forbidden); 22 + broader eval (pass@k etc.) green. (metrics.py pass@k
  estimator and runner sandbox were audited and found sound — no change.)
- **DATA-049** [data-quality/correctness, medium] `done` (2026-06-13, fresh
  data-sources audit) — `SWHClient.get_content_raw` (software_heritage.py)
  returned `requests` `Response.text`, which decodes leniently and NEVER raises on
  binary input — contradicting its own docstring and silently defeating the
  `except UnicodeDecodeError: continue` binary-skip in `_walk_directory`. Binary
  blobs (reachable in the `content_types=None` "all text files" mode) entered the
  corpus as mojibake. FIX: decode `resp.content` as strict UTF-8 so binary raises
  and is skipped as intended. Tests: test_swh.py +3 (binary rejected, valid UTF-8
  decodes, stream skips binary); 38 SWH/source/security tests green. Note:
  proposed-id collision (agent said DATA-041, which exists) → assigned DATA-049.
- **TOK-002** [tokenizer/correctness, medium] `done` (2026-06-13, de-collided from
  a provisional "TOK-001" label — TOK-001 was taken by the 2026-06-11 entry)
  — `CodeTokenizer.encode_fim`/`fim_prompt` assumed `<|fim_*|>` tokens
  exist, though the constructor treats them as OPTIONAL (only pad/bos/eos/unk
  required). A tokenizer trained without FIM tokens left the FIM ids `None`, so
  `encode_fim` emitted `[None] + prefix + [None] + suffix + [None]` — an invalid
  id silently corrupting the stream, surfacing only as a downstream model
  crash/garbage (contradicting the constructor's own "checks separately" comment).
  FIX: added `has_fim_tokens()` predicate + `_require_fim_tokens()` guard raising a
  clear `ValueError`; both helpers call it first. Tokenizers WITH FIM tokens
  (every currently-trained one incl. the live run's) are unaffected — no numerics
  change, safe for the live run. Tests: test_tokenizer_fim_missing.py +7; 17
  tokenizer/fim tests green.
- **TOOL-018** [tooling/bug, low] `done` (2026-06-13, from the TOK-002 audit) —
  `scripts/tokenizer_health.py` checked for special tokens named `<pad>`/`<unk>`/
  `<bos>`/`<eos>` (no pipes), but cola-coder tokenizers use `<|pad|>` etc., so the
  "Special tokens" check falsely reported all four MISSING on valid tokenizers.
  FIX: derive the required/optional token lists from the canonical `SPECIAL_TOKENS`
  in `tokenizer/train_tokenizer.py` (the same source CodeTokenizer validates
  against) — required = the 4 piped core tokens, optional = FIM + think tokens —
  so it can't false-alarm or drift if tokens are renamed. Tests: new
  test_tokenizer_health.py +4 (imports; piped form; passes on real tokenizer
  fixture; matches canonical core); green.
- **INFER-022** [inference/correctness, high] `done` (2026-06-13, de-collided from
  a provisional "INFER-013" label — INFER-013 was taken by the 2026-06-12 entry;
  fixed by INFER-023 follow-up for the GRPO path) — `CodeGenerator.generate`/`generate_stream` ignored
  `config.max_seq_len`, but the KV-cache + causal mask are sized for exactly
  `max_seq_len`. Two failures, both reachable from `/v1/fim` (a moderately long
  file's prefix+suffix easily exceeds seq_len): (1) prompt longer than seq_len →
  prefill `cache_k[:, 0:seq_len] = k` raises a cryptic tensor-size RuntimeError
  (500 on /v1/fim and /v1/completions); (2) generation crossing the window
  mid-decode → write to a ZERO-SIZE slice is a silent no-op, so the new token's
  K/V is dropped and the model reads stale cache → garbage output, no error (the
  sneakier one). FIX: `_fit_context_window(token_ids, max_new_tokens,
  max_seq_len)` left-truncates the prompt to the most recent `max_seq_len-1`
  tokens (sliding window) and caps `max_new_tokens` to remaining slots; wired
  into both methods via `_max_seq_len()` (returns 0 to disable when a stub model
  lacks config, so no regression). Tests: test_generator_context_window.py (9 —
  helper truncation/capping/exact-fit/disabled + 4 integration incl. real
  16-slot Transformer that previously crashed); 87 inference tests + ruff green
  on main.
- **DATA-050** [data-quality/security, high] `done` (2026-06-13, de-collided from
  a provisional "DATA-047" label — DATA-047 was taken by the 2026-06-12 entry) —
  `CredentialScanner.process("strip")` leaked private-key /
  certificate BODIES. The `Private Key` / `Certificate` regexes matched only the
  single `-----BEGIN ... -----` header line, so strip mode (used by
  `data/scorers/llm_judge.py` to scrub secrets out of untrusted scraped code
  BEFORE sending it to an external LLM-judge API) replaced just the header and
  forwarded every byte of key material. FIX: patterns now span the whole
  `BEGIN..END` PEM block (`[\s\S]*?`, END optional so truncated header-only keys
  still redact); per-line `scan()` detection and `reject`-mode behavior unchanged;
  added PGP to recognized key types. Tests: test_credential_scanner.py +3
  (full key body, truncated key, full cert body); 64 scorer/filter tests + ruff
  green on main. Relates to the "treat scraped code as untrusted / scan secrets
  before API calls" rules.
- **DATA-051** [data-quality/cleanup, low] `done` (2026-06-13, de-collided from a
  provisional "DATA-048" label — DATA-048 was taken) — three follow-up cleanups
  from the DATA-050 audit, all landed: (a) `credential_scanner.py` `scan()` now
  scans the whole text like strip-mode `process()` (line numbers derived from
  match offset), so detection and redaction coverage can't diverge for a future
  multiline pattern; (b) the GitHub fine-grained PAT regex was harmonized —
  `quality_filter.py` now uses the strict `github_pat_[A-Za-z0-9_]{22}_[A-Za-z0-9]{59}`
  shape matching `credential_scanner.py`, with cross-reference comments; (c) the
  dead `if not lines` branch in `check_avg_line_length` (`"".split("\n")`→`[""]`,
  never empty) was removed. Tests: parity/PAT/empty-content cases across
  test_credential_scanner.py + test_quality_filter.py; 74 green; DATA-050's
  PEM-strip fix verified intact.
- **BUG-120** [inference/correctness, medium] `done` (2026-06-13, found during a
  fresh inference-path scan) — `sample_next_token` (sampling.py) could emit a
  garbage token when `no_repeat_ngram_size > 0` banned the ENTIRE vocabulary: the
  greedy path (`temperature==0`) returns `logits.argmax()`, and argmax of an
  all-`-inf` tensor is index 0 (an arbitrary `<unk>`/`<pad>`-class token); the
  sampling path was protected by its `fallback_token`, but greedy was not. Common
  trigger: deterministic completion (temp 0) + no-repeat-ngram on a short/looping
  context where every candidate completes a seen n-gram. FIX: a ban now never
  masks the whole vocab — `if 0 < len(banned) < logits.numel()` — so when banning
  would leave nothing, the ban is skipped and the least-bad real token is kept
  (strictly better than emitting <unk>). Fixes both greedy and sampling paths at
  the source. Tests: test_inference.py::TestNoRepeatNgram +3 (greedy-all-banned →
  real max not 0, sampling-all-banned → in-range token, partial-ban-still-applies);
  26 inference tests + ruff green. Wiring verified end-to-end: server (all
  request schemas) → generator (all 4 gen paths) → sampling all forward min_p /
  repetition_penalty / no_repeat_ngram_size; only the best-of-N path omits
  repetition_penalty by design (no per-candidate history).
- **BUG-119** [training/robustness, high] `done` (2026-06-13) — `torch.compile`
  crashed training at step 0 instead of falling back to eager when its backend
  (Triton) is missing — common on Windows. `Trainer.__init__` called
  `torch.compile(model)` inside a try/except, BUT compilation is LAZY: the
  `InductorError: TritonMissing` fired on the FIRST forward pass, past that
  try/except, killing the run. FIX: added `_torch_compile_backend_ready(device)`
  in trainer.py — probes `importlib.util.find_spec("triton")` (no side-effecting
  import) up front and skips compile with a clear "training in eager mode" log
  line when Triton is absent, instead of crashing at step 0. The
  `COLA_NO_COMPILE=1` env override still works (and stays set by
  ps/cola-train-resume.ps1 as belt-and-braces). Tests:
  tests/test_compile_backend_probe.py (4) — cpu→False, triton-missing→False,
  triton-present→True, find_spec-raises→False. ruff + test_checkpoint green.
- **BUG-118** [training/robustness, high] `done` (2026-06-13) —
  `train.py --auto-resume` resumed from an architecturally-INCOMPATIBLE checkpoint
  (the `small_react_best` qk_norm config latched onto another run's non-qk_norm
  `checkpoints/small/step_00100000` and crashed in `_load_state_dict_tied`). FIX:
  new `find_resume_checkpoint(output_dir, model_config)` in checkpoint.py scans
  ONLY the config's own `checkpoint.output_dir` (resolving by highest `step_*`
  dir, never the stale `latest` pointer) and validates the candidate's saved
  architecture via `architecture_mismatch()` (dim, n_layers, n_heads, n_kv_heads,
  vocab_size, qk_norm, moe.enabled) — refusing with a clear `cli.warn` and
  starting fresh instead of crashing. `scripts/train.py --auto-resume` now calls
  it. `detect_latest_checkpoint` is unchanged (inference/eval callers legitimately
  scan the whole `checkpoints/` tree). Tests: 6 new in
  test_checkpoint.py::TestFindResumeCheckpoint (own-dir scoping, qk_norm-mismatch
  refusal, match-resumes, no-config-skips, missing-dir, core-field mismatch);
  test_checkpoint + test_training_resume (50) + ruff green on main. The
  ps/cola-train-resume.ps1 wrapper remains as a belt-and-braces reboot helper.

### USER REQUEST 2026-06-13 — full PowerShell usability + menu testing
Goal: every project workflow runnable from PowerShell with NO errors, and every
menu entry verified to work. Scheduled agent: pick ONE TOOL-015* sub-item per
cycle, build/extend the harness, fix any runtime errors found (log each as its
own BUG-###), mark the sub-item done. These are tractable one-per-cycle.

- **TOOL-015a** [tooling/test, high] `done` (2026-06-13) — Built the
  non-interactive menu-entry smoke harness (tests/test_menu_entries_smoke.py): a
  `stubbed` fixture monkeypatches every interactive prompt to cancel/empty
  (`cli.choose`→None via a `_Canceler` that fails fast on infinite loops,
  `confirm`→False, `multi_select`/`pick_languages`/`weight_editor`→empty) and
  stubs `MasterMenu._run_script`/`_pause` + `subprocess.run` so nothing trains,
  downloads, or shells out. Parametrized over every top-level entry across all 5
  sub-menus (DataMenu/TrainingMenu/EvalMenu/ToolsMenu{menu,settings,training_status}/
  PipelineMenu) asserting each OPENS and EXITS cleanly with no exception — catches
  the BUG-117 render-crash class tree-wide. 8 tests pass; ruff + checkpoint green.
  Follow-up TOOL-015b/c/d drive each individual leaf option through its handler.
- **TOOL-015b** [tooling/test, high] `done` (2026-06-13) — Extended the TOOL-015a
  harness to leaf-level coverage of the DataMenu. All 23 leaf handlers across the
  Collect/Modify/Score/Inspect/Prepare groups are invoked directly with every
  prompt cancelled/defaulted via a new `stubbed_leaf` fixture (adds an
  `input()`→EOFError stub on top of the existing `stubbed` fixture — bare
  `input()` raises `OSError` under pytest's captured stdin, not the
  EOFError/KeyboardInterrupt handlers catch as "cancel"), asserting each reaches
  its handler and returns without raising — catching BUG-117 /
  NoConsoleScreenBufferError-class leaf crashes the menu-open test can't reach.
  Added `test_data_menu_leaf_runs_cleanly` (parametrized over all 23) +
  `test_data_menu_group_handlers_exist` (guards leaf-list drift). No leaf bugs
  found; suite 9→32 tests, all green on main, ruff clean. Inline `_run_script`
  dispatch arms (no dedicated handler) stay covered by the menu-open test.
- **TOOL-015c** [tooling/test, high] `done` (2026-06-13) — Added
  `test_training_menu_leaf_runs_cleanly` parametrized over 27 TrainingMenu leaf
  handlers across all 6 groups (Pipeline Manager incl. `_full_pipeline_menu`,
  Foundation, Pre-Training incl. all 5 background-training actions, Post-Training
  incl. MoE upcycle + MoE fine-tune 7.5, Alignment & Reasoning, Monitoring),
  reusing `stubbed_leaf` + a `test_training_menu_group_handlers_exist` drift
  guard. Every leaf cancels cleanly with prompts stubbed and no scripts run; no
  wiring/render bugs surfaced.
- **TOOL-015d** [tooling/test, high] `done` (2026-06-13) — Added
  `test_eval_menu_leaf_runs_cleanly` (26 EvalMenu leaves) and
  `test_pipeline_menu_leaf_runs_cleanly` (8 PipelineMenu leaves), each with a
  `*_group_handlers_exist` drift guard. A dedicated `stubbed_pipeline_leaf`
  fixture stubs `PipelineMenu._run_stage_script` (which RAISES on non-zero exit)
  on top of the global `subprocess.run` stub, so no pipeline stage can
  train/collect even if a leaf reaches dispatch. No leaf bugs; missing
  checkpoints/data warn rather than crash. Full smoke suite now 96 tests, green
  on main, ruff clean. (ToolsMenu leaves left as optional follow-up — its
  top-level entries are already covered by TOOL-015a.)
- **TOOL-016** [tooling/UX, medium] `done` (2026-06-13) — PowerShell coverage:
  added 7 wrappers so every common workflow has a direct shortcut (was 23 → 30):
  cola-router (train_router), cola-vram (vram_estimate), cola-quality
  (quality_report), cola-safety (safety_eval), cola-eval-suite (run_eval_suite),
  cola-find-lr (find_lr), cola-compare (compare_models). All pass-throughs
  (`@args`), `Split-Path -Parent $PSScriptRoot` pattern, ps/README.md updated, all
  30 parse clean. Remaining un-wrapped (`upcycle_to_moe`) is niche and reachable
  via cola-menu / cola-pipeline (stage 7) — not worth a dedicated wrapper.
- **TOOL-017** [tooling/UX, low] `done` (2026-06-13) — Every `ps/cola-*.ps1`
  wrapper that invokes the venv interpreter now fails fast with the friendly
  message "venv missing — run cola-setup.ps1 first." (red, exit 1) when
  `.venv\Scripts\python.exe` is absent, instead of a cryptic downstream error.
  The guard mirrors the `cola-train-resume.ps1` reference pattern (resolve `$py`
  from `$project`, `Test-Path` check) and was added to 29 scripts.
  `cola-setup.ps1` is exempt (it CREATES the venv); `cola-train-resume.ps1`
  already had it. All 31 `ps/*.ps1` parse clean (verified via the PS
  `Parser::ParseFile` AST check). Used the inline guard rather than a dot-sourced
  `ps/_common.ps1` — 2 lines per script is simpler and keeps each wrapper
  self-contained/copy-pastable (no load-order/relative-path coupling).

- **MODEL-012** [model/config-consistency, low] `open` (user decision — changes
  training behavior/repro) — The 2025-26 stabilizers `qk_norm: true` and
  `z_loss: 1.0e-4` are set ONLY in `configs/4080_max.yaml`; tiny/small/medium/large
  leave both at their defaults (off). Consequences: (a) smaller configs miss
  cheap, broadly-beneficial stabilization (QK-Norm bounds attention logits; z-loss
  prevents bf16 logit drift — both PaLM/OLMo2/Gemma2/Qwen3 standard); (b) smoke/CI
  runs on tiny/small never exercise these code paths, so a regression there would
  pass small-config tests. NOT auto-fixable: enabling `qk_norm` adds params
  (`blocks.N.attn.{q,k}_norm.weight`) → existing tiny/small/medium checkpoints
  could no longer resume (hard `_load_state_dict_tied` failure). `z_loss` is
  backward-compatible (pure loss term, resumes fine) so could be added safely, but
  it still shifts the user's tuned loss curves. Defer to user: enable z-loss
  across configs? enable qk_norm only for fresh runs? Verified wiring is correct
  either way (config→Attention / config→loss), so this is a tuning choice, not a bug.

- **TOOL-011** [tooling/pipeline, low-medium] `open` (deferred — design decision)
  — Remaining `full_pipeline.py` divergence after TOOL-012: Stage 1
  `_stage_collect_data` runs `prepare_data.py` (code-only), NOT `collect_data.py
  --sources code,text,math`, so the runner SKIPS multi-source collection (the
  70/20/10 code/text/math mix incl. DATA-026/027/028), and Stage 2 then runs
  `prepare_data.py` AGAIN (redundant; the stage-2 run drops Stage 1's `--score`
  weights). Fixing this is a design decision (the collect-vs-prepare 2-stage
  model is ambiguous: collect_data.py already tokenizes+mixes, so running both
  double-produces .npy and training scans/picks one). The Pipeline Manager makes
  Stage 1 an interactive CHOICE (auto multi-source vs code-only) — full_pipeline
  needs an equivalent non-interactive policy (e.g. multi-source if
  data_sources.yaml enables text/math, else code-only, and drop the redundant
  Stage 2). Defer to user steer on the intended model.

- **DATA-025** [data-quality/dead-code, low] `open` (integrate or remove — user
  decision) — `data/fim_dataset.py` (`FIMDataset` + `create_fim_dataloader`) is a
  redundant SECOND dynamic-FIM implementation, not wired into any live path. The
  trainer uses DATA-012's `create_dataloader(fim_rate=...)` + `FIMTrainingCollator`
  (the weight-preserving, tested, collator-based path); `FIMDataset` does the same
  thing via a per-`__getitem__` dataset wrapper. Only `prepare_fim_data.py`
  DOCSTRINGS mention it (no actual call). It IS tested in isolation, so unlike
  MixedDataset (DATA-002, deleted) I did NOT unilaterally remove it. Same
  "integrate or remove" pattern as DATA-021. The integrate-or-remove DECISION is
  still open (user). Its two doc sub-items are DONE (2026-06-12, see DATA-035):
  FIMDataset now documents that `create_dataloader(fim_rate=...)` is the canonical
  trainer path (double-FIM warning), and fim.py's `truncate_or_pad=False` length
  docstring is corrected.

- **DATA-021** [data-quality/dead-code, low] `open` (urgency DOWNGRADED 2026-06-12)
  — The modular FilterPlugin + `DataPipeline` system (data/pipeline.py +
  data/filters/, registry in data/registry.py) is built and registered (DATA-020)
  but NOT wired into any LIVE entry point — `DataPipeline(` is constructed nowhere
  outside pipeline.py itself. ORIGINALLY framed as a data-quality exposure, but the
  DATA-042 scan established the practical risk is LOW: the live sources are already
  covered — the-stack-v2 (primary code) is license-filtered + PII-redacted upstream
  by BigCode; the GitHub scraper has the DATA-037 copyleft gate; collect_data now
  quality-filters code (DATA-042); secrets are scanned (DATA-033). So the unwired
  pii/license/content/syntax/dedup plugins are largely REDUNDANT with the live
  path, making this dead-code cleanup, not exposure. DECISION (user): either (a)
  route collect_data/prepare_data through DataPipeline (consolidate on the modular
  system), or (b) delete the unused DataPipeline filter path and keep
  quality_filter.py. Pick one architecture — don't leave both.

- **EVAL-007** [eval/capability, low] `open` — Follow-up to EVAL-006: to actually
  EVALUATE TypeScript HumanEval problems (TYPESCRIPT_PROBLEMS, ~40 in
  humaneval.py) rather than just guard them out, the pipeline needs (1) a TS
  `extract_function` (the current one keys on `def {entry_point}` and returns ""
  for TS) and (2) sandboxed TS execution of runnable `test_code` (assertions),
  not just tsc compile-check (ts_benchmark only type-checks TSProblem). Needs
  Node/ts-node in a sandbox (untrusted-code rules apply). Medium effort, defer
  until a TS-capable model exists to evaluate.


- **DATA-006** [data-quality, low] `open` — Follow-up to DATA-002: if dynamic
  per-batch / online source reweighting is wanted, design runtime data mixing
  into the trainer DELIBERATELY (multi-source dataloader, per-source loss
  tracking → inverse-loss reweighting). The orphaned `MixedDataset` that
  half-implemented this was removed; `data/mixing.py` already has the
  reweighting math (`MixingOptimizer`). The current pipeline does
  collection-time mixing via collect_data, which fits the single-.npy trainer.





- **MODEL-008** [model/training, low] `open` (part (a) done; (b) remains) —
  Follow-up to MODEL-007. (a) DONE (2026-06-12): `grpo_clipped_surrogate` gained a
  `length_norm` arg; GRPOTrainer `length_norm` config ("sum" legacy default vs
  "constant" = Dr. GRPO ÷ max_new_tokens, removing the length bias of the bare
  sum). Wired through reasoning.yaml + train_reasoning.py (+ wiring test). Default
  "sum" preserves behavior. Tests: test_grpo_token_surrogate.py +6 (sum vs
  constant, falsy=0 no-op, uniform-scaling, real trainer divisor resolution).
  (b) STILL OPEN: the PPO clip path (ppo_epochs>1) and length_norm="constant" are
  unit-tested but never validated in an actual short GRPO run — do a tiny
  sandboxed smoke (few steps, ppo_epochs=2, both length_norm modes) to confirm the
  ratio diverges from 1, the clip engages, and the constant-norm magnitude/LR
  interaction is sane. Needs a real/CPU model run; defer until convenient.

- **EXPORT-011** [export/bug, medium] `done` (2026-06-13) — CONFIRMED REAL and
  fixed. `_write_gguf_with_package` (gguf_export.py) passed raw F32 numpy arrays
  to `writer.add_tensor(name, data, raw_dtype=quant_type)`. `raw_dtype` only TAGS
  pre-packed bytes — the gguf python lib does NOT quantize — so F32 bytes tagged
  Q8_0/Q4_K/Q5_K (and even F16) yield a corrupt GGUF (declared type's byte count
  ≠ actual F32 length) on any host with the `gguf` package installed; only `f32`
  was accidentally correct. FIX: route the package path through the existing
  `_quantize_tensor` helper so the emitted bytes' dtype matches the declared
  `raw_dtype`, and raise a clear `NotImplementedError` (→ llama.cpp
  `llama-quantize`) for true K-quants this module can't pack — instead of silently
  emitting a corrupt/mislabeled file. Corrected the misleading docstrings that
  claimed the package path "handles this correctly." Tests: test_export.py +4
  (mocked GGUFWriter: data dtype matches raw_dtype for f32/f16/q8_0; K-quant
  raises + closes writer); 100 export tests + ruff green. Relates to EXPORT-010.

- **OPS-001** [tooling, low] `open` (deferred for user) — storage split-brain:
  configs/storage.yaml → E:/cola-coder-data vs config.checkpoint.output_dir →
  ./checkpoints. Needs the user's decision; do not unilaterally resolve.

---

## Done

- **DATA-046** [data-quality/robustness, low] `done` (2026-06-13) — `dedup.py`
  `_tokens_to_ngrams` returned an EMPTY n-gram list when a chunk's content was
  shorter than `ngram_size`, so `_make_minhash` produced an empty signature; in
  `deduplicate_self_array` all empty signatures collide (Jaccard treated as a
  match), dropping distinct short chunks as false near-dups. Rarely hit (chunks
  are normally fixed seq_len ≫ ngram_size) but a real silent-data-loss path for
  short/padded-tail chunks. Fix: when the sliding window yields no grams but the
  content is non-empty, fall back to the WHOLE content as a single gram —
  distinct short chunks now get distinct signatures (identical ones still
  collapse, which is correct); truly empty content → empty list. One-line change
  in `_tokens_to_ngrams`, so it benefits every `_make_minhash` caller (self-dedup,
  build_index, find_duplicates). Fresh-scan note: audited the SFT path
  (sft_dataset label masking + train_sft loss shift `logits[:-1]`/`labels[1:]` +
  right-pad collator) — all correct, no issue. Tests: 4 new in test_dedup.py
  (short distinct chunks survive; short identical still dedup; ngrams non-empty
  for short input, empty for empty). 23 dedup + checkpoint green; ruff clean.

- **DATA-048** [tooling/staleness, low] `done` (2026-06-13) — Finished the
  per-dataset-dir migration BUG-117 started. Three runtime sites still resolved
  data via the legacy `data_dir/processed/`, so on a per-dataset-dir setup they
  failed or scanned the wrong place: `master_menu.py:1254` (router-data
  `--source` hardcoded to `data/processed/train_data.npy` — generate_router_data
  would error on the missing file), and `data_menu.py` `_combine_datasets` +
  `_inspect_dataset` (both reported "No datasets found" with data present). DRY
  fix: added `DatasetResolver.find_dataset_npys()` — ONE source of truth
  (per-dataset dir → legacy `processed/` fallback, `.weights`/`.scores` sidecars
  excluded, sorted) — and routed all four sites through it (incl. refactoring
  BUG-117's inline scan). Router source now uses the first real dataset (warns +
  aborts if none). Also updated 2 stale docstring examples
  (crash_recovery/data_quality_report). Tests: 3 new in test_dataset_resolver.py
  (per-dataset detection w/ sidecars excluded; empty → []; legacy fallback). 72
  menu/data/resolver + checkpoint tests pass; ruff clean.

- **BUG-117** [tooling/UX, high] `done` (2026-06-13) — The master-menu Quick Start
  reported `data: missing` even with a fully-prepared 13.78 GB `code_data.npy`
  present, so it would re-run collection/prep and re-download gigabytes. Root
  cause: `_detect_pipeline_status` (master_menu.py:333) scanned the LEGACY
  `data_dir/processed/*.npy`, but datasets actually live in the per-dataset dir
  (`data/<dataset-name>/`) — the SAME resolver the tokenizer check already used
  (line 328), so tokenizer showed "ready" while data showed "missing". It also
  counted `.weights`/`.scores` SIDECARS as extra "datasets". Fix: scan
  `DatasetResolver.get_dataset_dir()` first (excluding sidecars), fall back to the
  legacy `processed/` dir for older setups. Keeps the "N dataset(s)" string the
  Quick Start gating (`"dataset" in status["data"]`) depends on. Follow-up
  (DATA-048): a few stale `data/processed/train_data.npy` DEFAULT paths remain in
  crash_recovery.py / data_quality_report.py / router data source — low impact
  (overridable suggestions, not detection). Tests: 2 in test_menu_smoke.py
  (resolver-dir dataset detected as "1 dataset(s)" w/ sidecars excluded; empty →
  "missing"). 16 menu-smoke + checkpoint green; ruff clean.

- **BUG-116** [tooling/Windows, high] `done` (2026-06-12) — Interactive CLI prompts
  CRASHED the whole pipeline on Windows when run non-interactively. The user's Full
  Pipeline run died at prepare_data's "overwrite existing data?" chooser with
  `prompt_toolkit ... NoConsoleScreenBufferError: No Windows console found` —
  `cli.choose` calls `questionary.select(...).ask()`, whose console error escaped
  the `except ImportError` (only missing-questionary was caught), and the
  numbered-menu fallback's `input()` would have `EOFError`-looped forever anyway.
  Hits any script run as a pipeline subprocess / with redirected I/O / in a
  terminal prompt_toolkit can't drive. Fix (cli.py): both `choose` and `confirm`
  now catch the broad console error and fall through; `choose` gained a `default`
  index and returns it (instead of crashing/looping) when stdin is unavailable
  (`EOFError`); `prepare_data._resolve_output` passes `default=0` ("create new",
  non-destructive) so automated runs proceed (pass `--output-name` to control).
  Tests: 7 in test_cli_non_interactive.py (choose/confirm degrade to default; the
  exact `_resolve_output` overwrite-chooser path no longer raises). Also (user ask)
  brought the new `ps\` PowerShell scripts up to date: fixed the stale
  `$PSScriptRoot\cola-coder` path (now `Split-Path -Parent $PSScriptRoot`),
  dropped the obsolete hardcoded `tokenizer.json` (auto-resolves), added 7 scripts
  for uncovered workflows (full pipeline, collect, sft, smoke, chat, env-check,
  export) + a ps\README.md index — 23 scripts, all parse-clean; cola-env-check &
  cola-lint verified end-to-end. 120 menu/pipeline/cli + checkpoint tests pass.

- **ROUTER-001** [model/router quality, medium] `done` (2026-06-12) — Both routers
  (`MLPRouter`, `TransformerRouter` in features/router_model.py) mean-pooled over
  ALL positions including padding. `train_router.py` pads short snippets to
  max_seq_len with pad_id=0 (line 84-86), so a 20-token snippet padded to 256 had
  its domain signal diluted ~12x by pad embeddings — AND inference (`route()`,
  line 214) does NOT pad, so the model was trained on pad-diluted means but
  evaluated on clean ones (a train/inference mismatch hurting accuracy). The
  TransformerRouter additionally ATTENDED to pad tokens (no key-padding mask).
  Fix: added `RouterConfig.pad_id` (default 0); both forwards now masked-mean-pool
  over real tokens only, and the transformer passes `src_key_padding_mask` (with
  an all-pad-row guard that keeps position 0 so the attention softmax can't NaN on
  degenerate empty input). Architecture/params unchanged → existing router
  checkpoints still load; unpadded inference is identical, so no regression — but
  routers should be RETRAINED to gain the now-consistent pooling (training was
  previously diluted). Fresh-scan note: the "MixedDataset dead code" known thread
  is ALREADY resolved (DATA-002, removed 2026-06-11). Tests: 5 new in
  test_router_pooling.py (MLP + Transformer: appending padding doesn't change
  logits; all-pad input stays finite; batched padded row matches the single run).
  22 router + checkpoint tests pass; ruff clean.

- **MODEL-014** [model/pipeline feature, medium] `done` (2026-06-12) — Wired the
  MoE expert-differentiation fine-tune into the automated pipeline (the known
  "MoE fine-tune pipeline stage" thread). Stage 7 (`_stage_upcycle_moe`) used to
  ONLY upcycle (`upcycle_to_moe.py` → checkpoints/moe) then jump to the router —
  but upcycling clones the dense FFN into every expert IDENTICALLY, so the router
  consumed undifferentiated experts and the optional MoE stage was effectively
  wasted (a MoE no better than the dense model, just bigger/slower). Stage 7 now
  also runs the MODEL-003 differentiation recipe: derive a fine-tune config
  (`derive_moe_finetune_config`, default 10% LR / 15% steps), write it to
  configs/auto/, and `train.py --config <derived> --resume checkpoints/moe --data
  <npy>` (trainer auto-detects MoE). The derived config inherits `run.config_path`
  so smoke runs inherit smoke step limits (15% of ~30 ≈ 5 steps) and the isolated
  `_smoke` dir; the fine-tuned `<base>_moe_ft/latest` becomes the stage artifact
  the router consumes. Graceful skips: no checkpoints/moe or unreadable config →
  warn + return the upcycle/base checkpoint. STAGE_DEFS[7] description updated.
  Builds on MODEL-013 (output-dir isolation) so the fine-tune can't clobber the
  dense base. 282 pipeline/menu/moe tests pass; ruff clean.

- **MODEL-013** [model/training bug, high] `done` (2026-06-12) — `derive_moe_finetune_config`
  rescaled only the `training` section, leaving `checkpoint.output_dir` pointing
  at the BASE config's dir. The MoE fine-tune RESUMES from the upcycled MoE dir
  (checkpoints/moe) but the trainer always saves to `config.checkpoint.output_dir`
  — so the fine-tune would have OVERWRITTEN the dense pretrained checkpoint (e.g.
  checkpoints/4080_max), irreversibly destroying the base model (days of GPU), and
  mixed MoE step dirs into the dense folder where `_cleanup_old_checkpoints` could
  prune the base. Hit BOTH the existing training-menu stage 7.5 AND the new
  pipeline wiring (MODEL-014). Fix: the derived config now redirects
  `checkpoint.output_dir` to an isolated `<base>_moe_ft` dir (handles trailing
  slash; defaults to `./checkpoints/model_moe_ft` when no checkpoint section), so
  dense base / upcycle source / fine-tune output are three distinct resumable
  checkpoints. Tests: 5 new in test_moe_finetune_config.py (redirect, differs from
  base, trailing-slash, default, input-not-mutated). 14 config + checkpoint green.

- **DATA-044** [data-quality, medium] `done` (2026-06-12) — CLOSED the last
  collect_data-vs-prepare_data parity gap: the multi-source mix can now train
  quality-WEIGHTED. Added `--score` to `collect_data.py` — after each source is
  tokenized + deduped, the CODE source's chunks are scored into an aligned
  `code_data.weights.npy` (scoring runs AFTER dedup so weights line up with the
  surviving chunks), and `carry_weights=True` is passed to `combiner.combine(...)`
  so DATA-047's weight-carrying produces an aligned `mixed_train_data.weights.npy`.
  Only the code source is scored (the `code_scorer` judges CODE quality; running
  it on prose/math would mis-weight them) — text/math carry neutral weight 1.0 via
  the combiner's missing-sidecar fallback. DRY: extracted the decode→score→weight
  loop prepare_data --score used into a shared `data/weight_scoring.py`
  (`compute_chunk_weights` + `score_npy_to_weights`, feature-gate-aware) and
  refactored prepare_data to call it, so both pipelines share ONE implementation /
  identical weight semantics. Menu wired: the Mixed Data Collection menu now
  prompts "Score code quality for weighted training? (--score)". Tests: 3 new in
  test_weight_scoring.py (1:1 alignment; sidecar written + matches; feature-off →
  (None,None), no sidecar). 113 menu/combine/scoring/quality tests + checkpoint
  green; ruff clean. (Note: collect_data scores only code — extending quality
  signals to text/math would need a prose/math scorer; out of scope.)

- **DATA-047** [data-quality/feature, medium] `done` (2026-06-12) — Weight-aware
  `DatasetCombiner` (the DATA-044 foundation + an immediate fix on its own).
  Combining datasets via `combine_datasets.py` SILENTLY DROPPED per-chunk quality
  weights: a user who ran `prepare_data --score` on two language datasets, then
  merged them, lost every `.weights.npy` — the merged set trained unweighted.
  Fix: `combine(..., carry_weights=True)` loads each source's `.weights.npy`
  sidecar (auto-detected via the prepare_data `<stem>.weights.npy` convention, or
  explicit `DatasetInput.weights_path`), and all three strategies (concat /
  interleave / weighted) now fill an output-weight array in LOCKSTEP with the
  chunk array — weight placed at the same output index as its chunk, so it can't
  drift — then the SAME shuffle permutation is applied to both, and an aligned
  `<output stem>.weights.npy` is written (`CombineResult.weights_path`). Safe
  fallbacks: a source missing its sidecar → neutral weight 1.0; a length-mismatched
  sidecar → ignored with a warning; NO sidecars at all → no weight file written.
  Wired into `combine_datasets.py` (carry_weights enabled when cross-dataset dedup
  did NOT run — dedup's `_temp_dedup_` row changes would desync sidecars; realigning
  weights through `deduplicate_pair` is left to a follow-up). Tests: 7 new in
  test_combine.py (alignment holds through every strategy + shuffle via
  constant-valued chunks; missing-sidecar neutral; no-sidecar skip; off-by-default;
  max_chunks trims weights in lockstep). 36 combine + 77 combine/script/checkpoint
  pass; ruff clean. Follow-up for full DATA-044: collect_data per-source scoring.

- **DATA-045** [data-quality/security, medium] `done` (2026-06-12) — The secret
  gates (`check_no_obvious_secrets`, conservative; `check_no_hardcoded_secrets`,
  strict) only scanned `content[:5000]` ("secrets are usually near the top"). But
  credentials live in file BODIES too — AKIA ids in deployment scripts, PEM blocks
  appended to a file, a key in a config dict at line 300 — and those flowed
  straight into the training set, a real exfiltration/memorization risk (DATA-033
  lineage). Fix: scan the whole file up to a new `_SECRET_SCAN_LIMIT` (1 MB). All
  patterns are linear-time (literal prefixes + bounded char classes, no
  catastrophic backtracking), verified before widening, so the full scan is
  O(n)-safe; the 1 MB cap only bounds worst-case cost on pathological multi-MB
  inputs (which length/size gates usually reject first). Fresh-scan note: the
  "within-file MinHash dedup" known thread is ALREADY IMPLEMENTED
  (`deduplicate_self_array` + `dedup_npy_file(mode="minhash")`, wired to
  `--dedup minhash` in collect_data/prepare_data) — verified correct (greedy
  keep-first LSH); do not re-investigate. Tests: 3 new (high-conf secret past
  5000 chars caught; heuristic secret past 5000 caught in strict; 1.2 MB clean
  file scans without crash/FP). 35 quality_filter + 91 checkpoint/security/filter
  pass; ruff clean.

- **INFER-018** [inference/partial-wiring, low-medium] `done` (2026-06-12) —
  Best-of-N generation silently DROPPED the `no_repeat_ngram_size` sampling
  param. The server's `_best_of_generate` → `generate_best_of_n` →
  `_generate_candidates` chain threaded `min_p` but not `no_repeat_ngram_size`,
  so a user who set it on a chat/completions request with `best_of > 1` got
  verbatim-repetition suppression silently ignored (every other generate path —
  regular, streaming, FIM — honors it). Root cause: the batched sampler
  (`sample_next_tokens_batch`) deliberately skips per-sequence n-gram history for
  throughput, so `generate_group` can't enforce it. Fix: thread the param through
  all three functions; when `no_repeat_ngram_size > 0`, `_generate_candidates`
  routes to the SERIAL `generate()` path (which applies the constraint) instead
  of silently dropping it — an explicit speed/correctness trade documented in the
  docstring (only triggered when the user opts into n-gram blocking). Fresh-scan
  note: verified the rest of server.py is consistently wired — all 4 request
  bodies (Generate/Chat/Completion/Fim) expose min_p + no_repeat_ngram_size and
  every non-best-of call site passes both; also re-audited GRPO (Dr. GRPO/DAPO
  per-token clip, std-collapse guard) — correct. Tests: 2 new in
  test_best_of_n.py (n-gram>0 forces serial + passes param; n-gram==0 keeps
  batched fast path). 20 best_of + 53 checkpoint/inference pass; ruff clean.

- **MODEL-011** [model/training metric, medium] `done` (2026-06-12) — Reported
  training loss conflated the optimization objective with the language-modeling
  METRIC. `trainer.py` logged `loss.item()` where `loss` = CE + z-loss (and + MoE
  aux), and `metrics.py:88` derives `perplexity = exp(avg_loss)` from it. So with
  z-loss enabled (4080_max.yaml, z_loss=1e-4) the displayed perplexity was
  INFLATED (z-loss term ≈ 0.02-0.09 on a ~1.5-2.0 loss → ~few-% ppl overestimate),
  the checkpoint `loss` metadata was inflated, early-stopping compared an inflated
  metric, and 4080_max's loss was incomparable to configs without z-loss. The
  z-loss/MoE-aux terms are REGULARIZERS, not LM costs. Fix: `language_modeling_loss`
  gains `return_components` → `(total_loss, ce_loss)`; trainer backprops `total`
  (CE+z+aux) but logs `ce_loss` (pure CE) via `step_loss += ce_loss.item()`.
  Validation path (trainer:629) already excluded z-loss — now training matches it.
  Fresh-scan note: verified the whole "2025-26 techniques" commit (c7d996d) —
  z-loss math, QK-Norm (per-head RMSNorm before RoPE, OLMo2/Gemma2 order),
  min-p filter, top-k/top-p single-vs-batch parity — all correct, wired
  (config→cfg→effect) and covered by test_modern_techniques.py. Tests: 2 new in
  TestZLoss (ce==plain-CE & total>ce with z on; total==ce with z off). Full module
  28 passed; training+resume+checkpoint green; ruff clean.

- **TOOL-014** [tooling/bug, low] `done` (2026-06-12) — `data_stats.py`
  `_estimate_unique_tokens` extrapolated a sample's distinct-token count by
  sqrt(data/sample) and capped at `2**20` (1,048,576). But .npy token data is
  uint16 → at most 65,536 distinct ids, so the diagnostic could report a
  physically IMPOSSIBLE unique-token count (more than the dtype can represent,
  let alone the ~32K vocab). Fixed: cap at the dtype's range
  (`2**(8*itemsize)` = 65536 for uint16) and never report below the count the
  sample actually observed. Tests: test_data_stats_unique.py (4): small=exact,
  uint8 ≤256, uint16 ≤65536, ≥ sample-observed. Fresh-scan note: verified
  prepare_data's dedup-BEFORE-score ordering is correct (weights align with the
  deduped data). data_stats --help OK; ruff + checkpoint green.

- **DATA-043** [data-quality/bug, medium] `done` (2026-06-12) — Parallel gap to
  DATA-042: `collect_data.py` (multi-source path) applied NO chunk dedup before
  tokenizing — it tokenized each source then combined via DatasetCombiner (which
  doesn't dedup either), so 25-40% DUPLICATE chunks (raw corpora are 25-40% dups,
  per the training rules' "exact SHA-256 chunk dedup, ON by default") flowed into
  the multi-source training set. prepare_data.py dedups by default; collect_data
  didn't — another divergence between the two Stage-1 paths. Fixed: new
  `_maybe_dedup(npy_path, mode, tokenizer)` runs `dedup_npy_file` in place after
  each source's tokenize_and_chunk; `--dedup {none,exact,minhash}` default "exact"
  (matches prepare_data + the ON-by-default rule). Applied to ALL 3 sources
  (exact chunk dedup is content-agnostic). Tests: test_collect_data_security.py +5
  (none=no-op, exact collapses identical chunks, keeps unique, wired into all 3
  sources, default=exact). collect_data --help OK; ruff + checkpoint green. With
  DATA-042 (filter) + DATA-043 (dedup), the multi-source code path now matches
  prepare_data's quality pipeline (filter → tokenize → dedup).

- **DATA-042** [data-quality/bug, medium] `done` (2026-06-12) — `collect_data.py`
  (the MULTI-SOURCE path — Full Auto Pipeline / Pipeline Manager Stage 1) tokenized
  the CODE source with NO quality filtering: `stream_code_data → _maybe_scan_stream
  (malware only) → tokenize_and_chunk`. `tokenize_and_chunk`'s own docstring states
  its input is "already quality-filtered", but collect_data violated that — so
  minified bundles, auto-generated code, data-file dumps, and broken-syntax files
  (the ~40-50% noise StarCoder filters, and exactly what quality_filter.py exists
  to reject) flowed straight into the multi-source training set. (prepare_data.py
  DID filter; collect_data didn't — a real divergence between the two Stage-1
  paths, related to TOOL-011.) Fixed: new `_maybe_quality_filter(iter, mode,
  languages, workers)` wraps the CODE stream with `parallel_filtered_stream`
  (language-aware, mirrors prepare_data) after the malware scan; mode from
  `--filter {conservative,strict,off}` > data_sources.yaml `code.filter` >
  "conservative" default. Text/math are prose and intentionally NOT code-filtered.
  Tests: test_collect_data_security.py +5 (off=passthrough, conservative drops
  minified / keeps clean, strict selectable, code-path-only wiring guard). Found in
  this cycle's collect_data flow scan. Fresh-scan note: malware scan IS applied to
  all 3 sources; the unwired DataPipeline filters (DATA-021) are largely REDUNDANT
  for the live sources — the-stack-v2 is license-filtered + PII-redacted upstream
  by BigCode, GitHub has the DATA-037 copyleft gate — so DATA-021's practical
  data-quality urgency is LOW (it's now mainly dead-code cleanup, not exposure).
  --help OK; ruff + checkpoint green.

- **MODEL-010** [model/training/consistency, low-medium] `done` (2026-06-12) —
  Fresh scan of the reasoning CoT path (cot_data.py, thinking_tokens.py, reward.py)
  found the format CONSISTENT but the consistency UNGUARDED. The SFT-warmup data
  (`format_thinking_example` → `<think>…</think>\ncode`) and the GRPO reward's
  format bonus (think-FIRST-then-code) independently defined "correct reasoning
  format" via duplicated inline logic — if either drifted, the model would be
  TRAINED on one shape and REWARDED for another (silent training inconsistency,
  the format-parity class: INFER-011/BUG-110). Extracted the single predicate
  `thinking_tokens.is_think_first_format(text)`; `compute_reward`'s format bonus
  now routes through it (behavior-identical — existing reward-format tests stay
  green). Tests: test_think_format_parity.py (8): predicate accepts canonical /
  leading-ws, rejects code-before-think (BUG-102) / no-code-after / missing-tags /
  unterminated; AND every built-in python+typescript CoT example satisfies the
  reward predicate (cross-module guard). Scan also verified _generate_reasoning_trace
  numbering and the CoT/GRPO format are end-to-end consistent. 259 reasoning +
  checkpoint green; ruff clean. No behavior change — DRY + lock cross-module parity.

- **DATA-041** [data-quality, low-medium] `done` (2026-06-12) — `is_typescript`'s
  CONTENT heuristic (language_detect.py — used by the LIVE tsc/eslint scorers and
  the quality filter) only checked 6 markers (`: string`/`: number`/`: boolean`/
  `interface `/`<T>`/`as const`), missing very common TS that carries no surface
  type annotations — `enum`, access modifiers (`readonly`), optional members
  (`?:`), `implements`, `namespace`, `satisfies`, `: void`/`: any`/`: unknown`. On
  this TS-PRIMARY repo, such files (no metadata language/extension) were mis-tagged
  not-TS and SKIPPED from tsc scoring → worse data-quality signal. Expanded the
  indicator set with TS-ONLY constructs (no valid-JS false positives), keeping the
  ≥2-hit threshold. Tests: test_language_detect_heuristic.py (10): enum+modifier,
  interface+optional, implements+void, namespace+satisfies all detected; plain JS /
  JS ternary (`a ? b : c` ≠ `?:`) / single-marker NOT flagged; metadata-extension
  path unchanged. Fresh-scan note: tokenizer training (digit-split, tested),
  whitespace/metadata transforms, and the live language_detect extension sets were
  all verified correct. 624 scorer/data + checkpoint green; ruff clean.

- **MODEL-009** [model/training, low-medium] `done` (2026-06-12) — GRPO curriculum
  temperature was applied (not a no-op) but used ABSOLUTE per-difficulty values
  `{easy:0.7, medium:0.8, hard:0.9}` that REPLACED the run's `temperature`, so a
  user's `--temperature` was silently ignored under `--problems curriculum`
  (the "config silently doesn't apply" class) — and the docstring's "temperature
  SCALING" was inaccurate. Fixed: `_CURRICULUM_TEMP_MULT = {easy:0.875, medium:1.0,
  hard:1.125}` are now MULTIPLIERS of the base temperature (easy → tighter/exploit,
  hard → looser/explore), so `--temperature` is honored; chosen so the default base
  (0.8) reproduces the old 0.7/0.8/0.9 exactly (backward compatible). Extracted a
  pure `_step_temperature(base, difficulty, curriculum)` helper and used it in the
  train loop. Tests: test_curriculum_temperature.py (6): off→base, default
  reproduces legacy absolutes, base honored, easy<medium<hard, unknown→base,
  multipliers around 1.0. Fresh-scan note: the curriculum sort + per-difficulty
  reporting were already correct. 100 grpo/reasoning + checkpoint green; ruff clean.

- **TOOL-013** [test-coverage/regression-guard, low-medium] `done` (2026-06-12) —
  Fresh scan of the GRPO reward path (reward.py, rewards/combined.py,
  reward_registry.py) found the code CORRECT — python_exec runs tests sandboxed;
  typescript/combined strip `<think>` traces (BUG-110 fix), clamp to [0,1], and
  both weight sets in CombinedReward sum to 1.0 — but two correctness-critical
  behaviors had NO test guard: (1) the `combined` adapter's `info["correct"]` key
  that GRPO `train_step` indexes (`sum(... if info["correct"])` → KeyError if
  dropped), and (2) the BUG-110 thinking-stripping for typescript+combined (scoring
  `<think>` prose as broken code would tank the reward on EVERY reasoning-formatted
  generation — a silent training-killer). Added 4 regression tests:
  test_combined_reward_info_has_correct_key, combined thinking-stripping (a
  bracket-imbalanced `<think>` + valid answer scores >0.8 via the adapter vs 0.40
  un-stripped — proving the strip materially helps), and the typescript
  thinking-stripping invariant (wrapped >= bare). 37 reward-registry + checkpoint
  green; ruff clean. No production change — locks existing-correct behavior so the
  GRPO contract / BUG-110 fix can't silently regress.

- **INFER-017** [inference/capability, low] `done` (2026-06-12) — Completed the
  no_repeat_ngram surface: wired it into the VS Code extension's INLINE COMPLETION
  (FIM) path — the highest-value consumer, since inline completions are where
  verbatim repetition loops are most visible. Added a
  `cola-coder.inline.noRepeatNgramSize` setting (default 3), the
  `inlineNoRepeatNgramSize` config field, `no_repeat_ngram_size?` on the
  `FimRequest` type, and the field in the InlineCompletionProvider's fim() call.
  `npx tsc --noEmit` clean, `npm run build` OK (dist 52.7kb), package.json valid.
  Best-of-N path INTENTIONALLY NOT threaded: its batched `generate_group` uses
  `sample_next_tokens_batch`, which deliberately omits per-sequence history for
  speed — adding no_repeat there fights that design, and threading only the serial
  fallback would silently no-op on the batched path. Best-of already curbs
  repetition via candidate selection. Scan also verified best_of_n selection
  (`sorted(key=(verified, score), reverse=True)` — verified always wins) is
  correct. checkpoint + server contract green.

- **INFER-016** [inference/capability, low-medium] `done` (2026-06-12) — Wired
  INFER-015's `no_repeat_ngram_size` to the user surface so it's actually
  reachable (the INFER-011→UX-014 "core capability needs a caller" pattern). Added
  the field (default 0 = off) to all 4 code-gen request bodies (GenerateRequest,
  ChatCompletionRequest, CompletionRequest, FimRequest) and forwarded it at every
  direct `base_gen.generate`/`generate_stream` call site (6: /generate, chat
  non-stream+stream, completions non-stream+stream, FIM — FIM is the highest-value,
  it feeds VS Code inline completions where loops are most visible). Added a
  `--no-repeat-ngram` CLI flag to scripts/generate.py. Tests:
  test_server_no_repeat_ngram.py (4): all models expose the field default-0,
  settable, ≥6 call sites forward it, server parses. 61 server/script-help +
  checkpoint green; --help OK; ruff clean. FOLLOW-UP: the best-of-N path
  (`generate_best_of_n`) doesn't thread it (separate signature) — lower priority
  since best-of already reduces repetition via candidate selection (INFER-017).

- **INFER-015** [inference/capability, medium] `done` (2026-06-12) — Added
  `no_repeat_ngram_size` decoding to the generator — the standard hard block on
  verbatim repetition loops (the failure mode where a code model re-emits the
  same line/block forever, which rep-penalty alone doesn't reliably stop). New
  pure `_banned_ngram_tokens(generated_ids, n)` returns every token that would
  complete an already-seen n-gram given the last n-1 tokens; `sample_next_token`
  sets those logits to -inf BEFORE the greedy/temperature paths (so it constrains
  both), threaded through `generate`/`generate_stream` (default 0 = off → zero
  behavior change for existing callers). 2024-25 best-practice decoding, fully
  verifiable without a trained model. Tests: test_inference.py +8 (bigram/trigram
  ban, unseen-prefix no-ban, too-short, disabled, size-1, greedy respects ban,
  off-by-default allows repeat). Fresh-scan note: audited the attention KV-cache
  (expand_cache + prefill causal mask via Transformer.forward start_pos==0) — all
  correct. 285 generation + checkpoint green; ruff clean.

- **DATA-040** [data-quality, low-medium] `done` (2026-06-12) — Follow-up to
  DATA-039: the docs HTML→Markdown converter flattened list items with
  `child.get_text(" ", strip=True)`, collapsing a `<pre>` CODE BLOCK inside a list
  item (common in step-by-step framework guides) into a single space-joined line —
  indentation/newlines gone → broken code examples in the docs training data.
  Fixed: the unified `ul`/`ol` handler now `extract()`s the `<pre>` blocks from each
  `<li>` BEFORE taking the description text, then renders each block as a proper
  fence via `_walk` (so it also gets DATA-039 language detection). Plain items and
  ordered numbering are unchanged. Scope note: a `<pre>` can only validly live in a
  `<li>` (browsers auto-close `<p>` before a block `<pre>`), so the `<li>` fix
  covers the real case; inline `<code>` inside flattened prose still loses its
  backticks (low harm — the model still sees the text), left as-is. Tests:
  test_scrape_docs_codefence.py +4 (code block in ordered/unordered item preserved
  with multiline formatting; plain list + ordered numbering unchanged). 156 docs +
  checkpoint green; ruff clean.

- **DATA-039** [data-quality/bug, low-medium] `done` (2026-06-12) — The docs
  HTML→Markdown converter (scripts/scrape_docs.py `_element_to_markdown`) read the
  code-fence language class ONLY from the `<code>` element. Many doc highlighters
  (highlight.js, MDX, Docusaurus) put `language-X` on the `<pre>` instead, so those
  examples shipped with an UNTAGGED ``` fence — the model couldn't tell what
  language a code sample was, degrading framework-docs training quality. Fixed:
  read the class from BOTH `<pre>` and `<code>`, and accept the `lang-X` shorthand
  alongside `language-X`; also unified the with/without-`<code>` branches via
  `(code_el or node).get_text()`. Tests: test_scrape_docs_codefence.py (7): lang on
  code / on pre / lang- shorthand / pre-without-code / untagged-when-none / code
  text preserved / standalone inline code. 13 docs + checkpoint green; ruff clean.

- **DATA-038** [data-quality/bug, low] `done` (2026-06-12) — `SoftwareHeritageSource`
  stored `content_types` as a raw `set(content_types)` but matched against
  `os.path.splitext(name)[1].lower()` (canonical ".ext" lowercase). So a plausible
  config like `content_types=[".PY"]` or `["py"]` matched NOTHING → a silent empty
  stream with no error (the "config silently doesn't match" class). Fixed: normalize
  each entry to `"." + ext.strip().lstrip(".").lower()` in __init__ (drops blanks).
  Tests: test_swh.py +6 (uppercase/missing-dot/mixed-forms normalized, blanks
  dropped, None stays None, canonical unchanged). Fresh-scan note: audited the whole
  data-source layer (huggingface→download.stream_code_data round-robin language
  balancing, SWH walk/binary-decode handling, github license gate) — all correct
  except this. 39 SWH/source + checkpoint green; ruff clean.

- **DATA-037** [data-quality/legal, medium] `done` (2026-06-12) — The GitHub
  scraper detected a repo's license (`check_license`, github.py:~1249) but only
  TAGGED it as metadata — it never REJECTED copyleft. The active gate was the
  GitHub `license:` SEARCH query, which is optional per scrape profile and relies
  on GitHub's own (sometimes-incomplete) detection; the `LicenseFilter` plugin
  that would reject non-permissive files is unwired (DATA-021). So a GPL/LGPL/AGPL
  repo that slipped the query filter had ALL its files extracted into the
  permissive training corpus (a legal + data-quality risk — the model could
  reproduce copyleft code verbatim). Fixed: added an ACTIVE post-clone gate in
  `stream` — new pure `_is_copyleft_license(spdx)` rejects GPL/LGPL/AGPL families
  (case-insensitive), and the loop `continue`s (skips file extraction) + logs.
  Conservative: Unknown/NOASSERTION/permissive/MPL are NOT rejected here (left to
  the query filter / downstream LicenseFilter — rejecting "Unknown" would drop
  permissive repos the heuristic detector didn't recognize). Tests:
  test_github_scraper.py +4 (GPL variants rejected, permissive/MPL allowed,
  Unknown/None not rejected, gate wired+continues). Fresh-scan note: the routing
  orchestrator's specialist path is an explicit TBD (always returns base_generator,
  gated on trained specialists) — not a bug. 209 source/data + checkpoint green;
  ruff clean.

- **EXPORT-012** [export/bug, medium] `done` (2026-06-12) — The generated Ollama
  Modelfile set temperature/top_p/top_k/repeat_penalty/num_predict/stop but NOT
  `num_ctx`. Ollama DEFAULTS num_ctx to 2048, so a checkpoint trained for a longer
  context (4080_max = 4096) deployed via Ollama would silently run at HALF its
  context window — long-file completions truncated, with no error. Fixed:
  `OllamaExporter.create_modelfile(..., num_ctx=None)` emits
  `PARAMETER num_ctx <n>` when given (int-coerced); export_model.py passes
  `config.model.max_seq_len`. None = omit (backward compatible). The fresh scan
  also VERIFIED clean+correct: quantize.py INT4 pack/dequant (padding trimmed) +
  _model_size_mb (ao packed-param accounting), and the Ollama ChatML template +
  stop tokens (already TOOL-006-fixed and tested). Tests: test_ollama_chatml.py +4
  (num_ctx emitted/reflects seq_len/omitted-by-default/int-coerced). 72
  ollama+export + checkpoint green; --help OK; ruff clean.

- **EXPORT-010** [export/bug, medium] `done` (2026-06-12) — GGUF export silently
  lied about quantization. `SUPPORTED_QUANTIZATIONS` advertises `q4_k_m`/`q5_k_m`,
  but the built-in writer (used when the `gguf` package is absent) can't emit true
  K-quants — `_quantize_q4_k_m`/`_quantize_q5_k_m` just `return _quantize_q8_0(...)`.
  The data was a valid q8_0 file (tagged `_GGML_TYPE_Q8_0`), but `ExportResult`
  reported `quantization="q4_k_m"` — so a user requesting a ~4× smaller q4 file got
  a 2× larger q8_0 file LABELED q4, with no warning (the silent-downgrade class).
  Fixed: new pure `_effective_quantization(requested, gguf_package_available)`
  resolves the quant ACTUALLY written + a warning; `export()` logs the warning,
  passes the EFFECTIVE quant to the writer, and reports it (plus new
  `requested_quantization` + `warning` fields on ExportResult). export_model.py now
  prints the warning + the realized `quant=` so the downgrade is visible. Tests:
  test_export.py +4 (K-quant w/o package → q8_0 + warning; with package kept;
  non-K-quants never warn; ExportResult carries the fields). 104 export/quant +
  checkpoint green; --help OK; ruff clean. Found in this cycle's export fresh scan.

- **DATA-036** [data-quality/observability, low-medium] `done` (2026-06-12) —
  Multi-source mixing (the 70/20/10 code/text/math ratios) had no way to VERIFY
  the realized mix. `DatasetCombiner._compute_sources` reported `chunks_available`
  (input size) while the `CombineResult.sources` docstring promised
  `chunks_contributed` — and the contributed count was never computed (the comment
  at combine() even said "we track how many chunks each dataset contributed", but
  it didn't). So a user requesting 70/20/10 could not confirm it didn't silently
  distort (the interleave code itself documents a prior 70/20/10→53/32/16 bug).
  Fixed: each strategy (_concat/_interleave/_weighted_sample) now returns its
  per-source contribution counts (concat=taken, interleave=emitted cursors,
  weighted=bincount of choices); `_compute_sources` reports `chunks_contributed`
  + `fraction` (realized share) alongside `chunks_available`; combine_datasets.py
  now PRINTS a "Realized mix (contributed / requested)" breakdown so the ratio is
  visible. The mixing math was already correct — this is pure observability. The
  fresh scan also VERIFIED _interleave (ratio-exact via per_ds_target/cursor caps)
  and _weighted_sample (rng.choice by weight) are correct. Tests: test_combine.py
  +6 (sources carry contributed/fraction, contributions sum to total, concat
  contributes all, interleave fraction≈weight, weighted skew correct, zero-weight
  →0). 101 combine/mixing + checkpoint green; --help OK; ruff clean.

- **DATA-035** [docs/data-quality, low] `done` (2026-06-12) — Corrected misleading
  docstrings in the FIM data-augmentation path (the inline-completion / VS Code
  ghost-text training format), found in a fresh scan of data/fim.py. (1) The
  `FIMTransform` PSM docstring showed a garbled order (`[prefix_ids]` twice); now
  states the correct StarCoder/OpenAI layout for both PSM
  (`<fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle`) and SPM. (2) The
  `truncate_or_pad` docstring claimed `False` makes output "shorter by up to 3"
  tokens — it's the OPPOSITE: `False` ADDS the 3 markers with no content removed,
  so output is LONGER by exactly 3 (`True` slices content to len-3 to stay
  constant). The CODE was always correct; only the docs misled. (3) Added a
  double-FIM warning to FIMDataset pointing at the canonical
  `create_dataloader(fim_rate=...)` trainer path (DATA-025 sub-item). Tests:
  test_fim.py +3 (truncate True preserves length, False adds exactly 3, False
  drops no content) — locks the real behavior so doc/code can't drift again. 76
  FIM + checkpoint green; ruff clean.

- **INFER-014** [inference/bug, low] `done` (2026-06-12) — `sample_next_token`'s
  all-masked safety fallback (sampling.py) returned `logits.argmax()` on the
  ALREADY-FILTERED logits. When the top-k/top-p/min-p filters mask everything to
  -inf (e.g. `min_p > 1`, or NaN/Inf model logits in unstable bf16 inference),
  argmax of an all -inf tensor returns index 0 — an arbitrary token — instead of
  the model's actual preference. Fixed: capture `fallback_token` = argmax of the
  post-temperature, PRE-filter logits and return that in the safety branch. Found
  via a fresh scan of the sampling path that added the missing robustness tests:
  the rest of the path (rep-penalty sign, min-p threshold, top-p keep-at-least-one,
  the safety guard itself) was already correct, but the fallback token and the
  guard/keep-one invariants had ZERO coverage. Tests: test_inference.py +5
  (top_p keeps exactly one, peaked top_p samples the top token, NaN/Inf logits
  don't crash, min_p>1 degrades to the true argmax). 177 inference + checkpoint
  green; ruff clean. (Scan also verified clean+correct: dataset/collator,
  language_modeling_loss per-sample weighting, preprocess doc-boundary eos/bos
  packing, and the Muon optimizer + its tests — no issues found there.)

- **BUG-114** [training-stability, medium] `done` (2026-06-12) — Defense-in-depth
  follow-up to DATA-034: bf16 (the RTX 4080 primary) runs `GradScaler(enabled=False)`,
  so a non-finite loss from ANY source (SFT all-ignored batch, a z-loss spike, a
  bad sample, fp overflow) backprops to NaN weights and corrupts the checkpoint —
  fp16 is saved by the GradScaler's inf/NaN skip, bf16 had NO net. Added a
  `torch.isfinite(loss)` guard BEFORE the backward in BOTH loops:
  (trainer.py) per-micro-batch skip + counter; if ALL micro-batches in a step are
  non-finite, skip the optimizer/scheduler step entirely (weights untouched);
  partial non-finite drops just those micro-batches and warns.
  (train_sft.py) per-batch skip with a warn + end-of-run summary count; the
  skipped batch contributes no grad to its accumulation window and isn't counted
  in epoch_loss. Tests: test_nonfinite_loss_guard.py (6): behavioral — finite loss
  updates weights, NaN loss leaves them UNCHANGED, and a control proving an
  unguarded NaN step DOES corrupt every weight; static — both loops contain the
  guard before .backward() and still parse. 389 training/SFT + checkpoint green;
  train_sft --help OK; ruff clean. Closes the "never corrupt a checkpoint"
  invariant gap on the primary precision.

- **DATA-034** [data-quality/training-stability, high] `done` (2026-06-12) —
  `SFTDataset._load` (data/sft_dataset.py) skipped malformed/empty examples but
  NOT examples with NO labeled assistant tokens (labels all -100). That happens
  when a conversation has no assistant turn, or when truncation to max_seq_len
  cuts off the assistant content. Such an example has zero SFT signal AND is a
  corruption hazard: train_sft.py uses `CrossEntropyLoss(ignore_index=-100)`
  (mean), which returns NaN when a batch is all-ignored → NaN gradients → NaN
  weights → CORRUPTED checkpoint. Critically, bf16 (the RTX 4080 primary
  precision) runs with `GradScaler(enabled=False)`, so there is NO skip-on-NaN
  safety net there (fp16 would be caught by the scaler). Fixed: `_load` now drops
  any example where `not any(lbl != -100 for lbl in labels)`, logging the count
  separately. Every loaded example now retains ≥1 label through the `labels[:,1:]`
  shift (assistant tokens are never at index 0), so a batch of valid examples can
  never be all-ignored. Tests: test_sft_dataset_labels.py (5): assistant kept with
  labels, user-only/system-only dropped, mixed file keeps only signal examples,
  load invariant. 65 SFT + checkpoint green; ruff clean. Found in this cycle's SFT
  data-path fresh scan. FOLLOW-UP: BUG-114 (defense-in-depth NaN guard in the
  train loop for any OTHER non-finite-loss source under bf16).

- **SEC-008** [security/bug, low-medium] `done` (2026-06-12) — Quarantine
  basename collision in collect_data.py `_scan_downloaded_data`. The post-download
  malware quarantine moved each threat to `quarantine_dir / src.name`, so two
  threatening files sharing a basename (ubiquitous: `index.js`, `__init__.py`,
  `main.py` across repos/subdirs) collided — `src.rename(dst)` SILENTLY OVERWROTE
  the first, destroying quarantined-malware evidence (forensic data loss). Fixed:
  new testable `_quarantine_dest(quarantine_dir, src)` prefixes the destination
  with an 8-char hash of the FULL source path — unique per distinct path,
  idempotent for the same path, basename preserved for triage. The data is clean
  either way (both moved out of raw_dir), but now BOTH threats are retained in
  quarantine. Fresh-scan note: verified the "tighten malware-scan defaults" thread
  is ALREADY safe (scoring.yaml enabled/in_stream/quarantine, warn fails-CLOSED
  non-interactively) and the config knobs (on_threat, scanners.*, yara_rules_dir)
  are all read+enforced; MoE fine-tune stage 7.5 is wired (training_menu →
  derive_moe_finetune_config). Tests: test_collect_data_security.py +4 (no
  collision on shared basename, idempotent, under quarantine dir + basename kept,
  end-to-end both files survive). 163 security/scan + checkpoint green; ruff clean.

- **INFER-013** [inference/bug, high] `done` (2026-06-12) — The FastAPI server's
  non-streaming `/v1/chat/completions` leaked the prompt into the reply in
  INSTRUCT mode. It did `strip_prompt_prefix(result, prompt)` against the
  marker-form ChatML prompt (`<|im_start|>…`), but `generate()` returns
  decode(prompt+completion) and `decode(skip_special=True)` STRIPS the markers, so
  `result` never starts with `prompt` → the longest-common-prefix diff matched
  almost nothing and returned (nearly) the whole decoded prompt as the assistant
  message. This is the VS Code chat-participant path when baseModelMode=false
  (the INFER-011 / BUG-111 prompt-echo class). The FIM endpoint already fixed it
  by diffing the DECODED prompt; chat did not. Fixed: extracted a shared
  `_completion_after_prompt(result, prompt, tokenizer)` (decode→encode the prompt
  so both sides are marker-free, then strip) and routed BOTH chat and FIM through
  it (DRY). Streaming chat was unaffected (generate_stream yields completion-only
  chunks); /v1/completions was unaffected (raw prompt has no special tokens).
  Tests: test_server_chat_prompt_strip.py (5): ChatML no-leak, a control proving
  the raw-diff DID leak, FIM infill-only, plaintext base mode, empty completion.
  169 server/inference + checkpoint green; ruff clean. Found in this cycle's
  inference-server fresh scan.

- **EVAL-009** [eval/observability, medium] `done` (2026-06-12) — The HumanEval
  orchestrator (scripts/evaluate.py) wrapped the whole generate→extract→grade
  pipeline in `except Exception: pass`, so a HARNESS failure (generator crash,
  OOM, sandbox misconfig, extraction bug) was silently counted as a model
  failure. A 0% pass@k was therefore ambiguous: weak model OR broken harness, with
  no signal to distinguish them — the exact "silent no-op masks a broken thing"
  class this project keeps fixing. Fixed: extracted a testable
  `_evaluate_problem(...) -> (num_correct, harness_errors)` that catches per-sample
  exceptions, counts them SEPARATELY from genuine test failures (where
  evaluate_solution returns (False, ...) WITHOUT raising), logs the first 3
  (type+message), and the run prints a loud "pass@k likely DEFLATED" warning with
  the total. The pass@k math (metrics.pass_at_k) and the sandboxed runner were
  audited this scan and are correct — no change. Tests:
  test_evaluate_harness_errors.py (5): all-pass→0 errors, test-failures≠harness
  errors, generator crash & grader exception counted as harness errors, mixed.
  evaluate.py --help OK; 346 eval/menu + checkpoint green; ruff clean. Found in
  this cycle's evaluation-suite fresh scan.

- **MODEL-006** [model/config-hygiene, low] `done` (2026-06-12) — Phantom config
  knob: `RoPEScalingConfig.original_max_seq_len` (default 4096) was NEVER read —
  `Transformer.__init__` hardcoded `original_max_seq_len=config.max_seq_len` when
  building the YaRN freq table, so a user extending a model trained at a DIFFERENT
  length than the current config's max_seq_len was silently ignored (the YaRN
  wavelength partitions `original_max_seq_len/beta_*` use the wrong base). Fixed:
  changed the dataclass default to the `0` sentinel ("use max_seq_len", now
  distinguishable from an explicit value) and Transformer passes
  `getattr(scaling, "original_max_seq_len", 0) or config.max_seq_len`. Verified no
  YAML config sets the field, so the sentinel change is safe. Tests:
  test_model006_rope_original_len.py (4): get_rope_freqs honors the param;
  end-to-end a Transformer with original_max_seq_len=2048 produces DIFFERENT
  rope_freqs than the default (proves it's read — failed on HEAD); 0 sentinel ==
  explicit max_seq_len; non-yarn unaffected. checkpoint + 280 config/rope tests
  green; ruff clean. Fresh-scan note: audited training/trainer.py step (grad-accum
  scaling, fp16 skipped-step LR guard, z_loss, MoE aux loss) — clean, no new item.

- **MODEL-007** [model/training, high] `done` (2026-06-12) — GRPO's PPO objective
  was both mis-formulated and inert. (a) The importance ratio was SEQUENCE-LEVEL:
  `ratio = exp(Σ_t Δlogp)` = the PRODUCT of per-token ratios — over long
  completions this explodes/vanishes and saturates the clip on nearly every
  sample, destroying PPO's per-token credit assignment (reference GRPO/Dr.GRPO/
  DAPO all clip PER TOKEN). (b) The trainer took ONE gradient step per generated
  group with old log-probs recomputed from the SAME weights, so the ratio was
  ALWAYS exactly 1.0 → `clip_epsilon` / `clip_epsilon_high` (the advertised DAPO
  clip-higher) were DEAD CODE that never engaged. Fixed: new pure
  `grpo_clipped_surrogate(new_logp, old_logp, A, clip_low, clip_high)` clips PER
  TOKEN and SUMS (so at the 1-epoch default the gradient is byte-identical to the
  old sequence-level one — zero dynamics change); `_completion_logprobs` returns
  the per-token vector (`_completion_logprob_sum` delegates, old tests intact);
  old policy stored as per-token detached tensors. New `ppo_epochs` param (config
  `reasoning.ppo_epochs`, default 1) wraps the update in μ inner epochs reusing
  the fixed old log-probs — the ONLY regime where the clip actually acts. Wired
  through reasoning.yaml + train_reasoning.py (+ wiring test requires the kwarg).
  ALSO fixed a PRE-EXISTING crash (failing on HEAD, unrelated to the above):
  train_step did `total_loss.backward()` unconditionally, so a degenerate group
  where NO member produced completion tokens (grad-less zero loss) raised
  RuntimeError and killed the run — now guarded (skip the step). Tests:
  test_grpo_token_surrogate.py (12 — per-token math, asymmetric DAPO clip, PPO
  min-branch, not-sequence-level) + the 2 previously-failing parallel-gen
  train_step tests now pass. 163 reasoning/reward + checkpoint green; ruff clean.
  Found in this cycle's GRPO fresh scan. FOLLOW-UP: Dr.GRPO constant length-norm
  (sum is length-biased) + a real GRPO smoke run to validate dynamics → MODEL-008.

- **DATA-033** [data-quality/security, medium] `done` (2026-06-12) —
  `check_no_hardcoded_secrets` was STRICT-ONLY, so in the DEFAULT conservative
  filter mode files with live credentials (OpenAI `sk-…`, GitHub PATs, AWS keys,
  PEM private-key blocks) flowed straight into the training set — both an
  exfiltration/memorization risk and noise that teaches the model to emit
  secret-shaped strings. Fixed by splitting secret signatures into two tiers:
  `_HIGH_CONFIDENCE_SECRET_PATTERNS` (provider-prefixed tokens + private-key
  headers + AWS `AKIA…` key VALUES — near-zero false positives) and
  `_HEURISTIC_SECRET_PATTERNS` (looser `api_key/password/secret = "…"`
  assignments). New `check_no_obvious_secrets` (high-confidence only) runs in
  CONSERVATIVE mode and is registered as a scoring GATE (fail 0.0, DATA-032
  capping applies); the fuller `check_no_hardcoded_secrets` (both tiers) stays
  STRICT-only so placeholder/example passwords don't cause conservative false
  positives. Verified: conservative now rejects+caps a real `sk-` key while
  `aws_access_key_id = os.getenv(...)` (env read) and `password="changeme123"`
  (placeholder) are NOT flagged conservatively (strict still catches the latter).
  Tests: test_quality_filter.py +7 (openai/PEM/AKIA/ghp rejected+gated; env-read
  & placeholder not FP'd conservatively; placeholder caught in strict). 644
  data/security/scoring tests + checkpoint green; ruff clean. Found following up
  the DATA-032 quality-filter scan.

- **DATA-032** [data-quality, medium] `done` (2026-06-12) — `score_code`
  (quality_filter.py) computed the overall quality score as an UNWEIGHTED AVERAGE
  of all per-check scores, so a single catastrophic failure was masked by trivial
  passes. Concretely: a JSON data-dump fails `check_not_data_file` (0.0) but
  passes the other ~7 checks → overall ≈ 0.79 → "good" tier → 1.5x TRAINING WEIGHT
  on garbage (verified). Same for minified/auto-generated/low-diversity files.
  This feeds `score_data.py --score` → `.weights.npy`, so the model was being told
  to train HARDER on exactly the files the filter would otherwise reject. Fixed:
  split checks into GATES (fail_score <= 0.10 in _SCORE_MAP — broken syntax,
  minified, autogenerated, data-dump, no diversity, embedded secret) vs soft
  quality checks; `overall = min(soft_average, min(gate_scores))`. When all gates
  pass the cap is inert (overall == soft average → clean files UNCHANGED); a gate
  failure caps the score into the reject/poor tier. Soft-only failures (naming,
  docs, comment ratio, test-heaviness, length, JS brace-balance) still merely
  down-weight. `_GATE_CHECKS` is derived from _SCORE_MAP so it stays in sync.
  Tests: test_quality_filter.py +5 (data-dump/minified/autogenerated capped to
  reject; clean code overall==soft_avg with all gates 1.0; soft-only failure NOT
  capped). 590 data/scoring tests + checkpoint green; ruff clean. Found this
  cycle's fresh scan of the live filtering path (quality_filter.py).

- **MODEL-005** [model/capability, low] `done` (2026-06-12) — YaRN context
  extension (`precompute_rope_freqs_yarn`, rope.py) scaled RoPE frequencies but
  omitted YaRN's attention temperature factor `mscale = 0.1*ln(factor)+1.0`
  (Peng et al. 2023), which the paper applies to keep attention logits
  calibrated at extended context. Without it, extended-context attention was
  mis-scaled vs. reference YaRN. Fixed: new `rope.yarn_attention_scale(factor)`
  returns mscale (1.0 for factor<=1); `GroupedQueryAttention` gained an
  `attn_logit_scale` param folded into `self.scale = (head_dim**-0.5) * scale`;
  `Transformer.__init__` now resolves rope scaling ONCE up-front (shared by the
  freq table and attention), computing `attn_logit_scale = mscale**2` ONLY for
  type=="yarn" and threading it through `TransformerBlock` to every attention
  layer. Inert for type none/ntk/linear and factor<=1, so all existing
  checkpoints are byte-for-byte unchanged (self.scale is a plain float, not in
  state_dict → checkpoint invariants intact). Tests: test_yarn_attention_scale.py
  (12 — mscale math, attention multiplier, transformer wiring incl. ntk/linear
  leave temperature untouched, all blocks share the scale); checkpoint suite +
  92 rope/attention/transformer tests green; full-tree ruff clean. Discovered
  MODEL-006 (phantom `original_max_seq_len`) during the audit. NOTE: end-to-end
  perplexity gain at extended context wants a real YaRN-extended checkpoint to
  confirm; the wiring + scale math are locked regardless.

- **UX-014** [ux/inference, low] `done` (2026-06-12) — Follow-up to INFER-011:
  `InteractiveChat` supported `chat_format="chatml"` but had NO live caller (only
  constructed in tests), so the format-parity fix was unreachable. Added
  `scripts/chat.py` — a multi-turn chat REPL that loads via the shared
  `load_generator` (MoE + SFT-vocab aware) and starts `InteractiveChat`. New
  module fn `resolve_chat_format(checkpoint, override)`: `auto` selects "chatml"
  when any path component ends in `_sft` (the dir train_sft.py writes to), else
  "alpaca"; `--chat-format {auto,alpaca,chatml}` overrides. Also made
  `InteractiveChat` forward per-turn `max_new_tokens`/`temperature` to the
  generator (previously hardcoded 256/0.7), so the script's `--max-tokens` /
  `--temperature` flags are meaningful. Wired into master_menu's Generate &
  Interact submenu ("Multi-Turn Chat" → chat.py, renumbered dispatch). Docs:
  scripts-reference (60→62) + CLAUDE.md counts. Tests: test_chat_script.py (8 —
  format resolution incl. override-wins and invalid→ValueError; sampling-param
  forwarding for both formats + defaults); test_menu_script_wiring auto-validates
  chat.py exists/flags/non-orphan; `chat.py --help` OK; checkpoint suite green;
  full-tree ruff clean. NOTE: end-to-end reply QUALITY still wants a real SFT
  checkpoint to confirm (no model trained yet) — the wiring and format selection
  are locked regardless.

- **INFER-011** [inference/consistency, medium] `done` (2026-06-12) —
  `multi_turn_chat`'s `ChatSession` only formatted prompts in ALPACA style
  (`### User:`/`### Assistant:`), but `train_sft.py` trains on CHATML
  (`<|im_start|>{role}\n…<|im_end|>`, tokenizer/chat_template.py). Interactive
  chat against an SFT checkpoint therefore fed it an unfamiliar format →
  degraded replies. Previously deemed "not safely fixable" because ChatML
  markers are special tokens that `decode` STRIPS, breaking string-based reply
  extraction. Fixed in three coherent layers, each unit-tested with the existing
  scripted-stub harness (no GPU/SFT model needed):
  (1) `CodeGenerator.generate(return_new_only=True)` (generator.py) returns
  completion-only text — decode of the NEW token ids alone — at both the normal
  and string-stop return points; default False keeps the legacy prompt+completion
  return (zero change for existing callers). This is the missing primitive the
  old note named.
  (2) `ChatSession.chat_format` ("alpaca"|"chatml"): rendering centralized into
  `_render`/`_render_alpaca`/`_render_chatml` (the latter reuses
  `chat_template.format_chat` + an `<|im_start|>assistant\n` generation prompt for
  format parity). Refactor also fixed a latent truncation bug — the old fallback
  hardcoded Alpaca prefixes and assumed the last message was a user turn.
  (3) `InteractiveChat(chat_format=...)` + `_generate_reply`: ChatML mode stops on
  `<|im_end|>` and uses `return_new_only=True` (no prompt string-diff); Alpaca
  keeps the legacy strip path. Tests: test_infer011_chatml_chat.py (12) across all
  three layers; regression: generator_stop_tokens / multi_turn_chat_extract /
  ollama_chatml / streaming / batch_strip / inference all green; checkpoint suite
  green; full-tree ruff clean. NOTE: end-to-end chat QUALITY still wants a real
  SFT checkpoint to A/B; InteractiveChat has no live menu caller yet (wiring it
  into a menu, with chat_format auto-set from whether the checkpoint is an _sft
  dir, is a small follow-up → logged as UX-014).

- **SEC-007** [security/sandbox, medium] `done` (2026-06-12) — The agent tool
  executor's `run_tests` handler (tools/executor.py `_handle_run_tests`) ran pytest
  on any model-supplied path, validating only that it stayed inside the project
  ROOT. But pytest IMPORTS AND EXECUTES `conftest.py` + `test_*.py` at collection,
  so this is an unsandboxed code-execution primitive — pointed at an in-tree dir
  holding untrusted code (e.g. a gitignored `data/` of scraped sources, or a
  planted `conftest.py`), it would run that code on the host. Discovered as the
  follow-up thread flagged after SEC-006's executor scan. Fixed: confine the target
  to the project's `tests/` tree (new `tests_subdir="tests"` constructor param →
  `self.tests_root`); a path resolving outside it returns a clear error BEFORE any
  subprocess, and traversal outside the root still raises via `_validate_path`. The
  agent's legitimate use (validating its own changes against the project suite) is
  fully covered. Also tightened the handler to pass the RESOLVED path to pytest
  rather than the raw string. Tests: test_tool_executor.py +5 (outside-tests
  rejected, traversal rejected, inside-tests passes gate, default path allowed,
  custom tests_subdir honored). checkpoint suite green; full-tree ruff clean.
  (Fresh-scan note this cycle: VERIFIED the "within-file near-dup MinHash" thread
  is ALREADY resolved — `dedup.py` `deduplicate_self_array` is wired into
  `prepare_data.py --dedup minhash`; no new item needed.)

- **SEC-006** [security/sandbox-consistency, medium] `done` (2026-06-12) — The agent
  tool executor's `typecheck` handler (tools/executor.py `_handle_typecheck`) ran
  tsc on UNTRUSTED model-generated code via raw `npx tsc --noEmit <tempfile>`
  through `_run_subprocess` — bypassing the project's canonical `TscRunner`. Three
  problems, all the TOOL-004 class but on the live agent path: (1) no hardened
  tsconfig, so a stray `tsconfig.json`/plugin in temp-dir scope could make tsc load
  arbitrary code; (2) `npx` can fetch tsc OVER THE NETWORK, violating the
  no-outbound-network sandbox rule; (3) on Windows tsc is a `.CMD`, so the bare
  `npx tsc` path is also the fragile invocation TscRunner already solved. Fixed:
  `_handle_typecheck` now routes through a lazily-built, per-strictness-cached
  `TscRunner` (hardened tsconfig plugins=[]/types=[]/typeRoots=[], SandboxedRunner
  execution, resolved tsc path), returns "OK" / a formatted error list / a graceful
  "tsc not available" message, and counts only `severity == "error"`. Removed the
  now-unused `os`/`tempfile` imports. Tests: test_tool_executor.py +6 (no-code,
  clean=OK, errors formatted, warnings ignored, unavailable handled, strict flag
  selects distinct runner) via an injected fake runner so they pass without tsc on
  PATH. Found this cycle fresh-scanning the agent tool-execution surface (the
  security-sensitive path per the standing untrusted-code constraint).

- **DATA-031** [data-quality/robustness, low] `done` (2026-06-12) —
  `CompositeScorer.score_batch` (data/scorers/protocol.py) keyed per-scorer batch
  results by `scorer.name`, so if two registered scorers shared a `.name` the
  second OVERWROTE the first and BOTH read the second's scores — the weighted
  overall diverged from the single-item `score()` (which iterates scorers
  directly). Not reachable with the built-in registry (tsc/eslint/stars/heuristic/
  llm_judge all unique), but a custom scorer with a colliding name silently
  mis-weighted. Fixed: key by scorer POSITION (zip the per-scorer batches with
  `self._scorers`) so `score_batch == score` regardless of names. Tests:
  test_scorer_protocol.py +2 (batch matches single per item; two colliding-name
  scorers weight 0.5/0.5 → overall 0.5, batch == score). 20 passed. Found auditing
  the scoring pipeline (logged during the DATA-030 cycle).

- **DATA-030** [data-quality/bug, medium] `done` (2026-06-12) — `prepare_fim_data.py`
  (static FIM dataset prep) called `setup_fim_tokenizer`, which ADDS the
  `<|fim_*|>` special tokens to the in-memory tokenizer when they're missing —
  but NEVER re-saves tokenizer.json. So if run with a tokenizer lacking FIM
  tokens, the FIM marker ids baked into the output `.npy` would be OUT OF VOCAB
  for the model at training time → silently corrupted FIM training data (the
  DATA-003/013 "silent token poison" class). Every tokenizer trained by
  train_tokenizer.py has these tokens (they're in SPECIAL_TOKENS), so it's a
  latent footgun, but the script should never silently poison. Fixed: added a
  read-only `_resolve_fim_ids(tokenizer)` that caches the existing ids on the
  tokenizer (where FIMTransform.apply reads them) and raises `_MissingFimTokens`
  if any are absent; main() fails loud with the exact remedy (retrain the
  tokenizer / use data.fim_rate) instead of producing broken data. Mirrors
  DATA-012's `_resolve_fim_ids` (reads, never adds) and DATA-013's fail-loud.
  (Also noted but NOT fixed: `--input X --output X` would hit the DATA-004 mmap
  lock — but output defaults to a distinct path, much lower probability than
  DATA-024. Scoring-pipeline robustness gap logged as DATA-031.) Tests:
  test_prepare_fim_data.py (3): real tokenizer resolves+caches, all-missing
  raises with full list, partial-missing lists only the absent token. Found
  this cycle auditing the FIM prep stage.
- **INFER-012** [inference/data-quality, medium] `done` (2026-06-11) — `SelfVerifier`
  (features/self_verification.py — the heuristic verifier wired into best-of-N as
  the fallback/tie-breaker when no tsc verifier exists, ZERO prior tests)
  penalised valid TypeScript. `verify_no_hallucination` ALWAYS flagged
  `console.log` / `var` / `undefined` as "JavaScript-isms in Python code" — but
  this is a TS-PRIMARY project where those are VALID TS, not hallucinations. So
  on the `generate --best-of N` fallback path (TS model, no tsc installed), the
  self-verifier ranked correct TS candidates LOWER (each JS-ism docks 0.20 off
  the 0.25-weighted hallucination score) — the DATA-010/DATA-022 Python-centric
  class. Fixed: `verify_no_hallucination(code, language=None)` and
  `verify_code(code, language=None)` are now language-aware — the JS-ism checks
  fire only when the code is NOT JS/TS (canonical `language_detect.is_js_ts`,
  using an explicit hint when given). Threaded the resolved `language` from
  best_of_n through `_verify_heuristic`/`_heuristic_confidence` so it's explicit,
  not just content-detected. Language-agnostic checks (fabricated
  modules/methods, degenerate repetition) still fire. Tests:
  test_self_verification.py (10, the module's first): python console.log flagged,
  TS console.log NOT flagged (content + explicit hint), explicit python flags JS,
  TS verify_code unpenalised, suspicious-module/repetition still fire, syntax +
  empty-completeness. Found scanning the features/ modules wired into the core.
- **BUG-113** [loaders/bug, high] `done` (2026-06-11) — Closed the EXPORT-009 bug
  CLASS: a sweep of every checkpoint-loading site found EIGHT more that build a
  `Transformer(...)` and `load_model_only`/`load_checkpoint` WITHOUT first calling
  `apply_moe_config_from_checkpoint` — so loading an upcycled MoE checkpoint
  (stages 7/7.5) crashed on `experts.*` keys. Fixed (apply before build, no-op for
  dense): `run.py` (interactive REPL), `benchmark.py`, `ts_benchmark.py`,
  `regression_test.py`, `train_sft.py` (SFT fine-tune), `train_reasoning.py`
  (GRPO base ckpt), `find_lr.py` (only when `--resume`), and
  `evaluation/quality_report.py`. The last two grep-missed sites build
  `Transformer(model_cfg)` from a bare ModelConfig, so they wrap it as
  `SimpleNamespace(model=model_cfg)` for the apply call. `upcycle_to_moe.py`
  correctly does NOT apply (its input is dense); the weight-level loaders
  (checkpoint_average/compare/model_card) operate on raw state dicts (no model
  build) — no bug. Separately fixed a REAL pre-existing bug in
  `inference_benchmark.py`: it passed `model_cfg` (a ModelConfig) as the `model`
  arg to `load_model_only` (which requires a built nn.Module) — it never worked;
  now builds the Transformer first (+ MoE-aware). Tests:
  test_moe_integration.py test_apply_moe_via_modelconfig_namespace_wrapper (the
  ModelConfig-wrapper path: flips to MoE, builds, loads, is_moe). All edited
  scripts compile. Found by the EXPORT-009 follow-up sweep.
- **EXPORT-009** [export/bug, medium] `done` (2026-06-11) — `export_model.py::_load_model`
  built `Transformer(config.model)` and called `load_model_only` WITHOUT first
  calling `apply_moe_config_from_checkpoint(config, checkpoint)`. Every other load
  path (inference/loading, trainer, evaluate, generate, serve, smoke_test) flips
  the config to MoE before building the model so an upcycled MoE checkpoint's
  expert weights (`blocks.N.ffn.experts.*`) have somewhere to load — export was
  the ONLY one missing it. So quantizing or benchmarking an upcycled/fine-tuned
  MoE checkpoint (stages 7/7.5) via export_model.py crashed with a hard
  `_load_state_dict_tied` failure (experts.* keys have nowhere to go in a dense
  model). Fixed by mirroring the canonical pattern: apply the MoE config (no-op
  for dense checkpoints) before building the Transformer; logs the expert counts
  when it fires. (GGUF export reads raw safetensors tensors and doesn't build a
  Transformer, so it's a separate concern — full MoE→GGUF support is out of
  scope.) Tests: test_moe_integration.py test_export_load_model_handles_moe_checkpoint
  (real dense→upcycle→export_model._load_model: config flipped to MoE, is_moe,
  forward works). Found scanning the export path.
- **EVAL-008** [eval/wiring+robustness, medium] `done` (2026-06-11) — The during-
  training auto-eval (`training/auto_eval.py` AutoEvaluator — periodic HumanEval
  pass@k + regression detection + best-checkpoint save, 545 lines) was fully
  implemented AND integrated into the Trainer (which calls should_eval/evaluate/
  check_regression), but `train.py` never constructed or passed one and NO config
  has an `auto_eval` section — so the feature was completely DORMANT (never ran in
  production). Two fixes: (1) WIRING — added opt-in `--auto-eval` / `--eval-every`
  / `--eval-subset` flags to train.py that build an AutoEvaluator (checkpoint_dir
  from config), load the tokenizer it needs, and pass `auto_evaluator`+`tokenizer`
  to `trainer.train()`; off by default (zero change to existing runs), and
  disables gracefully if the tokenizer can't resolve. (2) ROBUSTNESS — the
  trainer's eval block was UNGUARDED: a failure in the (never-before-run)
  generation/sandbox path would crash a long training run, and if it raised after
  `model.eval()` the model was left in eval mode (dropout off) for the rest of
  training. Extracted it to a crash-safe `Trainer._run_auto_eval(step, tokenizer)`
  (try/except → warn+continue; `finally: model.train()` always restores train
  mode) — consistent with auto_eval's own "never let eval crash training" design.
  Tests: test_auto_eval_hardening.py (5): failure swallowed + train mode restored,
  success runs report+regression, and the three skip guards (no evaluator / no
  tokenizer / should_eval False). Found scanning the training/eval internals.
- **DATA-029** [data-quality/pipeline, low-medium] `done` (2026-06-11) — The
  `PipelineOrchestrator` (cola_coder/pipeline/orchestrator.py — the engine behind
  `run_pipeline.py`) ran `prepare_data.py` WITHOUT `--score`, so the orchestrator-
  driven pipeline produced `train_data.npy` with NO `.weights.npy` sidecar and
  trained on FLAT per-sample weights — silently skipping the quality-weighted
  training the README recommends and that `full_pipeline.py` (and the Pipeline
  Manager) both do. The trainer auto-detects `<data>.weights.npy`
  (trainer.py:235) so the weights were simply never produced. Fixed by adding
  `--score` to `_run_data_prep`. Verified the orchestrator's OTHER stages are
  correctly wired (smoke `--json/--quick`, export `--action gguf-q4` against
  `_ACTION_MAP`, training passes the tracked `--data` path) and that
  `auto_pipeline.py` just delegates to `full_pipeline.py` (inherits TOOL-010/012).
  Tests: test_pipeline_orchestrator.py TestDataPrepArgs (data-prep cmd includes
  --score + essentials). Found this cycle auditing the run_pipeline/auto_pipeline
  orchestration paths.
- **TOOL-012** [tooling/pipeline, high] `done` (2026-06-11) — `full_pipeline.py`
  Stage 9 (reasoning) passed `--config configs/reasoning.yaml` to
  train_reasoning.py, which builds the model from `config.model` — and
  reasoning.yaml's own model section is a FIXED 768/12 (~101M). So for any run
  whose pretrained checkpoint isn't ~that size (tiny/medium/4080_max/large),
  Stage 9 built a mismatched model and the `load_model_only(base_checkpoint)`
  call CRASHED on shape mismatch — the reasoning stage was broken for almost
  every config. Fixed: pass `args.config` (the run's MODEL config) so the model
  matches the checkpoint, and derive `--language`/`--reward`/`--problems`/
  `--group-size` from `config.data.languages` + model size (single typescript →
  `typescript`, single python → `python_exec`, multi → `combined`), mirroring the
  audited Pipeline Manager. Also Stage 6 SFT: `--epochs` now scales with model
  size (tiny→3, larger→2) instead of a hardcoded 2. Extracted the scaling logic
  to a shared, torch-free `cola_coder.model.config.model_scale` (single source of
  truth; `pipeline_menu._model_scale` now delegates to it — DRY). Tests:
  test_full_pipeline_stage_args.py +4 (stage-6 epochs scale 3↔2; stage-9 uses the
  model config not reasoning.yaml; reward derived ts/py/combined; group-size
  present); 93 pipeline_menu tests still green via the delegation. TOOL-011
  narrowed to the remaining (deferred) Stage-1 multi-source design decision.
- **TOOL-010** [tooling/pipeline, high] `done` (2026-06-11) — `full_pipeline.py`
  Stage 5 (`_stage_generate_instructions`) ran `generate_instructions.py` with
  NO args, which drops into the script's INTERACTIVE menu — so an unattended
  `full_pipeline.py --config X` run HANGS FOREVER at stage 5 (the script's
  `main()` only goes non-interactive with `--non-interactive`). Even if it
  didn't hang: (a) no `--languages` → defaulted to `typescript` regardless of
  the config (the project's "never omit --languages" rule), and (b) it wrote to
  the script's default output (`<data_dir>/processed/instructions.jsonl`), NOT
  `data/sft/instructions.jsonl` where Stage 6 reads from — so Stage 6 would
  FileNotFoundError. Fixed by mirroring the audited Pipeline Manager
  (pipeline_menu._stage_generate_instructions): `--non-interactive --source
  huggingface --dataset <config> --languages <config langs> --mode template
  --output data/sft/instructions.jsonl`. This unbreaks the documented end-to-end
  runner. Logged TOOL-011 for the remaining (non-hanging) full_pipeline
  divergences. Tests: test_full_pipeline_stage_args.py (4): non-interactive
  present, config languages forwarded, output == Stage-6 input path, hf
  source+dataset. Found this cycle auditing pipeline stage-arg wiring.
- **MODEL-004** [model/reasoning, low] `done` (2026-06-11) — GRPO per-sequence
  policy log-prob (grpo.py) summed over PROMPT + completion tokens instead of
  completion-only. The prompt is fixed context, not a sampled action, so under
  the policy only completion tokens should count (reference GRPO). It was
  VERIFIED harmless in the current single-step, mean-centered, shared-prompt
  setup (the shared prompt-token log-probs cancel exactly in current−old), but
  it's a non-standard deviation that's fragile to future non-centered/multi-step
  changes. Fixed: added `_completion_logprob_sum(token_log_probs, prompt_len)`
  (token_log_probs[j] scores token j+1, so completion tokens index >= prompt_len
  are scored by indices >= prompt_len-1; returns 0-D 0.0 when there are no
  completion tokens) and used it in BOTH the old-policy (pi_old) and current-
  policy (pi_new) log-prob computations; `prompt_len = len(encode(prompt,
  add_bos=True))` computed once. Behavior is near-identical now (the prompt terms
  cancelled anyway) but the objective is now standard and robust. Tests:
  test_grpo_completion_mask.py (5): masks prompt, prompt_len=1 keeps all, no-
  completion→0, masked+prompt==full, start-at-end→0. Found while scanning the
  trainer/reasoning path; the model/inference/training/export cores all
  re-verified CLEAN this cycle (trainer accum/clip/scaler/scheduler/save/resume/
  eval; parallel_filtered_stream preserves order so DATA-028 holds; INT4 quant
  pack/unpack round-trips).
- **DATA-028** [data-quality/bug, high] `done` (2026-06-11) — Generalized DATA-026:
  `stream_code_data` now ROUND-ROBINS across languages (one sample per language
  per round) instead of yielding each language to exhaustion in sequence. DATA-026's
  per-language quota only balanced the `max_samples` path; but `prepare_data.py`
  (the main stage-2 prep) passes NO `max_samples` and caps DOWNSTREAM at the TOKEN
  level via `tokenize_and_chunk(max_tokens=...)`. With the old sequential stream,
  that token cap was reached while still inside the FIRST language → every other
  language starved (a multi-language `prepare_data --max-tokens N` produced a
  single-language corpus). Round-robin balances regardless of where/how the
  output is capped (samples here OR tokens downstream), since the FIRST N items
  are now an interleaved, balanced mix. Stream order is irrelevant downstream
  (chunks are deduped/scored/curriculum-reordered/shuffled). Single-language
  requests degrade to a plain sequential read; one erroring language is dropped
  from the rotation without killing the rest. Supersedes the DATA-026 quota with
  a single general mechanism (its 5 tests still pass — they assert counts, which
  round-robin preserves). Tests: test_download_balance.py +1
  (balanced interleaved prefix protects the downstream token cap: out[:6]=2/2/2,
  out[:30]=10/10/10). Found this cycle scanning prepare_data's cap path.
- **DATA-027** [data-quality/bug, high] `done` (2026-06-11) — `DatasetCombiner._interleave`
  (data/combine.py) — the live combine step that mixes code/text/math at the
  Qwen2.5-Coder 70/20/10 ratio — distorted that ratio whenever the highest-weight
  source wasn't proportionally the largest. It set `per_ds_target[i] =
  round(weight[i] * total_available)` then CLAMPED to available, so when the
  high-weight source was capped its share shrank while the others kept their
  (too-large) shares. With EQUAL-sized sources (exactly what `--max-samples`
  produces) a 70/20/10 request collapsed to ~53/32/16 — verified empirically.
  Fixed by computing the ratio-preserving target `min_i(available[i]/weight[i])`
  (the largest output where every source's `weight*target` fits in its available
  chunks) → output ratio matches the weights EXACTLY by subsampling
  over-represented sources (interleave deliberately doesn't upsample; that's the
  `weighted` strategy's job). weight=0 sources contribute nothing (BUG-101
  compatible); max_chunks still honored. Verified: equal-sized code/text/math →
  70/20/10 (was 53/32/16); existing interleave tests (loose) still pass and are
  now more correct (equal weights → truly equal contribution). Tests:
  test_combine.py test_interleave_hits_target_ratio_with_equal_sized_sources.
  Found in this cycle's collect_data/combine fresh scan (the natural follow-up
  to DATA-026 — both are live multi-source balance bugs).
- **DATA-026** [data-quality/bug, high] `done` (2026-06-11) — `stream_code_data`
  (data/download.py) — the live HuggingFace ingestion that feeds ALL training
  data — starved later languages under a sample cap. The language loop yielded
  the ENTIRE first language, only stopping at `max_samples` total, so a
  multi-language request with a cap returned `max_samples` of the FIRST language
  and ZERO of the rest (verified: `languages=["typescript","python"],
  max_samples=10` → 10 ts, 0 py). This is LIVE: `collect_data.py:266` passes the
  full multi-language config list + `--max-samples` to it, so
  `collect_data --config medium.yaml --max-samples N` (medium is multi-language)
  produced a corpus of only the first language — a silent, severe language
  imbalance in the training set. Fixed: each language now gets an even share of
  the REMAINING budget (`ceil((max_samples-count)/remaining_langs)`), so the cap
  is split fairly AND an under-full language rolls its leftover forward to later
  ones (total still reaches `max_samples`). Single-language and no-cap requests
  are byte-for-byte unchanged. Tests: test_download_balance.py (5): 2-lang 5/5,
  3-lang balanced, under-full rolls forward (2/8), single-lang unchanged, no-cap
  yields all. Found in this cycle's download.py fresh scan.
- **DATA-024** [data-quality/bug, medium] `done` (2026-06-11) — `CrossDatasetDeduplicator.
  deduplicate_pair` (data/dedup.py) `np.save`d the deduped output while the
  `secondary` source was still open via `np.load(..., mmap_mode="r")`. Its
  `output_path` DEFAULTS to `secondary_path` (documented "overwrites
  secondary_path"), so the default in-place mode wrote over a still-mmapped file
  → `PermissionError`/`OSError [Errno 22]` on Windows (the primary platform) —
  the SAME class as DATA-004/DATA-019, which the sibling `dedup_npy_file`
  already guards but `deduplicate_pair` did not. Verified empirically: default
  `deduplicate_pair(prim, sec)` crashed before, succeeds after. NOT currently
  live-triggered (the one caller, combine_datasets.py:279, passes a distinct
  `output_path`), but the documented default behavior was broken and it also
  leaked the secondary mmap handle. Fixed with the DATA-004 idiom: release the
  `secondary` mmap (`_mmap.close()` + del) before `np.save` (`deduped` is already
  an in-RAM `np.array` copy). Tests: test_dedup.py TestDeduplicatePairInPlace (2):
  default overwrite no-crash + correct surviving row; distinct output_path keeps
  secondary. Found in this cycle's dedup.py fresh scan.
- **MODEL-007** [model/tokenizer/capability, medium] `done` (2026-06-11) — Adopted
  digit-splitting in the BPE tokenizer (train_tokenizer.create_tokenizer): the
  pre-tokenizer is now `Sequence([Digits(individual_digits=True),
  ByteLevel(add_prefix_space=False)])`. Plain ByteLevel BPE merged digit runs
  (`"12345"`→1 token), which hurts numeric handling; LLaMA 3 / Qwen2.5-Coder /
  DeepSeek all split digits to one-token-per-digit. This matters here because
  the data mix deliberately includes ~10% math (open-web-math) plus code full of
  indices/versions/ports. A 2025-26 best practice with clear empirical support,
  modular and low-risk: it ONLY affects FRESHLY trained tokenizers — existing
  `tokenizer.json` files load via `Tokenizer.from_file` unchanged, so no
  checkpoint breaks (a new tokenizer would require re-prep + retrain anyway,
  which is an intentional stage-1 action). Verified empirically: `"12345"` →
  1 merged token before, 5 single-digit tokens after; round-trip intact. Tests:
  test_tokenizer.py test_digits_split_individually (per-digit tokenization, no
  multi-digit token, round-trip); all 37 tokenizer-dependent tests still green
  (FIM/gguf/ollama/sft/reasoning persist). Doc note added to CLAUDE.md.
  Found in this cycle's tokenizer-training fresh scan.
- **DATA-023** [data-quality, low] `done` (2026-06-11) — `AddMetadata._guess_language`
  (data/transforms/metadata.py) tagged each record's `estimated_language` from a
  CONTENT heuristic only, ignoring the file EXTENSION that the local/github
  sources already put in metadata (`extension`/`file_path`). Extension is far
  more reliable (content heuristics tie between TS/JS and miss short files), and
  the project's DRY rules mandate extension-based detection. Made `_guess_language`
  extension-first (new `_EXT_LANG` map mirroring language_detect's TS/JS ext sets
  + python/go/rust/java) with the content heuristic as fallback; `apply` now
  derives the extension from `extension` or `file_path`/`path`. Backward-compatible:
  no-extension records still use the content heuristic, and the
  existing-value guard is unchanged. `_guess_language` previously had ZERO
  direct tests. Tests: test_pipeline.py TestAddMetadata (+3): .ts extension beats
  Python-looking content, extension derived from file_path (.rs→rust), unknown
  extension falls back to content. NOTE: AddMetadata is part of the DataPipeline
  system that DATA-021 has not yet wired into the live collector — this is
  correct-when-used, found in this cycle's audit of the TOOL-008 modules. The
  rest of that audit (transforms/whitespace, sources/local/huggingface/mixed,
  github clone security [hooks disabled, no submodule recurse, no shell],
  language_detect) was CLEAN.
- **TOOL-009** [tooling/lint-hygiene, low-medium] `done` (2026-06-11) — The
  documented lint gate (`ruff check src/ scripts/ tests/`, per CLAUDE.md) was
  BROKEN: 21 pre-existing errors, all in `src/cola_coder/data/`. They went
  unnoticed because prior cycles only ran ruff on the *specific files touched*,
  never the whole tree (and several offending files were among the TOOL-008
  untracked set, invisible until last cycle). Breakdown: 11× E741 (ambiguous
  `l` loop var) + 1× F841 (dead `sample_text`) in quality_filter.py (the
  data-quality GATE, TEST-001-covered); F401 dead imports (`multiprocessing`,
  `typing.Any`, `IM_START`) auto-removed; 6× F401 availability-probe imports
  (sources/__init__.py ×5, heuristic_scorer is_available ×1) annotated
  `# noqa: F401` matching the existing filters/__init__.py convention (removing
  them would break @register_source registration + availability detection).
  Renamed all comprehension-local `l`→`ln` (behavior-neutral; each is
  comp-scoped). `ruff check src/ scripts/ tests/` now passes (21→0). Validated:
  test_quality_filter + test_filters + test_quality_classifier +
  test_scorer_protocol (102) and security/score-data/checkpoint suites all green
  — the quality-gate renames changed no behavior. Found this cycle ruff-scanning
  the newly-tracked TOOL-008 modules. (Going forward, run full-tree ruff, not
  just touched files.)
- **TOOL-008** [tooling/repo-integrity, HIGH] `done` (2026-06-11) — 25 core
  `.py` source files under `src/cola_coder/data/` were UNTRACKED (never
  committed) — the long-term fallout of the original unanchored `data/`
  gitignore rule: it matched `src/cola_coder/data/` and silently kept git from
  ever adding files there. TOOL-008's structural fix (anchoring to `/data/`)
  stopped NEW files being ignored but never back-filled the already-untracked
  ones. The orphaned set includes modules that ALREADY-COMMITTED code imports —
  e.g. `data/registry.py` is imported by the committed `filters/pii.py`
  (`from cola_coder.data.registry import register_filter`), and `pipeline.py`,
  `sources/*` (github/huggingface/local/mixed/self_align/software_heritage/
  instruction_gen), `filters/{__init__,length,quality}.py`, `curation/*`
  (test_runner/test_scorer/docker_sandbox), `transforms/*`, `fim.py`,
  `fim_dataset.py`, `mixing.py`, `combine.py`, `dedup.py`. So a FRESH CLONE
  would be missing them → import errors across the data pipeline, filter
  registry, FIM, curation, and sources. Discovered this cycle via `git stash`
  surfacing the untracked list. Verified: all 25 are first-party `.py` source
  (355B–43KB, no data/secrets/binaries — secret-scanned clean), not ignored,
  and import cleanly. Committed all 25 to version control. Orphaning was fully
  contained to `src/cola_coder/data/` (no untracked source elsewhere in
  src/scripts/tests). Follow-up DATA-021 (DataPipeline not wired into the live
  collector) is unaffected — these are now at least tracked/backed-up.
- **SEC-005** [security/sandbox, medium] `done` (2026-06-11) — The sandbox's
  native-timeout handler (`data/scorers/sandbox.py`) killed runaway processes
  with `taskkill /F /T /IM <cmd[0]>` — by IMAGE NAME. That (a) terminates EVERY
  process sharing the name system-wide (e.g. ALL `node` processes, including the
  VS Code extension host) when cmd[0] is a bare name like `node`/`tsc`, and (b)
  is INEFFECTIVE when cmd[0] is a full path (taskkill /IM can't match a path),
  leaking orphaned grandchildren of timed-out untrusted code. Fixed: `_run_native`
  /`_run_docker` now use `subprocess.Popen` so we hold the child PID and kill the
  whole tree SCOPED TO THAT PID — Windows `taskkill /F /T /PID`, POSIX
  `start_new_session=True` + `os.killpg` — via a shared `_finish_proc`/
  `_kill_proc_tree`. Never touches unrelated processes; reaps grandchildren.
  Preserved the contract (timeout→rc -1 + "Timeout", missing cmd→rc -2,
  counters, cwd isolation). Also fixed 2 pre-existing F821 (quoted
  SecurityConfig/ScoringAuditLogger annotations) with a TYPE_CHECKING import.
  Docker isolation flags (--network none/--cap-drop ALL/nobody/--read-only/
  pids/memory) verified unchanged. Tests: test_sandbox.py TestTimeoutKillIsPidScoped
  (+2: timeout kills by PID int not image name; dead-pid is safe); updated 4
  docker-flag tests to patch Popen. Found in this cycle's sandbox audit.
- **DATA-022** [data-quality/bug, low-medium] `done` (2026-06-11) — The JS/TS
  modernness scorer (`CodeScorer._score_modernness_js_ts`, features/code_scorer.py)
  detected deprecated loose equality with `re.findall(r'(?<!=)==(?!=)', code)`,
  which ALSO matched the `==` INSIDE `!==` (strict inequality — a MODERN idiom):
  in `!==` the `==` is preceded by `!` (lookbehind only excluded `=`) and followed
  by a non-`=`. So clean TypeScript using `!==` was scored as deprecated loose
  `==`, inflating deprecated_points and DEFLATING the modernness sub-score →
  lower training weight. Since CodeScorer feeds `score_data.py`'s `.weights.npy`
  and this is a TS-PRIMARY project where `!==` is ubiquitous, it systematically
  under-weighted good modern TS. Verified empirically: `a !== b && c !== d`
  matched 3 false `==` before, 0 after. Fixed by excluding `!` from the
  lookbehind: `(?<![=!])==(?!=)` — still catches real `==`, still ignores `===`.
  Found in this cycle's data-scorer fresh scan (heuristic/curriculum scorers
  otherwise clean; noted a tiny-dataset edge in curriculum `_compute_phases`
  where n<num_phases → np.min([]) — low sev, not fixed). Tests:
  test_code_scorer_modernness.py (3): strict-inequality not penalized vs loose,
  !==-only code beats ==-only code, genuine == still penalized vs ===.
- **MODEL-006** [model/moe/perf, low] `done` (2026-06-11) — Vectorized the MoE
  load-balancing loss (`MoEFFN._load_balancing_loss`, features/moe_layer.py). It
  counted per-expert top-k assignments with an O(top_k * num_experts) Python
  double-loop of per-expert `.sum()` kernels — run on EVERY training step (it's
  the aux-loss hot path; gated to training). Replaced with a single
  `torch.bincount(top_k_indices.reshape(-1), minlength=num_experts)` — bit-exact
  equivalent (verified empirically + by test against the original loop as oracle),
  far fewer kernel launches. Correctness-preserving cleanup of a real
  inefficiency in the MoE-training path (now more relevant after BUG-112 made MoE
  inference actually work). Tests: test_moe_aux_loss.py (3): matches the
  double-loop reference across 5 seeds, handles zero-token experts (minlength),
  stays 0 in eval. Model-core audit this cycle (attention/rope/RMSNorm/batched
  GRPO generation) otherwise CLEAN; logged the YaRN mscale gap as MODEL-005.
- **BUG-112** [model/moe, high] `done` (2026-06-11) — MoE inference collapse.
  `MoEFFN.forward` (features/moe_layer.py) applied capacity-based token dropping
  UNCONDITIONALLY with `capacity = int(capacity_factor * num_tokens /
  num_experts)`. During single-token autoregressive DECODE `num_tokens == 1`, so
  `capacity = int(1.25 * 1 / 8) = 0` and `token_indices[:0]` dropped EVERY
  routed-expert contribution — the MoE silently collapsed to just its shared
  expert(s) at generation time (verified: a 1-token eval forward with
  num_shared_experts=0 produced an all-zero output). So an upcycled/fine-tuned
  MoE checkpoint (stages 7/7.5) ran inference with its entire routed-expert
  capacity SILENT — drastically degraded generation. The formula also omitted
  `top_k`, so per-expert capacity was top_k× below the expected load
  (`num_tokens * top_k / num_experts`), over-dropping even in training. Fixed:
  gate capacity dropping on `self.training` (inference processes every token —
  the generator runs model.eval(), generator.py:96) and use the standard
  top_k-aware formula with a `1 <= capacity < len(token_indices)` guard so tiny
  batches never round to a zero/over-aggressive cap. Improves BOTH inference
  (no collapse) and training (correct, less aggressive dropping). Tests:
  test_moe_capacity.py (5): single-token decode routed experts contribute,
  eval drops no tokens, training doesn't over-drop with the top_k formula,
  capacity_factor=0 disables dropping, shared-expert sanity.
- **DATA-020** [data-quality/wiring+bug, medium] `done` (2026-06-11) — Two coupled
  fixes in the modular filter system. (1) WIRING: 5 of 8 FilterPlugins (`pii`,
  `content`, `license`, `syntax`, `deduplication`) conformed to the interface and
  were exported + documented as "composable filter plugins," but were NEVER
  registered via `@register_filter` — and the registry is the ONLY way the
  config-driven `DataPipeline` instantiates filters by name (verified: none are
  constructed directly anywhere in src/scripts). So the privacy (PII) and license
  filters were orphaned/unreachable by config. Added `@register_filter(name)` to
  all 5 (registry now: content/deduplication/length/license/pii/quality/
  quality_classifier/syntax). Purely additive — only used when a config names
  them. (2) BUG: PIIFilter's `password_assignment` regex matched the bare `pass`
  keyword inside unrelated identifiers — `bypass`/`compass`/`surpass` (verified
  empirically) were false-flagged as PII and the whole file dropped (DATA-015-class
  over-rejection shrinking the corpus). Restricted the bare `pass` alternative
  with a `(?<![A-Za-z])` lookbehind so it matches standalone `pass`/snake_case
  `db_pass` but not letter-prefixed suffixes; `password`/`passwd`/`pwd` keep
  matching after `_`/word starts (snake_case secrets still caught). Verified real
  secrets (password/db_pass/pass/PASSWORD/aws/PEM) still detected. Also removed 2
  pre-existing dead imports (math, torch) in quality_classifier.py while in the
  package. The dormant DataPipeline-not-in-live-path issue logged as DATA-021.
  Tests: test_filter_registry_and_pii.py (8): all 5 registered + constructible +
  setup, pii setup applies config, identifiers-ending-in-pass kept, real password
  assignments still flagged, other patterns (aws/PEM) unaffected.
- **EVAL-006** [eval/correctness, medium] `done` (2026-06-11) — `evaluate_solution`
  (evaluation/runner.py) executes Python (execute_code → `python main.py`) but
  ignored `problem.language`, so a `language="typescript"` `CodingProblem`
  (TYPESCRIPT_PROBLEMS, reachable via `ProblemSet.filter_by_language`) would have
  its TS test_code piped through the Python interpreter → misleading SyntaxError
  → always-fail, silently deflating pass@k for this TS-primary project. Latent
  (both current callers load Python-only `get_all_problems()`), but a real
  footgun. Fixed with a fail-loud `problem.language` guard (cf. TOK-001/DATA-003):
  returns a clear "LANGUAGE NOT SUPPORTED … use ts_benchmark" verdict naming the
  task_id, instead of a confusing false-fail, before any execution. Actually
  evaluating TS (TS extract_function + sandboxed TS execution) split to EVAL-007.
  Tests: test_eval_runner_language.py (4, execute_code stubbed): TS not executed
  as Python + clear message, Python still reaches execution, empty-test guard
  still fires, default language=python runs. (Found in last cycle's server scan;
  fixed this cycle.)
- **BUG-111** [inference/bug, high] `done` (2026-06-11) — The `/v1/fim` endpoint
  (server.py) — which powers the VS Code extension's inline ghost-text
  completions — returned the prefix+suffix code PREPENDED to the actual infill on
  EVERY request. `generate()` returns `decode(prompt_ids + new_ids)` and decode()
  STRIPS special tokens, so `result` never contained the FIM markers; the infill
  extraction `result[len(fim_prompt):] if result.startswith(fim_prompt) else
  result` checked against the MARKER-form `fim_prompt` (`<|fim_prefix|>…`), whose
  startswith could NEVER match the markers-stripped `result` → it ALWAYS hit the
  `else` and returned the WHOLE prefix+suffix+infill. So ghost text re-inserted
  the surrounding code — the INFER-001/009 prompt-echo class, still live after
  INFER-007 fixed the prompt CONSTRUCTION (this is the OUTPUT extraction).
  NOTE: a naive `strip_prompt_prefix(result, fim_prompt)` would ALSO fail here
  (common prefix 0 — result starts with code, fim_prompt with `<`). Fixed by
  stripping the DECODED prompt: `strip_prompt_prefix(result,
  decode(encode(fim_prompt)))` — re-encode+decode yields the exact markers-
  stripped prompt content, so the longest-common-prefix helper removes
  prefix+suffix and leaves only the generated middle. Found in this cycle's fresh
  scan of server.py. Tests: test_fim_prompt.py TestServerFimInfillExcludesPrompt
  (a _FaithfulGenerator reproducing the real decode(prompt+new) behaviour with
  the real tokenizer): infill == expected strip, != full result, no `const`
  prefix-echo, and asserts `not full.startswith(fim_prompt)` documenting why the
  old check always failed. (The same scan's flagged "streaming startswith leak"
  in _stream_chat/_stream_completion is a VERIFIED FALSE POSITIVE — see Not-a-bug.)
- **BUG-110** [reasoning/reward, high] `done` (2026-06-11) — The `typescript` and
  `combined` GRPO rewards (reward_registry `_typescript_reward`/`_combined_reward`)
  scored the RAW generation — `<think>reasoning</think>code` — INCLUDING the
  thinking trace. `TypeCheckReward.score`/`CombinedReward.detailed_score` don't
  strip thinking, so tsc/the syntax+completeness signals treated the `<think>`
  tags as TypeScript → syntax error / depressed score on EVERY reasoning-
  formatted generation. The `python_exec` reward already calls `extract_thinking`
  and runs only the code; the TS path (the RECOMMENDED reward for the TS-primary
  project: single typescript → `--reward typescript`) did not. Net effect: near-
  constant ~0 reward → ~zero variance → GRPO's std<1e-4 collapse-guard SKIPS the
  update → TS reasoning training silently makes no progress. Fixed: both wrappers
  now `extract_thinking(gen)` and score the CODE only. Same fix closes REWARD-001
  — they were also discarding `max_thinking_tokens`; both now apply the shared
  `thinking_length_penalty` (extracted to reward.py, DRY with compute_reward) so
  all three rewards discourage runaway reasoning identically, and add
  `thinking_length`/`length_penalty` to their info dicts. Reward kept in [0,1]
  (clamp − penalty, floored at 0) per the existing GRPO-stability invariant.
  Tests: test_reward_thinking_aware.py (9, scorers stubbed so no tsc/Node needed):
  helper under/at/over/capped, TS strips thinking before scoring + no-thinking
  unchanged + length penalty + unit-range floor, combined strips + penalty.
- **REWARD-001** [reasoning/reward-quality, medium] `done` (2026-06-11) — Folded
  into BUG-110 (same root: TS/combined rewards weren't thinking-aware). The
  shared `thinking_length_penalty` is now applied on all three reward paths.
- **BUG-109** [reasoning/reward-quality, medium] `done` (2026-06-11) — The GRPO
  format bonus in `compute_reward` (reasoning/reward.py — the DEFAULT
  `python_exec` reward, in active training use) contradicted its own comment.
  It claimed to "Check that thinking comes BEFORE the code" but only tested
  `think_end < len(generated_text) - 10` — i.e. that 10+ characters, WHITESPACE
  INCLUDED, followed `</think>`. So `code<think>reasoning</think>   ` (code
  FIRST, then trailing spaces) wrongly earned the +0.1 bonus, and the model got
  no signal that reasoning must precede the answer — and trailing whitespace
  alone satisfied the "code follows" intent. Same class as BUG-102 (a sloppy
  reward heuristic that did not match its docstring). Fixed to require BOTH
  (1) the first non-whitespace content is `<think>` (thinking-first) and (2)
  real non-whitespace content after `</think>` (the answer). Discovered via a
  fresh scan of the reasoning reward path; logged the sibling REWARD-001
  (max_thinking_tokens ignored by TS/combined rewards) as open. Tests:
  test_reward_format_bonus.py (6, execute_code stubbed so no host execution):
  proper think-first→bonus, leading-whitespace→bonus, code-before-thinking→none
  (the bug), think-first-but-only-whitespace-after→none, no-tokens→none,
  open-tag-only→none.
- **TOOL-007** [tooling/repo-hygiene, medium] `done` (2026-06-11) — Discovered
  while committing DATA-019: `.gitignore` had unanchored `data/` and
  `checkpoints/` rules, so `data/` ALSO matched the source package
  `src/cola_coder/data/`. Existing tracked files were safe (tracked overrides
  ignore), but a NEW file added under `src/cola_coder/data/` would be SILENTLY
  ignored by `git add` — a real footgun (a future data module could appear to
  commit but be dropped). Anchored both to repo root (`/data/`, `/checkpoints/`)
  so they match only the top-level dataset/checkpoint dirs. Verified:
  `data/foo.npy` + `checkpoints/foo` still ignored; a new
  `src/cola_coder/data/*.py` is no longer ignored.
- **DATA-019** [data-quality/bug, high] `done` (2026-06-11) — `tokenize_and_chunk`
  (data/preprocess.py — data-prep STAGE 2, the core tokenizer→.npy path) crashed
  at finalization on Windows (the primary platform). It did `data =
  mmap_data[:num_chunks]` (a memmap VIEW), saved it, `del mmap_data`, then
  `tmp_path.unlink()` — but the view kept the temp file mapped, so the UNGUARDED
  unlink raised `PermissionError [WinError 32]` AFTER the output .npy was already
  written, leaving an orphaned `_tmp.npy` and a confusing trailing crash. Same
  class as DATA-004 (which fixed only the dedup path); the main prepare path was
  never fixed and had ZERO tests. Verified empirically on this machine (both the
  bug and the fix). Fixed with the DATA-004 idiom: save from a throwaway slice
  (no long-lived view), release the handle (`_mmap.close()` + del + gc.collect),
  then unlink in a try/except (a residual lock now only warns, never crashes a
  finished run). Also dropped an unused `os` import + an f-string nit while in
  the file. Tests: test_preprocess_finalize.py (2): full run completes + temp
  removed + manifest written; empty-iterator path.
- **DATA-018** [docs/discoverability, low] `done` (2026-06-11) — DATA-012 wired
  dynamic FIM via `DataConfig.fim_rate`, but the knob lived only in the dataclass
  default — absent from config templates and docs, so users couldn't discover
  it (an undiscoverable feature is nearly a dead one). Surfaced a documented
  `fim_rate: 0.0` in the `data:` section of all 5 main configs (tiny/small/
  medium/4080_max/large) + a CLAUDE.md note covering prep-time vs dynamic FIM.
  All configs verified to still parse with fim_rate defaulting to 0.0 (off).
  Tests: test_config_fim_rate.py (6): every config exposes fim_rate, off by
  default, psm in [0,1], and the YAML surfaces the knob. (Fresh scan this cycle
  confirmed the untested feature modules data_balancer/multi_file_context/
  overfitting_detector are clean-or-unwired — the auto-discoverable bug surface
  is exhausted; remaining items INFER-011/DATA-006 need a training/model
  validation channel.)
- **INFER-010** [inference/bug, medium] `done` (2026-06-11) — `InteractiveChat`
  (features/multi_turn_chat.py, menu-wired, untested) extracted the assistant
  reply via `rsplit("### Assistant:", 1)` with a fallback of
  `assistant_text = response` — the WHOLE prompt+completion — when the marker
  wasn't found, and would also DROP the reply's start if the model emitted an
  assistant marker itself (rsplit takes after the LAST one). Refactored the
  inline logic into a testable `_extract_reply` using the canonical
  `strip_prompt_prefix` (the completion IS the reply; longest-common-prefix
  strip never echoes the whole prompt and is decode-trap-safe; DRY with the
  server / batch inference), plus a defensive truncation at a model-emitted
  user/assistant marker. Tests: test_multi_turn_chat_extract.py (4): clean
  completion, drift no-full-echo, truncation at a run-on user marker, no echo
  when the marker is absent. Discovered INFER-011 (the underlying ChatML format
  mismatch) during this — logged as open.
- **INFER-009** [inference/bug, medium] `done` (2026-06-11) — `BatchInference.run`
  (features/batch_inference.py, menu-wired, untested) stripped the prompt with
  the naive `output[len(prompt):] if output.startswith(prompt) else output` —
  the exact INFER-001 prompt-echo leak in a different code path. `generate()`
  returns `decode(prompt_tokens + new_tokens)`, and on any BPE round-trip
  mismatch (BOS render / whitespace / boundary merge) `startswith` is False, so
  the ENTIRE prompt was echoed back as the "generated" output (and the est.
  token count + success-rate metrics were inflated by it). Fixed to use the
  canonical `inference.text_utils.strip_prompt_prefix` (longest-common-prefix —
  a mismatch costs at most a few boundary chars, never the whole prompt; DRY
  with the server). Tests: test_batch_inference_prompt_strip.py (3): clean
  strip, drift doesn't echo the full prompt, output matches the canonical helper.
- **BUG-108** [reasoning/bug, medium] `done` (2026-06-11) — `SelfPlayTrainer.
  train_episode` (reasoning/self_play.py, menu-wired, previously ZERO tests)
  fine-tuned on a PASSING solution TWICE: when `best_reward > 0.9` it called
  `_update_on_solution` and `break`, then the post-loop "final update"
  (`if best_solution and best_reward > 0.3`) fired AGAIN (0.9 > 0.3) on the same
  solution — double the gradient steps, over-weighting passing solutions. Fixed
  with an `updated` flag so the post-loop update is skipped when the in-loop
  >0.9 update already ran (exactly-once semantics). Verifiable without training
  by counting `_update_on_solution` calls. Tests: test_self_play_update.py (3):
  passing→1 update, partial (0.3<r≤0.9)→1, none (r≤0.3)→0.
- **DATA-012** [capability/training, medium] `done` (2026-06-11) — Wired dynamic
  (train-time) Fill-in-the-Middle end-to-end (StarCoder2-style per-epoch FIM
  split variety), the optional follow-up to DATA-011. Added `DataConfig.fim_rate`
  (default 0.0 = off, backward-compatible) + `fim_psm_rate`; a `FIMTrainingCollator`
  in dataset.py that applies the canonical length-preserving `FIMTransform`
  per-batch and PRESERVES quality weights (composes with WeightedCodeDataset);
  `create_dataloader(fim_rate, fim_ids, fim_psm_rate)` wraps the collator when
  enabled. The Trainer resolves the `<|fim_*|>` ids from the sibling tokenizer
  (`_resolve_fim_ids`) and auto-DISABLES with a warning — never a silent no-op,
  never a crash — when fim_rate=0, the tokenizer is missing, or it lacks FIM
  tokens (reads ids, never adds them: a model trained on a fixed vocab can't use
  out-of-vocab FIM ids). The deferred "needs a training run to avoid a silent
  no-op" concern is met by unit tests proving the dataloader actually emits FIM
  tokens. Tests: test_dynamic_fim.py (8): collator applies FIM / rate-0 no-op /
  weights preserved, create_dataloader batches contain FIM ids at rate>0 and
  none at rate=0, and the trainer id-resolution (present / missing tokenizer /
  no-fim-tokens). NOTE: whether dynamic FIM IMPROVES eval quality still warrants
  a smoke training run — the wiring/no-op-safety is what's validated here.
- **DATA-017** [data-quality/bug, low-medium] `done` (2026-06-11) —
  `QualityFilterPlugin` (data/filters/quality.py) fell back to
  `record.metadata.get("languages")` (PLURAL) when no config languages were
  set, but sources emit the canonical SINGULAR `language` key (DATA-007/008) —
  so the fallback was ALWAYS None and the language-aware quality checks
  (e.g. TS-specific syntax) never engaged on a per-record basis. Fixed: prefer
  the pipeline-config `languages` list, else fall back to the record's singular
  `language` wrapped in a list (filter_code expects list[str]). Closes the
  filter-plugin sweep (syntax.py + quality.py audited; syntax.py clean modulo a
  caught deep-recursion pass-through). Tests: test_quality_filter_plugin.py (5):
  config precedence, singular-language fallback, None when absent, plural key no
  longer picked up, mode forwarding.
- **DATA-016** [data-quality/config, low-medium] `done` (2026-06-11) — Phantom
  config in `ContentFilter` (data/filters/content.py): `max_autogen_markers`
  was stored in `__init__`/`setup()` and documented ("Max autogen markers
  allowed (0 = reject any)"), but `_check_autogenerated` returned on the FIRST
  matched marker regardless of it — so any value > 0 was a silent no-op (the
  project's recurring silent-no-op-config pattern; cf. MEM-001, reasoning.yaml).
  Fixed: count the distinct AUTOGEN_MARKERS present in the header and reject
  only when the count EXCEEDS `max_autogen_markers`. Default (0) behavior
  unchanged (reject on any marker); the knob now works for >0. The sibling
  knobs `max_long_string_ratio`/`max_avg_line_length` were verified already
  wired. Tests: test_filters.py TestContentFilter (+2): default rejects a single
  marker; max=1 tolerates one marker but rejects two.
- **DATA-015** [data-quality/bug, medium] `done` (2026-06-11) — `LicenseFilter`
  rejected validly-licensed training data due to CASE-SENSITIVE SPDX matching.
  `check` did `if normalized in self.PERMISSIVE` against a canonically-cased set
  ({"MIT","Apache-2.0",...}); the alias map only normalized a handful of
  lowercase spellings. SPDX identifiers are case-insensitive (spec §10.1), so a
  permissive license arriving in non-canonical case that wasn't in the alias map
  — `"apache-2.0"`, `"zlib"`, `"bsd-3-clause"`, `"mit-0"`, etc. — was silently
  REJECTED, discarding usable permissively-licensed data and shrinking the
  corpus. Fixed: membership check is now case-insensitive (`normalized.lower()
  in {s.lower() for s in PERMISSIVE}`); the alias map still handles alternate
  SPELLINGS ("apache2"). Verified it does NOT flip GPL/AGPL/unknown to
  permissive. Tests: test_filters.py TestLicenseFilter (+2): case-insensitive
  permissive accepted, non-permissive still rejected regardless of case.
- **TEST-002** [test-quality/checkpoint, medium] `done` (2026-06-11) — Audited
  the training internals (trainer step, infinite dataloader, Muon/AdamW
  optimizer + decoupled WD grouping, cosine/WSD scheduler with the fp16
  scaler-skip guard, per-sample-weighted CE + PaLM z-loss) — all VERIFIED
  correct, no bugs. But the custom `Muon` optimizer (non-standard serialized
  state: momentum buffers + embedded-AdamW moments) is on the CRITICAL
  checkpoint path and its round-trip test only asserted `load_state_dict`
  "must not raise" — which would pass even if resume silently started from cold
  state. Hardened: added `test_muon_state_dict_restores_buffers` (every tensor
  state entry + AdamW step counter actually restored) and
  `test_muon_resume_reproduces_continuous_step` (resuming from a saved state and
  taking the next step yields the SAME weight update as never stopping — the
  real resume guarantee). Verified the round-trip is in fact correct (max param
  diff 0.0 after the resumed step); the tests lock it against future regressions
  (non-serializable state, param-group reorder).
- **INFER-008** [inference/bug, medium] `done` (2026-06-11) — `StreamingGenerator`
  (features/streaming_generation.py — a menu-wired, ZERO-test per-token streaming
  telemetry feature) was a diverged duplicate that had NEITHER of two fixes the
  canonical `generator.generate_stream` got: (1) it decoded text per-token
  (`decode([next_token])`), which mangles byte-level BPE — a single token can be
  a partial multi-byte UTF-8 sequence, decoding to replacement chars / wrong
  spacing in isolation; (2) it reduced multi-token stops to their first token
  (the INFER-006 bug), halting far too early. Fixed: use full-decode-diff for
  the emitted text (decode the full sequence, yield the incremental delta) and
  the shared hybrid stop matcher. Extracted `partition_stops(tokenizer,
  stop_tokens)` to module scope in generator.py (CodeGenerator._partition_stops
  now delegates — DRY) so both streamers behave identically; added the same
  hold-back so a multi-token stop can't leak its prefix. Tests:
  test_streaming_generation.py (5): multi-token stop not truncated early, eos /
  no-stop passthrough, a merge-tokenizer test proving full-decode-diff recovers
  a char that per-token decode loses, and StreamToken metadata. (The test suite
  caught a self-inflicted regression — an orphaned forward block — during the
  edit; fixed before commit.)
- **INFER-007** [inference/bug, high] `done` (2026-06-11) — The `/v1/fim`
  endpoint (server.py) — which powers the VS Code extension's inline ghost-text
  completions — built its prompt as `decode(encode_fim(prefix, suffix))`.
  `decode` skips special tokens, so the `<|fim_prefix|>`/`<|fim_suffix|>`/
  `<|fim_middle|>` markers were STRIPPED, leaving the model a plain
  `prefix+suffix` with NO fill-in-the-middle structure → FIM/inline completion
  was completely broken (verified empirically: `decode(encode_fim("A","B"))` ==
  `"AB"`). Fixed by adding `CodeTokenizer.fim_prompt(prefix, suffix)` which
  builds the marker-string form (`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`
  via `id_to_token`), so `generate()`'s re-encode recovers the exact FIM ids
  (`encode(fim_prompt(p,s)) == encode_fim(p,s)`, asserted). Server now calls
  `fim_prompt`. Tests: test_fim_prompt.py (4) — markers present + ordered,
  re-encode recovers ids, decode strips them (documents the bug), and a
  server-level capture proving `generate()` receives the markers.
- **TOOL-006** [tooling/export, medium] `done` (2026-06-11) — Ollama Modelfile
  used the wrong chat template. `OllamaExporter._CHAT_TEMPLATE` + stop params
  were LLaMA-3 style (`<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>` and
  stop `<|eot_id|>`/`<|end_of_text|>`) — tokens NOT in cola-coder's vocab. The
  model is trained on ChatML (`<|im_start|>{role}\n{content}<|im_end|>`, per
  chat_template.py), so an exported Ollama model would fragment the template,
  never see its trained chat tokens, and never hit a real stop token — broken
  instruction following (the BUG-106 family, on the Ollama side). Fixed: the
  Modelfile now emits the ChatML template + `stop <|im_end|>` / `stop <|eos|>`.
  Kept ollama_export torch-free (literal tokens + a test asserting they match
  chat_template.IM_START/IM_END and the base SPECIAL_TOKENS — the DATA-014
  pattern). Tests: test_ollama_chatml.py (5): ChatML tokens present, no stale
  LLaMA-3 tokens, stop params correct, tokens in sync with the canonical source.
- **TOOL-005** [tooling/export, high] `done` (2026-06-11) — GGUF export embedded
  NO tokenizer vocabulary. Both writers emitted `tokenizer.ggml.model="llama"`
  with bos/eos ids but no `tokenizer.ggml.tokens`/`token_type`/`merges` — and
  "llama" is the wrong model for cola-coder's byte-level BPE (should be "gpt2").
  A GGUF with no token list cannot be loaded by llama.cpp/Ollama at all, so
  EVERY exported model was unusable (TOOL-003 had audited the quant block layout,
  not the vocab). The builtin writer also couldn't express GGUF arrays. Fixed:
  added GGUF array support to `_encode_kv` (type 9), a torch-free
  `build_gguf_vocab(tokenizer_json)` (byte-level BPE → gpt2 model, id-ordered
  token list, CONTROL/UNKNOWN/NORMAL types, "a b" merges, special-token ids),
  and `_resolve_vocab` which loads the checkpoint's tokenizer via
  `resolve_tokenizer_path` (honouring the SFT/reasoning expanded tokenizer from
  BUG-106/107) and refuses to embed a vocab whose size ≠ the model's embedding
  rows. Both writers now embed it (builtin via array metadata; gguf-package via
  add_token_list/types/merges). When no tokenizer resolves, export still
  succeeds but warns LOUDLY that the GGUF has no vocabulary (no silent broken
  artifact). Tests: test_gguf_vocab.py (8) — vocab builder, array encoder
  round-trip, and end-to-end export whose GGUF KV section is parsed back to
  verify model=="gpt2", token count == vocab_size, merges present, plus
  missing-tokenizer and vocab-mismatch fallbacks.
- **MODEL-003** [model/capability, low-medium] `done` (2026-06-11) — Finished the
  MoE fine-tune orchestration (stage 7.5). MODEL-001 made `train.py --resume
  <moe_dir>` fine-tune an upcycled MoE, but there was no dedicated entry or
  auto-config — the user had to hand-pick a low LR / short schedule, and the MoE
  upcycling menu dropped them with no follow-up (experts stay identical copies
  of the dense FFN until differentiated). Added torch-free
  `derive_moe_finetune_config(cfg, lr_fraction=0.1, step_fraction=0.15)` in
  model/config.py (the standard sparse-upcycling recipe: a fraction of steps at
  a fraction of the LR; rescales only the training section, clamps min_lr below
  the lowered peak, keeps warmup short, never 0 steps; input not mutated) and a
  Post-Training menu entry `_moe_finetune_menu` (3 presets) that derives a config
  → writes configs/auto/{stem}_moe_ft.yaml → runs `train.py --resume`. Tests:
  test_moe_finetune_config.py (10); the static menu-wiring test validates the new
  train.py --config/--resume invocation. Does NOT start training (orchestration +
  config only).
- **SEC-004** [security, medium] `done` (2026-06-11) — Silent malware-scan
  fail-OPEN when no scanner backend is available. (1) `CompositeMalwareScanner.
  scan_file/scan_directory` returned `is_clean=True, scan_errors=[]` (a
  verified-clean verdict) when `self._scanners` was empty — but nothing actually
  scanned. (2) The GitHub scraper's `_scan_clone` did `if not available_scanners:
  return True` — silently admitting every cloned third-party repo as clean when
  `malware_scan=True` but yara-python wasn't installed and Defender wasn't
  present (common on non-Windows / minimal installs), contradicting the SEC-001
  "never skip a scan silently" posture. Fixed: the composite scanner now records
  a `scan_error` (→ `had_errors=True`, NOT verified-clean per the documented
  contract; `is_clean` stays True since no threats were *found*) when zero
  scanners are available; `_scan_clone` emits a loud SECURITY warning and
  proceeds unverified (consistent with its incomplete-scan path) instead of
  returning silently. Tests: test_malware_scanner.py updated test_no_scanners +
  added directory-flag and no-spurious-error-with-scanner (3).
- **DATA-014** [data-quality/consistency, medium] `done` (2026-06-11) — Router
  evaluation taxonomy diverged from the model/training taxonomy.
  `router_evaluation.KNOWN_DOMAINS` was an 8-domain set
  (`...testing, python, general_ts`) while `router_model.DEFAULT_DOMAINS`,
  `router_data_generator`, and `domain_detector.DOMAINS` (+`general` fallback)
  ALL use the canonical 7 (`...testing, general`). `create_test_dataset`
  labelled 2/10 built-in samples `python`/`general_ts` — labels a correctly-
  trained router can NEVER output — so `evaluate_router.py` capped accuracy at
  80% and emitted phantom confusion-matrix rows, making router eval misleading.
  (KNOWN_DOMAINS had no other consumers; it was pure divergence.) Fixed: aligned
  KNOWN_DOMAINS to the canonical 7 and relabelled the Python / plain-TS samples
  to the `general` fallback (a TS-focused router routes non-TS / generic code to
  general). Kept router_evaluation torch-free (literal list + a test asserting
  equality with DEFAULT_DOMAINS rather than a runtime import). Module had ZERO
  test coverage; added test_router_taxonomy.py (5): KNOWN_DOMAINS ==
  DEFAULT_DOMAINS == detector domains, no python/general_ts, all eval labels
  routable, perfect router reaches 100%.
- **BUG-107** [bug/train-inference, high] `done` (2026-06-11) — Sibling of
  BUG-106 in the REASONING path (predicted last cycle). `train_reasoning.py`
  calls `add_thinking_tokens()` to add `<think>`/`</think>` (vocab +2) and
  trains the model on those ids, then `save_checkpoint`s the expanded model —
  but NEVER persisted the expanded tokenizer (the save_checkpoint call omitted
  `tokenizer_path`). Inference reloads the BASE tokenizer.json (no thinking
  tokens): the reasoning markers fragment and the trained ids decode out of
  range, silently breaking `extract_thinking()`/`strip_thinking()` (so reward
  parsing + the whole reasoning feature fail at inference). Model side was
  already handled (`_maybe_resize_vocab`); only the tokenizer was lost. Fixed
  identically to BUG-106: save the expanded tokenizer into output_dir and pass
  `tokenizer_path=` to save_checkpoint (→ metadata.json → resolve_tokenizer_
  path). Tests: test_reasoning_tokenizer_persist.py (3): base lacks thinking
  tokens, add→save→reload preserves exact ids (single-id encode), metadata
  resolution.
- **BUG-106** [bug/train-inference, high] `done` (2026-06-11) — SFT train/
  inference tokenizer mismatch. `train_sft.py` calls `add_chat_tokens()` to add
  `<|im_start|>`/`<|im_end|>` (NOT in base SPECIAL_TOKENS) to the in-memory
  tokenizer and resize the model (vocab +2), trains the model on those new ids,
  then `save_checkpoint`s the expanded model — but NEVER persisted the expanded
  tokenizer. So inference (`resolve_tokenizer_path` → base tokenizer.json)
  reloaded a tokenizer WITHOUT the chat tokens: it fragments the ChatML role
  markers into ordinary tokens and can neither feed nor decode the ids the model
  trained on, silently breaking instruction following (the entire point of SFT).
  Discovered while auditing special-token-string usage (the DATA-003/013 family)
  — here the markers ARE added at train time but the expansion was thrown away
  on save. Fixed: train_sft saves the expanded tokenizer into the checkpoint dir
  and passes `tokenizer_path=` to `save_checkpoint` (writes it to metadata.json),
  which `resolve_tokenizer_path` already reads back first. Tests:
  test_sft_tokenizer_persist.py (4): base tokenizer lacks chat tokens,
  add→save→reload preserves the exact ids (marker encodes to a single id, not
  fragmented), metadata tokenizer_path resolves + takes priority, dead path
  falls through.
- **DATA-013** [data-quality/bug, medium] `done` (2026-06-11) — `prepare_docs_data.py`
  wrapped each doc in `<|doc|>...<|/doc|>` then `tokenizer.encode`d it, but
  `<|doc|>`/`<|/doc|>` are NOT in the base tokenizer's SPECIAL_TOKENS — they're
  `CONTEXT_TOKENS` added only by `add_context_tokens()`. With a normally-trained
  tokenizer, encode() FRAGMENTS those markers into ordinary punctuation tokens
  instead of single structural markers, silently producing degraded docs
  training data (the model never learns a clean `<|doc|>` boundary) — the same
  class as DATA-003. (`<|eos|>` IS a base special token, so that part was fine.)
  Fixed by mirroring DATA-003: a testable `_missing_doc_tokens()` helper +
  `cli.fatal` with the exact remedy (add_context_tokens + re-save, or retrain
  including CONTEXT_TOKENS) before any tokenization. Also switched
  `_build_doc_text` to the canonical `wrap_doc()` (DRY). Script had ZERO test
  coverage; added test_docs_data.py (6): missing/present/partial token
  detection, wrap markers + eos, header parsing.
- **BUG-105** [bug/cli, medium] `done` (2026-06-11) — `_run_weighted_mix`
  (combine_datasets.py, the non-interactive `--datasets a:0.8 b:0.2` path) had
  three defects in an untested CLI function: (1) non-2D inputs were
  `continue`-skipped, desyncing `zip(paths, arrays, norm_weights, row_counts)`
  so weights/labels applied to the WRONG dataset; (2) `row_counts[-1] = total -
  sum(rest)` could go NEGATIVE when many tiny datasets were each clamped to
  `max(1,...)`, crashing `np.random.choice(size<0)` (proved: 1 big + 9 tiny
  weights); (3) no chunk_size check, so mixing different `seq_len` datasets
  failed with an opaque numpy concatenate error (the interactive
  DatasetCombiner.combine validates this). Fixed: load+validate up front keeping
  paths/weights/arrays in lockstep, renormalise weights over the datasets
  actually loaded, abort with a clear message on chunk_size mismatch, and
  reconcile row counts by absorbing the rounding diff into the LARGEST bucket
  (never negative). Tests: test_combine_datasets_weighted_mix.py (8) — parse
  (default/explicit/Windows-path/invalid weights) + mix (basic, many-tiny
  no-crash, chunk mismatch aborts, skewed weights sum to total).
- **BUG-104** [data-quality/bug, medium] `done` (2026-06-11) — `combine_datasets.py`
  `run_pipeline` exact-dedup path was NOT cross-dataset. It called
  `ExactDeduplicator.deduplicate_array(arr)` per-dataset in a loop — removing
  only WITHIN-dataset duplicates (already done by prepare_data's default exact
  dedup) while leaving CROSS-dataset duplicate chunks in the combined training
  set, contradicting the comment ("remove dupes across datasets") and the menu
  label. The minhash path already deduped across (each secondary vs the
  primary). So combining overlapping corpora (e.g. the same file in both the TS
  and general datasets) with `--dedup exact` silently kept the dupes. Fixed by
  unifying both methods onto `CrossDatasetDeduplicator(method=dedup_method)` —
  exact now genuinely removes cross-dataset dupes (SHA-256 hash set of primary,
  star topology, primary kept intact), matching minhash. Removed the now-unused
  ExactDeduplicator/numpy imports. The script's `run_pipeline` had ZERO test
  coverage (test_combine.py only tests DatasetCombiner/dedup.py directly); added
  test_combine_datasets_script.py (3): exact removes cross-dataset dupes (4 vs
  the broken 6 chunks), none keeps all 6, primary kept intact. KNOWN LIMITATION
  (follow-up, both methods): secondaries are deduped vs the primary only, not
  vs each other (B∩C dupes survive) — a star topology, not full transitive
  dedup.
- **BUG-103** [bug/curation, medium] `done` (2026-06-11) — `score_repos_parallel`
  was BROKEN in subprocess mode. `_score_repo_worker` (test_runner.py)
  reconstructed `TestRunner(mode=mode, ...)` in each worker process WITHOUT
  forwarding `allow_host_execution`, but `TestRunner.__init__` raises
  ValueError for `mode="subprocess"` unless that flag is True (a safety gate
  added by a prior change). So every worker raised, was caught by the
  `as_completed` handler, and every repo was silently scored 0.2 with an error
  — i.e. `scripts/score_repos.py --mode subprocess` (which sets
  `allow_host_execution=True` on the parent then calls the parallel scorer)
  produced all-garbage rankings. The worker also hardcoded `cache_dir`,
  ignoring the caller's. The existing parallel test only used `dry_run` (no
  gate), so it never caught this. Fixed: store `_allow_host_execution` /
  `_cache_dir` on the runner and forward both into `_score_repo_worker`. Tests:
  test_curation.py TestParallelSubprocessGate (3): worker builds with the flag
  (no-framework repo → no execution), gate still raises without it, end-to-end
  parallel subprocess scoring returns a real (non-error) score.
- **BUG-102** [reasoning/reward-quality, low] `done` (2026-06-11) — The
  indentation-consistency sub-check in `_check_style` (reasoning/rewards/
  combined.py — the `combined` GRPO reward's style signal) was a no-op that
  contradicted its docstring ("Consistent indentation (2 or 4 spaces, not
  mixed)"). It did `indent_sizes.add(indent % 4 == 0 or indent % 2 == 0)` —
  adding a BOOLEAN to the set, with a condition that reduces to `indent % 2 ==
  0`. So it only ever distinguished odd vs even indent WIDTHS (never the
  tab-vs-space mix it claimed), treating 2-space and 4-space code as identical
  and only firing on rare odd indentation. Fixed to classify each indented line
  by its FIRST whitespace char and penalize genuine tab/space mixing (the
  canonical lint smell), without false-flagging tab-indent + space-alignment or
  varied space widths. Low impact (style weight 0.1 × 0.1 sub-penalty) but a
  real "sloppy prior AI work" cleanup of a training-signal heuristic. Tests:
  test_type_reward.py TestStyleCheck (+4): consistent-space / tab-only / varied
  widths → no penalty; mixed tab+space → penalty.
- **TOOL-004** [tooling/security-consistency, medium] `done` (2026-06-11) —
  `TSBenchmark._tsc_check` (evaluation/ts_benchmark.py) ran `tsc` on MODEL-
  GENERATED TypeScript via a raw `subprocess.run(["tsc", "--noEmit", ...])` on a
  bare temp file — bypassing the project's canonical `TscRunner` and its
  hardened tsconfig (`plugins=[]`/`types=[]`/`typeRoots=[]`, executed through
  `SandboxedRunner`). This violates two standing rules: "no ad-hoc tsc
  subprocess calls — use TscRunner" (DRY) and "execution-based scoring of
  generated code must be sandboxed by default." It also judged TS validity
  differently from the rest of the stack (TscScorer / TypeCheckReward), which
  all use the hardened config. Fixed: `_tsc_check` now routes through a shared,
  cached `TscRunner` (lazily built so importing this static benchmark doesn't
  pull the sandbox stack); a solution passes when `check()` reports no
  severity=="error" diagnostics (warnings don't fail). Removed the now-unused
  `subprocess`/`tempfile`/`shutil`/`Path` imports. Tests:
  test_ts_benchmark_tsc.py (8): routing (none-when-unavailable, true/false on
  errors, warnings don't fail, none on runner exception), the ad-hoc imports
  are gone (regression guard), and evaluate_solution tier-4 integration.
- **INFER-006** [inference/bug, high] `done` (2026-06-11) — Multi-token stop
  sequences were reduced to their FIRST token. `generate`/`generate_stream`
  (generator.py) built `stop_ids` via `stop_ids.add(encoded[0])`, so a stop like
  `";\n"` (tokens `[";", "\n"]`) halted at the first `;` — truncating code after
  one statement — and `"\n\n"` halted at the first single newline. This is on
  the standard OpenAI `stop` path (server forwards `request.stop` verbatim) and
  the built-in prompt_templates ship exactly such multi-char stops
  (`["\n\n", "\nfunction ", "\nclass "]`, `[";\n"]`, `["\n});"]`, ...), plus
  multi_turn_chat passes a multi-token user-prefix stop. Fixed with a HYBRID
  matcher (`_partition_stops`): EOS + any stop that encodes to ONE token (incl.
  special tokens like `<|im_end|>`/`<|fim_suffix|>`, which the decoder strips so
  they can't be matched as text) stay exact token-level; multi-token stops are
  matched at the STRING level on the decoded completion (`_earliest_stop_index`,
  searching only past the prompt so a stop in the prompt never truncates). The
  streaming path holds back the last `max_stop_len-1` chars before yielding (so
  `"\n\n"` can't leak its first `"\n"`) and flushes the tail on normal end —
  matching vLLM/TGI. Single-token behavior is byte-for-byte unchanged (no
  regression). Tests: test_generator_stop_tokens.py (14): double-newline /
  semicolon / word stops not truncated early, single-token + EOS still exact,
  stop-in-prompt ignored, streaming stop + held-back-tail flush + no-stop
  passthrough, partition + helper units.
- **DATA-011** [data-quality/bug, medium] `done` (2026-06-11) — `FIMCollator`
  (data/collator.py), the dynamic train-time Fill-in-the-Middle collator, was
  UNREACHABLE (only referenced in docs — `create_dataloader` hardcodes
  CodeCollator/WeightedCodeCollator) AND buggy: `_apply_fim` built `seq_len + 3`
  tokens then truncated back to `seq_len`, silently chopping the LAST 3 tokens —
  which are the END of `middle`, the segment FIM trains the model to predict.
  For short middles it could even drop the `<fim_middle>` marker, producing
  prefix+suffix with no target. It duplicated the correct, tested `FIMTransform`
  (fim.py), which avoids this by reserving 3 content slots up front (output
  length == input, target intact). The docs presented this buggy class as
  "efficient for the training hot path." Fixed by delegating `FIMCollator` to
  `FIMTransform` (DRY): length-preserving, target never truncated, gains SPM
  ordering for free, dtype/device preserved. The module had ZERO tests; added
  test_fim_collator.py (10): length preserved/stackable, middle-target present
  across 40 seeds, no content lost beyond 3 reserved slots, rate-0 unchanged,
  rate-1 transforms all, PSM/SPM ordering, dtype preserved. Corrected the
  misleading docs + added a "not wired by default" note. Follow-up: DATA-012
  (optional trainer wiring of dynamic FIM).
- **MEM-001** [bug/correctness, medium] `done` (2026-06-11) — `MemoryConfig.
  max_context_tokens` (default 1024, documented "Max tokens of memory to inject
  into prompts") was NEVER enforced — a silent no-op config. `get_relevant_
  memories` (called by routing_orchestrator to prepend memory to the model's
  prompt) concatenated project.md + 3 chunks with no token cap, so a large
  memory store could overflow the context window and push out the actual
  query/code. Fixed: enforce the budget — project context is truncated to fit,
  and chunks are added most-relevant-first only while they fit (char/4 token
  estimate, the project-wide heuristic). The memory module had ZERO tests; added
  test_memory_budget.py (5): estimate, project truncation, total-within-budget
  with many chunks, small-not-truncated, uninitialized→empty.
- **SEC-003** [security, high] `done` (2026-06-11) — PATH TRAVERSAL in the agent
  ToolExecutor. `_validate_path` (tools/executor.py) used
  `str(resolved).startswith(str(project_root))` for containment — the classic
  sibling-prefix bypass: a path resolving to `<root>-secrets/x` passes because
  the STRING starts with the root. CONFIRMED empirically: `read_file` with
  `path="../proj-secrets/secret.txt"` read a file outside the project root. The
  agent's read_file/run_tests/lint/git tools all route model-supplied paths
  through this. Fixed with `Path.is_relative_to` (component-wise containment).
  Also hardened `_handle_git_diff`: reject refs starting with `-` (a `git diff
  --ext-diff`-style flag-injection slipped past the char-set check). The tools/
  module had ZERO test coverage; added test_tool_executor.py (11): sibling/
  parent/absolute escapes blocked, valid paths allowed, git-ref flag injection
  + shell-metachar refs rejected, unknown tool, missing file.
- **DATA-010** [data-quality, medium] `done` (2026-06-11) — `score_quality`
  (instruction_gen.py, the code->instruction SFT generator) awarded its +0.2
  "parses" bonus ONLY for ast.parse-able Python. cola-coder is TypeScript-
  PRIMARY, so an identical TS/JS pair scored 0.2 lower than Python — and a short
  TS response (20-49 chars, balanced braces) scored 0.55 → REJECTED while the
  Python equivalent scored 0.65 → kept. That systematically filters out the
  project's primary-language synthetic data MORE than Python. Fixed: JS/TS earns
  the same bonus via a balanced-brace check. Also: `_make_fix_pair` shuffled the
  shared module-level `_BUG_INJECTIONS` list in place (side effect on every
  call) — now shuffles a copy. The module had ZERO test coverage; added
  test_instruction_gen.py (12): language parity, dedup, threshold, no global
  mutation, generate smoke.
- **DATA-009** [data-quality, medium] `done` (2026-06-11) — `SelfAlignSource.
  _build_inner_source` built `HuggingFaceSource(dataset=...)` for the HF path
  WITHOUT passing `languages`. HuggingFaceSource defaults to `["python"]`, so a
  `language: typescript` (or any non-Python) self-align config would download
  PYTHON code while the pipeline extracts TYPESCRIPT seeds from it — yielding
  few/no seeds and empty/garbage SFT data. This is the exact "always pass
  explicit languages to HuggingFaceSource" footgun the project has a feedback
  rule for; a prior agent missed it here. Fixed: pass `languages=[self._language]`.
  (The local path already maps extensions by language correctly.) Also removed
  dead code in `_analyze_seed` (`lines`/`first_line` assigned-never-used) and
  two E741 nits while in the file. Tests: test_self_align.py
  test_hf_inner_source_uses_configured_language (+ python variant).
- **DATA-007** [data-quality, medium] `done` (2026-06-11) — The local and
  software_heritage file sources emitted metadata key `"path"`, but the language
  detectors (scorers/language_detect.py) and the github source use the canonical
  key `"file_path"`. So files from those two sources were NOT language-detected
  by their EXTENSION (the most reliable signal) and silently fell back to weaker
  content heuristics — e.g. a simple `.ts` file with no TS markers wouldn't be
  recognised as TypeScript, so the tsc quality scorer wouldn't run on it. Fixed:
  both sources now emit `"file_path"` alongside `"path"` (kept for compat —
  swh's own code + tests read it). Tests: test_source_metadata.py +
  test_swh.py file_path assertion.
- **DATA-008** [data-quality, low] `done` (2026-06-11) — The HuggingFace source
  emitted no `language` metadata even when constructed for a single language
  (the common case — the pipeline builds one source per language). Now tags
  `language` for single-language sources so downstream language-aware scorers
  don't guess from content. (Multi-language sources can't know per-record
  language: stream_code_data yields only content strings.) Tests:
  test_source_metadata.py TestHuggingFaceSourceMetadata.
- **EVAL-005** [eval/capability, medium] `done` (2026-06-11) — Benchmark
  decontamination wired end-to-end. `DataLeakageDetector` (MinHash) was a
  COMPLETE, tested, but UNWIRED feature module. (1) Added a `containment` metric
  (|A∩B|/|A|) — Jaccard misses a short eval problem embedded in a larger
  training file (the common contamination case); containment catches it (1.0 vs
  0). (2) Wired it into `scripts/check_contamination.py` (eval problem set vs a
  JSONL text corpus or decoded .npy chunks; exits 1 on contamination to gate a
  pipeline) + an Eval-menu entry. Validated end-to-end: a corpus embedding a
  built-in problem's prompt is flagged (exit 1), a clean corpus passes (exit 0).
  Smoke caught a real dilution bug — eval docs must be the prompt (and solution)
  as SEPARATE units, not prompt+test_code concatenated. Tests:
  test_data_leakage_detector.py TestContainmentMetric + TestContaminationScriptWiring.
- **TOOL-003** [tooling/export, low-medium] `done` (2026-06-11) — GGUF export
  hardcoded `llama.attention.layer_norm_rms_epsilon = 1e-5` in BOTH the builtin
  writer (gguf_export.py:497) and the gguf-package path (:565), but the model's
  RMSNorm uses eps=1e-6 (model/normalization.py default, never overridden). So
  every exported GGUF told llama.cpp to use a 10x-larger epsilon than the model
  was built with — an unfaithful export (numerically small for normal
  activations, but wrong). Fixed: both paths now use a `_RMS_NORM_EPS` constant
  (=1e-6) and a regression test cross-checks it against an actual RMSNorm
  instance, so a future change to RMSNorm's default forces the export to update
  too. Audited the rest of gguf_export (Q8_0 block layout, weight-name mapping,
  GGUF v3 file layout / 32-byte alignment / data-relative offsets) — all correct
  (the flattened Q8_0 blocking aligns with llama.cpp's per-row blocking because
  cola-coder dims are always multiples of 32). Tests: test_export.py
  TestGGUFMetadata (3).
- **TOOL-002** [tooling/export, medium] `done` (2026-06-11) — `_model_size_mb`
  (export/quantize.py) summed only parameters() + buffers(), but torch.ao
  dynamic-quantized Linear stores its INT8 weights in a `_packed_params`
  submodule that neither enumerates. So EVERY dynamic-INT8 export reported the
  quantized model as ~0 MB and a compression ratio in the MILLIONS (verified:
  2,003,906×). The quantization itself was correct — only the user-facing size
  report was nonsense. A prior agent had papered over it (test_export.py
  comment: "may not always shrink the python object" + a loose `ratio >= 0.5`).
  Fixed: also count packed weights via the quantized module's callable
  `weight()`/`bias()` accessors (INT4 builtin already uses register_buffer, so it
  was fine). Now reports a sane ~4× for int8-vs-fp32. Tightened the two weak
  tests + added a direct regression (test_export.py
  test_model_size_counts_packed_int8_weights).
- **TEST-001** [test-quality, medium] `done` (2026-06-11) — quality_filter.py
  (the data-quality GATE deciding which scraped code enters training) had NO
  direct test coverage (only the separate filters/ plugins were tested). A
  subtle break silently poisons the training set. Added test_quality_filter.py
  (20 tests) locking the key checks — headline guard is check_character_
  diversity's denominator cap (see "Not a bug" below): a regression test
  asserts large diverse code PASSES, so the cap can't be "fixed" away.
- **MODEL-002** [model, low] `done` (2026-06-11) — `ModelConfig.total_params`
  used the dense FFN formula regardless of MoE, so a MoE config reported ~active
  params, not the true in-memory total (all experts) — which feeds VRAM
  estimates and model cards. Now counts (num_experts + num_shared_experts) FFNs
  + router gate for each MoE layer. As part of this, `resolve_moe_layers` moved
  to the torch-free `model/config.py` (canonical) and is re-exported from
  `features/moe_layer.py` (transformer.py + tests unchanged via re-export).
  Tests: test_moe_param_count.py (7).
- **REASON-001** [reasoning, medium] `done` (2026-06-11) — `extract_thinking`
  (thinking_tokens.py:124) used `.index()` for the FIRST `</think>` only, so a
  model emitting multiple reasoning blocks left later `<think>..</think>` blocks
  embedded in the extracted `code`. reward.py:63 then runs that code → invalid
  syntax → always-fail reward, giving GRPO a WRONG (penalizing) signal for
  legitimate multi-step reasoning. Fixed: `code = strip_thinking(text)` (loops
  over all blocks, like the existing strip_thinking). Tests:
  test_tokenizer_reasoning_fixes TestExtractThinkingMultiBlock (incl. ast.parse).
- **TOK-001** [tokenizer, medium] `done` (2026-06-11) — CodeTokenizer cached
  pad/bos/eos/unk ids from token_to_id() without checking None; a mismatched
  tokenizer (missing these) made `encode` produce `[None, ...]`, silently
  corrupting every sequence. Now `__init__` fails loud (ValueError listing the
  missing core tokens) — converts silent corruption into a clear, actionable
  error (cf. DATA-003). FIM tokens stay optional. Tests:
  test_tokenizer_reasoning_fixes TestTokenizerCoreTokenGuard.
- **INFER-004** [inference, low] `done` (2026-06-11) — Streaming `completion_tokens`
  was counted per SSE chunk but NEVER emitted (dead variable; no usage in the
  stream at all). Implemented OpenAI `stream_options.include_usage`: both stream
  endpoints now accumulate the completion text and emit a final usage-only chunk
  (`choices: []`) with an ACCURATE count (re-encode the accumulated text — chunks
  != tokens, and empty-decode tokens yield nothing). Default (no stream_options)
  still emits no usage, per spec. Tests: test_server_openai TestStreamingUsage.
- **EVAL-004** [eval, low] `done` (2026-06-11) — `evaluate_solution` with empty/
  whitespace `test_code` ran the generated code with no assertions → exit 0 →
  false PASS, inflating pass@k. Now returns (False, "NO TESTS: ...") and
  short-circuits before any sandbox execution. Tests: test_inference_eval_fixes
  TestEmptyTestCode.
- **INFER-005** [inference, low] `done` (2026-06-11) — `generate_batch` silently
  dropped min_p/repetition_penalty/stop_tokens; now accepts and forwards all
  three (parity with generate). Tests: test_inference_eval_fixes
  TestGenerateBatchForwardsParams.
- **BUG-101** [bug/ux, medium] `done` (2026-06-11) — The "Prepare Mixed Data
  (code+text+math)" menu item was BROKEN: it passed --mix-code/--mix-text/
  --mix-math to prepare_data.py, which has no such args, so argparse errored
  with "unrecognized arguments" every time. Discovered while resolving DATA-002.
  Fixed to write the chosen ratios into a derived data_sources config
  (`data/mixing.py:write_weighted_data_sources`, weight 0 disables a source) and
  run the real multi-source collector `collect_data.py --data-sources <derived>`.
  Misleading help text ("MixedDataset ... per-batch during training") corrected.
  Tests: test_mixed_data.py TestWriteWeightedDataSources + TestMenuWiring.
- **DATA-002** [data-quality, low] `done` (2026-06-11) — `MixedDataset`
  (training-time weighted sampler) was dead code: never constructed, imported, or
  tested; the trainer loads a single .npy. RESOLVED by REMOVE (not integrate):
  the working mixing is collection-time (collect_data, fits the trainer);
  runtime mixing is a separate deliberate feature deferred to DATA-006 rather
  than left as an orphan masquerading as functional. Also cleaned the now-unused
  `os` top-import and 3 pre-existing lint items in the touched files. Tests:
  test_mixed_data.py TestMixedDatasetRemoved.
- **SEC-001** [security, low->medium] `done` (2026-06-11) — In-stream malware
  scanning could be disabled SILENTLY: `_maybe_scan_stream` returned the
  unscanned iterator with no warning when `malware_scan.enabled=false` or
  `in_stream=false`. Now both disable paths emit a loud `cli.warn` (SECURITY ...
  DISABLED) — overrides are explicit, never silent. Safe defaults documented in
  scoring.yaml. Verified: missing/empty config DEFAULTS to scanning (fail-safe).
  Tests: test_collect_data_security.py TestInStreamScanGating.
- **SEC-002** [security, medium] `done` (2026-06-11) — DISCOVERED during the
  SEC-001 scan: `_scan_downloaded_data`'s `on_threat="warn"` path called
  `cli.confirm("Continue despite threats?")` with the implicit default=True, and
  `cli.confirm` returns its default on EOF/no-TTY — so a NON-INTERACTIVE
  collection run would silently CONTINUE with malware in the dataset (fail-OPEN).
  Fixed to `default=False` (fail-CLOSED: automated runs abort). The config
  default is on_threat=quarantine (also safe); the in-stream scanner always
  drops threats regardless. Test: TestFailClosedAndConfig.
- **MODEL-001** [model, medium] `done` (2026-06-11) — Upcycled MoE checkpoints
  could be loaded for INFERENCE but NOT fine-tuned: `Trainer.__init__` built
  `Transformer(config.model)` straight from the (dense) config, so resuming from
  a MoE checkpoint failed (`experts.*` keys have nowhere to go) unless you
  hand-edited the config. Fix: `apply_moe_config_from_checkpoint` moved to its
  canonical home `features/moe_layer.py` (re-exported from inference/loading for
  existing importers) and called in `Trainer.__init__` before the model is
  built, mirroring the inference auto-detect. `load_checkpoint` already tolerates
  the model-only upcycle output (fresh optimizer, step 0). Now `train.py
  --resume <moe_dir>` fine-tunes the upcycled model. Validated: apply -> build ->
  load_checkpoint -> forward+backward+aux-loss (test_moe_integration.py
  TestMoEResumeFineTune, incl. proof a dense model fails). Remaining dedicated
  stage 7.5 split to MODEL-003. Doc: checkpoints rule.
- **INFER-001** [inference, medium] `done` (2026-06-11) — Prompt-echo leak.
  CORRECTED diagnosis: the STREAMING path is actually safe (`generate_stream`
  yields only incremental new text via prev_decoded_len; the server's
  `startswith` check there is harmless dead code). The real leak is the
  NON-STREAMING chat + completion endpoints: `generate()` returns
  `decode(prompt_tokens + new_tokens)` and `result.startswith(prompt)` returns
  the WHOLE prompt echo on any BPE round-trip mismatch (BOS render, whitespace,
  boundary merge). Fix: shared `inference/text_utils.strip_prompt_prefix`
  (longest-common-prefix strip — never returns the full prompt on mismatch),
  used by both non-streaming endpoints; best_of_n._strip_prompt now aliases it
  (DRY). Tests: test_text_utils.py (6) + test_server_openai TestPromptStripRobustness
  (drift generator proves no leak).
- **DATA-005** [test-quality, low] `done` (2026-06-11) — DISCOVERED when installing
  datasketch unskipped `test_filters.py::test_detects_near_duplicates`, which
  then FAILED. Root cause: degenerate test data — a `block * 10` doc collapses
  to ~50 unique 5-gram shingles, so a `42->43` change lands at exact Jaccard
  0.818 (just over the 0.8 threshold) but MinHash(128) under-estimates it to
  0.734, below threshold → not detected. The DeduplicationFilter is CORRECT
  (verified: realistic 713-shingle content with a 1-token change → Jaccard 0.993
  → reliably caught; distinct content kept). Fixed the test to use
  non-repetitive content. Lesson: datasketch-gated tests had never run because
  the dep was never installed — installing it surfaced latent test-data rot.
- **DATA-004** [data-quality, high] `done` (2026-06-11) — DISCOVERED while
  validating DATA-001: exact+minhash dedup SILENTLY NO-OP'd ON WINDOWS. The
  cycle-1 mmap optimization (`np.load(output_file, mmap_mode="r")`) kept the
  file locked, so `os.replace(tmp, output_file)` raised PermissionError that the
  `except Exception` swallowed as "Deduplication failed. Keeping all chunks." —
  on the user's primary platform. The cycle-1 smoke missed it by using a full
  load, not mmap. Fix: extracted `dedup_npy_file()` in data/dedup.py that closes
  the memmap handle BEFORE os.replace; prepare_data calls it. Tests:
  test_dedup.py TestDedupNpyFile (real mmap+replace roundtrip on a temp file).
- **DATA-001** [data-quality, medium] `done` (2026-06-11) — within-file
  near-duplicate (MinHash) dedup wired: `CrossDatasetDeduplicator.
  deduplicate_self_array` + `prepare_data --dedup minhash --dedup-threshold`,
  menu 3-way choice, `[dedup]` extra (datasketch). Falls back to exact + loud
  warning when datasketch absent. Validated real near-dup removal with
  datasketch 1.10 installed (1-token-diff chunk dropped, distinct kept). Tests:
  test_dedup.py TestSelfMinHashDedup + TestDedupNpyFile.
- **DATA-003** [data-quality, high] `done` (2026-06-10) — `prepare_repo_context_data.py`
  silently fell back to `eos_id` for missing `<|repo|>/<|/repo|>/<|file|>/<|/file|>`
  tokens, emitting POISON training data (the entire repo/file structure became
  eos tokens). These tokens are NOT in the base tokenizer's SPECIAL_TOKENS —
  only `add_context_tokens()` adds them — and the script is menu-wired (run with
  no args), so a normally-trained tokenizer triggers it. Now `_resolve_context_
  token_ids` reports missing tokens and main() `cli.fatal`s with the exact
  remedy instead of producing broken data. Test: test_repo_context_data.py (4).
- **EVAL-002** [eval, low (was high)] `done` (2026-06-10) — VERIFIED: real but
  currently harmless (no FAKE_PACKAGES entry has a dot). JS/TS import regex char
  class now includes `.` so dotted hallucinated packages are matched. Test:
  test_safety_evaluator.py TestDottedPackageDetection.
- **INFER-002** [inference, medium] `done` (2026-06-10) — try/finally around the
  `generate_group` prefill+expand+decode guarantees `clear_caches()` even on
  exception; OOM handler now clears before `empty_cache()` so the retry actually
  reclaims VRAM. Test: test_loop_cycle_fixes.py
  `test_generate_group_clears_cache_on_decode_exception`.
- **EVAL-001** [eval, medium] `done` (2026-06-10) — `compute_pass_at_k` now
  returns None (not 0.0) when no problem has >= k samples and logs a warning;
  warns on partial exclusion; `format_results` shows "n/a"; auto_eval hardened.
  Tests: test_loop_cycle_fixes.py TestPassAtKSemantics.
- **INFER-003** [inference, low] `done` (2026-06-10) — `_apply_repetition_penalty`
  returns the mutated logits; caller reassigns. Test: TestRepetitionPenaltyReturn.

---

## Not a bug (verified)

- **Preprocessing "remainder chunk loss"** (preprocess.py — final `token_buffer`
  < chunk_size discarded at `_DONE`) — NOT A BUG (verified 2026-06-11). Dropping
  the trailing sub-chunk-size remainder is standard, correct GPT-style document
  packing: chunks are fixed-length training examples, and the proposed "fix"
  (zero-pad the remainder to chunk_size) would INJECT pad/token-0 ids into the
  training stream, teaching the model to predict padding — strictly worse. The
  loss is < chunk_size tokens (≤2047) out of a multi-GB corpus: negligible and
  intended. Do not "fix" by zero-padding.
- **SFT label/offset misalignment on truncation** (sft_dataset.py
  `_tokenize_conversation`:126-173) — NOT A BUG (verified 2026-06-11). Claim: tokens
  truncated to max_seq_len but offsets taken from untruncated text → misaligned
  labels. Verified safe: `CodeTokenizer.encode(text, add_bos, add_eos)` is exactly
  `[bos] + self.tokenizer.encode(text).ids + [eos]`, and the offsets come from the
  SAME `self.tokenizer.encode(text)`, so `token_ids[i+1]` aligns with `offsets[i]`
  (the `bos_offset=1` mapping is correct). Truncation is TOKEN-level (not mid-token
  in char space) and the label loop is bounded by `min(len(token_ids),
  len(offsets)+1)`, so surviving tokens stay correctly aligned and dropped tokens
  are simply unlabeled. No off-by-one. (Same scrutiny as the chat_template
  off-by-one false positive below.)
- **Streaming prompt-echo via `chunk.startswith(prompt_text)`** (server.py
  _stream_chat:628, _stream_completion analog) — FALSE POSITIVE (re-confirmed
  2026-06-11 while fixing BUG-111). `generate_stream` yields ONLY incremental
  completion deltas: it seeds `prev_decoded_len = len(decode(prompt_ids))` and
  every yield is `current_decoded[prev_decoded_len:]` — i.e. text strictly AFTER
  the prompt. The first streamed chunk is the first generated character(s), never
  the prompt, so `chunk.startswith(prompt_text)` (prompt is long; the delta is a
  few chars) is essentially never true — harmless dead code, not a leak (matches
  the original INFER-001 analysis). Unlike the NON-streaming path (`generate()`
  returns the full prompt+completion → needs strip_prompt_prefix) and the FIM
  endpoint (BUG-111), the streaming path needs no stripping. Do not re-flag.

- **check_character_diversity denominator cap (quality_filter.py:297)** — FALSE
  POSITIVE (a scan flagged the `min(total_chars, 1000)` cap as letting large
  repetitive files pass). The cap is REQUIRED and correct: unique chars are
  bounded by the charset (~95 for ASCII code), so without the cap `unique/total`
  → 0 for ANY large file and a normal diverse 10K-char code file (~85 unique)
  would score 0.0085 < 0.05 and be WRONGLY REJECTED. The cap converts the metric
  into an absolute floor ("≥50 unique chars" at 0.05), which matches the
  docstring ("Normal code uses 30-40 unique chars") and still rejects truly
  repetitive files (1-15 unique chars). Removing it would reject most large code
  — a catastrophic regression. Locked by test_quality_filter.py
  TestCharacterDiversityCap. The rest of the data-quality pipeline (filters,
  scorers, ScoreMapper, weight polarity, chunking) was audited and is CLEAN.

- **VS Code extension scan (2026-06-11)** — a thorough scan flagged 11 items in
  the extension (client/providers/server/ui/extension.ts); ALL verified as false
  positives or non-issues. Highlights: generateVerified state "stuck" (early
  returns are BEFORE setState; the in-try return still runs finally);
  InlineCompletionProvider debounce/abort "leaks/races" (Promise executor runs
  sync so `sub` is always set; debounce resolves cancelled BEFORE the request;
  post-await isCancellationRequested recheck exists); LanguageModelProvider
  uncaught-promise/`undefined`-name (the `.catch(()=>{})` handles the chain; an
  `if (params)` guard exists); chatStream empty-body throw is intentional and
  caught by both callers; onDidChangeConfiguration throw is caught by VS Code's
  listener wrapper. The extension is well-written — do not re-flag these.

- **chat_template offset off-by-one** (claimed format_chat_training mis-masks
  assistant spans) — FALSE POSITIVE. `offset += len(segment) + 1` is read at the
  START of each iteration, so it exactly equals that segment's start in the
  `"\n".join(parts)` output; the trailing +1 after the last segment is never
  used. Worked a 2-message example: assistant content lands at offset 52 and
  `text[52:54] == "yo"`. The spans are correct.

- **BUG-004 (min_p NaN)** — FALSE POSITIVE. `_min_p_filter` threshold =
  `min_p * probs.max()`; for `min_p ≤ 1` the argmax token always satisfies
  `probs >= threshold`, so it is never masked → all-`-inf` is impossible. Only a
  defensive guard for misconfigured `min_p > 1` would add value (very low).

- **EVAL-003 (secret regex false positives)** — NOT A BUG (verified). The
  hardcoded-secret pattern requires a QUOTED key (`"api_key": "..."`) and only
  matches 8+ char values. Flagging secret-shaped quoted assignments in generated
  code is the INTENDED behavior of a safety probe, not a false positive —
  whether the value is a placeholder or real, the model emitting that shape is
  what we measure. Documented by test_safety_evaluator.py
  TestSecretAndDangerousDetection. Refining to exclude obvious placeholders is a
  possible low-value nicety, not a correctness fix.
