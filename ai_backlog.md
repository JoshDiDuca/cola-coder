# Cola-Coder AI Improvement Backlog

Persistent issue/opportunity log for the autonomous improvement loop. Append new
items with a stable ID; only mark `done` after a validated fix. Severity:
critical / high / medium / low. Status: open / in-progress / done / dropped.

Ratings reflect VERIFICATION against the code, not just an agent's first guess —
e.g. BUG-004 was downgraded to not-a-bug after checking the math.

---

## Open

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

- **DATA-021** [data-quality/architecture, medium] `open` — The modular
  FilterPlugin + `DataPipeline` system (data/pipeline.py + data/filters/, registry
  in data/registry.py) is built and now fully registered (DATA-020), but is NOT
  wired into any LIVE entry point: `collect_data.py` filters via
  `stream_code_data`/`tokenize_and_chunk` + the monolithic `quality_filter.py`,
  not `DataPipeline`; `DataPipeline(` is constructed nowhere outside pipeline.py
  itself (self_align.py only mentions it in docstrings). So the privacy (pii),
  license, content, syntax, dedup filters — though now selectable by name — have
  no config that actually invokes them in the real collection path. DECISION
  NEEDED (larger, staged refactor): either (a) route collect_data/prepare_data
  through DataPipeline with a `filters:` section in data_sources.yaml, or (b)
  consolidate onto quality_filter.py and retire the unused DataPipeline filter
  path. Don't do blindly — pick a single filtering architecture. Found this cycle.

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

- **EXPORT-011** [export/bug?, medium] `open` (needs validation — gguf package +
  llama.cpp, not installable/runnable here) — In `_write_gguf_with_package`
  (gguf_export.py ~line 785) the gguf-package path does
  `writer.add_tensor(name, data, raw_dtype=quant_type)` where `data` is the F32
  numpy array and `quant_type` is e.g. `GGMLQuantizationType.Q4_K`. The gguf
  PYTHON library does NOT quantize — `raw_dtype` declares the bytes are ALREADY in
  that format. Passing F32 bytes tagged as Q4_K/Q5_K/Q8_0 likely produces a file
  llama.cpp reads as garbage (size/format mismatch) or rejects. If confirmed, the
  gguf-package path can only safely emit F32/F16 unless it pre-quantizes the data
  to the exact K-quant block layout (which is what llama.cpp's llama-quantize
  does). Suggested resolution: for the package path, write F16 and instruct the
  user to run `llama-quantize` for K-quants (or pre-quantize q8_0 ourselves and
  pass the real q8_0 bytes with raw_dtype=Q8_0). Could not validate in this
  environment (gguf package absent, EXPORT-010 fresh scan). Verify before
  shipping a fix.

- **OPS-001** [tooling, low] `open` (deferred for user) — storage split-brain:
  configs/storage.yaml → E:/cola-coder-data vs config.checkpoint.output_dir →
  ./checkpoints. Needs the user's decision; do not unilaterally resolve.

---

## Done

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
