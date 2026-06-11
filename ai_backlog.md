# Cola-Coder AI Improvement Backlog

Persistent issue/opportunity log for the autonomous improvement loop. Append new
items with a stable ID; only mark `done` after a validated fix. Severity:
critical / high / medium / low. Status: open / in-progress / done / dropped.

Ratings reflect VERIFICATION against the code, not just an agent's first guess —
e.g. BUG-004 was downgraded to not-a-bug after checking the math.

---

## Open


- **DATA-006** [data-quality, low] `open` — Follow-up to DATA-002: if dynamic
  per-batch / online source reweighting is wanted, design runtime data mixing
  into the trainer DELIBERATELY (multi-source dataloader, per-source loss
  tracking → inverse-loss reweighting). The orphaned `MixedDataset` that
  half-implemented this was removed; `data/mixing.py` already has the
  reweighting math (`MixingOptimizer`). The current pipeline does
  collection-time mixing via collect_data, which fits the single-.npy trainer.


- **MODEL-003** [model, low] `open` — Follow-up to MODEL-001: fine-tuning an
  upcycled MoE now works via `train.py --resume <moe_dir> --config <cfg>`, but
  there is no DEDICATED pipeline stage 7.5 / menu entry / auto-config (low LR,
  fraction of steps) to orchestrate it. Optional convenience wrapper.


- **MODEL-004** [model, low] `open` — GRPO sequence log-prob (grpo.py:230-241,
  297-308) includes PROMPT tokens, not just completion tokens. VERIFIED this is
  currently harmless: advantages are mean-centered (Σaᵢ=0), the prompt is shared
  across the group, and the trainer does one step per group with ratios≈1, so
  the shared prompt-token gradients cancel exactly (only bf16-precision noise
  remains). Masking prompt tokens would match standard GRPO and be robust to
  future multi-step/non-centered changes — a cleanup, NOT a bug. Low priority.

- **OPS-001** [tooling, low] `open` (deferred for user) — storage split-brain:
  configs/storage.yaml → E:/cola-coder-data vs config.checkpoint.output_dir →
  ./checkpoints. Needs the user's decision; do not unilaterally resolve.

---

## Done

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
