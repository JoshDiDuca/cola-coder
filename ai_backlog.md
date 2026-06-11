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

- **MODEL-002** [model, low] `open` — `ModelConfig.total_params` reports the dense
  FFN count for MoE configs (display only; undercounts expert params).

- **OPS-001** [tooling, low] `open` (deferred for user) — storage split-brain:
  configs/storage.yaml → E:/cola-coder-data vs config.checkpoint.output_dir →
  ./checkpoints. Needs the user's decision; do not unilaterally resolve.

---

## Done

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
