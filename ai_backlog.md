# Cola-Coder AI Improvement Backlog

Persistent issue/opportunity log for the autonomous improvement loop. Append new
items with a stable ID; only mark `done` after a validated fix. Severity:
critical / high / medium / low. Status: open / in-progress / done / dropped.

Ratings reflect VERIFICATION against the code, not just an agent's first guess —
e.g. BUG-004 was downgraded to not-a-bug after checking the math.

---

## Open

- **INFER-001** [inference, medium] `open` — Server SSE streaming strips the
  prompt echo with `chunk.startswith(prompt_text)` on the raw string, but
  `decode(encode(prompt))` is not guaranteed byte-identical (BPE), so the echo
  strip can fail and leak the prompt into the stream. `server.py` ~557, ~702.
  `generate_stream` itself handles this correctly via prev_decoded_len; the
  server re-implements it more fragilely. Fix: strip by tokenized prompt length.

- **INFER-004** [inference, low] `open` — Streaming `completion_tokens` is
  incremented per SSE chunk, not per token, so OpenAI-compatible usage stats are
  wrong for streamed responses. `server.py` ~565, ~626. Fix: count via
  tokenizer or track token count in the stream loop.

- **EVAL-002** [eval, high] `open` — SafetyEvaluator package-hallucination regex
  `(?:require|from)\s*\(?['"]([@\w/-]+)['"]` misses package names with dots
  (`@babel/core`, `lodash.memoize`), so hallucinated dotted packages are never
  flagged (false negatives inflate safety score). `evaluation/safety_eval.py`
  ~245. Fix: add `.` to the char class. NEEDS VERIFICATION before fix.

- **EVAL-003** [eval, high] `open` — SafetyEvaluator secret-leak regex matches in
  any context incl. comments/test fixtures (`test_api_key = "test123..."`),
  inflating `secret_leak_rate` with false positives and hiding real leaks.
  `evaluation/safety_eval.py` ~81. NEEDS VERIFICATION (check against the suite).

- **EVAL-004** [eval, low] `open` — `execute_code` with empty `test_code` trivially
  "passes" (no tests to fail) and counts toward pass@k. `evaluation/runner.py`
  ~125-149. Fix: skip/warn on empty tests.

- **INFER-005** [inference, low] `open` — `generate_batch` doesn't accept/forward
  `min_p` (silently uses 0.0). `generator.py` ~471-508. API inconsistency.

- **DATA-001** [data-quality, medium] `open` — within-file near-duplicate
  (MinHash) dedup not available; only exact (within-file, just wired) and
  cross-dataset MinHash (combine_datasets.py) exist. Near-dups (20-30% of code)
  pass through prepare_data. Opportunity: self-MinHash in prepare_data --dedup
  minhash.

- **DATA-002** [data-quality, low] `open` — `MixedDataset` (data/dataset.py) is
  dead code — never constructed anywhere. Either integrate into the data path or
  remove. (mix_temperature config already removed.)

- **SEC-001** [security, low] `open` — Tighten malware-scan config defaults so
  in-stream scanning can't be silently disabled into an unsafe state; document
  the safe defaults. (From a prior audit; verify current defaults.)

- **MODEL-001** [model, medium] `open` — MoE is now loadable/runnable but upcycle
  only INITIALIZES experts; there is no pipeline stage to FINE-TUNE the upcycled
  MoE (needs ~10-20% training compute to differentiate experts). Opportunity:
  add stage 7.5.

- **MODEL-002** [model, low] `open` — `ModelConfig.total_params` reports the dense
  FFN count for MoE configs (display only; undercounts expert params).

- **OPS-001** [tooling, low] `open` (deferred for user) — storage split-brain:
  configs/storage.yaml → E:/cola-coder-data vs config.checkpoint.output_dir →
  ./checkpoints. Needs the user's decision; do not unilaterally resolve.

---

## Done

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
