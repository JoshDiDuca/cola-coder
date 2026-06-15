# Cola-Coder Web UI — Full Rebuild Spec & Plan

Source of truth for the UI rebuild. EVERY requirement the user has stated is captured here.
Nothing gets dropped. Update statuses as work lands. Owner: autonomous loop.

Last updated: 2026-06-15.

---

## 0. Non-negotiable principles

- Make it a proper application with real UX — not a wall of cards, not a script-runner table.
- Completely redesign the whole thing — a ground-up rebuild, not incremental restyling.
- **Master-detail screens** (user-approved): each section = ONE coherent screen — a list/picker on the
  left, the selected item's full detail + actions on the right. No grid-of-N-cards per page.
- Everything has a UI — even run arguments are real form fields (config dropdown, checkpoint picker,
  number/float inputs, flag checkboxes, choice selects) — never a raw `--flag value` text box.
- Everything must be 1:1 and STAY 1:1 with the CLI master menu / each script's argparse.
- Polished, consistent, and correct; every flow actually WORKS (no "Failed to fetch", no dead buttons).

---

## 1. Requirements (each is a tracked item)

### R1 — App shell & navigation  ✅ DONE (UI-063)
Sidebar (grouped nav) + routed pages (hash router) + topbar w/ live status. One page at a time.

### R2 — Master–detail screens for every section  ◧ 3 INTEGRATED, rest pending
INTEGRATED + live: Checkpoints, Data, Eval (UI-068) + Run & Jobs + System & Tools (UI-069) now render
master-detail screens (MasterDetail.tsx primitive). Run = ActionsPanel launcher + jobs→live-log detail;
System = tool list → reused panel detail. Dropped tools kept in a temporary "More tools" area on
Checkpoints/Data (1:1 preserved) until folded in. REMAINING screens: Configs & Pipeline, Tokenizer
(Dashboard stays a dashboard — now with the live monitor).
Replace each page's card-grid with a single master→detail screen.
- Checkpoints — ✅ screen built (CheckpointsScreen): list → Health/Compare/Export. **Gap:** Model Card,
  Training Manifest, Router, Average (model-soup), Export-artifacts not yet in the screen (kept in a
  temporary "More tools" area — must be folded into the screen, 1:1).
- Data — ✅ screen built (DataScreen): datasets → stats/preview/scores + Collect/Prepare launchers.
  **Gap:** Combine, DataSources, SFT data, Vector index, Security scan, Filters catalog, Scoring config,
  Repo scores not yet in the screen ("More tools" temp area — fold in, 1:1).
- Evaluation — ✅ screen built (EvalScreen): unified artifacts + history/benchmarks/safety/regression.
- Run & Jobs — ◻ master-detail: jobs list → job detail/live-log; actions via typed forms (see R3).
- Configs & Pipeline — ✅ screen built (PipelineScreen, UI-071): runs list → stage timeline detail +
  lifecycle actions (reset/override/delete) + New Run; one-click PipelineLauncher embedded at top.
  Configs/VRAM/LR/ConfigDiff kept in "More tools" (fold in next).
- Tokenizer — ✅ screen built (TokenizerScreen, UI-073): tabbed info/health/tokenize/vocab single screen
  replacing the 4-panel grid. Resolver fixed (R6).
- System & Tools — ◻ low-quality, needs full rebuild as master-detail (list of tools → detail).
- Dashboard (overview) — ✅ hero + GPU gauges + metrics chart + health + sysinfo (stays a dashboard,
  not master-detail) — but must become the LIVE TRAINING MONITOR (see R5).

### R3 — Typed argument forms for ALL actions (1:1 with argparse)  ✅ DONE (UI-070)
- ✅ Backend: `ActionParam` model + `ActionDef.params` in schemas.py; `action_params.py` holds the 1:1
  argparse spec for all 29 actions (full arg set, not just defaults), validated into `ActionParam` at
  import (fail-fast). `/api/actions` merges `ACTION_PARAMS` per key. ActionParam in gen_ts_types +
  types.gen.ts; drift test extended (111 pass).
- ✅ Frontend: `ActionForm.tsx` renders the right control per param (config select from /api/configs,
  checkpoint picker from snapshot, int/float number, bool checkbox, choice select; exhaustive
  never-check). `ActionsPanel` renders it when `params.length > 0`; raw text box only as a fallback for
  un-spec'd actions. Pure `buildArgs()` with store_true semantics.
- ✅ Acceptance met: every action launchable with all options as controls; trainer-guard preserved;
  tsc + vite build green.
- Follow-up: a parity test asserting every `ACTIONS` key has a non-empty `ACTION_PARAMS` entry (and that
  each param's `flag` actually appears in the script's argparse) would lock 1:1 permanently — backlog.

### R4 — One-click FULL TRAINING PIPELINE launcher  ✅ DONE (UI-071)
- **"A full pipeline for training, one click with all of the options available for me to choose. EVERYTHING."**
- ✅ `PipelineLauncher.tsx`: mode toggle (Full pipeline `full_pipeline` vs hardware-Auto `auto_pipeline`);
  a 10-stage checkbox selector (All/None, optional-stage marks 4 & 7) compiling to `--stages` CSV; all
  remaining options (start/stop/skip-optional/auto-resume/dry-run/tokenizer/config) as typed ActionForm
  controls; a live read-only command preview; trainer-guard warning banner + 409 "second trainer refused"
  surfaced as an error chip. One "Run pipeline" button → background job.
- ✅ Backend: `train` + `auto_pipeline` added to ACTIONS (trainer-class, guarded); typed specs were
  already in action_params.py so the launcher gets full form definitions.
- ✅ Acceptance met: every `full_pipeline`/`auto_pipeline`/`train` option is a control; one click launches
  as a background job; trainer guard protects the live run (verified). Embedded at top of PipelineScreen.

### R5 — Live training monitor (dashboard)  ✅ DONE
- LiveTrainingPanel on the Dashboard (top): run name + live pulsing status, big step/total + progress
  bar, loss/ppl/tok-s/s-it tiles, AND a LIVE auto-refreshing (5s) tail of train_small_react_best.log in
  a scroll box. "See what's currently training, its logs, progress" — front and center. (Latest
  eval/tests for the current checkpoint still to add — minor follow-up.)
- **"I want to see what's currently training and its logs, progress, tests, everything."** (asked repeatedly)
- Dashboard must prominently show: current run (config, step/total, loss/ppl/tok-s, ETA, progress bar
  — ✅ hero), the **LIVE TRAINING LOG tail** (train_small_react_best.log — backend already lists it in
  /api/logs; surface a live tail on the dashboard, NOT buried on the Run page), the metrics chart
  (✅), and the latest eval/test/smoke results for the current checkpoint.
- Acceptance: open the app → immediately see what's training, its live log, progress, and latest tests.

### R6 — Tokenizer page is BROKEN — fix it  ✅ DONE
- Fixed: `_resolve_tokenizer_file`/`_candidate_paths` now probe `checkpoints/<run>/tokenizer.json` FIRST.
  Verified: /api/tokenizer vocab 32768, /api/tokenize count 8, /api/tokenizer-health ok — no more "not found".
- (Original symptom below kept for history.)
- Symptom: Tokenizer/TokenizerHealth/Tokenize/Vocab endpoints all return
  `{"error":"tokenizer.json not found: <default locations>"}` — the UI's `_resolve_tokenizer_file`
  default-location probing does not find the tokenizer the live run actually uses.
- Fix: resolve the tokenizer from the real location (storage.yaml tokenizer path / the run's data dir /
  the training manifest). Tie into R7.

### R7 — Preserve the current run's tokenizer next to its output + use it forever (backend)  ◧ PARTIAL
- ✅ DONE (safe, no train disruption): copied the live run's exact tokenizer
  (E:\cola-coder-data\data\typescript-text-math\tokenizer.json, per manifest train_file) →
  checkpoints/small_react_best/tokenizer.json + a tokenizer.source.txt provenance note. The UI now
  resolves it (R6).
- ◻ REMAINING: (a) the trainer copies tokenizer.json into the output dir on checkpoint save for every
  run going forward; (b) resume-training / generate / serve / eval resolve tokenizer next-to-checkpoint
  FIRST (mirror the UI resolver) so a checkpoint is self-describing; (c) record tokenizer path + hash in
  the manifest. These touch checkpoint.py/train.py — do carefully WITH test_checkpoint.py, never disrupt
  the live run.
- Requirement: the tokenizer currently used for the live train must be stored NEXT TO the output so it
  is never lost, and that same tokenizer must be used in the future for continuing training, running
  models, eval, and serving. Hard constraint: must NOT disrupt the current train.
- Copy the EXACT `tokenizer.json` the live `small_react_best` run uses INTO its output dir
  (`checkpoints/small_react_best/tokenizer.json`) so it is never lost. Also record its path + content
  hash in `training_manifest.yaml`.
- Resolution order for ALL future resume-training / generate / serve / eval / UI: tokenizer
  next-to-checkpoint FIRST, then storage.yaml/data-dir fallback. So a checkpoint is self-describing
  about its tokenizer and can be moved/copied as a unit.
- Make the trainer copy the tokenizer into the output dir on checkpoint save for every run going forward.
- Surface "tokenizer used" in the Checkpoints screen detail; this also fixes R6 (UI resolves it).
- **HARD CONSTRAINT: do NOT disrupt the live train.** Copying a file into the output dir is safe (the
  trainer only writes `step_*`/`latest`/`training_manifest.yaml`); NEVER touch the running process,
  its python tree, or its checkpoints. No relaunch.

### R8 — Everything WORKS (no broken flows)  ◻ ONGOING
- No "Failed to fetch" with no explanation (✅ connection banner added). Each screen's actions verified
  against a live server. Fix bugs as found (e.g. R6).

### R9 — Full CLI master-menu parity (every menu item has a UI, 1:1)  ◻ ONGOING / AUDIT NEEDED
- Original standing goal: "full parity with the CLI master menu — every menu item -> typed FastAPI
  endpoint with response_model + typed api client call + polished React view; every action runnable as
  a background job."
- Audit `features/menus/*` (data/training/eval/tools/pipeline) AND `master_menu.py` against the UI;
  produce a parity checklist; cover every gap. The UI must reach everything the CLI can do.
- Known CLI bugs surfaced while building must be fixed (BUG-130/131 done; scrape_github interactive
  BUG-132 open — its launcher is blocked until it has non-interactive flags).

### R10 — Generate / Chat / Inference playground (use the model from the UI)  ◧ GEN+CHAT+FIM DONE
- ✅ `InferenceScreen` + `POST /api/generate` (UI-074): one-shot code gen, sampling controls, completion
  + token/elapsed stats.
- ✅ `ChatScreen` + `POST /api/chat` (UI-075): multi-turn chat, ChatML-or-plain formatting (`use_chat_template`),
  per-reply stats, transcript bubbles, system-prompt seed.
- ✅ `FimScreen` + `POST /api/fim` (UI-076): fill-in-the-middle (prefix/suffix), infill + stitched preview;
  clear message when the tokenizer lacks `<|fim_*|>` tokens.
- All three: a "Playground" nav group. Model loaded per request and freed (no persistent VRAM). GPU-aware —
  REFUSED (409) while training is live via the elevation-proof guard (OPS-002, shared `_training_busy()`);
  frontend disables the action + shows a banner when `trainingAlive`.
- ◻ REMAINING: best-of-N + --repo context, streaming output (SSE). Build on the same gated pattern.

### R11 — Settings / features / storage editing  ◻ PARTIAL
- features.yaml toggles (FeaturesPanel ✅ exists), storage.yaml view (StoragePanel ✅), config EDIT
  (write path) is still read-only (backlog UI-018). Fold into the relevant screens.

---

## 2. Phased plan (execution order)

- **Phase A (now):** integrate the 3 built screens (Checkpoints/Data/Eval) into routing, keeping dropped
  tools in a temporary "More tools" area so NOTHING is lost (R2). Fix the Tokenizer resolver (R6).
- **Phase B:** Live training monitor on the dashboard — live log tail + progress + latest tests (R5).
- **Phase C:** Typed `ActionForm` system + populate `ActionParam` for all actions, 1:1 (R3); rebuild
  Run & Jobs as master-detail using it.
- **Phase D:** One-click full Training pipeline launcher with all options (R4).
- **Phase E:** System & Tools master-detail rebuild (R2); Configs & Pipeline + Tokenizer screens (R2).
- **Phase F:** Fold the "More tools" extras into their screens so each page is fully 1:1 and clean (R2).
- **Phase G (backend):** Tokenizer preservation — copy current run's tokenizer next-to-output now (safe,
  no train disruption), resolve-from-there everywhere, trainer copies on save going forward (R7).
- **Phase H:** Inference playground — generate / chat / FIM, GPU-aware, never disrupts the train (R10).
- **Phase I:** Full CLI-menu parity audit + close every gap; config write/edit (R9, R11).
- **Throughout:** verify each flow against a running server; tsc + vite build green; keep CLI↔UI 1:1.

---

## 3. Status log
- 2026-06-15: Spec created from the user's requirements. App shell (R1) done; 3 master-detail screens
  built (Checkpoints/Data/Eval) pending integration; Dashboard hero + gauges + chart done. Tokenizer
  resolver bug (R6) diagnosed. Everything else above is open and tracked here.
- 2026-06-15: R3 done (UI-070). Typed argument forms shipped — `ActionParam`/`ActionDef.params` schema,
  `action_params.py` 1:1 spec for all 29 actions (validated at import), `/api/actions` merge,
  `ActionForm.tsx` typed controls + `ActionsPanel` rewire. Verified: ruff clean, 111 drift tests,
  /api/actions ActionDef-validated, tsc + vite build green.
- 2026-06-15: R4 + R2-pipeline done (UI-071). One-click full pipeline launcher (PipelineLauncher: mode
  toggle, 10-stage checkboxes, typed options, command preview, trainer-guard) + Configs & Pipeline
  master-detail screen (PipelineScreen: runs list → stage timeline + lifecycle). `train` + `auto_pipeline`
  added to ACTIONS (trainer-guarded). Built via 2 parallel agents on disjoint files. Verified: ruff,
  111 drift tests, tsc clean, vite build green (86 modules). Next: R2 remaining = Tokenizer screen +
  fold "More tools" extras into Checkpoints/Data/Pipeline screens; then R10 inference playground.
