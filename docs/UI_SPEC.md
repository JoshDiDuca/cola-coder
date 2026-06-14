# Cola-Coder Web UI — Specification

> **North-star document.** A fast, beautiful, lightweight **local** dashboard that gives full
> parity with the cola-coder CLI master menu. You can *see* everything (live training, GPU,
> datasets + previews + score histograms, streaming job logs) and *execute* every CLI action as
> a background job — the UI need not stay open for the work to continue.

- **Frontend:** React 18 + TypeScript 5 + Vite 6 — `webui/`
- **Backend:** FastAPI, thin wiring over status/jobs/datasets helpers — `src/cola_coder/ui/`
- **Dev:** `npm run dev` (Vite :5173) proxies `/api` → FastAPI `:8800`
- **Prod:** FastAPI serves the built bundle from `src/cola_coder/ui/static/`

---

## 1. Vision & Principles

The CLI master menu is powerful but serial and ephemeral: one terminal, one action at a time, no
live picture of training, and every long task pins the terminal. The Web UI keeps the CLI's
*model of the world* (the same scripts, the same artifacts) while adding **observability** and
**non-blocking execution**.

| Principle | What it means in practice |
|-----------|---------------------------|
| **Fast** | Sub-second first paint. Polling, not heavyweight realtime frameworks. No client-side router, no state library, no CSS framework. |
| **Lightweight** | Zero runtime deps beyond `react` + `react-dom`. The whole bundle is hand-written components + one CSS file. No Tailwind/MUI/chart libs — histograms and bars are CSS. |
| **Standalone** | The UI is read-mostly and survives the trainer. It reads log/err files and `nvidia-smi`; it does not embed the trainer. Closing the browser never affects a running job. |
| **Background-job based** | Every "do something" action spawns a detached subprocess via `JobManager` and returns immediately. Progress is observed by tailing the job's log file. |
| **Beautiful UX** | A cohesive dark theme (GitHub-dark palette), monospace metrics, calm motion, card grid that reflows responsively. |
| **Never disrupts live training** | `POST /api/train/start` **refuses** with HTTP 409 if any `train.py` process is already running (`is_training_running()` via psutil). Checkpoint corruption loses days of GPU time — the guard is non-negotiable. |
| **Safe by construction** | The runner only launches scripts from a server-side allow-list (`ACTIONS`). The localhost UI cannot be coaxed into running an arbitrary binary. |

---

## 2. Architecture

### 2.1 Frontend (`webui/`)

```
webui/
  index.html               Vite entry
  vite.config.ts           dev proxy /api → 127.0.0.1:8800, build → dist/
  tsconfig.json            strict: true, noUnusedLocals/Parameters, bundler resolution
  package.json             react + react-dom only; scripts: dev, build, typecheck, preview
  src/
    main.tsx               React root mount
    App.tsx                useStatus() 2.5s poll + useClock() 1s; renders the card grid
    api.ts                 typed fetch client — one function per endpoint
    types.ts               shared response interfaces (1:1 with backend JSON)
    index.css              the entire design system (CSS variables + class vocabulary)
    components/
      TrainingPanel.tsx    live step/loss/ppl/tok-s + progress bar
      SystemPanel.tsx      GPU name/util/mem/power
      CheckpointsPanel.tsx per-model checkpoint table
      ActionsPanel.tsx     allow-listed script runner
      JobsPanel.tsx        job list + live log tail
      DatasetsPanel.tsx    dataset browser + preview + score histogram
```

**Component model.** Plain function components, local `useState`/`useEffect`. No context, no
reducer, no router. `App.tsx` owns the single shared poll (`/api/status`) and passes slices down
as props; panels that own their own data (Jobs, Datasets, Actions) poll independently. This keeps
each panel a self-contained, individually-shippable unit — ideal for parallel agents.

**The typed client (`api.ts`).** Every endpoint has exactly one exported function returning a
typed `Promise`. A private `j<T>(url, opts)` helper does `fetch` + `res.ok` check + `json()` cast;
`postJson(url, body)` builds the JSON `RequestInit`. New endpoints add one function + one type —
never an inline `fetch` in a component.

**Polling cadences.**

| Data | Source | Cadence | Owner |
|------|--------|---------|-------|
| Training + GPU + checkpoints | `/api/status` | 2.5 s | `App.useStatus` |
| Clock | local | 1 s | `App.useClock` |
| Jobs list | `/api/jobs` | ~2 s (planned: pause when no running jobs) | `JobsPanel` |
| Selected job log tail | `/api/jobs/{id}/log` | ~1.5 s while open | `JobsPanel` |
| Datasets list | `/api/datasets` | on mount + manual refresh | `DatasetsPanel` |

Errors are swallowed quietly to **keep the last good state** — a transient backend hiccup must not
blank the dashboard.

### 2.2 Backend (`src/cola_coder/ui/`)

```
ui/
  app.py        create_app(...) — FastAPI factory, thin wiring + ACTIONS allow-list
  status.py     pure lib: parse log/err, nvidia-smi, enumerate checkpoints (never raises)
  jobs.py       JobManager: detached subprocess launch, log tee, poll, stop, training guard
  datasets.py   read-only .npy/.jsonl listing, preview, .weights.npy score summary
  static/       built frontend bundle (served at /) — present in prod only
```

**Thin-wiring philosophy.** `app.py` is just route → helper. All real logic lives in
`status`/`jobs`/`datasets`, which are torch-free, Rich-free, and trivially unit-testable. The
factory `create_app(*, job_manager, project_root, data_root, ckpt_root, log_path, err_path)`
takes injectable roots so tests can run against temp dirs.

**Job model.** `JobManager.start(name, cmd, cwd)` opens a per-job log file
(`ui_jobs/{name}-{id}.log`), launches `subprocess.Popen` with `stdout`+`stderr` merged into it and
`stdin=DEVNULL`, and records `{id, name, pid, status, cmd, log, started}`. `_refresh()` polls
`proc.poll()` → `running`/`done`/`failed`. Jobs are detached: they outlive the browser and the
server process restart is the only thing that forgets the in-memory registry (logs persist on
disk — see roadmap "job persistence").

### 2.3 Contract & types

The backend JSON shapes are mirrored 1:1 in `types.ts`. Discipline: **change a response shape →
change the interface in the same PR**. Current contract types: `TrainingStatus`, `SystemStatus`,
`Checkpoint`, `StatusResponse`, `Dataset`, `Job`, `ActionDef`, `ScoreSummary`, `Preview`.

### 2.4 Dev proxy & prod static-serve

- **Dev:** `npm run dev` runs Vite on :5173 with a proxy: `'/api' → 'http://127.0.0.1:8800'`. Start
  the backend (`uvicorn`/serve script) on :8800 separately. HMR for the frontend; backend reloads
  independently.
- **Prod:** `npm run build` (`tsc -b && vite build`) emits `webui/dist/`. Those assets are copied
  to `src/cola_coder/ui/static/`; `GET /` returns `static/index.html` and the bundle is served from
  the same origin, so no proxy/CORS is needed. (Build-to-static copy step is a planned menu action.)

---

## 3. Design System

All tokens and classes live in `webui/src/index.css`. **No utility framework** — components compose
this fixed vocabulary, which keeps the bundle tiny and the look consistent.

### 3.1 Color tokens (CSS variables on `:root`)

| Variable | Value | Role |
|----------|-------|------|
| `--bg` | `#0d1117` | page background |
| `--panel` | `#161b22` | card background |
| `--panel2` | `#1c2230` | inset / input / hover surface |
| `--border` | `#2d3340` | hairline dividers, card borders |
| `--text` | `#e6edf3` | primary text |
| `--muted` | `#8b949e` | labels, secondary text |
| `--accent` | `#4f9cf9` | primary action, links, bars |
| `--good` | `#3fb950` | live / done / success |
| `--warn` | `#d29922` | warnings |
| `--bad` | `#f85149` | dead / failed / errors |
| `--mono` | `ui-monospace, …, Consolas, monospace` | all numeric/metric/log text |

### 3.2 Class vocabulary

| Class | Purpose |
|-------|---------|
| `app-header`, `app-header h1`, `.clock` | sticky top bar with status dot, title, live clock |
| `app-grid` | responsive `auto-fit minmax(330px,1fr)` card grid, max-width 1200px |
| `card`, `card-wide`, `card-title` | the panel primitive; `card-wide` spans full grid row; uppercase tracked title |
| `stat-big`, `stat-sub` | hero metric (30px mono) + sub-caption |
| `row`, `.k`, `.v` | key/value row with bottom hairline; `.k` muted label, `.v` mono value |
| `bar` > `.fill` | progress bar; gradient accent→good, animated width |
| `tbl`, `th`, `td`, `.right` | dense data table, uppercase headers, row hover, right-align numerics |
| `mono`, `muted`, `err` | inline text modifiers |
| `btn`, `btn-primary`, `btn-danger`, `:disabled` | button system |
| `input`, `select` | form controls with accent focus ring |
| `tag`, `.running`, `.done`, `.failed` | pill status badges (colored per job state) |
| `pre`, `.scroll` | log viewer (`#0a0d12` bg, mono, wrap, max-height scroll) |
| `hist`, `.hist .b` | flex-end CSS-bar histogram (score distribution) |
| `dot`, `.live`, `.dead` | status indicator; `.live` pulses (keyframe), `.dead` static red |

### 3.3 Typography, spacing, motion, accessibility

- **Typography:** system sans for chrome, `--mono` for every number, metric, path, and log line.
- **Spacing:** 16–24px gutters, 18px card padding, 10–12px intra-card gaps; one rhythm everywhere.
- **Motion:** restrained — 0.15s control transitions, 0.3–0.4s bar/histogram growth, a 2s `pulse`
  on the live dot. Motion communicates state; it never decorates.
- **Accessibility (target):** keyboard-focusable controls with visible accent focus ring (present);
  *planned* — `aria-live="polite"` on the training metric + job-status regions, semantic table
  headers (present), color-plus-text status (tags carry text labels, not color alone), and a
  `prefers-reduced-motion` guard to disable the pulse.

---

## 4. CLI → UI Parity Map

Every master-menu item and sub-menu action, mapped to its UI view, the FastAPI endpoint that backs
it, and the script it ultimately runs. **Status:** ✅ shipped · 🟡 partial · ⬜ to-build.

Legend for endpoint column: existing endpoints are named; `POST /api/run {action}` means it flows
through the generic allow-listed runner (the `action` key must be added to `ACTIONS`).

### 4.1 Top-level master menu

| # | CLI item | UI view | Endpoint | Backing script | Status |
|---|----------|---------|----------|----------------|--------|
| 1 | Quick Start Pipeline | Pipeline view → Quick Start | `POST /api/run` (chain) | train_tokenizer / prepare_data / train | ⬜ |
| 2 | Full Auto Pipeline | Pipeline view → Auto | `POST /api/pipeline/auto` | auto_pipeline.py | ⬜ |
| 3 | Data Pipeline | Datasets + Data Wizard | several (below) | collect/prepare/score | 🟡 |
| 4 | Training | Training view | `POST /api/train/start` | train.py | ✅ |
| 5 | Instruction Tuning | Post-Train view | `POST /api/run` | generate_sft_data / train_sft / train_reasoning | ⬜ |
| 6 | Generate & Interact | Generate view | `POST /api/run` + serve | run/generate/chat/serve | ⬜ |
| 7 | Evaluate & Benchmark | Eval view | `POST /api/run` | evaluate / ts_benchmark / smoke_test / … | 🟡 (smoke/evaluate shipped) |
| 8 | Router & Specialists | Router view | `POST /api/run` | train_router / evaluate_router | ⬜ |
| 9 | Tools & Utilities | Tools view | `POST /api/run` | tests / lint / env_check / export | 🟡 (vram/health shipped) |
| 10 | Project Memory | Memory view | `GET/POST /api/memory` | (memory store) | ⬜ |
| 11 | Retrieval & Search | RAG view | `POST /api/search` | retrieval.rag | ⬜ |
| 12 | Settings | Settings view | `GET/PUT /api/settings` | features.yaml / storage.yaml | ⬜ |
| 13 | Training Status | Training view (header dot + panel) | `GET /api/status` | log/err parse | ✅ |

### 4.2 Data Pipeline (data_menu)

| CLI action | UI view | Endpoint | Backing script | Status |
|------------|---------|----------|----------------|--------|
| Collect — GitHub API | Data Wizard → Collect | `POST /api/run` | scrape_github.py | ⬜ |
| Collect — Browse/Import HuggingFace (preview, lang filter, download) | Data Wizard → HF | `GET /api/hf/preview`, `POST /api/run` | HuggingFaceSource / prepare_data | ⬜ |
| Collect — Software Heritage | Data Wizard → SWH (info) | `GET /api/info/swh` | software_heritage source | ⬜ |
| Collect — Scrape Framework Docs | Data Wizard → Docs | `POST /api/run` | scrape_docs.py | ⬜ |
| Collect — Prepare Docs / Repo-context data | Data Wizard | `POST /api/run` | prepare_docs_data / prepare_repo_context_data | ⬜ |
| Collect — Text (FineWeb) / Math (OpenWebMath) | Data Wizard | `POST /api/run` | collect_data.py (sources) | ⬜ |
| Collect — GitHub issues/PRs / Instruction datasets | Data Wizard | `POST /api/run` | collect_data / downloaders | ⬜ |
| Modify — Combine datasets (weighted) | Datasets → Combine | `POST /api/run` | combine_datasets.py | ⬜ |
| Modify — Generate instructions | Data Wizard | `POST /api/run` | generate_instructions.py | ⬜ |
| Score — Code quality / repos | Datasets → Score | `POST /api/run` | score_repos / train_quality_classifier | ⬜ |
| Score — HuggingFace samples (histogram of grades) | Data Wizard → Score HF | `GET /api/hf/score` | HuggingFaceSource + scorers | ⬜ |
| Score — Run scoring pipeline (tsc+eslint+heuristic) | Datasets → Score | `POST /api/run` | score_data.py | 🟡 (`score_data` in ACTIONS) |
| Score — LLM-as-Judge annotate / train judge classifier | Datasets → Score | `POST /api/run` | train_judge_classifier.py | ⬜ |
| Score — Apply curriculum ordering | Datasets → Score | `POST /api/run` | score_data.py --curriculum | ⬜ |
| Score — Scan data for malware | Datasets → Scan | `POST /api/datasets/scan` | security.scanner | ⬜ |
| Score — Advanced filters (info) | Datasets (info panel) | `GET /api/info/filters` | filters registry | ⬜ |
| Inspect — Inspect dataset (decode samples) | **Datasets browser + preview** | `GET /api/datasets`, `/preview` | datasets.py | ✅ |
| Inspect — Data statistics | Datasets → Stats | `POST /api/run` | data_stats.py | 🟡 (`data_stats` in ACTIONS) |
| Inspect — Prepare FIM data | Data Wizard | `POST /api/run` | prepare_fim_data.py | ⬜ |
| Prepare — quick modes / interactive / enhanced wizard | **Data collection wizard** | `POST /api/run` | prepare_data.py / prepare_data_interactive.py | 🟡 (`prepare_data` in ACTIONS) |
| Prepare — Mixed (code+text+math) / repo-level | Data Wizard | `POST /api/run` | collect_data.py | 🟡 (`collect_data` in ACTIONS) |
| Inspect — score histogram (`.weights.npy`) | **Datasets → Score Histogram** | `GET /api/datasets/scores` | datasets.score_summary | ✅ |

### 4.3 Training — the 10 stages (training_menu + pipeline_menu)

| Stage | CLI action | UI view | Endpoint | Backing script | Status |
|-------|------------|---------|----------|----------------|--------|
| 1 | Train Tokenizer | Foundation | `POST /api/run` | train_tokenizer.py | ⬜ |
| 2 | Prepare / mix data | Data Wizard | `POST /api/run` | prepare_data.py | 🟡 |
| 3 | Pre-Train (select size) | **Training view** | `POST /api/train/start` | train.py | ✅ |
| 3 | Resume training | Training → Resume (pick checkpoint) | `POST /api/train/start {resume}` | train.py --resume | ✅ |
| 3 | Background training (GPU throttle, schedule) | Training → Background | `POST /api/train/background` | background_train.py | ⬜ |
| 4 | Extend context (YaRN) | Post-Train | `POST /api/run` | train.py (rope_scaling) | ⬜ |
| 5 | Generate instruction data | Post-Train | `POST /api/run` | generate_instructions.py | ⬜ |
| 6 | Instruction tuning (SFT) | Post-Train | `POST /api/run` | train_sft.py | ⬜ |
| 7 | MoE upcycling | Post-Train | `POST /api/run` | upcycle_to_moe.py | ⬜ |
| 7.5 | Fine-tune upcycled MoE | Post-Train | `POST /api/run` | train.py --resume (derived cfg) | ⬜ |
| — | Generate distillation data | Post-Train | `POST /api/run` | generate_distillation_data.py | ⬜ |
| — | Generate RFT data (self-verified) | Post-Train | `POST /api/run` | generate_rft_data.py | 🟡 (`generate_rft` in ACTIONS) |
| 8 | Train semantic router | Router view | `POST /api/run` | train_router.py | ⬜ |
| 9 | Train reasoning (GRPO) | Reasoning/RFT view | `POST /api/run` | train_reasoning.py | ⬜ |
| 9 | Self-play training | Reasoning/RFT view | `POST /api/run` | train_reasoning.py --iterations | ⬜ |
| 10 | Evaluate (smoke + HumanEval + report) | Eval view | `POST /api/run` | smoke_test / evaluate / quality_report | 🟡 |
| — | VRAM estimation | Monitoring | `POST /api/run` | vram_estimate.py | 🟡 (`vram_estimate` in ACTIONS) |
| — | LR finder | Monitoring | `POST /api/run` | find_lr.py | ⬜ |
| — | Training dashboard / eval history | **Training view** (native) | `GET /api/status` (+ planned `/api/train/history`) | training_dashboard / training_eval_history | 🟡 |

### 4.4 Pipeline Manager (pipeline_menu — named runs)

| CLI action | UI view | Endpoint | Backing | Status |
|------------|---------|----------|---------|--------|
| Full Auto Pipeline | Pipeline → Auto | `POST /api/pipeline/auto` | hardware_profiler + run_manager | ⬜ |
| New / Resume / View runs | **Pipeline runs view** (stage state machine) | `GET/POST /api/pipeline/runs`, `/runs/{name}` | PipelineRunManager (`pipeline_runs/{name}.json`) | ⬜ |
| Run single stage / Reset to stage | Pipeline → run detail | `POST /api/pipeline/runs/{name}/stage` | run_manager | ⬜ |
| Delete run | Pipeline runs view | `DELETE /api/pipeline/runs/{name}` | run_manager | ⬜ |
| Quick / legacy full pipeline | Pipeline → Quick | `POST /api/run` | full_pipeline.py | ⬜ |

### 4.5 Evaluate & Benchmark (eval_menu)

| CLI action | UI view | Endpoint | Backing script | Status |
|------------|---------|----------|----------------|--------|
| TypeScript benchmark (+categories) | Eval view | `POST /api/run` | ts_benchmark.py | ⬜ |
| TS nano / TS-React benchmark | Eval view | `POST /api/run` | nano_benchmark / ts_benchmark | ⬜ |
| Python HumanEval (pass@k) | **Eval view** | `POST /api/run` | evaluate.py | 🟡 (`evaluate` in ACTIONS) |
| Python completion / mixed quick benchmark | Eval view | `POST /api/run` | completion_benchmark / benchmark | ⬜ |
| Run ALL benchmarks | Eval view | `POST /api/run` | run_eval_suite.py | ⬜ |
| Inference profiler | Eval view | `POST /api/run` | inference_benchmark.py | ⬜ |
| Smoke test | Eval view | `POST /api/run` | smoke_test.py | ✅ (`smoke_test` in ACTIONS) |
| Regression tests / quality report / model card | Eval view | `POST /api/run` | regression_test / quality_report / model_card | ⬜ |
| Compare checkpoints / models / diff / info | **Checkpoints view** → Compare | `POST /api/run` | compare_checkpoints / compare_models / checkpoint_diff / checkpoint_info | 🟡 (list shipped) |
| Safety evaluation (suites) | Eval view | `POST /api/run` | safety_eval.py | ⬜ |
| Routing accuracy | Router view | `POST /api/run` | evaluate_router.py | ⬜ |
| Data contamination | Eval view | `POST /api/run` | check_contamination.py | ⬜ |
| Domain detection / router accuracy (in-proc) | Router view | `GET /api/router/eval` | router_evaluation/domain_detector | ⬜ |

### 4.6 Router & Specialists / Generate / Tools / Retrieval

| CLI action | UI view | Endpoint | Backing | Status |
|------------|---------|----------|---------|--------|
| Generate router data / train / evaluate router | Router view | `POST /api/run` | generate_router_data / train_router / evaluate_router | ⬜ |
| Specialist registry / MoE config / domain test | Router view | `GET /api/router/specialists` | specialist_registry / moe_layer | ⬜ |
| Quick generate / interactive / best-of-N / context-aware | Generate view | `POST /api/run` | run / generate.py | ⬜ |
| Multi-turn chat | Generate → Chat | `POST /api/chat` (SSE) or serve | chat.py / serve.py | ⬜ |
| Serve API | Generate → Serve | `POST /api/run` | serve.py | ⬜ |
| Tests / lint | Tools view | `POST /api/run` | pytest / ruff | ⬜ |
| GPU status | **System panel** (native) | `GET /api/status` | nvidia-smi | ✅ |
| Env check / project health / tokenizer health | Tools view | `POST /api/run` | env_check / project_health / tokenizer_health | 🟡 (`project_health`,`tokenizer_health` in ACTIONS) |
| Feature toggles / storage paths | Settings view | `GET/PUT /api/settings` | features.yaml / storage.yaml | ⬜ |
| Export (GGUF/Ollama/quantized) / model card | Export view | `POST /api/run` | export_model / model_card | ⬜ |
| Index repo / semantic search / RAG config / vector stats | RAG view | `POST /api/search`, `GET /api/rag/stats` | retrieval.rag / vector_store | ⬜ |

**Parity-table row count: 84 rows** across the 6 sub-tables (13 top-level + 21 data + 19 training +
6 pipeline + 13 eval + 12 router/generate/tools/retrieval). Shipped today: the **Training**,
**System/GPU**, **Checkpoints (list)**, **Datasets (browse/preview/score-histogram)**, **Jobs (list
+ live log)**, and **Actions** views, backed by `/api/status`, `/api/datasets[/preview|/scores]`,
`/api/jobs[/{id}/log|/stop]`, `/api/actions`, `/api/run`, and `/api/train/start`.

---

## 5. Views

### Shipped

**Training panel** — props: `TrainingStatus`. Hero `stat-big` loss, sub metrics (step / total,
ppl, tok/s or s/it), a `bar` progress fill from `progress_pct`, and the raw `last_log_line` in a
mono row. Header `dot` reflects `alive`. *Planned:* a small sparkline of recent loss and a
**Start/Resume Training** form (config select + optional resume checkpoint) wired to
`/api/train/start`, disabled with an explanatory note when `alive` (the no-second-trainer guard).

**System (GPU) panel** — props: `SystemStatus`. GPU name, util %, mem used/total (with a `bar`),
power draw. All-null tolerant (renders "—" when `nvidia-smi` is unavailable).

**Checkpoints panel** — props: `Checkpoint[]`. `tbl` of model / step / loss / mtime, grouped by
model, newest first. *Planned:* select-two-to-compare → routes to compare endpoints; "set latest".

**Datasets browser + preview + score histogram** — owns its data. Left: `tbl` of datasets
(`name`, `kind`, samples, size, `has_weights` tag). Selecting a row calls `/preview` (renders npy
shape/dtype + first-N rows, or jsonl records) and, when `has_weights`, `/scores` →
`hist` bars over the 10-bin distribution with mean/min/max. *Planned:* decode npy token rows via
tokenizer for human-readable previews; malware-scan button.

**Jobs + live logs** — owns its data. `tbl` of jobs with `tag` status (running/done/failed),
name, pid, started, returncode. Selecting a job opens a `pre` log viewer that tails
`/api/jobs/{id}/log?lines=` every ~1.5s; a `btn-danger` **Stop** posts `/stop`. This is the
universal observation surface for *every* action.

**Actions runner** — fetches `/api/actions` and renders a card per allow-listed action with its
label and default args (editable, *planned*). Clicking runs `POST /api/run {action}` and the new
job appears in the Jobs panel. Server-side allow-list is the security boundary.

### Planned

| View | Data | Key interactions |
|------|------|------------------|
| **Pipeline runs** | `GET /api/pipeline/runs` (state machine per `pipeline_runs/{name}.json`) | Create/resume/reset/delete; per-stage status icons (pending/running/completed/failed/skipped); run a single stage; artifact chain display |
| **Eval results** | `GET /api/eval/results` (parsed report JSON) | Trigger eval suites; render pass@k tables, per-category bars, safety/contamination summaries |
| **Config editor** | `GET/PUT /api/configs/{name}` | View/edit `configs/*.yaml` with validation; never mutate a running run's config |
| **Data collection wizard** | HF preview/score endpoints | Multi-step: dataset → languages → filter mode → dedup → score → launch `prepare_data`/`collect_data` as a job |
| **Reasoning / RFT** | `POST /api/run` | GRPO setup (reward by language, problem set, group size), self-play iterations, distillation/RFT data generation |
| **Export** | `POST /api/run` | GGUF / Ollama / quantized export; model card generation |
| **Settings** | `GET/PUT /api/settings` | Feature toggles (`features.yaml`), storage paths (`storage.yaml`) |
| **Router & Generate** | `/api/router/*`, `/api/chat` | Router train/eval, specialist registry, code generation + chat (SSE) |

---

## 6. Endpoint Catalog

### Current (in `app.py`)

| Method | Path | Params / body | Response shape |
|--------|------|---------------|----------------|
| GET | `/` | — | HTML dashboard (`static/index.html`) |
| GET | `/api/status` | — | `{training: TrainingStatus, system: SystemStatus, checkpoints: Checkpoint[]}` |
| GET | `/api/datasets` | `?data_root=` | `Dataset[]` |
| GET | `/api/datasets/preview` | `?path=&n=20` | `Preview` (npy: shape/dtype/num_samples/preview; jsonl: num_samples/preview; or `{error}`) |
| GET | `/api/datasets/scores` | `?path=` | `ScoreSummary` `{n,mean,min,max,histogram[10],bins[11]}` or `{error}` |
| GET | `/api/jobs` | — | `Job[]` (freshly polled) |
| GET | `/api/jobs/{id}/log` | `?lines=200` | `{log: string}` (last N lines; `404` if unknown) |
| POST | `/api/jobs/{id}/stop` | — | `{stopped: bool}` (`404` if unknown) |
| GET | `/api/actions` | — | `ActionDef[]` `{key,script,label,args}` |
| POST | `/api/run` | `{action: string, args?: string[]}` | `Job` (`400` if `action` not in allow-list) |
| POST | `/api/train/start` | `{config?: string, resume?: string}` | `Job` (`200`) or `{error}` (**`409`** if a trainer already runs) |
| GET | `/api/docs` | — | FastAPI Swagger UI |

`TrainingStatus = {alive, step, total_steps, progress_pct, loss, ppl, tok_per_s, s_per_it,
last_log_line}` (all nullable except `alive`). `SystemStatus = {gpu_name, gpu_util_pct,
gpu_mem_used_mb, gpu_mem_total_mb, gpu_power_w}` (all nullable). `Checkpoint = {model, name, step,
loss, path, mtime}`. `Job = {id, name, pid, status, cmd[], log, started, returncode}`.

### Planned

| Method | Path | Body / params | Response |
|--------|------|---------------|----------|
| GET | `/api/jobs/{id}/stream` | SSE | streamed log lines (replaces polling for active job) |
| POST | `/api/train/background` | `{config, duration?, stop_at?, gpu_clock?, gpu_power?}` | `Job` / `409` |
| GET/POST | `/api/pipeline/runs` | list / `{name, config, skip_stages}` | run summaries / created run |
| GET | `/api/pipeline/runs/{name}` | — | full stage state machine |
| POST | `/api/pipeline/runs/{name}/stage` | `{stage, action: run\|reset\|override, value?}` | updated run |
| DELETE | `/api/pipeline/runs/{name}` | — | `{deleted: bool}` |
| POST | `/api/pipeline/auto` | `{mode: smoke\|full\|dry}` | created run / plan |
| GET | `/api/hf/preview` | `?dataset=&split=&languages=&n=` | sample previews |
| GET | `/api/hf/score` | `?dataset=&languages=&scorer=&n=` | grade histogram |
| POST | `/api/datasets/scan` | `{path, mode}` | scan result (clean / threats) |
| GET | `/api/eval/results` | `?checkpoint=` | parsed eval/report JSON |
| GET/PUT | `/api/configs/{name}` | yaml body | config contents / validation result |
| GET/PUT | `/api/settings` | features/storage | settings snapshot |
| GET | `/api/router/eval` | `?router=&domains=` | accuracy + per-domain P/R/F1 |
| POST | `/api/chat` | `{checkpoint, messages}` (SSE) | streamed tokens |
| POST | `/api/search` | `{query, top_k}` | RAG results |

---

## 7. Background-Job & Safety Model

**Launch.** Actions and training go through `JobManager`. `start()` writes a per-job log file,
spawns a detached `Popen` (`stdin=DEVNULL`, stdout+stderr → log), and returns metadata
immediately. The HTTP request never blocks on the work.

**Observe.** `list()`/`get()` re-poll `proc.poll()` on every call (no background thread needed),
mapping returncode → `running`/`done`/`failed` and closing the log handle on exit. The UI tails
`/api/jobs/{id}/log?lines=N`, which returns the last N lines of the on-disk log. *Planned:* an SSE
`/stream` endpoint for push-based tailing of the active job (lower latency, fewer requests).

**Stop.** `POST /api/jobs/{id}/stop` calls `proc.terminate()`; returns `{stopped:false}` if the
process already exited. Training jobs are meant to be stopped via their own graceful
checkpoint-on-stop path (background trainer lock file) rather than hard-killed.

**No-second-trainer guard.** `start_training()` first calls `is_training_running()` — psutil scans
process cmdlines for `train.py`. If found, it **returns `{"error": "training already running"}`**
and the route responds **HTTP 409**; no process is launched. This is the single most important
safety invariant (per `.claude/rules/checkpoints.md`: never interrupt an active training run). The
UI must surface this as a clear, non-alarming "training already in progress" state and disable the
start control.

**Allow-list.** `POST /api/run` rejects any `action` not in `ACTIONS` with HTTP 400. The script
name is server-controlled; only `args` come from the client. This prevents the localhost UI from
becoming an arbitrary-command-execution surface.

**Durability note (roadmap).** The job registry is in-memory, so a backend restart forgets live
jobs (logs persist on disk). Phase work adds a small on-disk job index so the Jobs view survives
restarts and can re-attach to still-running detached processes by PID.

---

## 8. Phased Roadmap

Each phase is a small, shippable batch. Items within a phase are independent — **one view +/or one
endpoint per agent** — so parallel agents can fan out.

**Phase 0 — Harden what's shipped** (foundation)
- `aria-live` + reduced-motion; `tsc --noEmit` + `npm run build` green in CI.
- Backend tests for every current endpoint (happy path + 404/400/409).
- Editable action args in `ActionsPanel`; pause Jobs polling when no job is running.

**Phase 1 — Training control & observability**
- Start/Resume Training form (config select + checkpoint picker) → `/api/train/start`, guard-aware.
- Loss sparkline from a rolling `/api/status` history.
- `GET /api/train/history` (eval snapshots) + Background-training start/stop/schedule.

**Phase 2 — Jobs durability & streaming**
- On-disk job index (survive restart, re-attach by PID).
- SSE `/api/jobs/{id}/stream`; switch the active-job viewer to push.

**Phase 3 — Data collection wizard**
- HF preview + score endpoints; multi-step wizard launching `prepare_data`/`collect_data` jobs.
- Dataset malware-scan button; tokenizer-decoded npy previews.

**Phase 4 — Pipeline runs**
- `/api/pipeline/runs*` over `PipelineRunManager`; stage state-machine view; Full Auto Pipeline.

**Phase 5 — Eval & Checkpoints**
- Eval suites → parsed results view (pass@k, categories, safety, contamination).
- Checkpoint compare/diff/info from the Checkpoints view.

**Phase 6 — Post-training & alignment**
- SFT / context-extend / MoE upcycle+finetune / distillation / RFT / GRPO / self-play launchers.

**Phase 7 — Generate, Router, Export, Settings, RAG**
- Generate + chat (SSE), router train/eval + specialists, export (GGUF/Ollama), settings editors,
  semantic search.

To add any allow-listed action: append a key to `ACTIONS` in `app.py`, add the run wiring, and (if
it needs a bespoke UI) add a panel. Each is a self-contained PR.

---

## 9. Quality Bar

| Dimension | Requirement |
|-----------|-------------|
| **TypeScript** | `strict: true` with `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`. No `any` in committed code; response casts go through `api.ts`/`types.ts` only. |
| **Type-check & build** | `npm run typecheck` (`tsc --noEmit`) and `npm run build` (`tsc -b && vite build`) must be **green** before any PR merges. Run both after every TS change. |
| **Backend tests** | Every endpoint has a pytest test via `create_app(...)` with injected temp roots + a fake `JobManager`. The 409 guard, 400 allow-list rejection, and 404s are explicitly covered. Status/datasets helpers tested against fixture files (incl. malformed → never raises). |
| **Performance budget** | Bundle stays dependency-light (react + react-dom only). First paint < 1s on localhost. `/api/status` returns in < 100ms typical (file reads + one `nvidia-smi` with a 5s timeout). Polling cadences as in §2.1; no busy-loops. |
| **Maintainability** | Thin `app.py` (wiring only); logic in pure helpers. One `api.ts` function + one `types.ts` interface per endpoint — never inline `fetch`. One CSS vocabulary (§3) — no per-component style sprawl, no new CSS framework. New panels follow the `card`/`card-title`/`row`/`tbl` pattern. |
| **Safety** | The allow-list and no-second-trainer guard are load-bearing — never weaken them. Untrusted/destructive scripts keep their existing sandbox/verification on the backend; the UI only *launches* allow-listed scripts, it does not bypass any safety in the scripts themselves. |

---

*This document is the build target for the autonomous scheduler and any parallel agents. Keep it in
sync: when a row in §4 ships, flip its status; when an endpoint lands, move it from "Planned" to
"Current" in §6.*
