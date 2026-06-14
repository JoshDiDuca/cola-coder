"""FastAPI app for the cola-coder local UI.

Endpoints (all JSON unless noted):
    GET  /                       -> the built React app (webui/dist) or a hint
    GET  /api/status             -> {training, system, checkpoints}
    GET  /api/events             -> Server-Sent Events stream of full snapshots
    GET  /api/datasets           -> [datasets]                (?data_root=)
    GET  /api/datasets/preview   -> preview                   (?path=&n=)
    GET  /api/datasets/scores    -> score summary             (?path=)
    GET  /api/jobs               -> [jobs]
    GET  /api/jobs/{id}/log      -> {log: "...tail..."}       (?lines=)
    POST /api/jobs/{id}/stop     -> {stopped: bool}
    GET  /api/actions            -> [runnable script actions]
    POST /api/run                -> start a script as a background job
    POST /api/train/start        -> start training (REFUSES if one already runs)

Live updates are server-PUSHED via ``/api/events`` (one SSE stream per client,
NO polling): a single 1s server-side tick recomputes the snapshot and pushes it
only when it changed. The app is built via ``create_app(...)`` so tests can
inject a temp JobManager, data_root, and checkpoint root. Heavy logic lives in
status/jobs/datasets — this module is thin wiring, which keeps it fast and
trivially testable.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import checkpoint_detail as cd
from . import checkpoints_compare as cc
from . import config_diff as cdf
from . import configs as cfg
from . import data_sources_view as dsv
from . import datasets as ds
from . import eval_history as eh
from . import evals as ev
from . import exports as ex
from . import features as ft
from . import features_write as fw
from . import health as hl
from . import logs as lg
from . import metrics_history as mh
from . import model_card as mc
from . import pipeline as pl
from . import reasoning as rs
from . import router as rt
from . import scripts_catalog as sc
from . import sft_data as sd
from . import status as st
from . import storage_view as sv
from . import system_info as si
from . import schemas as sch
from . import tokenize as tkz
from . import tokenizer_info as tk
from .jobs import JobManager

_MISSING_DIST_HTML = (
    "<h1>Cola-Coder UI</h1>"
    "<p>The React app has not been built yet. Run:</p>"
    "<pre>cd webui &amp;&amp; npm install &amp;&amp; npm run build</pre>"
)

# Scripts the UI is allowed to launch (mirrors the CLI menu actions). Each entry:
# key -> (script filename, human label, default args shown in the UI). Extend this
# as the UI grows toward full CLI parity; the runner validates against these keys so
# the localhost UI can't be coaxed into running an arbitrary binary.
ACTIONS: dict[str, dict] = {
    "prepare_data": {"script": "prepare_data.py", "label": "Prepare data (tokenize/filter)",
                     "args": ["--config", "configs/small.yaml", "--score"]},
    "collect_data": {"script": "collect_data.py", "label": "Collect multi-source data",
                     "args": ["--config", "configs/small.yaml"]},
    "score_data": {"script": "score_data.py", "label": "Score data quality",
                   "args": ["--data", "data/processed/train_data.npy"]},
    "evaluate": {"script": "evaluate.py", "label": "Evaluate (HumanEval pass@k)",
                 "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "smoke_test": {"script": "smoke_test.py", "label": "Smoke test (8 checks)", "args": []},
    "generate_rft": {"script": "generate_rft_data.py", "label": "Generate RFT data (self-verified)",
                     "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "data_stats": {"script": "data_stats.py", "label": "Dataset statistics", "args": []},
    "tokenizer_health": {"script": "tokenizer_health.py", "label": "Tokenizer health check", "args": []},
    "project_health": {"script": "project_health.py", "label": "Project health score", "args": []},
    "vram_estimate": {"script": "vram_estimate.py", "label": "VRAM estimate", "args": []},
    "env_check": {"script": "env_check.py", "label": "Environment check", "args": []},
    "quality_report": {"script": "quality_report.py", "label": "Quality report (syntax/types/tokens)",
                       "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "safety_eval": {"script": "safety_eval.py", "label": "Safety eval (secrets/dangerous patterns)",
                    "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml",
                             "--suite", "basic"]},
    "completion_benchmark": {"script": "completion_benchmark.py", "label": "Completion benchmark",
                             "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "benchmark": {"script": "benchmark.py", "label": "Throughput benchmark (tok/s)", "args": []},
    "regression_test": {"script": "regression_test.py", "label": "Regression test (quality tracking)",
                        "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "ts_benchmark": {"script": "ts_benchmark.py", "label": "TypeScript benchmark (tsc --strict)",
                     "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "depth_profile": {"script": "depth_profile.py", "label": "Depth / early-exit profile",
                      "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "robustness_eval": {"script": "robustness_eval.py", "label": "Robustness eval (perturbations)",
                        "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "generate_instructions": {"script": "generate_instructions.py", "label": "Generate instruction pairs",
                              "args": ["--config", "configs/small.yaml"]},
    # Trainer-class actions — REFUSED by /api/run while a training process is alive (never a 2nd trainer).
    "train_sft": {"script": "train_sft.py", "label": "Instruction tune (SFT)", "trainer": True,
                  "args": ["--data", "data/sft/instructions.jsonl", "--config", "configs/small.yaml",
                           "--checkpoint", "checkpoints/small/latest", "--epochs", "2", "--lr", "2e-5"]},
    "train_reasoning": {"script": "train_reasoning.py", "label": "GRPO reasoning training", "trainer": True,
                        "args": ["--config", "configs/small.yaml", "--sft-warmup", "--reward", "combined"]},
    "train_router": {"script": "train_router.py", "label": "Train semantic router", "trainer": True,
                     "args": ["--arch", "mlp"]},
    "upcycle_to_moe": {"script": "upcycle_to_moe.py", "label": "Upcycle dense -> MoE", "trainer": True,
                       "args": ["--config", "configs/small.yaml"]},
    "find_lr": {"script": "find_lr.py", "label": "LR range finder", "trainer": True,
                "args": ["--config", "configs/small.yaml"]},
    "full_pipeline": {"script": "full_pipeline.py", "label": "Full 10-stage pipeline", "trainer": True,
                      "args": ["--config", "configs/small.yaml"]},
}


def create_app(
    *,
    job_manager: JobManager | None = None,
    project_root: str | Path | None = None,
    data_root: str = "data",
    ckpt_root: str = "checkpoints",
    log_path: str = "train_small_react_best.log",
    err_path: str = "train_small_react_best.err",
) -> FastAPI:
    root = Path(project_root) if project_root else Path.cwd()
    jobs = job_manager or JobManager()
    app = FastAPI(title="Cola-Coder UI", docs_url="/api/docs")

    def _full_snapshot() -> dict:
        """The complete dashboard state pushed over SSE and returned by /api/status (+jobs)."""
        return {
            "training": st.get_training_status(log_path, err_path),
            "system": st.get_system_status(),
            "checkpoints": st.list_checkpoints(ckpt_root),
            "jobs": jobs.list(),
        }

    @app.get("/api/status", response_model=sch.StatusResponse)
    def status() -> dict:
        return {
            "training": st.get_training_status(log_path, err_path),
            "system": st.get_system_status(),
            "checkpoints": st.list_checkpoints(ckpt_root),
        }

    @app.get("/api/events")
    def events() -> StreamingResponse:
        async def gen() -> AsyncIterator[str]:
            last = ""
            try:
                snapshot = _full_snapshot()
                last = json.dumps(snapshot, sort_keys=True, default=str)
                yield f"data: {json.dumps(snapshot, default=str)}\n\n"
                while True:
                    await asyncio.sleep(1.0)
                    snapshot = _full_snapshot()
                    current = json.dumps(snapshot, sort_keys=True, default=str)
                    if current != last:
                        last = current
                        yield f"data: {json.dumps(snapshot, default=str)}\n\n"
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/datasets", response_model=list[sch.Dataset])
    def datasets(data_root: str = data_root) -> list[dict]:
        return ds.list_datasets(data_root)

    @app.get("/api/datasets/preview", response_model=sch.Preview | sch.ErrorResponse)
    def datasets_preview(path: str, n: int = 20) -> dict:
        return ds.dataset_preview(path, n)

    @app.get("/api/datasets/scores", response_model=sch.ScoreSummary | sch.ErrorResponse)
    def datasets_scores(path: str) -> dict:
        return ds.score_summary(path)

    @app.get("/api/jobs", response_model=list[sch.Job])
    def jobs_list() -> list[dict]:
        return jobs.list()

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, lines: int = 200) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        log = Path(job["log"])
        if not log.exists():
            return {"log": ""}
        text = log.read_text(encoding="utf-8", errors="replace")
        return {"log": "\n".join(text.splitlines()[-lines:])}

    @app.post("/api/jobs/{job_id}/stop")
    def job_stop(job_id: str) -> dict:
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"stopped": jobs.stop(job_id)}

    @app.get("/api/actions", response_model=list[sch.ActionDef])
    def actions() -> list[dict]:
        return [{"key": k, **v} for k, v in ACTIONS.items()]

    @app.post("/api/run")
    def run(payload: dict) -> JSONResponse:
        key = payload.get("action")
        if key not in ACTIONS:
            raise HTTPException(status_code=400, detail=f"unknown action: {key!r}")
        spec = ACTIONS[key]
        # Trainer-class actions (SFT/GRPO/router/MoE/LR-finder/full-pipeline) load the model
        # and optimize on the GPU — refuse to launch one while the live trainer is running, so
        # the UI can never spawn a second trainer that fights the pretraining run for VRAM.
        if spec.get("trainer") and jobs.is_training_running():
            return JSONResponse(
                {"error": "training already running — refusing to start a second trainer"},
                status_code=409,
            )
        args = payload.get("args")
        if args is None:
            args = spec["args"]
        cmd = [sys.executable, str(root / "scripts" / spec["script"]), *args]
        job = jobs.start(name=key, cmd=cmd, cwd=str(root))
        return JSONResponse(job)

    @app.post("/api/train/start")
    def train_start(payload: dict) -> JSONResponse:
        config = payload.get("config", "configs/small.yaml")
        resume = payload.get("resume")
        result = jobs.start_training(config=config, resume=resume)
        # start_training returns {"error": ...} when a trainer is already running.
        if "error" in result:
            return JSONResponse(result, status_code=409)
        return JSONResponse(result)

    @app.get("/api/configs", response_model=list[sch.ConfigFile])
    def configs_list() -> list[dict]:
        return cfg.list_configs(str(root / "configs"))

    @app.get("/api/config", response_model=sch.ConfigContent | sch.ErrorResponse)
    def config_get(path: str) -> dict:
        return cfg.read_config(path)

    @app.get("/api/pipeline/runs", response_model=list[sch.PipelineRun])
    def pipeline_runs() -> list[dict]:
        return pl.list_pipeline_runs(str(root / "pipeline_runs"))

    @app.get("/api/pipeline/run")
    def pipeline_run(path: str) -> dict:
        return pl.read_pipeline_run(path)

    @app.get("/api/evals", response_model=list[sch.EvalResult])
    def evals_list() -> list[dict]:
        return ev.list_eval_results(str(root))

    @app.get("/api/eval", response_model=sch.EvalDetail | sch.ErrorResponse)
    def eval_get(path: str) -> dict:
        return ev.read_eval_result(path)

    @app.get("/api/logs", response_model=list[sch.LogFile])
    def logs_list() -> list[dict]:
        return lg.list_logs(str(root))

    @app.get("/api/log", response_model=sch.LogTail | sch.ErrorResponse)
    def log_get(path: str, lines: int = 200) -> dict:
        return lg.tail_log(path, lines)

    @app.get("/api/features", response_model=sch.FeaturesView)
    def features_get() -> dict:
        return ft.list_features(str(root / "configs" / "features.yaml"))

    @app.post("/api/features/set", response_model=sch.FeatureSetResult | sch.ErrorResponse)
    def features_set(payload: dict) -> dict:
        return fw.set_feature(
            str(payload.get("key", "")),
            bool(payload.get("enabled", False)),
            str(root / "configs" / "features.yaml"),
        )

    @app.get("/api/reasoning", response_model=sch.ReasoningView | sch.ErrorResponse)
    def reasoning_get() -> dict:
        return rs.read_reasoning(str(root / "configs" / "reasoning.yaml"))

    @app.get("/api/tokenizer", response_model=sch.TokenizerInfo | sch.ErrorResponse)
    def tokenizer_get() -> dict:
        return tk.tokenizer_info()

    @app.get("/api/checkpoint", response_model=sch.CheckpointDetail | sch.ErrorResponse)
    def checkpoint_get(path: str) -> dict:
        return cd.checkpoint_detail(path)

    @app.get("/api/router", response_model=sch.RouterOverview)
    def router_get() -> dict:
        return rt.router_overview(str(root))

    @app.get("/api/exports", response_model=sch.ExportOverview)
    def exports_get() -> dict:
        return ex.export_overview(str(root))

    @app.get("/api/metrics/history", response_model=sch.MetricsHistory)
    def metrics_history() -> dict:
        return mh.training_history(log_path)

    @app.get("/api/data-sources", response_model=sch.DataSourcesView)
    def data_sources() -> dict:
        return dsv.read_data_sources(str(root / "configs" / "data_sources.yaml"))

    @app.get("/api/eval-history", response_model=sch.EvalHistoryView)
    def eval_history_get() -> dict:
        return eh.eval_history(str(root))

    @app.post("/api/tokenize", response_model=sch.TokenizeResult | sch.ErrorResponse)
    def tokenize_post(payload: dict) -> dict:
        return tkz.tokenize_text(str(payload.get("text", "")))

    @app.get("/api/health", response_model=sch.HealthSummary)
    def health_get() -> dict:
        return hl.project_health(str(root))

    @app.get("/api/sft", response_model=list[sch.SftFile])
    def sft_list() -> list[dict]:
        return sd.list_sft_files(str(root))

    @app.get("/api/sft/preview", response_model=sch.SftPreview | sch.ErrorResponse)
    def sft_preview(path: str, n: int = 10) -> dict:
        return sd.preview_sft(path, n)

    @app.get("/api/scripts", response_model=sch.ScriptsCatalog)
    def scripts_list() -> dict:
        return sc.list_scripts(str(root))

    @app.get("/api/model-card", response_model=sch.ModelCard | sch.ErrorResponse)
    def model_card_get(path: str) -> dict:
        return mc.build_model_card(path)

    @app.get("/api/config-diff", response_model=sch.ConfigDiff | sch.ErrorResponse)
    def config_diff_get(a: str, b: str) -> dict:
        return cdf.compare_configs(a, b)

    @app.get("/api/system-info", response_model=sch.SystemInfo)
    def system_info_get() -> dict:
        return si.system_info(str(root))

    @app.get("/api/storage", response_model=sch.StorageView)
    def storage_get() -> dict:
        return sv.read_storage(str(root))

    @app.get("/api/checkpoints/compare", response_model=sch.CompareResult | sch.ErrorResponse)
    def checkpoints_compare_get(a: str, b: str) -> dict:
        return cc.compare_checkpoints(a, b)

    # Serve the built React app. Mount LAST so /api/* routes resolve first —
    # StaticFiles at "/" otherwise shadows every API route.
    dist = root / "webui" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="webui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _MISSING_DIST_HTML

    return app
