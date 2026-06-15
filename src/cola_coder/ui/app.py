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

from . import backlog_view as blv
from . import benchmark_results_view as brv
from . import checkpoint_detail as cd
from . import lr_finder_view as lfv
from . import repo_scores_view as rscv
from . import filters_catalog_view as fcv
from . import checkpoint_health_view as chv
from . import checkpoints_compare as cc
from . import config_diff as cdf
from . import configs as cfg
from . import data_sources_view as dsv
from . import data_stats_view as dst
from . import datasets as ds
from . import docs_view as dv
from . import env_check_view as ecv
from . import eval_history as eh
from . import evals as ev
from . import exports as ex
from . import features as ft
from . import features_write as fw
from . import health as hl
from . import logs as lg
from . import memory_stats_view as msv
from . import metrics_history as mh
from . import model_card as mc
from . import pipeline as pl
from . import pipeline_ops as po
from . import project_health_view as phv
from . import reasoning as rs
from . import reasoning_problems_view as rpv
from . import regression_history_view as rhv
from . import research_log_view as rlv
from . import retrieval_stats_view as rsv
from . import retrieval_search_view as rsch
from . import router as rt
from . import safety_eval_view as sev
from . import security_scan_view as ssv
from . import scoring_config_view as scv
from . import scripts_catalog as sc
from . import specialists_view as spv
from . import sft_data as sd
from . import status as st
from . import storage_view as sv
from . import system_info as si
from . import schemas as sch
from . import tokenize as tkz
from . import tokenizer_health_view as thv
from . import tokenizer_info as tk
from . import training_manifest_view as tmv
from . import vocab_explorer_view as vxv
from . import vram_estimate_view as vev
from .action_params import ACTION_PARAMS
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
    "evaluate": {"script": "evaluate.py", "label": "Evaluate (HumanEval pass@k)", "gpu": True,
                 "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "smoke_test": {"script": "smoke_test.py", "label": "Smoke test (8 checks)", "gpu": True, "args": []},
    "generate_rft": {"script": "generate_rft_data.py", "label": "Generate RFT data (self-verified)",
                     "gpu": True,
                     "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "data_stats": {"script": "data_stats.py", "label": "Dataset statistics", "args": []},
    "tokenizer_health": {"script": "tokenizer_health.py", "label": "Tokenizer health check", "args": []},
    "project_health": {"script": "project_health.py", "label": "Project health score", "args": []},
    "vram_estimate": {"script": "vram_estimate.py", "label": "VRAM estimate", "args": []},
    "env_check": {"script": "env_check.py", "label": "Environment check", "args": []},
    "quality_report": {"script": "quality_report.py", "label": "Quality report (syntax/types/tokens)",
                       "gpu": True,
                       "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "safety_eval": {"script": "safety_eval.py", "label": "Safety eval (secrets/dangerous patterns)",
                    "gpu": True,
                    "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml",
                             "--suite", "basic"]},
    "completion_benchmark": {"script": "completion_benchmark.py", "label": "Completion benchmark", "gpu": True,
                             "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "benchmark": {"script": "benchmark.py", "label": "Throughput benchmark (tok/s)", "gpu": True, "args": []},
    "regression_test": {"script": "regression_test.py", "label": "Regression test (quality tracking)",
                        "gpu": True,
                        "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "ts_benchmark": {"script": "ts_benchmark.py", "label": "TypeScript benchmark (tsc --strict)", "gpu": True,
                     "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "depth_profile": {"script": "depth_profile.py", "label": "Depth / early-exit profile", "gpu": True,
                      "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "robustness_eval": {"script": "robustness_eval.py", "label": "Robustness eval (perturbations)", "gpu": True,
                        "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml"]},
    "generate_instructions": {"script": "generate_instructions.py", "label": "Generate instruction pairs",
                              "args": ["--config", "configs/small.yaml"]},
    # Trainer-class actions — REFUSED by /api/run while a training process is alive (never a 2nd trainer).
    "train": {"script": "train.py", "label": "Pretrain (from scratch / resume)", "trainer": True,
              "args": ["--config", "configs/small.yaml"]},
    "auto_pipeline": {"script": "auto_pipeline.py", "label": "Auto pipeline (detect hardware -> best config)",
                      "trainer": True, "args": ["--yes"]},
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
    # CPU weight-averaging (model soup) — not a trainer; the UI panel supplies --checkpoints explicitly.
    "average_checkpoints": {"script": "average_checkpoints.py", "label": "Average checkpoints (model soup)",
                            "args": ["--method", "uniform", "--output", "checkpoints/soup"]},
    # CPU dataset mixing — the UI panel supplies "--datasets PATH:WEIGHT ... --output PATH" explicitly
    # (default args empty: a bare launch would drop into the interactive TUI, which a job can't answer).
    "combine_datasets": {"script": "combine_datasets.py", "label": "Combine datasets (weighted mix)",
                         "args": []},
    # CPU export (GGUF/Ollama/quantize) — non-interactive when --action is passed (the UI panel always sends it).
    "export_model": {"script": "export_model.py", "label": "Export model (GGUF / Ollama / quantize)",
                     "args": ["--checkpoint", "checkpoints/small/latest", "--config", "configs/small.yaml",
                              "--action", "gguf-f16"]},
}


def _build_chat_prompt(messages: list[tuple[str, str]], use_chat_template: bool) -> str:
    """Build a generation prompt from chat turns.

    Mirrors the inference server: ChatML (with a trailing empty assistant turn,
    minus its closing marker, so the model continues the reply) for instruction-
    tuned checkpoints, or plain newline concatenation for base models. Falls back
    to concatenation if the ChatML template is unavailable.
    """
    if use_chat_template:
        try:
            from cola_coder.tokenizer.chat_template import format_chat

            dicts = [{"role": role, "content": content} for role, content in messages]
            dicts.append({"role": "assistant", "content": ""})
            prompt = format_chat(dicts)
            for marker in ("<|im_end|>\n", "<|im_end|>"):
                if prompt.endswith(marker):
                    return prompt[: -len(marker)]
            return prompt
        except ImportError:
            pass
    return "\n".join(content for _role, content in messages)


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

    @app.get("/api/jobs/{job_id}/stream")
    def job_stream(job_id: str, tail: int = 200) -> StreamingResponse:
        """Server-push a job's log: an initial tail, then new bytes as they land.

        Each SSE frame is a JSON ``JobLogChunk`` ({text, done}). The stream ends
        with a final ``done=true`` frame once the job process exits. Reading is
        byte-offset based (binary seek) so multi-byte boundaries never corrupt.
        """
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        log_path = Path(job["log"])

        def _frame(text: str, done: bool) -> str:
            return f"data: {sch.JobLogChunk(text=text, done=done).model_dump_json()}\n\n"

        async def gen() -> AsyncIterator[str]:
            pos = 0
            if log_path.exists():
                data = log_path.read_bytes()
                pos = len(data)
                lines = data.decode("utf-8", errors="replace").splitlines()
                yield _frame("\n".join(lines[-tail:]), False)
            try:
                while True:
                    await asyncio.sleep(0.5)
                    status = jobs.get(job_id)
                    done = status is None or status["status"] != "running"
                    chunk = ""
                    if log_path.exists():
                        size = log_path.stat().st_size
                        if size > pos:
                            with open(log_path, "rb") as fh:
                                fh.seek(pos)
                                raw = fh.read()
                                pos = fh.tell()
                            chunk = raw.decode("utf-8", errors="replace")
                    if chunk or done:
                        yield _frame(chunk, done)
                    if done:
                        return
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/stop")
    def job_stop(job_id: str) -> dict:
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"stopped": jobs.stop(job_id)}

    @app.get("/api/actions", response_model=list[sch.ActionDef])
    def actions() -> list[dict]:
        # Merge each action's typed argument spec (1:1 with the script's argparse,
        # source of truth in action_params.py) so the UI renders real form fields.
        return [
            {"key": k, **v, "params": ACTION_PARAMS.get(k, [])}
            for k, v in ACTIONS.items()
        ]

    @app.post("/api/run", response_model=sch.Job | sch.ErrorResponse)
    def run(req: sch.RunRequest) -> dict | JSONResponse:
        if req.action not in ACTIONS:
            raise HTTPException(status_code=400, detail=f"unknown action: {req.action!r}")
        spec = ACTIONS[req.action]
        # Trainer-class actions (SFT/GRPO/router/MoE/LR-finder/full-pipeline) load the model
        # and optimize on the GPU — refuse to launch one while the live trainer is running, so
        # the UI can never spawn a second trainer that fights the pretraining run for VRAM.
        if spec.get("trainer") and jobs.is_training_running():
            return JSONResponse(
                {"error": "training already running — refusing to start a second trainer"},
                status_code=409,
            )
        args = req.args if req.args is not None else spec["args"]
        cmd = [sys.executable, str(root / "scripts" / spec["script"]), *args]
        return jobs.start(name=req.action, cmd=cmd, cwd=str(root))

    @app.post("/api/train/start", response_model=sch.Job | sch.ErrorResponse)
    def train_start(req: sch.TrainStartRequest) -> dict | JSONResponse:
        result = jobs.start_training(config=req.config, resume=req.resume)
        # start_training returns {"error": ...} when a trainer is already running.
        if "error" in result:
            return JSONResponse(result, status_code=409)
        return result

    def _training_busy() -> JSONResponse | None:
        """Return a 409 JSONResponse if training is live, else None.

        Robust guard: the cmdline scan misses a trainer launched at higher OS
        integrity (OPS-001), so also trust the per-step progress freshness — the
        same signal the dashboard 'alive' flag uses. Either positive refuses, so a
        UI generation can never contend with the live trainer for the GPU.
        """
        if jobs.is_training_running() or st.is_training_active(log_path, err_path):
            return JSONResponse(
                {"error": "training is running — generation refused to protect the "
                          "live run (free the GPU first)"},
                status_code=409,
            )
        return None

    def _run_generation(
        prompt: str, checkpoint: str, config: str,
        *, max_tokens: int, temperature: float, top_p: float, top_k: int,
    ) -> dict | JSONResponse:
        """Load a checkpoint, generate from ``prompt``, free the model, return result.

        Shared by /api/generate, /api/chat, /api/fim. The model is loaded per
        request and freed afterward so the UI server holds no GPU memory between
        calls. Callers MUST check ``_training_busy()`` first.
        """
        try:
            import time

            from cola_coder.inference.loading import load_generator

            ckpt = checkpoint if Path(checkpoint).is_absolute() else str(root / checkpoint)
            conf = config if Path(config).is_absolute() else str(root / config)
            generator, _config, _tok = load_generator(ckpt, conf)
            start = time.perf_counter()
            completion = generator.generate(
                prompt, max_new_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, return_new_only=True,
            )
            elapsed = time.perf_counter() - start
            n_tokens = len(_tok.encode(completion)) if hasattr(_tok, "encode") else 0
            del generator
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return sch.InferenceResult(
                completion=completion, prompt=prompt, checkpoint=checkpoint,
                tokens_generated=n_tokens, elapsed_s=elapsed,
            ).model_dump()
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - surface any load/generate failure to the UI
            return JSONResponse({"error": f"generation failed: {exc}"}, status_code=500)

    @app.post("/api/generate", response_model=sch.InferenceResult | sch.ErrorResponse)
    def generate_text(req: sch.InferenceRequest) -> dict | JSONResponse:
        """One-shot code generation for the inference playground (gated, see above)."""
        busy = _training_busy()
        if busy is not None:
            return busy
        return _run_generation(
            req.prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    @app.post("/api/chat", response_model=sch.InferenceResult | sch.ErrorResponse)
    def chat(req: sch.ChatRequest) -> dict | JSONResponse:
        """Multi-turn chat. Gated like /api/generate. ChatML formatting (best with
        instruction-tuned checkpoints) or plain concatenation for base models."""
        busy = _training_busy()
        if busy is not None:
            return busy
        prompt = _build_chat_prompt(
            [(m.role, m.content) for m in req.messages], req.use_chat_template
        )
        return _run_generation(
            prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    @app.post("/api/fim", response_model=sch.InferenceResult | sch.ErrorResponse)
    def fim(req: sch.FimRequest) -> dict | JSONResponse:
        """Fill-in-the-middle completion. Gated like /api/generate. Requires the
        tokenizer's <|fim_*|> tokens; returns a clear error if absent."""
        busy = _training_busy()
        if busy is not None:
            return busy
        try:
            from cola_coder.inference.loading import resolve_tokenizer_path
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

            ckpt = req.checkpoint if Path(req.checkpoint).is_absolute() else str(root / req.checkpoint)
            conf = req.config if Path(req.config).is_absolute() else str(root / req.config)
            tok = CodeTokenizer(str(resolve_tokenizer_path(Path(ckpt), conf)))
            if not tok.has_fim_tokens():
                return JSONResponse(
                    {"error": "tokenizer has no <|fim_*|> tokens — this checkpoint's "
                              "tokenizer was not trained with FIM support"},
                    status_code=400,
                )
            prompt = tok.fim_prompt(req.prefix, req.suffix)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"FIM prompt build failed: {exc}"}, status_code=500)
        return _run_generation(
            prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    def _stream_response(
        prompt: str, checkpoint: str, config: str,
        *, max_tokens: int, temperature: float, top_p: float, top_k: int,
    ) -> StreamingResponse:
        """Shared SSE streamer for generate/chat/fim. Load → stream deltas → free.

        Yields ``GenStreamChunk`` frames; load/generate errors are surfaced as a final
        frame with ``error`` set. Callers MUST check ``_training_busy()`` first.
        """
        def _frame(delta: str, done: bool, error: str | None = None) -> str:
            return f"data: {sch.GenStreamChunk(delta=delta, done=done, error=error).model_dump_json()}\n\n"

        async def gen() -> AsyncIterator[str]:
            generator = None
            try:
                from cola_coder.inference.loading import load_generator

                ckpt = checkpoint if Path(checkpoint).is_absolute() else str(root / checkpoint)
                conf = config if Path(config).is_absolute() else str(root / config)
                generator, _config, _tok = load_generator(ckpt, conf)
                for delta in generator.generate_stream(
                    prompt, max_new_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k,
                ):
                    yield _frame(delta, False)
                    await asyncio.sleep(0)  # cooperatively yield so the client flushes
                yield _frame("", True)
            except FileNotFoundError as exc:
                yield _frame("", True, error=str(exc))
            except asyncio.CancelledError:
                return  # client disconnected — stop quietly
            except Exception as exc:  # noqa: BLE001 - surface load/generate failure to the UI
                yield _frame("", True, error=f"generation failed: {exc}")
            finally:
                if generator is not None:
                    del generator
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/generate/stream", response_model=None)
    def generate_stream_endpoint(req: sch.InferenceRequest) -> StreamingResponse | JSONResponse:
        """Stream a one-shot generation token-by-token (gated, see _stream_response)."""
        busy = _training_busy()
        if busy is not None:
            return busy
        return _stream_response(
            req.prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    @app.post("/api/chat/stream", response_model=None)
    def chat_stream_endpoint(req: sch.ChatRequest) -> StreamingResponse | JSONResponse:
        """Stream a chat reply token-by-token. Gated; same ChatML/plain prompt build as /api/chat."""
        busy = _training_busy()
        if busy is not None:
            return busy
        prompt = _build_chat_prompt(
            [(m.role, m.content) for m in req.messages], req.use_chat_template
        )
        return _stream_response(
            prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    @app.post("/api/fim/stream", response_model=None)
    def fim_stream_endpoint(req: sch.FimRequest) -> StreamingResponse | JSONResponse:
        """Stream a FIM infill token-by-token. Gated; 400 if the tokenizer lacks <|fim_*|>."""
        busy = _training_busy()
        if busy is not None:
            return busy
        try:
            from cola_coder.inference.loading import resolve_tokenizer_path
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

            ckpt = req.checkpoint if Path(req.checkpoint).is_absolute() else str(root / req.checkpoint)
            conf = req.config if Path(req.config).is_absolute() else str(root / req.config)
            tok = CodeTokenizer(str(resolve_tokenizer_path(Path(ckpt), conf)))
            if not tok.has_fim_tokens():
                return JSONResponse(
                    {"error": "tokenizer has no <|fim_*|> tokens — this checkpoint's "
                              "tokenizer was not trained with FIM support"},
                    status_code=400,
                )
            prompt = tok.fim_prompt(req.prefix, req.suffix)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"FIM prompt build failed: {exc}"}, status_code=500)
        return _stream_response(
            prompt, req.checkpoint, req.config,
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k,
        )

    @app.get("/api/configs", response_model=list[sch.ConfigFile])
    def configs_list() -> list[dict]:
        return cfg.list_configs(str(root / "configs"))

    @app.get("/api/config", response_model=sch.ConfigContent | sch.ErrorResponse)
    def config_get(path: str) -> dict:
        return cfg.read_config(path)

    @app.post("/api/config/write", response_model=sch.ConfigWriteResult | sch.ErrorResponse)
    def config_write(req: sch.ConfigWriteRequest) -> dict | JSONResponse:
        """Validate + atomically write an edited YAML config (R11).

        Refuses (400) any path outside configs/ or any content that does not parse as
        YAML — a config is never corrupted from the UI. Safe vs. the live trainer,
        which read its config at launch.
        """
        result = cfg.write_config(req.path, req.content, configs_dir=str(root / "configs"))
        if "error" in result:
            return JSONResponse(result, status_code=400)
        return result

    @app.get("/api/pipeline/runs", response_model=list[sch.PipelineRun])
    def pipeline_runs() -> list[dict]:
        return pl.list_pipeline_runs(str(root / "pipeline_runs"))

    @app.get("/api/pipeline/run")
    def pipeline_run(path: str) -> dict:
        return pl.read_pipeline_run(path)

    @app.get("/api/pipeline/detail", response_model=sch.PipelineRunDetail | sch.ErrorResponse)
    def pipeline_detail(name: str) -> dict:
        return po.get_run_detail(name, str(root / "pipeline_runs"))

    @app.post("/api/pipeline/create", response_model=sch.PipelineRunDetail | sch.ErrorResponse)
    def pipeline_create(payload: dict) -> dict:
        skip = payload.get("skip_stages") or []
        return po.create_run(
            str(payload.get("name", "")),
            str(payload.get("config_path", "")),
            skip_stages=[int(n) for n in skip],
            runs_dir=str(root / "pipeline_runs"),
        )

    @app.post("/api/pipeline/reset", response_model=sch.PipelineRunDetail | sch.ErrorResponse)
    def pipeline_reset(payload: dict) -> dict:
        return po.reset_run(
            str(payload.get("name", "")),
            int(payload.get("stage_num", 0)),
            runs_dir=str(root / "pipeline_runs"),
        )

    @app.post("/api/pipeline/override", response_model=sch.PipelineRunDetail | sch.ErrorResponse)
    def pipeline_override(payload: dict) -> dict:
        return po.set_override(
            str(payload.get("name", "")),
            int(payload.get("stage_num", 0)),
            str(payload.get("path", "")),
            runs_dir=str(root / "pipeline_runs"),
        )

    @app.post("/api/pipeline/delete", response_model=sch.PipelineDeleteResult | sch.ErrorResponse)
    def pipeline_delete(payload: dict) -> dict:
        return po.delete_run(str(payload.get("name", "")), str(root / "pipeline_runs"))

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

    @app.get("/api/specialists", response_model=sch.SpecialistsView | sch.ErrorResponse)
    def specialists_get() -> dict:
        return spv.specialists_view(str(root))

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

    @app.get("/api/tokenizer-health",
             response_model=sch.TokenizerHealthReport | sch.ErrorResponse)
    def tokenizer_health_get(path: str | None = None) -> dict:
        return thv.tokenizer_health(path)

    @app.get("/api/checkpoint-health",
             response_model=sch.CheckpointHealth | sch.ErrorResponse)
    def checkpoint_health_get(model: str, step: str) -> dict:
        return chv.checkpoint_health(model, step)

    @app.get("/api/memory-stats",
             response_model=sch.MemoryStats | sch.ErrorResponse)
    def memory_stats_get() -> dict:
        return msv.memory_stats(str(root))

    @app.get("/api/retrieval/index-stats",
             response_model=sch.IndexStats | sch.ErrorResponse)
    def retrieval_index_stats_get() -> dict:
        return rsv.index_stats()

    @app.get("/api/retrieval/search",
             response_model=sch.RetrievalSearchResult | sch.ErrorResponse)
    def retrieval_search_get(q: str, top_k: int = 10) -> dict:
        return rsch.search_index(q, top_k=top_k, root=str(root))

    @app.get("/api/gpu/processes", response_model=sch.GpuProcesses)
    def gpu_processes_get() -> dict:
        return st.get_gpu_processes()

    @app.get("/api/security/scan",
             response_model=sch.MalwareScanResult | sch.ErrorResponse)
    def security_scan_get(path: str, max_files: int = 500) -> dict:
        return ssv.scan_summary(path, max_files)

    @app.get("/api/env-check",
             response_model=sch.EnvCheckReport | sch.ErrorResponse)
    def env_check_get() -> dict:
        return ecv.env_check()

    @app.get("/api/vram-estimate",
             response_model=sch.VramEstimate | sch.ErrorResponse)
    def vram_estimate_get(config: str) -> dict:
        return vev.vram_estimate(config)

    @app.get("/api/project-health",
             response_model=sch.ProjectHealthReport | sch.ErrorResponse)
    def project_health_get() -> dict:
        return phv.project_health()

    @app.get("/api/benchmark-results",
             response_model=sch.BenchmarkResults | sch.ErrorResponse)
    def benchmark_results_get() -> dict:
        return brv.benchmark_results(str(root))

    @app.get("/api/safety-eval-results",
             response_model=sch.SafetyEvalResults | sch.ErrorResponse)
    def safety_eval_results_get() -> dict:
        return sev.safety_eval_results(str(root))

    @app.get("/api/filters-catalog",
             response_model=sch.FiltersCatalog | sch.ErrorResponse)
    def filters_catalog_get() -> dict:
        return fcv.filters_catalog()

    @app.get("/api/reasoning-problems",
             response_model=sch.ReasoningProblemSet | sch.ErrorResponse)
    def reasoning_problems_get(which: str = "all") -> dict:
        return rpv.reasoning_problems(which)

    @app.get("/api/vocab-search",
             response_model=sch.VocabSearchResult | sch.ErrorResponse)
    def vocab_search_get(query: str = "", path: str | None = None, limit: int = 200) -> dict:
        return vxv.vocab_search(query, path, limit)

    @app.get("/api/scoring-config",
             response_model=sch.ScoringConfig | sch.ErrorResponse)
    def scoring_config_get() -> dict:
        return scv.scoring_config()

    @app.get("/api/regression-history",
             response_model=sch.RegressionHistory | sch.ErrorResponse)
    def regression_history_get() -> dict:
        return rhv.regression_history(str(root))

    @app.get("/api/lr-finder-results",
             response_model=sch.LrFinderResults | sch.ErrorResponse)
    def lr_finder_results_get() -> dict:
        return lfv.lr_finder_results(str(root))

    @app.get("/api/repo-scores",
             response_model=sch.RepoScoresResult | sch.ErrorResponse)
    def repo_scores_get() -> dict:
        return rscv.repo_scores(str(root))

    @app.get("/api/training-manifests",
             response_model=sch.TrainingManifests | sch.ErrorResponse)
    def training_manifests_get() -> dict:
        return tmv.training_manifests(str(root / "checkpoints"))

    @app.get("/api/backlog", response_model=sch.BacklogView | sch.ErrorResponse)
    def backlog_get() -> dict:
        return blv.backlog(str(root))

    @app.get("/api/research-log", response_model=sch.ResearchLog | sch.ErrorResponse)
    def research_log_get() -> dict:
        return rlv.research_log(str(root))

    @app.get("/api/docs", response_model=sch.DocsList | sch.ErrorResponse)
    def docs_list_get() -> dict:
        return dv.docs_list(str(root))

    @app.get("/api/doc", response_model=sch.DocContent | sch.ErrorResponse)
    def doc_get(path: str) -> dict:
        return dv.doc_content(path, str(root))

    @app.get("/api/data-stats", response_model=sch.DataStats | sch.ErrorResponse)
    def data_stats_get(
        data_path: str | None = None,
        weights_path: str | None = None,
        estimate_unique: bool = True,
    ) -> dict:
        return dst.data_stats(data_path, weights_path, estimate_unique)

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
