"""Local web UI for cola-coder — a fast, lightweight dashboard over the CLI.

Read-only views (training status, GPU, checkpoints, datasets, scores) plus a
background-job runner that drives the EXISTING scripts (never reimplements training
or data logic). FastAPI backend + a single static HTML page (no npm/build step).

Design notes:
- The UI is standalone: it reads files (logs, checkpoints, data) and runs scripts as
  detached jobs, so it does not need to stay open for jobs to keep running.
- Training safety: starting training from the UI REFUSES if a trainer is already
  running (JobManager.is_training_running) — never a second trainer.
"""

from .app import create_app
from .checkpoint_detail import checkpoint_detail
from .configs import list_configs, read_config
from .data_sources_view import read_data_sources
from .datasets import dataset_preview, list_datasets, score_summary
from .eval_history import eval_history
from .evals import list_eval_results, read_eval_result
from .exports import export_overview
from .features import list_features
from .health import project_health
from .jobs import JobManager
from .logs import list_logs, tail_log
from .metrics_history import training_history
from .pipeline import list_pipeline_runs, read_pipeline_run
from .reasoning import read_reasoning
from .router import router_overview
from .scripts_catalog import list_scripts
from .sft_data import list_sft_files, preview_sft
from .status import get_system_status, get_training_status, list_checkpoints
from .tokenize import tokenize_text
from .tokenizer_info import tokenizer_info

__all__ = [
    "create_app",
    "JobManager",
    "get_training_status",
    "get_system_status",
    "list_checkpoints",
    "list_datasets",
    "dataset_preview",
    "score_summary",
    "list_configs",
    "read_config",
    "list_pipeline_runs",
    "read_pipeline_run",
    "list_eval_results",
    "read_eval_result",
    "list_logs",
    "tail_log",
    "list_features",
    "read_reasoning",
    "tokenizer_info",
    "checkpoint_detail",
    "router_overview",
    "export_overview",
    "training_history",
    "read_data_sources",
    "eval_history",
    "tokenize_text",
    "project_health",
    "list_sft_files",
    "preview_sft",
    "list_scripts",
]
