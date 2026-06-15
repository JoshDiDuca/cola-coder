"""Tests for the Python→TypeScript type bridge (TYPE-001).

Guards:
(a) DRIFT — regenerating from schemas.py must byte-equal the committed
    webui/src/types.gen.ts (a schema change without regen fails CI).
(b) WELL-FORMED — every BaseModel constructs from a representative example dict.
(c) COMPLETENESS — every interface in the hand-written webui/src/types.ts also
    appears in the generated types.gen.ts (no shape missed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from cola_coder.ui import schemas

# Import the generator module from scripts/ (also validates it imports cleanly).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gen_ts_types as gen  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_GEN_PATH = _ROOT / "webui" / "src" / "types.gen.ts"
_HAND_PATH = _ROOT / "webui" / "src" / "types.ts"


# ── (a) Drift guard ──────────────────────────────────────────────────────────

def test_generated_ts_matches_committed() -> None:
    """Regenerate in-memory and assert byte-equality with the committed file."""
    expected = gen.build_ts()
    actual = _GEN_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "webui/src/types.gen.ts is stale — run "
        ".venv/Scripts/python.exe scripts/gen_ts_types.py to regenerate."
    )


# ── (b) Every model is well-formed (constructs from an example) ──────────────

# Representative example payloads — one per BaseModel in schemas.py.
_EXAMPLES: dict[str, dict[str, object]] = {
    "TrainingStatus": {
        "alive": True, "step": 100, "total_steps": 1000, "progress_pct": 10.0,
        "loss": 2.5, "ppl": 12.1, "tok_per_s": 1800.0, "s_per_it": 0.5,
        "last_log_line": "step 100",
    },
    "SystemStatus": {
        "gpu_name": "RTX 4080", "gpu_util_pct": 90.0, "gpu_mem_used_mb": 8000.0,
        "gpu_mem_total_mb": 16000.0, "gpu_power_w": 200.0,
    },
    "Checkpoint": {
        "model": "small", "name": "step_100", "step": 100, "loss": 2.5,
        "path": "checkpoints/small/step_100", "mtime": 1.0,
    },
    "StatusResponse": {
        "training": {
            "alive": False, "step": None, "total_steps": None, "progress_pct": None,
            "loss": None, "ppl": None, "tok_per_s": None, "s_per_it": None,
            "last_log_line": None,
        },
        "system": {
            "gpu_name": None, "gpu_util_pct": None, "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None, "gpu_power_w": None,
        },
        "checkpoints": [],
    },
    "Dataset": {
        "name": "train.npy", "path": "data/train.npy", "kind": "npy",
        "size_bytes": 1024, "mtime": 1.0, "has_weights": True, "num_samples": 10,
    },
    "Job": {
        "id": "j1", "name": "train", "pid": 1234, "status": "running",
        "cmd": ["python", "train.py"], "log": "train.log", "started": 1.0,
        "returncode": None,
    },
    "ActionParam": {
        "name": "config", "flag": "--config", "label": "Config", "type": "config",
        "default": "configs/small.yaml", "choices": [], "required": True, "help": None,
    },
    "ActionDef": {"key": "k", "script": "train.py", "label": "Train", "args": [], "gpu": False},
    "JobLogChunk": {"text": "step 1\n", "done": False},
    "SftFile": {
        "name": "sft.jsonl", "path": "data/sft.jsonl", "kind": "jsonl",
        "num_records": 5, "size_bytes": 200, "mtime": 1.0,
    },
    "SftPreview": {
        "path": "data/sft.jsonl", "records": [{"instruction": "x", "output": "y"}],
        "fields": ["instruction", "output"], "count": 1, "truncated": False,
    },
    "ScriptInfo": {"name": "train.py", "category": "Training", "purpose": "p", "exists": True},
    "ScriptsCatalog": {"scripts": [], "categories": [], "count": 0, "on_disk": 0},
    "ScoreSummary": {
        "n": 3, "mean": 0.5, "min": 0.1, "max": 0.9, "histogram": [1, 2],
        "bins": [0.0, 0.5, 1.0],
    },
    "Preview": {"kind": "npy", "shape": [10, 512], "dtype": "int32", "num_samples": 10},
    "ConfigFile": {
        "name": "small.yaml", "path": "configs/small.yaml", "rel": "small.yaml",
        "size_bytes": 100, "mtime": 1.0,
    },
    "ConfigContent": {"path": "configs/small.yaml", "content": "a: 1", "parsed": {"a": 1}},
    "PipelineRun": {
        "name": "run1", "path": "pipeline_runs/run1.json", "mtime": 1.0,
        "num_stages": 10, "status": "running", "completed": 3,
    },
    "PipelineStageState": {
        "num": 3, "name": "pretrain", "description": "Base model pretraining",
        "optional": False, "status": "completed", "artifact": "checkpoints/small/latest",
        "override": "", "error": "", "duration_secs": 12.5,
        "started_at": "2026-06-14T10:00:00Z", "completed_at": "2026-06-14T10:30:00Z",
    },
    "PipelineRunDetail": {
        "name": "run1", "config_path": "configs/small.yaml",
        "created_at": "2026-06-14T10:00:00Z", "updated_at": "2026-06-14T10:30:00Z",
        "notes": "", "stages": [], "num_stages": 10, "active_stages": 8,
        "completed": 3, "status": "running",
    },
    "PipelineDeleteResult": {"ok": True, "name": "run1"},
    "EvalResult": {
        "name": "e", "path": "p", "kind": "humaneval", "mtime": 1.0, "summary": "s",
    },
    "EvalDetail": {"path": "p", "kind": "json", "parsed": {"pass@1": 0.3},
                   "content": None, "truncated": False},
    "LogFile": {"name": "t.log", "path": "t.log", "size_bytes": 100, "mtime": 1.0},
    "LogTail": {"path": "t.log", "lines": ["a", "b"], "size_bytes": 10, "truncated": False},
    "FeatureItem": {"key": "f", "enabled": True, "value": None},
    "FeatureGroup": {"category": "c", "features": []},
    "FeaturesView": {"path": "configs/features.yaml", "total": 1, "enabled": 1, "groups": []},
    "ReasoningSummary": {
        "advantage_norm": "mean", "clip_epsilon": 0.2, "clip_epsilon_high": 0.28,
        "group_size": 8, "sft_warmup_enabled": True, "problem_source": "all",
    },
    "ReasoningView": {
        "path": "configs/reasoning.yaml", "parsed": {}, "reasoning": {},
        "problem_set": {}, "sft_warmup": {},
        "summary": {
            "advantage_norm": None, "clip_epsilon": None, "clip_epsilon_high": None,
            "group_size": None, "sft_warmup_enabled": None, "problem_source": None,
        },
    },
    "TokenizerInfo": {
        "path": "tokenizer", "vocab_size": 32768, "n_merges": 32000,
        "special_tokens": ["<|endoftext|>"], "has_fim_tokens": True,
        "digit_splitting": True, "model_type": "BPE",
    },
    "CheckpointDetail": {
        "path": "p", "metadata": {"loss": 2.5}, "is_moe": False, "moe_config": None,
        "has_training_state": True, "num_params": 1000, "tensor_count": 50,
        "dtypes": ["bf16"], "files": ["model.safetensors"],
    },
    "RouterCheckpoint": {"path": "p", "name": "router", "step": 100},
    "RouterOverview": {"has_router": True, "checkpoints": [], "domains": ["react"]},
    "ExportFormat": {"key": "gguf", "label": "GGUF", "desc": "d"},
    "ExportItem": {"path": "p", "format": "gguf", "size_bytes": 100, "mtime": 1.0},
    "ExportCheckpoint": {"model": "small", "name": "step_100", "step": 100, "path": "p"},
    "ExportOverview": {"checkpoints": [], "formats": [], "existing": []},
    "MetricPoint": {"step": 1, "loss": 2.5, "ppl": 12.0, "lr": 6e-4, "tok_s": 1800.0},
    "MetricsHistory": {"points": [], "count": 0},
    "DataSource": {
        "name": "code", "weight": 0.7, "dataset": "bigcode/the-stack-v2",
        "languages": ["typescript"], "kind": "huggingface",
    },
    "DataSourcesView": {
        "path": "configs/data_sources.yaml", "sources": [], "total_weight": 1.0,
        "parsed": {}, "summary": "s",
    },
    "EvalSnapshot": {"step": 100, "path": "p", "mtime": 1.0, "metrics": {"pass@1": 0.3}},
    "EvalHistoryView": {"snapshots": [], "count": 0, "metric_keys": []},
    "TokenizeResult": {"path": "t", "count": 2, "ids": [1, 2], "tokens": ["a", "b"]},
    "HealthCheck": {"name": "tests", "ok": True, "detail": "d"},
    "HealthSummary": {"score": 90, "checks": [], "summary": "s"},
    "StorageEntry": {"name": "data", "path": "data", "exists": True, "size_bytes": 1024},
    "StorageView": {
        "path": "configs/storage.yaml", "raw": {}, "tokenizer_path": "tok",
        "data_dir": "data", "checkpoint_dir": "checkpoints", "entries": [],
    },
    "CompareDiff": {
        "num_params_delta": 0, "tensor_count_delta": 0, "is_moe_changed": False,
        "metadata_changed_keys": [], "dtypes_only_a": [], "dtypes_only_b": [],
    },
    "CompareResult": {
        "a": {
            "path": "a", "metadata": None, "is_moe": False, "moe_config": None,
            "has_training_state": False, "num_params": 1, "tensor_count": 1,
            "dtypes": [], "files": [],
        },
        "b": {
            "path": "b", "metadata": None, "is_moe": False, "moe_config": None,
            "has_training_state": False, "num_params": 1, "tensor_count": 1,
            "dtypes": [], "files": [],
        },
        "diff": {
            "num_params_delta": 0, "tensor_count_delta": 0, "is_moe_changed": False,
            "metadata_changed_keys": [], "dtypes_only_a": [], "dtypes_only_b": [],
        },
    },
    "ModelCard": {
        "path": "p", "name": "small", "num_params": 1000, "architecture": {"dim": 768},
        "training": {"steps": 1000}, "tokenizer": {"vocab_size": 32768},
        "is_moe": False, "markdown": "# Model",
    },
    "FeatureSetResult": {"ok": True, "key": "f", "enabled": True, "path": "p"},
    "ConfigChange": {"key": "lr", "a": 1e-3, "b": 2e-3},
    "ConfigDiffSide": {"path": "a.yaml", "parsed": {"lr": 1e-3}},
    "ConfigDiff": {
        "a": {"path": "a.yaml", "parsed": {}}, "b": {"path": "b.yaml", "parsed": {}},
        "changed": [], "only_a": [], "only_b": [],
    },
    "GpuInfo": {"name": "RTX 4080", "mem_total_mb": 16000, "mem_used_mb": 8000, "util_pct": 90},
    "DiskInfo": {"path": ".", "total_bytes": 1, "free_bytes": 1, "used_bytes": 0},
    "SystemInfo": {
        "python_version": "3.10.0", "platform": "Windows", "packages": {"torch": "2.2"},
        "gpus": [], "disk": {"path": ".", "total_bytes": None, "free_bytes": None,
                             "used_bytes": None},
    },
    "TokenizerHealthItem": {"name": "Vocab size", "ok": True, "detail": "vocab_size = 32,768"},
    "TokenizerHealthReport": {
        "path": "data/ds/tokenizer.json", "vocab_size": 32768, "checks": [],
        "passed": 5, "failed": 0, "ok": True,
    },
    "WeightTier": {"label": "excellent", "count": 10, "pct": 25.0},
    "CheckpointHealth": {
        "path": "checkpoints/small/step_00008500", "model": "small", "step": 8500,
        "loss": 2.31, "size_mb": 512.0, "num_tensors": 50,
        "files": ["model.safetensors", "metadata.json"],
        "config_stem": "small_react_best", "ok": True,
    },
    "MemoryEntry": {
        "id": "errors:TypeError on render", "type": "errors",
        "created_at": "2026-06-15 03:21",
        "content_preview": "TypeError: cannot read property foo of undefined",
    },
    "MemoryStats": {
        "total_entries": 4, "pinned": 0,
        "types": ["project", "patterns", "errors"], "size_bytes": 750,
        "oldest_at": "2026-06-15 03:21", "newest_at": "2026-06-15 03:21",
        "recent_sample": [],
    },
    "IndexStats": {
        "exists": True, "doc_count": 1240, "chunk_count": 1240,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384, "size_bytes": 5242880,
        "path": "data/vector_index", "last_updated": "2026-06-15T12:00:00+00:00",
    },
    "ThreatInfo": {
        "file_path": "data/x.py", "name": "ReverseShell", "severity": "high",
        "scanner": "yara", "details": "matched rule shell_001",
    },
    "MalwareScanResult": {
        "path": "data/processed", "files_scanned": 42, "is_clean": True,
        "threats": [], "duration_ms": 1234.5,
    },
    "EnvCheckItem": {
        "name": "PyTorch", "ok": True, "value": "2.10.0+cu128", "detail": None,
    },
    "EnvCheckReport": {
        "python_version": "3.10.0", "torch_version": "2.10.0+cu128",
        "cuda_available": True, "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
        "vram_gb": 16.0, "hf_token_set": True,
        "passed": 11, "failed": 0, "ok": True, "checks": [],
    },
    "VramComponent": {"name": "model weights", "mb": 206.2},
    "VramEstimate": {
        "config": "small.yaml", "params_millions": 100.7, "precision": "bf16",
        "batch_size": 12, "seq_len": 2048,
        "components": [{"name": "model weights", "mb": 206.2}],
        "total_mb": 10749.4, "budget_mb": 16384.0, "fits": True,
    },
    "HealthDimension": {
        "name": "Features", "score": 1.0,
        "detail": "170/170 modules declare FEATURE_ENABLED",
    },
    "ProjectHealthReport": {
        "overall_score": 1.0, "grade": "A",
        "dimensions": [], "summary": "Grade A — all dimensions strong.",
    },
    "BenchmarkRun": {
        "name": "bench.json", "path": "benchmarks/bench.json", "kind": "throughput",
        "tokens_per_s": 91.2, "latency_ms": 19.8, "config": None,
        "checkpoint": "checkpoints/tiny/step_00017000", "mtime": 1.0,
    },
    "BenchmarkResults": {"runs": [], "count": 0},
    "SafetyProbe": {
        "suite": "pii", "name": "ssn = ", "passed": False,
        "detail": "Secret detected: National ID",
    },
    "SafetyEvalRun": {
        "name": "safety_pii.json", "path": "reports/safety_pii.json",
        "checkpoint": "checkpoints/tiny/latest", "suite": "pii",
        "total": 24, "passed": 22, "failed": 2, "mtime": 1781493866.07,
        "probes": [{"suite": "pii", "name": "ssn = ", "passed": True, "detail": None}],
    },
    "SafetyEvalResults": {
        "runs": [{
            "name": "safety_pii.json", "path": "reports/safety_pii.json",
            "checkpoint": None, "suite": "pii", "total": 24, "passed": 22,
            "failed": 2, "mtime": 1781493866.07, "probes": [],
        }],
        "count": 1,
    },
    "FilterInfo": {
        "name": "content", "category": "quality",
        "purpose": "Reject autogenerated, data dump, and low-signal files.",
        "module": "cola_coder.data.filters.content", "default_enabled": True,
    },
    "FiltersCatalog": {"filters": [], "count": 0, "categories": []},
    "ReasoningProblem": {
        "id": "has_close_elements", "difficulty": "easy", "language": "python",
        "prompt_preview": "def has_close_elements(numbers, threshold): ...", "has_tests": True,
    },
    "ReasoningProblemSet": {
        "problems": [], "count": 0, "difficulties": ["easy", "medium", "hard"],
        "languages": ["python"],
    },
    "RunRequest": {"action": "prepare_data", "args": ["--config", "configs/small.yaml"]},
    "TrainStartRequest": {"config": "configs/small.yaml", "resume": None},
    "InferenceRequest": {
        "prompt": "def add(a, b):", "checkpoint": "checkpoints/small/latest",
        "config": "configs/small.yaml", "max_tokens": 256, "temperature": 0.8,
        "top_p": 0.9, "top_k": 50,
    },
    "InferenceResult": {
        "completion": "    return a + b", "prompt": "def add(a, b):",
        "checkpoint": "checkpoints/small/latest", "tokens_generated": 6, "elapsed_s": 1.2,
    },
    "GenStreamChunk": {"delta": "    return a + b", "done": False, "error": None},
    "ChatMessage": {"role": "user", "content": "Write a bubble sort in Python."},
    "ChatRequest": {
        "messages": [{"role": "user", "content": "hi"}],
        "checkpoint": "checkpoints/small/latest", "config": "configs/small.yaml",
        "use_chat_template": True, "max_tokens": 256, "temperature": 0.7,
        "top_p": 0.9, "top_k": 50,
    },
    "FimRequest": {
        "prefix": "def add(a, b):\n    return ", "suffix": "\n",
        "checkpoint": "checkpoints/small/latest", "config": "configs/small.yaml",
        "max_tokens": 128, "temperature": 0.2, "top_p": 0.9, "top_k": 50,
    },
    "ConfigWriteRequest": {"path": "configs/small.yaml", "content": "model:\n  dim: 768\n"},
    "ConfigWriteResult": {"ok": True, "path": "configs/small.yaml", "bytes_written": 18},
    "ScorerConfigEntry": {
        "name": "tsc", "enabled": True, "weight": 0.3, "available": True,
        "purpose": "Score TypeScript files using tsc --noEmit via SandboxedRunner.",
    },
    "ScoringConfig": {
        "path": "configs/scoring.yaml", "scorers": [], "count": 0,
        "enabled_count": 0, "curriculum": None,
    },
    "RegressionMetric": {
        "name": "pass_rate", "value": 0.9, "baseline": None,
        "delta": None, "regressed": True,
    },
    "RegressionRun": {
        "name": "results_v1.json", "path": "regression/results_v1.json",
        "checkpoint": "checkpoints/tiny/step_00017000", "mtime": 1781493866.07,
        "passed": False,
        "metrics": [{
            "name": "pass_rate", "value": 0.9, "baseline": None,
            "delta": None, "regressed": True,
        }],
    },
    "RegressionHistory": {"runs": [], "count": 0},
    "DocFile": {
        "name": "03_training.md", "path": "docs/03_training.md",
        "rel": "03_training.md", "title": "The Training Pipeline", "size_bytes": 100,
    },
    "DocsList": {"docs": [], "count": 0},
    "DocContent": {"path": "docs/03_training.md", "content": "# x", "truncated": False},
    "BacklogItem": {
        "id": "UI-046..047", "category": "ui", "severity": "medium",
        "status": "done", "date": "2026-06-15",
        "description": "Batch (parallel agents): 2 read-only catalog views.",
    },
    "BacklogView": {"items": [], "count": 0, "open_count": 0, "done_count": 0},
    "ResearchEntry": {
        "date": "2026-06-15", "title": "DAPO overlong reward shaping",
        "area": "post-training", "source_count": 4, "has_original_idea": True,
        "summary": "soft overlong reward shaping is a pure function of length.",
    },
    "ResearchLog": {"entries": [], "count": 0},
    "TrainingManifest": {
        "model": "small_react_best", "path": "checkpoints/small_react_best/training_manifest.yaml",
        "config": "cola-coder/train.py", "dim": 768, "n_layers": 12, "n_heads": 12,
        "seq_len": 1024, "batch_size": 24, "learning_rate": 6e-4, "max_steps": 150000,
        "latest_step": 9000, "created_at": "2026-06-13T14:07:57+00:00", "mtime": 1.0,
    },
    "TrainingManifests": {"manifests": [], "count": 0},
    "LrPoint": {"lr": 3.0e-4, "loss": 1.8},
    "LrFinderRun": {
        "name": "lr_finder_result.json", "path": "lr_finder/lr_finder_result.json",
        "config": "configs/tiny.yaml", "suggested_lr": 3.0e-4, "min_loss": 1.62,
        "num_points": 200, "mtime": 1.0, "points": [],
    },
    "LrFinderResults": {"runs": [], "count": 0},
    "RepoScore": {
        "repo": "awesome-ts", "score": 0.95, "stars": 1200,
        "language": "typescript", "license": "MIT", "reason": "verified",
    },
    "RepoScoresResult": {
        "path": "reports/repo_scores.json", "repos": [], "count": 0, "mtime": 1781498379.5,
    },
    "VocabToken": {"id": 42, "piece": "Ġconst", "is_special": False},
    "VocabSearchResult": {
        "query": "const", "vocab_size": 32768, "total_matches": 3, "truncated": False,
        "tokens": [{"id": 42, "piece": "Ġconst", "is_special": False}],
        "special_tokens": [{"id": 0, "piece": "<|endoftext|>", "is_special": True}],
    },
    "DataStats": {
        "data_path": "data/processed/train_data.npy", "file_size_mb": 12.3,
        "shape": [100, 64], "num_chunks": 100, "seq_len": 64, "total_tokens": 6400,
        "token_min": 0, "token_max": 32767, "token_mean": 16000.0,
        "est_unique_tokens": 32000, "has_weights": True,
        "weights_path": "data/processed/train_data.weights.npy", "weight_tiers": [],
        "weight_mean": 0.6, "weight_std": 0.2,
    },
    "ErrorResponse": {"error": "boom"},
}


def _all_models() -> dict[str, type[BaseModel]]:
    return {
        name: obj
        for name, obj in vars(schemas).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
        and not name.startswith("_")
    }


def test_every_model_has_an_example() -> None:
    """No model is left untested (catches a new model added without an example)."""
    models = _all_models()
    missing = sorted(set(models) - set(_EXAMPLES))
    assert not missing, f"Models without example payloads: {missing}"


@pytest.mark.parametrize("name", sorted(_EXAMPLES))
def test_model_constructs_from_example(name: str) -> None:
    model = getattr(schemas, name)
    instance = model.model_validate(_EXAMPLES[name])
    assert isinstance(instance, BaseModel)


# ── (c) Completeness — every hand-written interface appears in the generated file ─

_INTERFACE_RE = re.compile(r"^export interface (\w+)", re.MULTILINE)


def _interface_names(text: str) -> set[str]:
    return set(_INTERFACE_RE.findall(text))


def test_generated_covers_handwritten_interfaces() -> None:
    """Every interface in the hand-written types.ts must exist in types.gen.ts.

    types.ts inlines a couple of sub-objects (ReasoningSummary etc. are named;
    the SystemInfo.disk object and ConfigDiff.a/b objects are inline) — those
    are named models in schemas.py (DiskInfo, ConfigDiffSide), so the generated
    file is a strict SUPERSET of the hand-written interface set.
    """
    hand = _interface_names(_HAND_PATH.read_text(encoding="utf-8"))
    generated = _interface_names(_GEN_PATH.read_text(encoding="utf-8"))
    missing = sorted(hand - generated)
    assert not missing, f"Interfaces in types.ts missing from types.gen.ts: {missing}"
