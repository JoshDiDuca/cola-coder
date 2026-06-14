"""Pydantic v2 schemas — the SINGLE SOURCE OF TRUTH for every cola-coder web UI
HTTP response shape.

Each ``BaseModel`` below maps 1:1 to a TypeScript interface in
``webui/src/types.gen.ts``, which is GENERATED from this module by
``scripts/gen_ts_types.py``. Never hand-edit the generated TS; edit these models
and regenerate (``tests/test_ui_types_generated.py`` guards against drift).

Rules (see ``.claude/rules/typing.md``):
- ``str | None`` ↔ TS ``string | null``; a field with a default ↔ optional.
- ``int``/``float`` ↔ TS ``number`` (int for counts/steps, float for loss/ratios/mtime).
- ``bool`` ↔ ``boolean``; ``list[X]`` ↔ ``X[]``; nested model ↔ its interface.
- Genuinely open JSON uses the shared ``JsonValue`` alias — NEVER ``Any``.
- ``ErrorResponse`` maps to the TS ``ApiError``.
- ``model_config = ConfigDict(extra="forbid")`` so a model can't silently gain stray keys.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeAliasType

# The ONE allowed principled open-JSON type (parsed YAML configs, checkpoint
# metadata, previewed JSONL rows, config-diff values, model-card sections).
# Maps to the recursive TS ``JsonValue`` union emitted once in types.gen.ts.
# Defined via ``TypeAliasType`` so the self-reference is a true recursive alias
# (Pydantic v2 handles this without infinite schema expansion).
JsonValue = TypeAliasType(
    "JsonValue",
    "Union[str, int, float, bool, None, list[JsonValue], dict[str, JsonValue]]",
)


class _UiModel(BaseModel):
    """Base for all UI response models — forbids stray/undeclared keys."""

    model_config = ConfigDict(extra="forbid")


class TrainingStatus(_UiModel):
    alive: bool
    step: int | None
    total_steps: int | None
    progress_pct: float | None
    loss: float | None
    ppl: float | None
    tok_per_s: float | None
    s_per_it: float | None
    last_log_line: str | None


class SystemStatus(_UiModel):
    gpu_name: str | None
    gpu_util_pct: float | None
    gpu_mem_used_mb: float | None
    gpu_mem_total_mb: float | None
    gpu_power_w: float | None


class Checkpoint(_UiModel):
    model: str
    name: str
    step: int
    loss: float | None
    path: str
    mtime: float


class StatusResponse(_UiModel):
    training: TrainingStatus
    system: SystemStatus
    checkpoints: list[Checkpoint]


class Dataset(_UiModel):
    name: str
    path: str
    kind: Literal["npy", "jsonl"]
    size_bytes: int
    mtime: float
    has_weights: bool
    num_samples: int | None


class Job(_UiModel):
    id: str
    name: str
    pid: int
    status: Literal["running", "done", "failed"]
    cmd: list[str]
    log: str
    started: float
    returncode: int | None


class ActionDef(_UiModel):
    key: str
    script: str
    label: str
    args: list[str]
    trainer: bool = False


class SftFile(_UiModel):
    name: str
    path: str
    kind: str
    num_records: int
    size_bytes: int
    mtime: float


class SftPreview(_UiModel):
    path: str
    records: list[dict[str, JsonValue]]
    fields: list[str]
    count: int
    truncated: bool


class ScriptInfo(_UiModel):
    name: str
    category: str
    purpose: str
    exists: bool


class ScriptsCatalog(_UiModel):
    scripts: list[ScriptInfo]
    categories: list[str]
    count: int
    on_disk: int


class ScoreSummary(_UiModel):
    n: int
    mean: float
    min: float
    max: float
    histogram: list[int]
    bins: list[float]


class Preview(_UiModel):
    kind: str
    num_samples: int | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    preview: list[JsonValue] | None = None
    error: str | None = None


class ConfigFile(_UiModel):
    name: str
    path: str
    rel: str
    size_bytes: int
    mtime: float


class ConfigContent(_UiModel):
    path: str | None = None
    content: str | None = None
    parsed: JsonValue = None
    truncated: bool | None = None
    error: str | None = None


class PipelineRun(_UiModel):
    name: str
    path: str
    mtime: float
    num_stages: int | None
    status: str | None
    completed: int | None
    error: str | None = None


class EvalResult(_UiModel):
    name: str
    path: str
    kind: str
    mtime: float
    summary: str


class EvalDetail(_UiModel):
    path: str
    kind: str
    parsed: JsonValue
    content: str | None
    truncated: bool


class LogFile(_UiModel):
    name: str
    path: str
    size_bytes: int
    mtime: float


class LogTail(_UiModel):
    path: str
    lines: list[str]
    size_bytes: int
    truncated: bool


class FeatureItem(_UiModel):
    key: str
    enabled: bool
    value: JsonValue


class FeatureGroup(_UiModel):
    category: str
    features: list[FeatureItem]


class FeaturesView(_UiModel):
    path: str
    total: int
    enabled: int
    groups: list[FeatureGroup]


class ReasoningSummary(_UiModel):
    advantage_norm: JsonValue
    clip_epsilon: JsonValue
    clip_epsilon_high: JsonValue
    group_size: JsonValue
    sft_warmup_enabled: bool | None
    problem_source: JsonValue


class ReasoningView(_UiModel):
    path: str
    parsed: dict[str, JsonValue]
    reasoning: dict[str, JsonValue]
    problem_set: dict[str, JsonValue]
    sft_warmup: dict[str, JsonValue]
    summary: ReasoningSummary


class TokenizerInfo(_UiModel):
    path: str
    vocab_size: int
    n_merges: int
    special_tokens: list[str]
    has_fim_tokens: bool
    digit_splitting: bool
    model_type: str


class CheckpointDetail(_UiModel):
    path: str
    metadata: dict[str, JsonValue] | None
    is_moe: bool
    moe_config: dict[str, JsonValue] | None
    has_training_state: bool
    num_params: int
    tensor_count: int
    dtypes: list[str]
    files: list[str]


class RouterCheckpoint(_UiModel):
    path: str
    name: str
    step: int | None


class RouterOverview(_UiModel):
    has_router: bool
    checkpoints: list[RouterCheckpoint]
    domains: list[str]


class ExportFormat(_UiModel):
    key: str
    label: str
    desc: str


class ExportItem(_UiModel):
    path: str
    format: str
    size_bytes: int
    mtime: float


class ExportCheckpoint(_UiModel):
    model: str
    name: str
    step: int | None
    path: str


class ExportOverview(_UiModel):
    checkpoints: list[ExportCheckpoint]
    formats: list[ExportFormat]
    existing: list[ExportItem]


class MetricPoint(_UiModel):
    step: int
    loss: float | None
    ppl: float | None
    lr: float | None
    tok_s: float | None


class MetricsHistory(_UiModel):
    points: list[MetricPoint]
    count: int


class DataSource(_UiModel):
    name: str
    weight: float | None
    dataset: str | None
    languages: list[str]
    kind: str | None


class DataSourcesView(_UiModel):
    path: str
    sources: list[DataSource]
    total_weight: float | None
    parsed: dict[str, JsonValue]
    summary: str


class EvalSnapshot(_UiModel):
    step: int | None
    path: str
    mtime: float
    metrics: dict[str, JsonValue]


class EvalHistoryView(_UiModel):
    snapshots: list[EvalSnapshot]
    count: int
    metric_keys: list[str]


class TokenizeResult(_UiModel):
    path: str
    count: int
    ids: list[int]
    tokens: list[str]
    truncated: bool = False


class HealthCheck(_UiModel):
    name: str
    ok: bool
    detail: str


class HealthSummary(_UiModel):
    score: int
    checks: list[HealthCheck]
    summary: str


class StorageEntry(_UiModel):
    name: str
    path: str
    exists: bool
    size_bytes: int | None


class StorageView(_UiModel):
    path: str
    raw: dict[str, JsonValue]
    tokenizer_path: str | None
    data_dir: str | None
    checkpoint_dir: str | None
    entries: list[StorageEntry]


class CompareDiff(_UiModel):
    num_params_delta: int
    tensor_count_delta: int
    is_moe_changed: bool
    metadata_changed_keys: list[str]
    dtypes_only_a: list[str]
    dtypes_only_b: list[str]


class CompareResult(_UiModel):
    a: CheckpointDetail
    b: CheckpointDetail
    diff: CompareDiff


class ModelCard(_UiModel):
    path: str
    name: str
    num_params: int
    architecture: dict[str, JsonValue]
    training: dict[str, JsonValue]
    tokenizer: dict[str, JsonValue] | None
    is_moe: bool
    markdown: str


class FeatureSetResult(_UiModel):
    ok: Literal[True]
    key: str
    enabled: bool
    path: str


class ConfigChange(_UiModel):
    key: str
    a: JsonValue
    b: JsonValue


class ConfigDiffSide(_UiModel):
    path: str
    parsed: dict[str, JsonValue]


class ConfigDiff(_UiModel):
    a: ConfigDiffSide
    b: ConfigDiffSide
    changed: list[ConfigChange]
    only_a: list[str]
    only_b: list[str]


class GpuInfo(_UiModel):
    name: str
    mem_total_mb: int | None
    mem_used_mb: int | None
    util_pct: int | None


class DiskInfo(_UiModel):
    path: str
    total_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None


class SystemInfo(_UiModel):
    python_version: str
    platform: str
    packages: dict[str, str | None]
    gpus: list[GpuInfo]
    disk: DiskInfo


class ErrorResponse(_UiModel):
    """Maps to the TS ``ApiError`` interface."""

    error: str
