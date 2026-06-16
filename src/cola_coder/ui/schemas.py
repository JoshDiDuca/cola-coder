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
    # Seconds since the training step last advanced (0 right after an advance);
    # None until a step is seen. Surfaces hung-vs-slow on the dashboard (OPS-003).
    step_stalled_s: float | None = None
    # Human-readable time-remaining from the trainer's "ETA …" log suffix
    # (e.g. "338h 58m 51s"), or None if the log carries no ETA (BUG-138).
    eta: str | None = None


class LossStability(_UiModel):
    """Dashboard loss-stability meter (ZClip z-score spike idea on the loss curve)."""

    current_loss: float | None
    ema_loss: float | None
    trend: Literal["improving", "flat", "worsening", "unknown"]
    spike_count: int
    recent_max_z: float | None
    verdict: Literal["stable", "watch", "spiking", "insufficient_data"]
    points_used: int


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


class ScoreSnippetRequest(_UiModel):
    """Body for ``POST /api/data/score-snippet`` — score ad-hoc code (pure-Python scorers)."""

    code: str


class ScorerBreakdown(_UiModel):
    """One scorer's 0–1 quality score + its tier for a snippet."""

    name: str
    score: float
    tier: str


class SnippetScores(_UiModel):
    """Per-scorer quality breakdown for a snippet + the unweighted mean."""

    scorers: list[ScorerBreakdown]
    mean_score: float
    mean_tier: str
    count: int


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


class ActionParam(_UiModel):
    """One CLI argument of an action, described for a typed UI form control.

    Maps 1:1 to a script's argparse argument so the UI can render the right control
    (config/checkpoint dropdown, number, flag checkbox, choice select) instead of a
    raw flag string. The frontend builds the args list from these + the form values.
    """

    name: str
    flag: str  # e.g. "--config"; "" for a positional argument
    label: str
    type: Literal["string", "int", "float", "bool", "choice", "config", "checkpoint", "path"]
    default: str | None = None
    choices: list[str] = []
    required: bool = False
    help: str | None = None


class ActionDef(_UiModel):
    key: str
    script: str
    label: str
    args: list[str]
    trainer: bool = False
    gpu: bool = False
    params: list[ActionParam] = []
    # Backend-assigned group for the Run screen (schema-first, replaces the
    # frontend's brittle name-regex heuristic). The UI groups actions by this.
    category: Literal[
        "Data", "Training", "Pipeline", "Evaluation", "Inspection", "Export", "Tools"
    ] = "Tools"


class JobLogChunk(_UiModel):
    """One Server-Sent-Events frame from a job's live log stream."""

    text: str
    done: bool


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


class PipelineStageState(_UiModel):
    num: int
    name: str
    description: str
    optional: bool
    status: str
    artifact: str
    override: str
    error: str
    duration_secs: float
    started_at: str | None = None
    completed_at: str | None = None


class PipelineRunDetail(_UiModel):
    name: str
    config_path: str
    created_at: str
    updated_at: str
    notes: str
    stages: list[PipelineStageState]
    num_stages: int
    active_stages: int
    completed: int
    status: str


class PipelineDeleteResult(_UiModel):
    ok: bool
    name: str


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


class CheckpointNote(_UiModel):
    """A user annotation for one checkpoint (keyed by its path). Sidecar-stored."""

    key: str
    label: str
    note: str
    updated_at: str


class CheckpointNotes(_UiModel):
    """All checkpoint notes (``GET /api/checkpoints/notes``)."""

    notes: list[CheckpointNote]


class CheckpointNoteSetRequest(_UiModel):
    """Body for ``POST /api/checkpoints/notes/set`` — upsert one note by key (path)."""

    key: str
    label: str = ""
    note: str = ""


class CheckpointNoteDeleteRequest(_UiModel):
    """Body for ``POST /api/checkpoints/notes/delete`` — remove a note by key."""

    key: str


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


class TokenizerHealthItem(_UiModel):
    name: str
    ok: bool
    detail: str


class TokenizerHealthReport(_UiModel):
    path: str
    vocab_size: int
    checks: list[TokenizerHealthItem]
    passed: int
    failed: int
    ok: bool


class WeightTier(_UiModel):
    label: str
    count: int
    pct: float


class DataStats(_UiModel):
    data_path: str
    file_size_mb: float
    shape: list[int]
    num_chunks: int
    seq_len: int | None = None
    total_tokens: int
    token_min: int
    token_max: int
    token_mean: float
    est_unique_tokens: int | None = None
    has_weights: bool = False
    weights_path: str | None = None
    weight_tiers: list[WeightTier] = []
    weight_mean: float | None = None
    weight_std: float | None = None


class CheckpointHealth(_UiModel):
    path: str
    model: str
    step: int
    loss: float | None
    size_mb: float
    num_tensors: int | None
    files: list[str]
    config_stem: str | None
    ok: bool


class MemoryEntry(_UiModel):
    id: str
    type: str
    created_at: str
    content_preview: str


class MemoryStats(_UiModel):
    total_entries: int
    pinned: int
    types: list[str]
    size_bytes: int
    oldest_at: str | None
    newest_at: str | None
    recent_sample: list[MemoryEntry]


class MemoryFile(_UiModel):
    """Full content of one themed memory markdown file (``GET /api/memory/export``)."""

    type: str
    name: str
    content: str
    truncated: bool
    entry_count: int


class MemoryExport(_UiModel):
    """All memory files for the read view; ``initialized`` false when empty."""

    initialized: bool
    files: list[MemoryFile]


class MemoryAddRequest(_UiModel):
    """Body for ``POST /api/memory/add`` — append one entry to a theme.

    ``primary`` is the main text; ``secondary`` the optional second field
    (example/fix/rationale/content/domain depending on ``kind``).
    """

    kind: Literal["pattern", "error", "decision", "domain", "session"]
    primary: str
    secondary: str = ""


class MemorySearchRequest(_UiModel):
    """Body for ``POST /api/memory/search`` — TF-IDF query (CPU only)."""

    query: str
    max_chunks: int = 5


class MemoryChunkOut(_UiModel):
    """One TF-IDF search hit from the memory store."""

    content: str
    source_file: str
    section: str
    relevance_score: float


class MemorySearchResult(_UiModel):
    """Result of ``POST /api/memory/search``."""

    query: str
    results: list[MemoryChunkOut]


class MemoryFileCount(_UiModel):
    """Per-file duplicate-removal count from a compaction."""

    name: str
    removed: int


class MemoryCompactResult(_UiModel):
    """Result of ``POST /api/memory/compact``."""

    removed_total: int
    removed: list[MemoryFileCount]


class IndexStats(_UiModel):
    exists: bool
    doc_count: int
    chunk_count: int | None
    embedding_model: str | None
    embedding_dim: int | None
    size_bytes: int
    path: str | None
    last_updated: str | None


class ThreatInfo(_UiModel):
    file_path: str
    name: str
    severity: str
    scanner: str
    details: str | None


class MalwareScanResult(_UiModel):
    path: str
    files_scanned: int
    is_clean: bool
    threats: list[ThreatInfo]
    duration_ms: float


class EnvCheckItem(_UiModel):
    name: str
    ok: bool
    value: str
    detail: str | None


class EnvCheckReport(_UiModel):
    python_version: str
    torch_version: str | None
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    hf_token_set: bool
    passed: int
    failed: int
    ok: bool
    checks: list[EnvCheckItem]


class VramComponent(_UiModel):
    name: str
    mb: float


class VramEstimate(_UiModel):
    config: str
    params_millions: float
    precision: str
    batch_size: int
    seq_len: int
    components: list[VramComponent]
    total_mb: float
    budget_mb: float
    fits: bool


class HealthDimension(_UiModel):
    name: str
    score: float
    detail: str


class ProjectHealthReport(_UiModel):
    overall_score: float
    grade: str
    dimensions: list[HealthDimension]
    summary: str


class BenchmarkRun(_UiModel):
    name: str
    path: str
    kind: str
    tokens_per_s: float | None
    latency_ms: float | None
    config: str | None
    checkpoint: str | None
    mtime: float


class BenchmarkResults(_UiModel):
    runs: list[BenchmarkRun]
    count: int


class SafetyProbe(_UiModel):
    suite: str
    name: str
    passed: bool
    detail: str | None


class SafetyEvalRun(_UiModel):
    name: str
    path: str
    checkpoint: str | None
    suite: str
    total: int
    passed: int
    failed: int
    mtime: float
    probes: list[SafetyProbe]


class SafetyEvalResults(_UiModel):
    runs: list[SafetyEvalRun]
    count: int


class FilterInfo(_UiModel):
    name: str
    category: str
    purpose: str
    module: str
    default_enabled: bool


class FiltersCatalog(_UiModel):
    filters: list[FilterInfo]
    count: int
    categories: list[str]


class ReasoningProblem(_UiModel):
    id: str
    difficulty: str
    language: str
    prompt_preview: str
    has_tests: bool


class ReasoningProblemSet(_UiModel):
    problems: list[ReasoningProblem]
    count: int
    difficulties: list[str]
    languages: list[str]


class VocabToken(_UiModel):
    id: int
    piece: str
    is_special: bool


class VocabSearchResult(_UiModel):
    query: str
    vocab_size: int
    total_matches: int
    truncated: bool
    tokens: list[VocabToken]
    special_tokens: list[VocabToken]


class ScorerConfigEntry(_UiModel):
    name: str
    enabled: bool
    weight: float
    available: bool
    purpose: str


class ScoringConfig(_UiModel):
    path: str
    scorers: list[ScorerConfigEntry]
    count: int
    enabled_count: int
    curriculum: str | None


class RegressionMetric(_UiModel):
    name: str
    value: float | None
    baseline: float | None
    delta: float | None
    regressed: bool


class RegressionRun(_UiModel):
    name: str
    path: str
    checkpoint: str | None
    mtime: float
    passed: bool
    metrics: list[RegressionMetric]


class RegressionHistory(_UiModel):
    runs: list[RegressionRun]
    count: int


class DocFile(_UiModel):
    name: str
    path: str
    rel: str
    title: str
    size_bytes: int


class DocsList(_UiModel):
    docs: list[DocFile]
    count: int


class DocContent(_UiModel):
    path: str
    content: str
    truncated: bool


class BacklogItem(_UiModel):
    id: str
    category: str
    severity: str
    status: str
    date: str | None
    description: str


class BacklogView(_UiModel):
    items: list[BacklogItem]
    count: int
    open_count: int
    done_count: int


class ResearchEntry(_UiModel):
    date: str
    title: str
    area: str | None
    source_count: int
    has_original_idea: bool
    summary: str


class ResearchLog(_UiModel):
    entries: list[ResearchEntry]
    count: int


class ResearchLogAppendRequest(_UiModel):
    """Body for ``POST /api/research-log/append`` — add a dated entry (append-only)."""

    title: str
    body: str


class BacklogAppendRequest(_UiModel):
    """Body for ``POST /api/backlog/append`` — file a new backlog item (append-only)."""

    item_id: str
    category: str
    description: str
    severity: str = ""
    status: Literal["open", "in-progress", "done", "dropped"] = "open"


class TrainingManifest(_UiModel):
    model: str
    path: str
    config: str | None
    dim: int | None
    n_layers: int | None
    n_heads: int | None
    seq_len: int | None
    batch_size: int | None
    learning_rate: float | None
    max_steps: int | None
    latest_step: int | None
    created_at: str | None
    mtime: float


class TrainingManifests(_UiModel):
    manifests: list[TrainingManifest]
    count: int


class LrPoint(_UiModel):
    lr: float
    loss: float


class LrFinderRun(_UiModel):
    name: str
    path: str
    config: str | None
    suggested_lr: float | None
    min_loss: float | None
    num_points: int
    mtime: float
    points: list[LrPoint]


class LrFinderResults(_UiModel):
    runs: list[LrFinderRun]
    count: int


class RepoScore(_UiModel):
    repo: str
    score: float
    stars: int | None
    language: str | None
    license: str | None
    reason: str | None


class RepoScoresResult(_UiModel):
    path: str
    repos: list[RepoScore]
    count: int
    mtime: float


class RunRequest(_UiModel):
    """Body for ``POST /api/run`` — launch an allow-listed action as a job."""

    action: str
    args: list[str] | None = None


class TrainStartRequest(_UiModel):
    """Body for ``POST /api/train/start``."""

    config: str = "configs/small.yaml"
    resume: str | None = None


class InferenceRequest(_UiModel):
    """Body for ``POST /api/generate`` — one-shot code generation from the UI.

    Refused (HTTP 409) while a training run is live so a UI generation can never
    contend with the live trainer for the GPU.
    """

    prompt: str
    checkpoint: str
    config: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    # Advanced sampling (0 = disabled): min-p keeps tokens >= min_p*max_prob
    # (adapts to peakedness; good for small models); top-nσ truncates on raw
    # logits at mean + n·std. Both supported by the generator (Nguyen et al. 2024).
    min_p: float = 0.0
    top_n_sigma: float = 0.0


class InferenceResult(_UiModel):
    """Result of a one-shot ``/api/generate`` / ``/api/chat`` / ``/api/fim`` call.

    ``completion`` holds the generated text (the reply for chat, the infill for
    FIM). Shared across all three inference endpoints to keep the contract small.
    """

    completion: str
    prompt: str
    checkpoint: str
    tokens_generated: int
    elapsed_s: float


class BestOfNRequest(_UiModel):
    """Body for ``POST /api/best-of`` — sandbox-verified best-of-N generation.

    Gated like ``/api/generate``. Generates N candidates, verifies each (tsc/exec/
    parse, sandboxed), returns the best. Refused (409) while training is live.
    """

    prompt: str
    checkpoint: str
    config: str
    num_candidates: int = 4
    language: Literal["auto", "python", "typescript"] = "auto"
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    # min-p floor (0 = disabled); top-nσ is not wired for best-of.
    min_p: float = 0.0


class BestOfNCandidate(_UiModel):
    """One candidate from a best-of-N run (completion only + its verdict)."""

    completion: str
    verified: bool
    score: float


class BestOfNResponse(_UiModel):
    """Result of ``POST /api/best-of`` — the best candidate + all ranked candidates."""

    best_completion: str
    language: str
    verifier: str
    solved: bool
    candidates_used: int
    elapsed_s: float
    candidates: list[BestOfNCandidate]


class GenStreamChunk(_UiModel):
    """One Server-Sent-Events frame from ``POST /api/generate/stream``.

    ``delta`` is the incremental text produced since the previous frame. The final
    frame has ``done=true`` (and ``delta=""``); ``error`` is set instead of streaming
    when load/generation fails (e.g. the training-alive guard, a bad checkpoint).
    """

    delta: str
    done: bool
    error: str | None = None


class ChatMessage(_UiModel):
    """One turn in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(_UiModel):
    """Body for ``POST /api/chat`` — multi-turn chat. Gated like ``/api/generate``."""

    messages: list[ChatMessage]
    checkpoint: str
    config: str
    use_chat_template: bool = True
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    top_n_sigma: float = 0.0


class FimRequest(_UiModel):
    """Body for ``POST /api/fim`` — fill-in-the-middle. Gated like ``/api/generate``."""

    prefix: str
    suffix: str
    checkpoint: str
    config: str
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    top_n_sigma: float = 0.0


class RetrievalHit(_UiModel):
    """One lexical-search hit from the persisted retrieval index."""

    id: str
    score: float
    snippet: str
    source: str | None = None


class RetrievalSearchResult(_UiModel):
    """Result of ``GET /api/retrieval/search`` — lexical rank over indexed chunks."""

    query: str
    exists: bool
    total_indexed: int
    hits: list[RetrievalHit]


class GpuProcess(_UiModel):
    """One process holding the GPU (from nvidia-smi compute-apps)."""

    pid: int
    name: str
    used_memory_mb: float | None


class GpuProcesses(_UiModel):
    """Processes currently using the GPU. ``available`` is False when nvidia-smi
    is absent; ``restricted`` flags entries whose name is hidden by OS integrity
    (the elevated trainer shows as '[Insufficient Permissions]' under OPS-001)."""

    available: bool
    count: int
    processes: list[GpuProcess]
    restricted: bool


class SpecialistEntry(_UiModel):
    """One domain specialist from ``configs/specialists.yaml`` (router registry)."""

    domain: str
    checkpoint: str
    config: str | None = None
    keywords: list[str] = []
    confidence_threshold: float | None = None
    description: str | None = None


class SpecialistsView(_UiModel):
    """The router specialist registry (Vision: router + per-domain 50M specialists)."""

    path: str
    exists: bool
    count: int
    specialists: list[SpecialistEntry]


class DomainDetectRequest(_UiModel):
    """Body for ``POST /api/router/detect-domain`` — classify a code snippet.

    Pure regex heuristic (``features.domain_detector``); no model/GPU. ``filename``
    is optional extra context (e.g. ``Button.test.tsx`` boosts the testing domain).
    """

    code: str
    filename: str = ""


class DomainScoreOut(_UiModel):
    """One domain's match breakdown + normalized confidence (0–1)."""

    domain: str
    import_matches: int
    keyword_matches: int
    raw_score: float
    confidence: float


class RouteDecisionOut(_UiModel):
    """The margin-aware routing decision (MODEL-053) for the detected snippet.

    Shows what the specialist cascade would actually DO: dispatch to ``domain``,
    or abstain to ``general`` when the top pick's confidence/margin is too weak
    (``abstained`` true; ``reason`` names the guard).
    """

    domain: str
    confidence: float
    margin: float
    abstained: bool
    reason: str


class DomainDetectResult(_UiModel):
    """Result of ``POST /api/router/detect-domain`` — ranked scores + routing."""

    top_domain: str
    scores: list[DomainScoreOut]
    routing: RouteDecisionOut


class SpecialistSaveRequest(_UiModel):
    """Body for ``POST /api/specialists/save`` — upsert one registry entry.

    Adds the domain if new, updates it in place if it exists. The backend
    validates and atomically rewrites ``configs/specialists.yaml``.
    """

    domain: str
    checkpoint: str
    keywords: list[str] = []
    config: str | None = None
    confidence_threshold: float | None = None
    description: str | None = None


class SpecialistRemoveRequest(_UiModel):
    """Body for ``POST /api/specialists/remove`` — delete one entry by domain."""

    domain: str


class ConfigKV(_UiModel):
    """One label→value row in a config summary (value coerced to str at boundary)."""

    label: str
    value: str


class ConfigGroup(_UiModel):
    """A titled group of config rows (e.g. Model, Training)."""

    title: str
    items: list[ConfigKV]


class ConfigSummary(_UiModel):
    """Grouped hyperparameter summary of a YAML config (``GET /api/config/summary``)."""

    path: str
    name: str
    exists: bool
    groups: list[ConfigGroup]


class ConfigWriteRequest(_UiModel):
    """Body for ``POST /api/config/write`` — save edited YAML config text."""

    path: str
    content: str


class ConfigWriteResult(_UiModel):
    """Result of a successful config write (validated YAML, atomic on-disk replace)."""

    ok: bool
    path: str
    bytes_written: int


class ErrorResponse(_UiModel):
    """Maps to the TS ``ApiError`` interface."""

    error: str
