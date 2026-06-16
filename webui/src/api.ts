import type {
  StatusResponse,
  Dataset,
  Preview,
  ScoreSummary,
  Job,
  ActionDef,
  ConfigFile,
  ConfigContent,
  PipelineRun,
  PipelineRunDetail,
  PipelineDeleteResult,
  EvalResult,
  EvalDetail,
  LogFile,
  LogTail,
  FeaturesView,
  ReasoningView,
  TokenizerInfo,
  CheckpointDetail,
  RouterOverview,
  ExportOverview,
  MetricsHistory,
  DataSourcesView,
  EvalHistoryView,
  TokenizeResult,
  HealthSummary,
  SftFile,
  SftPreview,
  ScriptsCatalog,
  StorageView,
  CompareResult,
  ModelCard,
  FeatureSetResult,
  ConfigDiff,
  SystemInfo,
  TokenizerHealthReport,
  DataStats,
  CheckpointHealth,
  CheckpointNotes,
  CheckpointNoteSetRequest,
  MemoryStats,
  MemoryExport,
  MemoryAddRequest,
  MemorySearchRequest,
  MemorySearchResult,
  MemoryCompactResult,
  IndexStats,
  MalwareScanResult,
  EnvCheckReport,
  VramEstimate,
  ProjectHealthReport,
  BenchmarkResults,
  SafetyEvalResults,
  FiltersCatalog,
  ReasoningProblemSet,
  VocabSearchResult,
  ScoringConfig,
  RegressionHistory,
  DocsList,
  DocContent,
  BacklogView,
  BacklogAppendRequest,
  ResearchLog,
  ResearchLogAppendRequest,
  TrainingManifests,
  LrFinderResults,
  RepoScoresResult,
  RunRequest,
  TrainStartRequest,
  InferenceRequest,
  InferenceResult,
  ChatRequest,
  FimRequest,
  BestOfNRequest,
  BestOfNResponse,
  ConfigWriteRequest,
  ConfigWriteResult,
  SpecialistsView,
  SpecialistSaveRequest,
  DomainDetectRequest,
  DomainDetectResult,
  RetrievalSearchResult,
  GpuProcesses,
  ApiError,
  JsonValue,
} from './types';

async function j<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${url}`);
  }
  return (await res.json()) as T;
}

// Request bodies are keyed by string; values are JSON-serializable, and an
// optional field may be `undefined` (JSON.stringify simply drops it).
// Accepts any typed request object (named schema models like RunRequest, or
// inline literals). Generic over `object` so concrete interfaces — which lack an
// implicit index signature — are assignable; the body is JSON-serialized here at
// the single request boundary.
function postJson<T extends object>(body: T): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

export function getStatus(): Promise<StatusResponse> {
  return j<StatusResponse>('/api/status');
}

export function getDatasets(): Promise<Dataset[]> {
  return j<Dataset[]>('/api/datasets');
}

export function getPreview(path: string, n?: number): Promise<Preview> {
  const q = new URLSearchParams({ path });
  if (n !== undefined) q.set('n', String(n));
  return j<Preview>(`/api/datasets/preview?${q.toString()}`);
}

export function getScores(path: string): Promise<ScoreSummary> {
  return j<ScoreSummary>(`/api/datasets/scores?path=${encodeURIComponent(path)}`);
}

export function getJobs(): Promise<Job[]> {
  return j<Job[]>('/api/jobs');
}

export function getJobLog(id: string, lines?: number): Promise<{ log: string }> {
  const q = new URLSearchParams();
  if (lines !== undefined) q.set('lines', String(lines));
  const qs = q.toString();
  return j<{ log: string }>(
    `/api/jobs/${encodeURIComponent(id)}/log${qs ? `?${qs}` : ''}`
  );
}

export function stopJob(id: string): Promise<{ stopped: boolean }> {
  return j<{ stopped: boolean }>(
    `/api/jobs/${encodeURIComponent(id)}/stop`,
    { method: 'POST' }
  );
}

// Live job-log stream (SSE). The caller opens an EventSource on this URL and
// parses each frame as a `JobLogChunk` ({ text, done }). Returned as a URL (not
// a fetch) because EventSource manages its own connection + auto-reconnect.
export function jobLogStreamUrl(id: string, tail = 200): string {
  return `/api/jobs/${encodeURIComponent(id)}/stream?tail=${tail}`;
}

// One-shot inference for the playground. The backend refuses with HTTP 409
// while a training run is live (the j() helper then throws an Error whose
// message contains "409"); it may also resolve an ApiError body.
export function generateText(req: InferenceRequest): Promise<InferenceResult | ApiError> {
  return j<InferenceResult | ApiError>('/api/generate', postJson(req));
}

// Multi-turn chat. Gated like /api/generate (409 while training is live).
export function chatGenerate(req: ChatRequest): Promise<InferenceResult | ApiError> {
  return j<InferenceResult | ApiError>('/api/chat', postJson(req));
}

// Open the token-streaming generation endpoint. Returns the raw Response so the
// caller can read response.body as a stream of SSE `GenStreamChunk` frames
// (the shared j() helper buffers the whole body, so it can't be used here).
// `signal` lets the caller abort an in-flight stream. A 409 (training live) or
// other non-OK status arrives as `res.ok === false` with a JSON {error} body.
export function openGenerateStream(req: InferenceRequest, signal: AbortSignal): Promise<Response> {
  return fetch('/api/generate/stream', { ...postJson(req), signal });
}

// Streaming variants of chat + FIM (same SSE GenStreamChunk protocol + 409 gating
// as openGenerateStream). The caller reads response.body; `signal` aborts.
export function openChatStream(req: ChatRequest, signal: AbortSignal): Promise<Response> {
  return fetch('/api/chat/stream', { ...postJson(req), signal });
}

export function openFimStream(req: FimRequest, signal: AbortSignal): Promise<Response> {
  return fetch('/api/fim/stream', { ...postJson(req), signal });
}

// Fill-in-the-middle completion. Gated like /api/generate; 400 if the
// tokenizer lacks <|fim_*|> tokens.
export function fimGenerate(req: FimRequest): Promise<InferenceResult | ApiError> {
  return j<InferenceResult | ApiError>('/api/fim', postJson(req));
}

// Sandbox-verified best-of-N generation. Gated like /api/generate (409 while training).
export function bestOfN(req: BestOfNRequest): Promise<BestOfNResponse | ApiError> {
  return j<BestOfNResponse | ApiError>('/api/best-of', postJson(req));
}

export function getActions(): Promise<ActionDef[]> {
  return j<ActionDef[]>('/api/actions');
}

export function runAction(action: string, args?: string[]): Promise<Job> {
  const body: RunRequest = { action, args: args ?? null };
  return j<Job>('/api/run', postJson(body));
}

export function startTraining(
  config: string,
  resume?: string
): Promise<Job | ApiError> {
  const body: TrainStartRequest = { config, resume: resume ?? null };
  return j<Job | ApiError>(
    '/api/train/start',
    postJson(body)
  );
}

export function getConfigs(): Promise<ConfigFile[]> {
  return j<ConfigFile[]>('/api/configs');
}

export function getConfig(path: string): Promise<ConfigContent> {
  return j<ConfigContent>(`/api/config?path=${encodeURIComponent(path)}`);
}

// Router specialist registry (configs/specialists.yaml).
export function getSpecialists(): Promise<SpecialistsView | ApiError> {
  return j<SpecialistsView | ApiError>('/api/specialists');
}

// Upsert one specialist (add or update). Backend validates + atomically rewrites
// the registry and returns the refreshed view; 400 (ApiError) on bad input.
export function saveSpecialist(req: SpecialistSaveRequest): Promise<SpecialistsView | ApiError> {
  return j<SpecialistsView | ApiError>('/api/specialists/save', postJson(req));
}

// Remove one specialist by domain. Returns the refreshed registry; 400 if missing.
export function removeSpecialist(domain: string): Promise<SpecialistsView | ApiError> {
  return j<SpecialistsView | ApiError>('/api/specialists/remove', postJson({ domain }));
}

// Classify a code snippet by framework/domain (regex heuristic; no model/GPU).
// Returns ranked domain scores; 400 on empty code.
export function detectDomain(req: DomainDetectRequest): Promise<DomainDetectResult | ApiError> {
  return j<DomainDetectResult | ApiError>('/api/router/detect-domain', postJson(req));
}

// Lexical search over the persisted retrieval index (no model/GPU; empty when no index).
export function searchRetrieval(q: string, topK = 10): Promise<RetrievalSearchResult | ApiError> {
  return j<RetrievalSearchResult | ApiError>(
    `/api/retrieval/search?q=${encodeURIComponent(q)}&top_k=${topK}`,
  );
}

// Processes currently holding the GPU (nvidia-smi compute-apps) — GPU-contention view.
export function getGpuProcesses(): Promise<GpuProcesses> {
  return j<GpuProcesses>('/api/gpu/processes');
}

// Save an edited YAML config. Backend validates YAML + path (inside configs/)
// before writing atomically; returns 400 (ApiError) on invalid YAML / bad path.
export function writeConfig(req: ConfigWriteRequest): Promise<ConfigWriteResult | ApiError> {
  return j<ConfigWriteResult | ApiError>('/api/config/write', postJson(req));
}

export function getPipelineRuns(): Promise<PipelineRun[]> {
  return j<PipelineRun[]>('/api/pipeline/runs');
}

// The backend (`read_pipeline_run`) returns either the full arbitrary
// pipeline-run-state object or `{"error": str}`. There is no generated detail
// schema (the run JSON is open-ended), so this is the sanctioned `JsonValue`
// case: a string-keyed record of JSON, or an `ApiError`.
export function getPipelineRun(path: string): Promise<Record<string, JsonValue> | ApiError> {
  return j<Record<string, JsonValue> | ApiError>(
    `/api/pipeline/run?path=${encodeURIComponent(path)}`
  );
}

// ── Pipeline run lifecycle (pure state ops — never execute a stage) ──────────
export function getPipelineDetail(name: string): Promise<PipelineRunDetail | ApiError> {
  return j<PipelineRunDetail | ApiError>(`/api/pipeline/detail?name=${encodeURIComponent(name)}`);
}

export function createPipelineRun(
  name: string,
  configPath: string,
  skipStages: number[] = [],
): Promise<PipelineRunDetail | ApiError> {
  return j<PipelineRunDetail | ApiError>(
    '/api/pipeline/create',
    postJson({ name, config_path: configPath, skip_stages: skipStages }),
  );
}

export function resetPipelineRun(
  name: string,
  stageNum: number,
): Promise<PipelineRunDetail | ApiError> {
  return j<PipelineRunDetail | ApiError>(
    '/api/pipeline/reset',
    postJson({ name, stage_num: stageNum }),
  );
}

export function setPipelineOverride(
  name: string,
  stageNum: number,
  path: string,
): Promise<PipelineRunDetail | ApiError> {
  return j<PipelineRunDetail | ApiError>(
    '/api/pipeline/override',
    postJson({ name, stage_num: stageNum, path }),
  );
}

export function deletePipelineRun(name: string): Promise<PipelineDeleteResult | ApiError> {
  return j<PipelineDeleteResult | ApiError>('/api/pipeline/delete', postJson({ name }));
}

export function getEvals(): Promise<EvalResult[]> {
  return j<EvalResult[]>('/api/evals');
}

export function getEval(path: string): Promise<EvalDetail | ApiError> {
  return j<EvalDetail | ApiError>(`/api/eval?path=${encodeURIComponent(path)}`);
}

export function getLogs(): Promise<LogFile[]> {
  return j<LogFile[]>('/api/logs');
}

export function getLog(path: string, lines?: number): Promise<LogTail | ApiError> {
  const q = new URLSearchParams({ path });
  if (lines !== undefined) q.set('lines', String(lines));
  return j<LogTail | ApiError>(`/api/log?${q.toString()}`);
}

export function getFeatures(): Promise<FeaturesView | ApiError> {
  return j<FeaturesView | ApiError>('/api/features');
}

export function getReasoning(): Promise<ReasoningView | ApiError> {
  return j<ReasoningView | ApiError>('/api/reasoning');
}

export function getTokenizer(): Promise<TokenizerInfo | ApiError> {
  return j<TokenizerInfo | ApiError>('/api/tokenizer');
}

export function getCheckpointDetail(path: string): Promise<CheckpointDetail | ApiError> {
  return j<CheckpointDetail | ApiError>(
    `/api/checkpoint?path=${encodeURIComponent(path)}`
  );
}

export function getRouter(): Promise<RouterOverview | ApiError> {
  return j<RouterOverview | ApiError>('/api/router');
}

export function getExports(): Promise<ExportOverview | ApiError> {
  return j<ExportOverview | ApiError>('/api/exports');
}

export function getMetricsHistory(): Promise<MetricsHistory | ApiError> {
  return j<MetricsHistory | ApiError>('/api/metrics/history');
}

export function getDataSources(): Promise<DataSourcesView | ApiError> {
  return j<DataSourcesView | ApiError>('/api/data-sources');
}

export function getEvalHistory(): Promise<EvalHistoryView | ApiError> {
  return j<EvalHistoryView | ApiError>('/api/eval-history');
}

export function postTokenize(text: string): Promise<TokenizeResult | ApiError> {
  return j<TokenizeResult | ApiError>('/api/tokenize', postJson({ text }));
}

export function getHealth(): Promise<HealthSummary | ApiError> {
  return j<HealthSummary | ApiError>('/api/health');
}

export function getSftFiles(): Promise<SftFile[]> {
  return j<SftFile[]>('/api/sft');
}

export function getSftPreview(path: string, n?: number): Promise<SftPreview | ApiError> {
  const q = new URLSearchParams({ path });
  if (n !== undefined) q.set('n', String(n));
  return j<SftPreview | ApiError>(`/api/sft/preview?${q.toString()}`);
}

export function getScriptsCatalog(): Promise<ScriptsCatalog | ApiError> {
  return j<ScriptsCatalog | ApiError>('/api/scripts');
}

export function getStorage(): Promise<StorageView | ApiError> {
  return j<StorageView | ApiError>('/api/storage');
}

export function getCheckpointCompare(a: string, b: string): Promise<CompareResult | ApiError> {
  return j<CompareResult | ApiError>(
    `/api/checkpoints/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`
  );
}

// Checkpoint notes/tags (sidecar; never touches checkpoint dirs).
export function getCheckpointNotes(): Promise<CheckpointNotes> {
  return j<CheckpointNotes>('/api/checkpoints/notes');
}

export function setCheckpointNote(req: CheckpointNoteSetRequest): Promise<CheckpointNotes | ApiError> {
  return j<CheckpointNotes | ApiError>('/api/checkpoints/notes/set', postJson(req));
}

export function deleteCheckpointNote(key: string): Promise<CheckpointNotes | ApiError> {
  return j<CheckpointNotes | ApiError>('/api/checkpoints/notes/delete', postJson({ key }));
}

export function getModelCard(path: string): Promise<ModelCard | ApiError> {
  return j<ModelCard | ApiError>(`/api/model-card?path=${encodeURIComponent(path)}`);
}

export function setFeature(key: string, enabled: boolean): Promise<FeatureSetResult | ApiError> {
  return j<FeatureSetResult | ApiError>('/api/features/set', postJson({ key, enabled }));
}

export function getConfigDiff(a: string, b: string): Promise<ConfigDiff | ApiError> {
  return j<ConfigDiff | ApiError>(
    `/api/config-diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`
  );
}

export function getSystemInfo(): Promise<SystemInfo | ApiError> {
  return j<SystemInfo | ApiError>('/api/system-info');
}

export function getTokenizerHealth(path?: string): Promise<TokenizerHealthReport | ApiError> {
  const qs = path !== undefined ? `?path=${encodeURIComponent(path)}` : '';
  return j<TokenizerHealthReport | ApiError>(`/api/tokenizer-health${qs}`);
}

export function getCheckpointHealth(model: string, step: string): Promise<CheckpointHealth | ApiError> {
  return j<CheckpointHealth | ApiError>(
    `/api/checkpoint-health?model=${encodeURIComponent(model)}&step=${encodeURIComponent(step)}`,
  );
}

// Full markdown content per theme file (read view). Robust GET — never an error.
export function getMemoryExport(): Promise<MemoryExport> {
  return j<MemoryExport>('/api/memory/export');
}

// Append one entry (auto-inits the store); returns refreshed stats. MAIN-SAFE.
export function addMemory(req: MemoryAddRequest): Promise<MemoryStats | ApiError> {
  return j<MemoryStats | ApiError>('/api/memory/add', postJson(req));
}

// TF-IDF search the store (CPU only, no model/GPU). 400 on empty query.
export function searchMemory(req: MemorySearchRequest): Promise<MemorySearchResult | ApiError> {
  return j<MemorySearchResult | ApiError>('/api/memory/search', postJson(req));
}

// Drop duplicate entries; returns per-file removals. 400 if uninitialised.
export function compactMemory(): Promise<MemoryCompactResult | ApiError> {
  return j<MemoryCompactResult | ApiError>('/api/memory/compact', postJson({}));
}

export function getMemoryStats(): Promise<MemoryStats | ApiError> {
  return j<MemoryStats | ApiError>('/api/memory-stats');
}

export function getIndexStats(): Promise<IndexStats | ApiError> {
  return j<IndexStats | ApiError>('/api/retrieval/index-stats');
}

export function scanForMalware(
  path: string,
  maxFiles?: number,
): Promise<MalwareScanResult | ApiError> {
  const q = new URLSearchParams({ path });
  if (maxFiles !== undefined) q.set('max_files', String(maxFiles));
  return j<MalwareScanResult | ApiError>(`/api/security/scan?${q.toString()}`);
}

export function getEnvCheck(): Promise<EnvCheckReport | ApiError> {
  return j<EnvCheckReport | ApiError>('/api/env-check');
}

export function getVramEstimate(config: string): Promise<VramEstimate | ApiError> {
  return j<VramEstimate | ApiError>(`/api/vram-estimate?config=${encodeURIComponent(config)}`);
}

export function getProjectHealth(): Promise<ProjectHealthReport | ApiError> {
  return j<ProjectHealthReport | ApiError>('/api/project-health');
}

export function getBenchmarkResults(): Promise<BenchmarkResults | ApiError> {
  return j<BenchmarkResults | ApiError>('/api/benchmark-results');
}

export function getSafetyEvalResults(): Promise<SafetyEvalResults | ApiError> {
  return j<SafetyEvalResults | ApiError>('/api/safety-eval-results');
}

export function getFiltersCatalog(): Promise<FiltersCatalog | ApiError> {
  return j<FiltersCatalog | ApiError>('/api/filters-catalog');
}

export function getReasoningProblems(which?: string): Promise<ReasoningProblemSet | ApiError> {
  const qs = which !== undefined ? `?which=${encodeURIComponent(which)}` : '';
  return j<ReasoningProblemSet | ApiError>(`/api/reasoning-problems${qs}`);
}

export function searchVocab(query: string, limit?: number): Promise<VocabSearchResult | ApiError> {
  const q = new URLSearchParams({ query });
  if (limit !== undefined) q.set('limit', String(limit));
  return j<VocabSearchResult | ApiError>(`/api/vocab-search?${q.toString()}`);
}

export function getScoringConfig(): Promise<ScoringConfig | ApiError> {
  return j<ScoringConfig | ApiError>('/api/scoring-config');
}

export function getRegressionHistory(): Promise<RegressionHistory | ApiError> {
  return j<RegressionHistory | ApiError>('/api/regression-history');
}

export function getLrFinderResults(): Promise<LrFinderResults | ApiError> {
  return j<LrFinderResults | ApiError>('/api/lr-finder-results');
}

export function getRepoScores(): Promise<RepoScoresResult | ApiError> {
  return j<RepoScoresResult | ApiError>('/api/repo-scores');
}

export function getTrainingManifests(): Promise<TrainingManifests | ApiError> {
  return j<TrainingManifests | ApiError>('/api/training-manifests');
}

export function getBacklog(): Promise<BacklogView | ApiError> {
  return j<BacklogView | ApiError>('/api/backlog');
}

// File a new backlog item (append-only); returns the refreshed backlog. 400 on bad input.
export function appendBacklog(req: BacklogAppendRequest): Promise<BacklogView | ApiError> {
  return j<BacklogView | ApiError>('/api/backlog/append', postJson(req));
}

export function getResearchLog(): Promise<ResearchLog | ApiError> {
  return j<ResearchLog | ApiError>('/api/research-log');
}

// Append a dated research-log entry (append-only); returns the refreshed log. 400 on bad input.
export function appendResearchLog(req: ResearchLogAppendRequest): Promise<ResearchLog | ApiError> {
  return j<ResearchLog | ApiError>('/api/research-log/append', postJson(req));
}

export function getDocs(): Promise<DocsList | ApiError> {
  return j<DocsList | ApiError>('/api/docs');
}

export function getDocContent(path: string): Promise<DocContent | ApiError> {
  return j<DocContent | ApiError>(`/api/doc?path=${encodeURIComponent(path)}`);
}

export function getDataStats(
  dataPath?: string,
  estimateUnique = true,
): Promise<DataStats | ApiError> {
  const q = new URLSearchParams();
  if (dataPath !== undefined) q.set('data_path', dataPath);
  if (!estimateUnique) q.set('estimate_unique', 'false');
  const qs = q.toString();
  return j<DataStats | ApiError>(`/api/data-stats${qs ? `?${qs}` : ''}`);
}
