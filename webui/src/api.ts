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
type JsonRequestBody = { [key: string]: JsonValue | undefined };

function postJson(body: JsonRequestBody): RequestInit {
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

export function getActions(): Promise<ActionDef[]> {
  return j<ActionDef[]>('/api/actions');
}

export function runAction(action: string, args?: string[]): Promise<Job> {
  return j<Job>('/api/run', postJson({ action, args }));
}

export function startTraining(
  config: string,
  resume?: string
): Promise<Job | ApiError> {
  return j<Job | ApiError>(
    '/api/train/start',
    postJson({ config, resume })
  );
}

export function getConfigs(): Promise<ConfigFile[]> {
  return j<ConfigFile[]>('/api/configs');
}

export function getConfig(path: string): Promise<ConfigContent> {
  return j<ConfigContent>(`/api/config?path=${encodeURIComponent(path)}`);
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
