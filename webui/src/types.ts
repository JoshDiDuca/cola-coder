export interface TrainingStatus {
  alive: boolean;
  step: number | null;
  total_steps: number | null;
  progress_pct: number | null;
  loss: number | null;
  ppl: number | null;
  tok_per_s: number | null;
  s_per_it: number | null;
  last_log_line: string | null;
}

export interface SystemStatus {
  gpu_name: string | null;
  gpu_util_pct: number | null;
  gpu_mem_used_mb: number | null;
  gpu_mem_total_mb: number | null;
  gpu_power_w: number | null;
}

export interface Checkpoint {
  model: string;
  name: string;
  step: number;
  loss: number | null;
  path: string;
  mtime: number;
}

export interface StatusResponse {
  training: TrainingStatus;
  system: SystemStatus;
  checkpoints: Checkpoint[];
}

export interface Dataset {
  name: string;
  path: string;
  kind: 'npy' | 'jsonl';
  size_bytes: number;
  mtime: number;
  has_weights: boolean;
  num_samples: number | null;
}

export interface Job {
  id: string;
  name: string;
  pid: number;
  status: 'running' | 'done' | 'failed';
  cmd: string[];
  log: string;
  started: number;
  returncode: number | null;
}

export interface ActionDef {
  key: string;
  script: string;
  label: string;
  args: string[];
}

export interface ScoreSummary {
  n: number;
  mean: number;
  min: number;
  max: number;
  histogram: number[];
  bins: number[];
}

export interface Preview {
  kind: string;
  num_samples?: number;
  shape?: number[];
  dtype?: string;
  preview?: unknown[];
  error?: string;
}

export interface ConfigFile {
  name: string;
  path: string;
  rel: string;
  size_bytes: number;
  mtime: number;
}

export interface ConfigContent {
  path?: string;
  content?: string;
  parsed?: unknown;
  truncated?: boolean;
  error?: string;
}

export interface PipelineRun {
  name: string;
  path: string;
  mtime: number;
  num_stages: number | null;
  status: string | null;
  completed: number | null;
  error?: string;
}

export interface EvalResult {
  name: string;
  path: string;
  kind: string;
  mtime: number;
  summary: string;
}

export interface EvalDetail {
  path: string;
  kind: string;
  parsed: unknown | null;
  content: string | null;
  truncated: boolean;
}

export interface LogFile {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
}

export interface LogTail {
  path: string;
  lines: string[];
  size_bytes: number;
  truncated: boolean;
}

export interface FeatureItem {
  key: string;
  enabled: boolean;
  value: unknown;
}

export interface FeatureGroup {
  category: string;
  features: FeatureItem[];
}

export interface FeaturesView {
  path: string;
  total: number;
  enabled: number;
  groups: FeatureGroup[];
}

export interface ReasoningSummary {
  advantage_norm: unknown;
  clip_epsilon: unknown;
  clip_epsilon_high: unknown;
  group_size: unknown;
  sft_warmup_enabled: boolean | null;
  problem_source: unknown;
}

export interface ReasoningView {
  path: string;
  parsed: Record<string, unknown>;
  reasoning: Record<string, unknown>;
  problem_set: Record<string, unknown>;
  sft_warmup: Record<string, unknown>;
  summary: ReasoningSummary;
}

export interface TokenizerInfo {
  path: string;
  vocab_size: number;
  n_merges: number;
  special_tokens: string[];
  has_fim_tokens: boolean;
  digit_splitting: boolean;
  model_type: string;
}

export interface CheckpointDetail {
  path: string;
  metadata: Record<string, unknown> | null;
  is_moe: boolean;
  moe_config: Record<string, unknown> | null;
  has_training_state: boolean;
  num_params: number;
  tensor_count: number;
  dtypes: string[];
  files: string[];
}

export interface RouterCheckpoint {
  path: string;
  name: string;
  step: number | null;
}

export interface RouterOverview {
  has_router: boolean;
  checkpoints: RouterCheckpoint[];
  domains: string[];
}

export interface ExportFormat {
  key: string;
  label: string;
  desc: string;
}

export interface ExportItem {
  path: string;
  format: string;
  size_bytes: number;
  mtime: number;
}

export interface ExportCheckpoint {
  model: string;
  name: string;
  step: number | null;
  path: string;
}

export interface ExportOverview {
  checkpoints: ExportCheckpoint[];
  formats: ExportFormat[];
  existing: ExportItem[];
}

export interface MetricPoint {
  step: number;
  loss: number | null;
  ppl: number | null;
  lr: number | null;
  tok_s: number | null;
}

export interface MetricsHistory {
  points: MetricPoint[];
  count: number;
}

export interface ApiError {
  error: string;
}

export function isApiError(v: unknown): v is ApiError {
  return typeof v === 'object' && v !== null && typeof (v as { error?: unknown }).error === 'string';
}
