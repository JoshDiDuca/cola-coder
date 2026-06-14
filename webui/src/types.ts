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
