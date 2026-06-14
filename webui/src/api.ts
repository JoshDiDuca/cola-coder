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
} from './types';

async function j<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${url}`);
  }
  return (await res.json()) as T;
}

function postJson(body: unknown): RequestInit {
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

export function getActions(): Promise<ActionDef[]> {
  return j<ActionDef[]>('/api/actions');
}

export function runAction(action: string, args?: string[]): Promise<Job> {
  return j<Job>('/api/run', postJson({ action, args }));
}

export function startTraining(
  config: string,
  resume?: string
): Promise<Job | { error: string }> {
  return j<Job | { error: string }>(
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

export function getPipelineRun(path: string): Promise<unknown> {
  return j<unknown>(`/api/pipeline/run?path=${encodeURIComponent(path)}`);
}
