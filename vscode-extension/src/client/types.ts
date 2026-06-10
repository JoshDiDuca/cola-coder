/**
 * Shared types for the Cola-Coder API client.
 * These mirror the Pydantic models defined in src/cola_coder/inference/server.py.
 */

// ── Chat messages ─────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// ── Usage stats ───────────────────────────────────────────────────────────────

export interface UsageStats {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// ── Chat completions ──────────────────────────────────────────────────────────

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  stop?: string[] | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
  }[];
  usage: UsageStats;
}

/** The incremental content delta inside a streaming chat chunk. */
export interface ChatDelta {
  content?: string;
}

export interface ChatStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: ChatDelta;
    finish_reason: string | null;
  }[];
}

// ── Text completions ──────────────────────────────────────────────────────────

export interface CompletionRequest {
  model?: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  stop?: string[] | null;
  stream?: boolean;
  /** Best-of-N with sandboxed verification (server picks the best candidate).
   *  Non-streaming only — the server rejects best_of > 1 with stream=true. */
  best_of?: number;
  /** Verifier language for best_of: 'auto' | 'python' | 'typescript'. */
  verify_language?: string;
}

export interface CompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    text: string;
    finish_reason: string | null;
  }[];
  usage: UsageStats;
}

// ── Fill-in-the-middle ────────────────────────────────────────────────────────

export interface FimRequest {
  prefix: string;
  suffix: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  language?: string | null;
  file_path?: string | null;
}

export interface FimResponse {
  id: string;
  infill: string;
  finish_reason: string;
  usage: UsageStats;
}

// ── Repository context ────────────────────────────────────────────────────────

export interface ContextRequest {
  file_path: string;
  max_tokens?: number;
}

export interface ContextResponse {
  context: string;
  files_referenced: string[];
  project_name: string | null;
  frameworks: Record<string, string>;
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string;
  model: string | null;
  params: number | null;
  device: string | null;
  gpu_name: string | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  uptime_seconds: number | null;
}

// ── Models ────────────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  metadata?: {
    params: number;
    vocab_size: number;
    max_seq_len: number;
  };
}

export interface ModelListResponse {
  object: string;
  data: ModelInfo[];
}
