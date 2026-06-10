/**
 * Typed wrapper around VS Code extension settings.
 *
 * Like a TypeScript interface over a JSON config — reads from
 * VS Code's `workspace.getConfiguration('cola-coder')`.
 */

import * as vscode from 'vscode';

export interface ColaCoderConfig {
  // Server
  mode: 'auto' | 'external';
  serverUrl: string;
  pythonPath: string;
  projectRoot: string;
  checkpoint: string;
  configFile: string;

  // Inline completions
  inlineEnabled: boolean;
  inlineDebounceMs: number;
  inlineMaxTokens: number;
  inlineTemperature: number;
  inlineLanguages: string[];

  // Chat
  chatTemperature: number;
  chatMaxTokens: number;
  showThinking: boolean;
  baseModelMode: boolean;

  // Context
  contextEnabled: boolean;
  contextMaxTokens: number;

  // Verified generation (best-of-N)
  verifiedCandidates: number;
  verifiedMaxTokens: number;
}

export function getConfig(): ColaCoderConfig {
  const cfg = vscode.workspace.getConfiguration('cola-coder');

  return {
    mode: cfg.get<'auto' | 'external'>('mode', 'external'),
    serverUrl: cfg.get<string>('serverUrl', 'http://localhost:8000'),
    pythonPath: cfg.get<string>('pythonPath', ''),
    projectRoot: cfg.get<string>('projectRoot', ''),
    checkpoint: cfg.get<string>('checkpoint', ''),
    configFile: cfg.get<string>('configFile', ''),

    inlineEnabled: cfg.get<boolean>('inline.enabled', true),
    inlineDebounceMs: cfg.get<number>('inline.debounceMs', 300),
    inlineMaxTokens: cfg.get<number>('inline.maxTokens', 128),
    inlineTemperature: cfg.get<number>('inline.temperature', 0.2),
    inlineLanguages: cfg.get<string[]>('inline.languages', [
      'typescript', 'javascript', 'typescriptreact', 'javascriptreact', 'python',
    ]),

    chatTemperature: cfg.get<number>('chat.temperature', 0.8),
    chatMaxTokens: cfg.get<number>('chat.maxTokens', 1024),
    showThinking: cfg.get<boolean>('chat.showThinking', true),
    baseModelMode: cfg.get<boolean>('chat.baseModelMode', true),

    contextEnabled: cfg.get<boolean>('context.enabled', true),
    contextMaxTokens: cfg.get<number>('context.maxTokens', 2048),

    verifiedCandidates: cfg.get<number>('verified.candidates', 4),
    verifiedMaxTokens: cfg.get<number>('verified.maxTokens', 256),
  };
}
