/**
 * Chat participant for Cola-Coder, registered as `@cola-coder` in the VS Code
 * chat panel.
 *
 * Slash commands:
 *   /explain   — walk through the selected code step by step
 *   /fix       — diagnose and correct a bug
 *   /generate  — generate clean, well-typed code from a description
 *   /refactor  — suggest improvements with before/after code blocks
 *   (none)     — general-purpose assistant
 *
 * Thinking tokens: if `config.showThinking` is true and the model returns
 * <think>…</think> blocks, they are rendered as collapsible <details> sections
 * before the main answer.
 */

import * as vscode from 'vscode';
import { ColaCoderClient } from '../client/ColaCoderClient';
import { getConfig } from '../utils/config';
import { logger } from '../utils/logger';
import { getEditorContext, buildUserMessage } from '../context/ContextAssembler';
import {
  createInitialState,
  parseStreamChunk,
  formatThinkingBlock,
} from '../ui/ThinkingRenderer';
import type { ChatMessage } from '../client/types';

// ── Types ──────────────────────────────────────────────────────────────────────

interface ChatParticipantResult extends vscode.ChatResult {
  metadata?: { command?: string };
}

// ── System prompts by command ──────────────────────────────────────────────────

const SYSTEM_PROMPTS: Record<string, string> = {
  explain:
    'You are a code assistant. Explain the given code clearly and concisely. Show your reasoning in <think> tags.',
  fix:
    'You are a code assistant. Diagnose the bug and provide a corrected version as a code block.',
  generate:
    'You are a code assistant. Generate clean, well-typed code for the given description.',
  refactor:
    'You are a code assistant. Suggest refactoring improvements. Show before and after as code blocks.',
  '':
    'You are Cola-Coder, a code generation AI trained from scratch. Help the user with their code.',
};

// ── Follow-up suggestions by command ──────────────────────────────────────────

const FOLLOWUPS: Record<string, vscode.ChatFollowup[]> = {
  explain: [
    { prompt: 'Tell me more' },
    { prompt: 'Show an example' },
  ],
  fix: [
    { prompt: 'Explain the bug' },
    { prompt: 'Add tests for this fix' },
  ],
  generate: [
    { prompt: 'Add error handling' },
    { prompt: 'Add tests' },
  ],
  refactor: [
    { prompt: 'Explain the changes' },
    { prompt: 'Apply the refactoring' },
  ],
};

// ── ChatParticipant ────────────────────────────────────────────────────────────

export class ChatParticipant implements vscode.Disposable {
  private participant: vscode.ChatParticipant;

  private constructor(private readonly client: ColaCoderClient) {
    this.participant = vscode.chat.createChatParticipant(
      'cola-coder.chat',
      (request, context, stream, token) =>
        this.handleRequest(request, context, stream, token),
    );

    this.participant.iconPath = new vscode.ThemeIcon('sparkle');

    this.participant.followupProvider = {
      provideFollowups: (result: ChatParticipantResult, _context, _token) =>
        this.provideFollowups(result),
    };

    logger.info('Chat participant registered as @cola-coder');
  }

  // ── Static factory ─────────────────────────────────────────────────────────

  static register(client: ColaCoderClient): ChatParticipant {
    return new ChatParticipant(client);
  }

  // ── Request handler ────────────────────────────────────────────────────────

  private async handleRequest(
    request: vscode.ChatRequest,
    context: vscode.ChatContext,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
  ): Promise<ChatParticipantResult> {
    const command = request.command ?? '';
    const config = getConfig();

    logger.info(`Chat request — command: "${command}", prompt: "${request.prompt}"`);

    // ── Assemble messages ──────────────────────────────────────────────────

    const messages: ChatMessage[] = [];

    // System prompt (our server supports role: "system"; include it first)
    const systemText = SYSTEM_PROMPTS[command] ?? SYSTEM_PROMPTS[''];
    messages.push({ role: 'system', content: systemText });

    // History from previous turns in this chat session
    const historyMessages = buildHistoryMessages(context.history);
    messages.push(...historyMessages);

    // Current user turn: editor context + the user's prompt
    const editorCtx = getEditorContext();
    const userContent = buildUserMessage(request.prompt, editorCtx);
    messages.push({ role: 'user', content: userContent });

    // ── Guard: server must be reachable ────────────────────────────────────

    if (!this.client.isConnected()) {
      stream.markdown(
        '**Cola-Coder:** The inference server is not reachable. '
        + 'Check the status bar indicator or run `Cola-Coder: Start Server`.',
      );
      return { metadata: { command } };
    }

    // ── Stream the response ────────────────────────────────────────────────

    const controller = new AbortController();
    token.onCancellationRequested(() => controller.abort());

    let thinkingEmitted = false;
    let thinkingState = createInitialState();

    try {
      await this.client.chatStream(
        {
          messages,
          temperature: config.chatTemperature,
          max_tokens: config.chatMaxTokens,
        },
        (delta) => {
          if (token.isCancellationRequested) {
            return;
          }

          const text = delta.content;
          if (!text) {
            return;
          }

          const result = parseStreamChunk(text, thinkingState);
          thinkingState = result.state;

          // Emit a completed thinking block once (before the first content token)
          if (result.thinkingText !== null && config.showThinking && !thinkingEmitted) {
            thinkingEmitted = true;
            stream.markdown(formatThinkingBlock(result.thinkingText));
          }

          // Emit regular content
          if (result.contentText) {
            stream.markdown(result.contentText);
          }
        },
        controller.signal,
      );

      // Flush any thinking that never received a </think> close tag
      if (
        thinkingState.inThinking
        && thinkingState.thinkingBuffer
        && config.showThinking
        && !thinkingEmitted
      ) {
        stream.markdown(formatThinkingBlock(thinkingState.thinkingBuffer));
      }
    } catch (err) {
      if (!token.isCancellationRequested) {
        const message = err instanceof Error ? err.message : String(err);
        logger.error(`Chat stream error: ${message}`);
        stream.markdown(`\n\n**Error:** ${message}`);
      }
    }

    return { metadata: { command } };
  }

  // ── Follow-up provider ─────────────────────────────────────────────────────

  private provideFollowups(
    result: ChatParticipantResult,
  ): vscode.ChatFollowup[] {
    const command = result.metadata?.command ?? '';
    return FOLLOWUPS[command] ?? [];
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  dispose(): void {
    this.participant.dispose();
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Convert VS Code chat history turns into the flat ChatMessage[] format
 * expected by the Cola-Coder server.
 *
 * - ChatRequestTurn  → role: "user"
 * - ChatResponseTurn → role: "assistant" (text parts concatenated)
 */
function buildHistoryMessages(
  history: ReadonlyArray<vscode.ChatRequestTurn | vscode.ChatResponseTurn>,
): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const turn of history) {
    if (turn instanceof vscode.ChatRequestTurn) {
      messages.push({ role: 'user', content: turn.prompt });
    } else if (turn instanceof vscode.ChatResponseTurn) {
      // A response may contain markdown, anchors, buttons, etc.
      // Extract only the plain text / markdown parts.
      const textParts: string[] = [];
      for (const part of turn.response) {
        if (part instanceof vscode.ChatResponseMarkdownPart) {
          textParts.push(part.value.value);
        }
      }
      const content = textParts.join('');
      if (content) {
        messages.push({ role: 'assistant', content });
      }
    }
  }

  return messages;
}
