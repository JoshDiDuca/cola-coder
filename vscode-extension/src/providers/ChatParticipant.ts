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

/**
 * Optional sink for generation activity, used to drive the status bar
 * ('generating' spinner + tokens/sec) without coupling this provider to UI.
 */
export interface GenerationActivity {
  begin(): void;
  end(tokensPerSec: number | null): void;
}

// ── System prompts by command ──────────────────────────────────────────────────

// System prompts — only used if the model has been instruction-tuned.
// For base code models these are skipped (see isBaseModel flag below).
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

// Base code models (no instruction tuning) work best when given raw code
// to continue. We skip system prompts and send the prompt directly.
// Controlled by config.baseModelMode (cola-coder.chat.baseModelMode setting).

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

  private constructor(
    private readonly client: ColaCoderClient,
    private readonly activity?: GenerationActivity,
  ) {
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

  static register(
    client: ColaCoderClient,
    activity?: GenerationActivity,
  ): ChatParticipant {
    return new ChatParticipant(client, activity);
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
    const editorCtx = getEditorContext();

    // For /generate with a base model: treat the prompt as raw code to
    // complete. We echo the prefix back so the user sees the full output.
    const isRawCompletion = config.baseModelMode && command === 'generate';

    if (config.baseModelMode) {
      // Base model: send raw code, no system prompt (it's just noise)
      if (isRawCompletion) {
        // /generate: the user's prompt IS the code prefix to complete
        messages.push({ role: 'user', content: request.prompt });
      } else if (command === 'explain' || command === 'fix' || command === 'refactor') {
        // For explain/fix/refactor, send the selected code as a completion prompt
        const code = editorCtx.selectedCode || request.prompt;
        messages.push({ role: 'user', content: code });
      } else {
        // Default: send whatever the user typed + selected code
        const userContent = buildUserMessage(request.prompt, editorCtx);
        messages.push({ role: 'user', content: userContent });
      }
    } else {
      // Instruction-tuned model: use system prompts and structured messages
      const systemText = SYSTEM_PROMPTS[command] ?? SYSTEM_PROMPTS[''];
      messages.push({ role: 'system', content: systemText });
      const historyMessages = buildHistoryMessages(context.history);
      messages.push(...historyMessages);
      let userContent = buildUserMessage(request.prompt, editorCtx);
      // Repository context (context.enabled): ask the server's /v1/context
      // endpoint for related files/frameworks around the active file.
      const repoContext = await this.fetchRepoContext(config, editorCtx.filePath);
      if (repoContext) {
        userContent = `Repository context:\n${repoContext}\n\n${userContent}`;
      }
      messages.push({ role: 'user', content: userContent });
    }

    // ── Guard: server must be reachable ────────────────────────────────────

    if (!this.client.isConnected()) {
      stream.markdown(
        '**Cola-Coder:** The inference server is not reachable. '
        + 'Check the status bar indicator or run `Cola-Coder: Restart Server`.',
      );
      return { metadata: { command } };
    }

    // ── Stream the response ────────────────────────────────────────────────

    const controller = new AbortController();
    // Dispose the listener when the request settles (in finally) so it isn't
    // retained for the token's lifetime — EXT-003.
    const cancelSub = token.onCancellationRequested(() => controller.abort());

    // For /generate: echo the user's code prefix in a code block first
    if (isRawCompletion) {
      const lang = editorCtx.language || 'typescript';
      stream.markdown('```' + lang + '\n' + request.prompt);
    }

    let thinkingEmitted = false;
    let thinkingState = createInitialState();

    // Drive the status bar: spinner while generating, tok/s on completion.
    // Each SSE delta is approximately one token, so deltas/second is an
    // honest tokens-per-second estimate.
    this.activity?.begin();
    const startedAt = Date.now();
    let deltaCount = 0;

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
          deltaCount++;

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
      // Close the code block for /generate
      if (isRawCompletion) {
        stream.markdown('\n```');
      }
    } catch (err) {
      if (!token.isCancellationRequested) {
        const message = err instanceof Error ? err.message : String(err);
        logger.error(`Chat stream error: ${message}`);
        stream.markdown(`\n\n**Error:** ${message}`);
      }
    } finally {
      cancelSub.dispose();
      const elapsedSecs = (Date.now() - startedAt) / 1000;
      const tokensPerSec = deltaCount > 0 && elapsedSecs > 0.5
        ? Math.round(deltaCount / elapsedSecs)
        : null;
      this.activity?.end(tokensPerSec);
    }

    return { metadata: { command } };
  }

  /**
   * Fetch repository context from /v1/context when context.enabled is on.
   * Returns an empty string when disabled, disconnected, no active file,
   * or on any server error (context is an enhancement, never a blocker).
   */
  private async fetchRepoContext(
    config: ReturnType<typeof getConfig>,
    filePath: string,
  ): Promise<string> {
    if (!config.contextEnabled || !filePath || !this.client.isConnected()) {
      return '';
    }
    try {
      const response = await this.client.getContext(filePath, config.contextMaxTokens);
      return response.context ?? '';
    } catch (err) {
      logger.warn(`Repo context unavailable: ${String(err)}`);
      return '';
    }
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
