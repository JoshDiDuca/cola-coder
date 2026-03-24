/**
 * Language Model Chat Provider — registers cola-coder in VS Code's model picker.
 *
 * This makes cola-coder available as a model in VS Code's native AI features
 * and to other extensions via the `vscode.lm` API. Stable since VS Code 1.104.
 *
 * Note: Models contributed via this API currently only work for users on
 * individual GitHub Copilot plans (not enterprise/team).
 */

import * as vscode from 'vscode';
import { ColaCoderClient } from '../client/ColaCoderClient';
import { logger } from '../utils/logger';
import type { ChatMessage } from '../client/types';

export class LanguageModelProvider implements vscode.Disposable {
  private registration: vscode.Disposable | undefined;
  private modelId = 'cola-coder';
  private modelName = 'Cola-Coder';

  constructor(private client: ColaCoderClient) {}

  static register(client: ColaCoderClient): LanguageModelProvider {
    const provider = new LanguageModelProvider(client);
    provider.registration = vscode.lm.registerLanguageModelChatProvider(
      'cola-coder',
      provider,
    );

    // Fetch model info to update name
    client.getModels().then(models => {
      if (models.data.length > 0) {
        provider.modelId = models.data[0].id;
        const params = models.data[0].metadata?.params;
        if (params) {
          provider.modelName = `Cola-Coder ${params}`;
        }
      }
    }).catch(() => {
      // Server might not be ready yet — will update on next health check
    });

    return provider;
  }

  provideLanguageModelChatInformation(
    _options: { silent: boolean },
  ): vscode.LanguageModelChatInformation[] {
    return [{
      id: this.modelId,
      name: this.modelName,
      family: 'cola-coder',
      version: '1.0',
      maxInputTokens: 4096,
      maxOutputTokens: 2048,
      capabilities: {
        imageInput: false,
        toolCalling: false,
      },
    }];
  }

  async provideLanguageModelChatResponse(
    _modelInfo: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    _options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    // Convert VS Code messages to our format
    const chatMessages: ChatMessage[] = messages.map(msg => ({
      role: msg.role === vscode.LanguageModelChatMessageRole.User
        ? 'user' as const
        : 'assistant' as const,
      content: typeof msg.content === 'string'
        ? msg.content
        : msg.content
            .filter((part): part is vscode.LanguageModelTextPart =>
              part instanceof vscode.LanguageModelTextPart)
            .map(part => part.value)
            .join(''),
    }));

    // Wire up cancellation
    const controller = new AbortController();
    token.onCancellationRequested(() => controller.abort());

    try {
      await this.client.chatStream(
        {
          messages: chatMessages,
          temperature: 0.8,
          max_tokens: 2048,
        },
        (delta) => {
          if (delta.content) {
            progress.report(
              new vscode.LanguageModelTextPart(delta.content),
            );
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (!token.isCancellationRequested) {
        logger.error(`LM Provider error: ${err}`);
        throw err;
      }
    }
  }

  async provideTokenCount(
    _model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatRequestMessage,
    _token: vscode.CancellationToken,
  ): Promise<number> {
    // Rough BPE estimate: ~4 chars per token for code
    const str = typeof text === 'string' ? text : JSON.stringify(text);
    return Math.ceil(str.length / 4);
  }

  dispose(): void {
    this.registration?.dispose();
  }
}
