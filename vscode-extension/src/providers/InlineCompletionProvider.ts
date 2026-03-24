/**
 * Inline completion (ghost text) provider for Cola-Coder.
 *
 * Implements VS Code's InlineCompletionItemProvider interface to surface
 * FIM (Fill-in-the-Middle) suggestions as grey ghost text while the user
 * types. VS Code calls `provideInlineCompletionItems` automatically and
 * passes a CancellationToken that fires when the request is superseded by
 * a newer keystroke — we wire that straight to an AbortController so the
 * in-flight HTTP request is cancelled immediately.
 *
 * Telemetry is intentionally minimal: shown/partially-accepted events are
 * logged to the OutputChannel rather than sent anywhere.
 */

import * as vscode from 'vscode';
import { ColaCoderClient } from '../client/ColaCoderClient';
import { extractFimContext } from '../context/FimFormatter';
import { getConfig } from '../utils/config';
import { logger } from '../utils/logger';

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
  private enabled = true;
  private registration: vscode.Disposable | undefined;
  private lastDisconnectLog = 0;

  constructor(private readonly client: ColaCoderClient) {}

  // ── Registration ──────────────────────────────────────────────────────────

  /**
   * Create a provider instance, register it against the language selector
   * derived from `config.inlineLanguages`, and return the provider so the
   * caller can hold a reference for `setEnabled` / `dispose`.
   */
  static register(client: ColaCoderClient): InlineCompletionProvider {
    const provider = new InlineCompletionProvider(client);
    const config = getConfig();

    const selector: vscode.DocumentSelector = config.inlineLanguages.map((lang) => ({
      language: lang,
    }));

    provider.registration = vscode.languages.registerInlineCompletionItemProvider(
      selector,
      provider,
    );

    logger.info(
      `Inline completions registered for: ${config.inlineLanguages.join(', ')}`,
    );

    return provider;
  }

  // ── Toggle ────────────────────────────────────────────────────────────────

  /** Enable or disable completions without unregistering the provider. */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    logger.info(`Inline completions ${enabled ? 'enabled' : 'disabled'}`);
  }

  // ── Provider implementation ───────────────────────────────────────────────

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList | undefined> {
    if (!this.enabled) {
      return undefined;
    }

    if (!this.client.isConnected()) {
      // Throttle: log once every 30s so the user knows why nothing appears
      const now = Date.now();
      if (now - this.lastDisconnectLog > 30_000) {
        this.lastDisconnectLog = now;
        logger.info('Inline completion skipped: server not connected');
      }
      return undefined;
    }

    const config = getConfig();
    const fimContext = extractFimContext(document, position);

    // Link VS Code's cancellation token to an AbortController so the
    // underlying fetch is cancelled as soon as the token fires (i.e. the
    // user typed another character before we got a response).
    const controller = new AbortController();
    token.onCancellationRequested(() => controller.abort());

    // Bail out early if already cancelled before we even start the request.
    if (token.isCancellationRequested) {
      return undefined;
    }

    try {
      const response = await this.client.fim({
        prefix: fimContext.prefix,
        suffix: fimContext.suffix,
        max_tokens: config.inlineMaxTokens,
        temperature: config.inlineTemperature,
        language: fimContext.language,
        file_path: fimContext.filePath,
      }, controller.signal);

      // Check again after the await — the token may have fired while we
      // were waiting for the network response.
      if (token.isCancellationRequested || !response.infill) {
        return undefined;
      }

      const item = new vscode.InlineCompletionItem(
        response.infill,
        // Insert range: zero-width range at the cursor so the text is
        // inserted rather than replacing existing characters.
        new vscode.Range(position, position),
      );

      return { items: [item] };
    } catch (err) {
      // Swallow abort errors — they are expected when the user types quickly.
      // Only log genuine failures.
      if (!token.isCancellationRequested) {
        logger.error(`Inline completion error: ${String(err)}`);
      }
      return undefined;
    }
  }

  // ── Telemetry callbacks ───────────────────────────────────────────────────

  /**
   * Called by VS Code when a completion item is shown to the user.
   * Logged for basic visibility; no data is sent externally.
   */
  handleDidShowCompletionItem(_item: vscode.InlineCompletionItem): void {
    logger.info('Inline completion shown');
  }

  /**
   * Called by VS Code when the user accepts part of a completion (e.g.
   * accepts word-by-word via Ctrl+Right rather than Tab for the whole item).
   */
  handleDidPartiallyAcceptCompletionItem(
    _item: vscode.InlineCompletionItem,
    acceptedLength: number,
  ): void {
    logger.info(`Inline completion partially accepted: ${acceptedLength} chars`);
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  dispose(): void {
    this.registration?.dispose();
  }
}
