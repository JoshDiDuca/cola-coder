/**
 * Cola-Coder VS Code Extension — entry point.
 *
 * Activates on startup, connects to (or starts) the FastAPI server,
 * and registers all AI features: inline completions, chat participant,
 * code actions, and language model provider.
 */

import * as vscode from 'vscode';
import { ColaCoderClient } from './client/ColaCoderClient';
import { ServerManager } from './server/ServerManager';
import { HealthMonitor } from './server/HealthMonitor';
import { StatusBar } from './ui/StatusBar';
import { InlineCompletionProvider } from './providers/InlineCompletionProvider';
import { ChatParticipant } from './providers/ChatParticipant';
import { CodeActionProvider } from './providers/CodeActionProvider';
import { LanguageModelProvider } from './providers/LanguageModelProvider';
import { getConfig } from './utils/config';
import { logger } from './utils/logger';

let client: ColaCoderClient;
let serverManager: ServerManager;
let healthMonitor: HealthMonitor;
let statusBar: StatusBar;
let inlineProvider: InlineCompletionProvider;
let chatParticipant: ChatParticipant;
let codeActionProvider: CodeActionProvider;
let languageModelProvider: LanguageModelProvider;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  logger.info('Cola-Coder extension activating...');

  const config = getConfig();

  // ── Core infrastructure ─────────────────────────────────────────────
  client = new ColaCoderClient(config.serverUrl);
  healthMonitor = new HealthMonitor(client);
  statusBar = new StatusBar();
  serverManager = new ServerManager(client, healthMonitor);

  // Wire health monitor state changes to status bar
  healthMonitor.onStateChange(state => {
    statusBar.setState(state);
  });
  healthMonitor.onHealthUpdate(health => {
    statusBar.updateFromHealth(health);
  });

  // ── Start or connect to server ──────────────────────────────────────
  if (config.mode === 'auto') {
    try {
      statusBar.setState('starting');
      await serverManager.start();
    } catch (err) {
      logger.error(`Failed to start server: ${err}`);
      statusBar.setState('error');
      vscode.window.showErrorMessage(
        `Cola-Coder: Failed to start server. ${err}`,
        'Open Settings',
      ).then(choice => {
        if (choice === 'Open Settings') {
          vscode.commands.executeCommand(
            'workbench.action.openSettings',
            'cola-coder',
          );
        }
      });
    }
  } else {
    // External mode — just try to connect
    try {
      await client.healthCheck();
      statusBar.setState('ready');
    } catch {
      statusBar.setState('disconnected');
      logger.warn(
        `Cannot reach server at ${config.serverUrl}. `
        + 'Start the server or check cola-coder.serverUrl setting.',
      );
    }
  }

  // Start health polling
  healthMonitor.start();

  // ── Register providers ──────────────────────────────────────────────
  if (config.inlineEnabled) {
    inlineProvider = InlineCompletionProvider.register(client);
    context.subscriptions.push(inlineProvider);
  }

  chatParticipant = ChatParticipant.register(client);
  context.subscriptions.push(chatParticipant);

  codeActionProvider = CodeActionProvider.register();
  context.subscriptions.push(codeActionProvider);

  languageModelProvider = LanguageModelProvider.register(client);
  context.subscriptions.push(languageModelProvider);

  // ── Register commands ───────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.toggleInline', () => {
      const cfg = vscode.workspace.getConfiguration('cola-coder');
      const current = cfg.get<boolean>('inline.enabled', true);
      cfg.update('inline.enabled', !current, vscode.ConfigurationTarget.Global);

      if (inlineProvider) {
        inlineProvider.setEnabled(!current);
      }

      vscode.window.showInformationMessage(
        `Cola-Coder inline completions: ${!current ? 'ON' : 'OFF'}`,
      );
    }),

    vscode.commands.registerCommand('cola-coder.restartServer', async () => {
      try {
        await serverManager.restart();
        vscode.window.showInformationMessage('Cola-Coder server restarted.');
      } catch (err) {
        vscode.window.showErrorMessage(`Failed to restart: ${err}`);
      }
    }),

    vscode.commands.registerCommand('cola-coder.showLogs', () => {
      logger.show();
    }),

    vscode.commands.registerCommand('cola-coder.configureModel', () => {
      vscode.commands.executeCommand(
        'workbench.action.openSettings',
        'cola-coder',
      );
    }),

    vscode.commands.registerCommand('cola-coder.explainSelection', () => {
      vscode.commands.executeCommand(
        'workbench.action.chat.open',
        { query: '@cola-coder /explain' },
      );
    }),

    vscode.commands.registerCommand('cola-coder.fixSelection', () => {
      vscode.commands.executeCommand(
        'workbench.action.chat.open',
        { query: '@cola-coder /fix' },
      );
    }),

    vscode.commands.registerCommand('cola-coder.refactorSelection', () => {
      vscode.commands.executeCommand(
        'workbench.action.chat.open',
        { query: '@cola-coder /refactor' },
      );
    }),
  );

  // ── Push disposables ────────────────────────────────────────────────
  context.subscriptions.push(
    healthMonitor,
    statusBar,
    serverManager,
    { dispose: () => logger.dispose() },
  );

  logger.info('Cola-Coder extension activated.');
}

export function deactivate(): void {
  logger.info('Cola-Coder extension deactivating...');
}
