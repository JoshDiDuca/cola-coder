/**
 * Cola-Coder VS Code Extension — entry point.
 *
 * Key principle: register ALL providers immediately so the extension
 * is always visible in VS Code (chat, code actions, status bar).
 * Server connection happens in the background — features gracefully
 * show "server not connected" messages when it's unavailable.
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
let inlineProvider: InlineCompletionProvider | undefined;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  logger.info('Cola-Coder extension activating...');

  const config = getConfig();

  // ── Core infrastructure (these never throw) ─────────────────────────
  client = new ColaCoderClient(config.serverUrl);
  healthMonitor = new HealthMonitor(client);
  statusBar = new StatusBar();
  serverManager = new ServerManager(client, healthMonitor);

  // Wire health monitor → status bar
  healthMonitor.onStateChange(state => statusBar.setState(state));
  healthMonitor.onHealthUpdate(health => statusBar.updateFromHealth(health));

  // ── Register ALL providers FIRST — before any server work ───────────
  // This ensures @cola-coder, code actions, etc. are always visible
  // even if the server isn't running yet.

  // Inline completions
  if (config.inlineEnabled) {
    try {
      inlineProvider = InlineCompletionProvider.register(client);
      context.subscriptions.push(inlineProvider);
    } catch (err) {
      logger.error(`Failed to register inline completions: ${err}`);
    }
  }

  // Chat participant (@cola-coder)
  try {
    // Generation activity → status bar: HealthMonitor owns the state machine
    // (its poll loop already refuses to overwrite 'generating'), and the
    // tok/s figure is rendered by the status bar on completion.
    const chatParticipant = ChatParticipant.register(client, {
      begin: () => {
        statusBar.setTokensPerSec(null);
        healthMonitor.setState('generating');
      },
      end: (tokensPerSec) => {
        statusBar.setTokensPerSec(tokensPerSec);
        if (healthMonitor.state === 'generating') {
          healthMonitor.setState('ready');
        }
      },
    });
    context.subscriptions.push(chatParticipant);
    logger.info('Chat participant @cola-coder registered');
  } catch (err) {
    logger.error(`Failed to register chat participant: ${err}`);
  }

  // Code actions (lightbulb menu)
  try {
    const codeActionProvider = CodeActionProvider.register();
    context.subscriptions.push(codeActionProvider);
  } catch (err) {
    logger.error(`Failed to register code actions: ${err}`);
  }

  // Language model provider (VS Code model picker)
  try {
    const languageModelProvider = LanguageModelProvider.register(client);
    context.subscriptions.push(languageModelProvider);
  } catch (err) {
    logger.error(`Failed to register language model provider: ${err}`);
  }

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

  logger.info('Cola-Coder providers registered. Connecting to server...');

  // ── NOW try to connect to server (in background, never crashes) ─────
  connectToServer(config).catch(err => {
    logger.error(`Server connection failed: ${err}`);
  });

  logger.info('Cola-Coder extension activated.');
}

/**
 * Connect to (or start) the server. Runs in the background after
 * all providers are registered, so failures here never prevent
 * the extension from being visible in VS Code.
 */
async function connectToServer(
  config: ReturnType<typeof getConfig>,
): Promise<void> {
  if (config.mode === 'auto') {
    try {
      statusBar.setState('starting');
      await serverManager.start();
    } catch (err) {
      logger.error(`Failed to start server: ${err}`);
      statusBar.setState('disconnected');
      vscode.window.showWarningMessage(
        `Cola-Coder: Server not started. ${err}`,
        'Open Settings',
        'Show Logs',
      ).then(choice => {
        if (choice === 'Open Settings') {
          vscode.commands.executeCommand(
            'workbench.action.openSettings',
            'cola-coder',
          );
        } else if (choice === 'Show Logs') {
          logger.show();
        }
      });
    }
  } else {
    // External mode — try to connect
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

  // Start health polling regardless — will auto-detect when server comes up
  healthMonitor.start();
}

export function deactivate(): void {
  logger.info('Cola-Coder extension deactivating...');
}
