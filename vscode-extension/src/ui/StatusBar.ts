/**
 * Rich status bar item with live metrics and quick pick menu.
 *
 * Shows server state, model name, and tokens/sec during generation.
 * Clicking opens a quick pick menu for common actions.
 */

import * as vscode from 'vscode';
import type { ServerState } from '../server/HealthMonitor';
import type { HealthStatus } from '../client/types';

const COMMAND_ID = 'cola-coder.statusBarClick';

export class StatusBar implements vscode.Disposable {
  private item: vscode.StatusBarItem;
  private disposables: vscode.Disposable[] = [];
  private state: ServerState = 'disconnected';
  private modelName: string | null = null;
  private tokensPerSec: number | null = null;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100,
    );
    this.item.command = COMMAND_ID;
    this.item.show();

    // Register the click command
    this.disposables.push(
      vscode.commands.registerCommand(COMMAND_ID, () => this.showMenu()),
    );

    this.render();
  }

  setState(state: ServerState): void {
    this.state = state;
    this.render();
  }

  setModelName(name: string): void {
    this.modelName = name;
    this.render();
  }

  setTokensPerSec(tps: number | null): void {
    this.tokensPerSec = tps;
    this.render();
  }

  updateFromHealth(health: HealthStatus): void {
    if (health.model) {
      this.modelName = health.model;
    }
    this.render();
  }

  private render(): void {
    const modelSuffix = this.modelName ? ` (${this.modelName})` : '';

    switch (this.state) {
      case 'disconnected':
        this.item.text = '$(debug-disconnect) Cola-Coder: Disconnected';
        this.item.backgroundColor = new vscode.ThemeColor(
          'statusBarItem.errorBackground',
        );
        this.item.tooltip = 'Cola-Coder server is not connected';
        break;

      case 'starting':
        this.item.text = '$(loading~spin) Cola-Coder: Starting...';
        this.item.backgroundColor = new vscode.ThemeColor(
          'statusBarItem.warningBackground',
        );
        this.item.tooltip = 'Cola-Coder server is starting up';
        break;

      case 'ready': {
        const tps = this.tokensPerSec
          ? ` ${this.tokensPerSec} tok/s`
          : '';
        this.item.text = `$(sparkle) Cola-Coder: Ready${modelSuffix}${tps}`;
        this.item.backgroundColor = undefined;
        this.item.tooltip = 'Cola-Coder is ready. Click for options.';
        break;
      }

      case 'generating': {
        const tps = this.tokensPerSec
          ? ` (${this.tokensPerSec} tok/s)`
          : '';
        this.item.text = `$(loading~spin) Cola-Coder: Generating...${tps}`;
        this.item.backgroundColor = undefined;
        this.item.tooltip = 'Cola-Coder is generating code';
        break;
      }

      case 'error':
        this.item.text = '$(error) Cola-Coder: Error';
        this.item.backgroundColor = new vscode.ThemeColor(
          'statusBarItem.errorBackground',
        );
        this.item.tooltip = 'Cola-Coder encountered an error. Click for options.';
        break;
    }
  }

  private async showMenu(): Promise<void> {
    const items: vscode.QuickPickItem[] = [
      {
        label: '$(symbol-boolean) Toggle Inline Completions',
        description: vscode.workspace
          .getConfiguration('cola-coder')
          .get<boolean>('inline.enabled', true)
          ? 'Currently: ON'
          : 'Currently: OFF',
      },
      {
        label: '$(debug-restart) Restart Server',
        description: 'Stop and restart the server',
      },
      {
        label: '$(output) Show Server Logs',
        description: 'Open the output channel',
      },
      {
        label: '$(gear) Open Settings',
        description: 'Configure Cola-Coder',
      },
    ];

    const choice = await vscode.window.showQuickPick(items, {
      placeHolder: 'Cola-Coder Actions',
    });

    if (!choice) return;

    if (choice.label.includes('Toggle Inline')) {
      vscode.commands.executeCommand('cola-coder.toggleInline');
    } else if (choice.label.includes('Restart')) {
      vscode.commands.executeCommand('cola-coder.restartServer');
    } else if (choice.label.includes('Logs')) {
      vscode.commands.executeCommand('cola-coder.showLogs');
    } else if (choice.label.includes('Settings')) {
      vscode.commands.executeCommand(
        'workbench.action.openSettings',
        'cola-coder',
      );
    }
  }

  dispose(): void {
    this.item.dispose();
    this.disposables.forEach(d => d.dispose());
  }
}
