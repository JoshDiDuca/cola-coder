/**
 * OutputChannel-based logger for the extension.
 *
 * All server output and extension diagnostics go here.
 * Users can view it via "Cola-Coder: Show Server Logs" command.
 */

import * as vscode from 'vscode';

class Logger {
  private channel: vscode.OutputChannel | undefined;

  private getChannel(): vscode.OutputChannel {
    if (!this.channel) {
      this.channel = vscode.window.createOutputChannel('Cola-Coder');
    }
    return this.channel;
  }

  info(message: string): void {
    this.getChannel().appendLine(`[INFO] ${message}`);
  }

  warn(message: string): void {
    this.getChannel().appendLine(`[WARN] ${message}`);
  }

  error(message: string): void {
    this.getChannel().appendLine(`[ERROR] ${message}`);
  }

  show(): void {
    this.getChannel().show();
  }

  dispose(): void {
    this.channel?.dispose();
  }
}

export const logger = new Logger();
