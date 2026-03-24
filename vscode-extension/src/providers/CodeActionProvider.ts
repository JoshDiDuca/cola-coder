/**
 * AI-powered code actions (lightbulb menu).
 *
 * Registers actions that appear in VS Code's quick-fix menu:
 * - "Cola-Coder: Fix This" on diagnostics (errors/warnings)
 * - "Cola-Coder: Explain" on selected code
 * - "Cola-Coder: Refactor" on selected code
 *
 * Each action opens the chat panel with the relevant slash command.
 */

import * as vscode from 'vscode';

export class CodeActionProvider implements vscode.CodeActionProvider {
  static readonly providedCodeActionKinds = [
    vscode.CodeActionKind.QuickFix,
    vscode.CodeActionKind.RefactorRewrite,
  ];

  private registration: vscode.Disposable | undefined;

  static register(): CodeActionProvider {
    const provider = new CodeActionProvider();
    provider.registration = vscode.languages.registerCodeActionsProvider(
      { pattern: '**' },
      provider,
      { providedCodeActionKinds: CodeActionProvider.providedCodeActionKinds },
    );
    return provider;
  }

  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range | vscode.Selection,
    context: vscode.CodeActionContext,
    _token: vscode.CancellationToken,
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    // On diagnostics: offer "Fix This"
    if (context.diagnostics.length > 0) {
      const fix = new vscode.CodeAction(
        'Cola-Coder: Fix This',
        vscode.CodeActionKind.QuickFix,
      );
      fix.command = {
        command: 'cola-coder.fixSelection',
        title: 'Fix with Cola-Coder',
      };
      fix.diagnostics = [...context.diagnostics];
      fix.isPreferred = false;
      actions.push(fix);
    }

    // On selection: offer Explain and Refactor
    if (!range.isEmpty) {
      const explain = new vscode.CodeAction(
        'Cola-Coder: Explain',
        vscode.CodeActionKind.Empty,
      );
      explain.command = {
        command: 'cola-coder.explainSelection',
        title: 'Explain with Cola-Coder',
      };
      actions.push(explain);

      const refactor = new vscode.CodeAction(
        'Cola-Coder: Refactor',
        vscode.CodeActionKind.RefactorRewrite,
      );
      refactor.command = {
        command: 'cola-coder.refactorSelection',
        title: 'Refactor with Cola-Coder',
      };
      actions.push(refactor);
    }

    return actions;
  }

  dispose(): void {
    this.registration?.dispose();
  }
}
