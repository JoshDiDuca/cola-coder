/**
 * Assembles editor context for prompts.
 *
 * Gathers selected code, active file content, diagnostics, and
 * repo context to build rich prompts for the chat participant.
 */

import * as vscode from 'vscode';

export interface EditorContext {
  /** Currently selected text (empty string if no selection) */
  selectedCode: string;
  /** Language ID of the active editor */
  language: string;
  /** File path of the active editor */
  filePath: string;
  /** Full file content (for context) */
  fileContent: string;
  /** Diagnostics on the current file */
  diagnostics: string[];
}

/**
 * Gather context from the active editor.
 */
export function getEditorContext(): EditorContext {
  const editor = vscode.window.activeTextEditor;

  if (!editor) {
    return {
      selectedCode: '',
      language: 'plaintext',
      filePath: '',
      fileContent: '',
      diagnostics: [],
    };
  }

  const document = editor.document;
  const selection = editor.selection;

  // Get selected text
  const selectedCode = selection.isEmpty
    ? ''
    : document.getText(selection);

  // Get diagnostics for this file
  const allDiagnostics = vscode.languages.getDiagnostics(document.uri);
  const diagnosticStrings = allDiagnostics
    .filter(d => d.severity === vscode.DiagnosticSeverity.Error
      || d.severity === vscode.DiagnosticSeverity.Warning)
    .slice(0, 5) // limit to 5 most relevant
    .map(d => {
      const severity = d.severity === vscode.DiagnosticSeverity.Error
        ? 'Error' : 'Warning';
      return `${severity} (line ${d.range.start.line + 1}): ${d.message}`;
    });

  return {
    selectedCode,
    language: document.languageId,
    filePath: document.uri.fsPath,
    fileContent: document.getText(),
    diagnostics: diagnosticStrings,
  };
}

/**
 * Build a user message that includes editor context.
 */
export function buildUserMessage(
  prompt: string,
  context: EditorContext,
): string {
  const parts: string[] = [];

  if (context.selectedCode) {
    parts.push(`File: ${context.filePath} (${context.language})`);
    parts.push('```' + context.language);
    parts.push(context.selectedCode);
    parts.push('```');
  }

  if (context.diagnostics.length > 0) {
    parts.push('Diagnostics:');
    for (const d of context.diagnostics) {
      parts.push(`- ${d}`);
    }
  }

  if (prompt) {
    parts.push(prompt);
  }

  return parts.join('\n');
}
