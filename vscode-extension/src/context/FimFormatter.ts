/**
 * Extract FIM (Fill-in-the-Middle) prefix and suffix from the editor.
 *
 * Given a cursor position in a document, extracts the text before and
 * after the cursor to form the FIM prefix and suffix.
 */

import * as vscode from 'vscode';

export interface FimContext {
  prefix: string;
  suffix: string;
  language: string;
  filePath: string;
}

/** Max chars for prefix (roughly ~750 tokens at 4 chars/token) */
const MAX_PREFIX_CHARS = 3000;

/** Max chars for suffix (less needed than prefix) */
const MAX_SUFFIX_CHARS = 1000;

/**
 * Extract FIM context from the current cursor position.
 */
export function extractFimContext(
  document: vscode.TextDocument,
  position: vscode.Position,
): FimContext {
  const fullText = document.getText();
  const offset = document.offsetAt(position);

  // Get prefix (text before cursor, capped)
  const prefixStart = Math.max(0, offset - MAX_PREFIX_CHARS);
  const prefix = fullText.slice(prefixStart, offset);

  // Get suffix (text after cursor, capped)
  const suffixEnd = Math.min(fullText.length, offset + MAX_SUFFIX_CHARS);
  const suffix = fullText.slice(offset, suffixEnd);

  return {
    prefix,
    suffix,
    language: document.languageId,
    filePath: document.uri.fsPath,
  };
}
