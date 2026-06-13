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

const isHighSurrogate = (c: number) => c >= 0xd800 && c <= 0xdbff;
const isLowSurrogate = (c: number) => c >= 0xdc00 && c <= 0xdfff;

/**
 * Extract FIM context from the current cursor position.
 */
export function extractFimContext(
  document: vscode.TextDocument,
  position: vscode.Position,
): FimContext {
  const fullText = document.getText();
  const offset = document.offsetAt(position);

  // Get prefix (text before cursor, capped). If the cap lands on the LOW half
  // of a surrogate pair (emoji / CJK in code), advance one code unit so the
  // prefix doesn't begin with a lone surrogate that corrupts tokenization. (EXT-004)
  let prefixStart = Math.max(0, offset - MAX_PREFIX_CHARS);
  if (prefixStart > 0 && isLowSurrogate(fullText.charCodeAt(prefixStart))) {
    prefixStart += 1;
  }
  const prefix = fullText.slice(prefixStart, offset);

  // Get suffix (text after cursor, capped). If the cap would leave a dangling
  // HIGH surrogate at the end, pull the boundary back one code unit. (EXT-004)
  let suffixEnd = Math.min(fullText.length, offset + MAX_SUFFIX_CHARS);
  if (suffixEnd < fullText.length && isHighSurrogate(fullText.charCodeAt(suffixEnd - 1))) {
    suffixEnd -= 1;
  }
  const suffix = fullText.slice(offset, suffixEnd);

  return {
    prefix,
    suffix,
    language: document.languageId,
    filePath: document.uri.fsPath,
  };
}
