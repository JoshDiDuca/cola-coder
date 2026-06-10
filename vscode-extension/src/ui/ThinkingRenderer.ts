/**
 * Parses <think>...</think> tokens from the SSE stream.
 *
 * When the model produces thinking tokens, this parser splits the
 * stream into thinking content (rendered in a collapsible block)
 * and code content (rendered as the main response).
 */

export interface ThinkingState {
  inThinking: boolean;
  thinkingBuffer: string;
  contentBuffer: string;
}

export interface ParseResult {
  /** Complete thinking text to render (null if still accumulating) */
  thinkingText: string | null;
  /** Content text to append to chat output */
  contentText: string | null;
  /** Updated state for next chunk */
  state: ThinkingState;
}

export function createInitialState(): ThinkingState {
  return { inThinking: false, thinkingBuffer: '', contentBuffer: '' };
}

/**
 * Process a single stream chunk, tracking <think>/<\/think> boundaries.
 *
 * Returns any complete thinking blocks and content text to render.
 */
/** Length of the longest tag prefix found at the END of `text`, or 0. */
function partialTagSuffix(text: string, tag: string): number {
  let match = 0;
  for (let i = 1; i < tag.length && i <= text.length; i++) {
    if (text.endsWith(tag.slice(0, i))) {
      match = i;
    }
  }
  return match;
}

export function parseStreamChunk(chunk: string, state: ThinkingState): ParseResult {
  // Prepend any partial tag carried over from the previous chunk so a tag
  // split across chunks (e.g. '<thi' + 'nk>') is reassembled before
  // matching. Without this the buffered fragment used to be flushed as
  // visible content and the tag never matched.
  let remaining = state.contentBuffer + chunk;
  let thinkingText: string | null = null;
  let contentText: string | null = null;
  const newState = { ...state, contentBuffer: '' };

  while (remaining.length > 0) {
    if (newState.inThinking) {
      // Looking for </think>
      const closeIdx = remaining.indexOf('</think>');
      if (closeIdx >= 0) {
        // Found end of thinking block
        newState.thinkingBuffer += remaining.slice(0, closeIdx);
        thinkingText = newState.thinkingBuffer;
        newState.thinkingBuffer = '';
        newState.inThinking = false;
        remaining = remaining.slice(closeIdx + '</think>'.length);
      } else {
        // Still in thinking. Hold back a partial '</think>' at the end of
        // the chunk (it completes in the next chunk via the carry above).
        const partial = partialTagSuffix(remaining, '</think>');
        newState.thinkingBuffer += remaining.slice(0, remaining.length - partial);
        newState.contentBuffer = remaining.slice(remaining.length - partial);
        remaining = '';
      }
    } else {
      // Looking for <think>
      const openIdx = remaining.indexOf('<think>');
      if (openIdx >= 0) {
        // Content before <think>
        const before = remaining.slice(0, openIdx);
        if (before) {
          contentText = (contentText ?? '') + before;
        }
        newState.inThinking = true;
        remaining = remaining.slice(openIdx + '<think>'.length);
      } else {
        // Check for partial <think> at end of chunk
        // (e.g., chunk ends with "<thi" — don't emit yet)
        const partial = partialTagSuffix(remaining, '<think>');
        const safe = remaining.slice(0, remaining.length - partial);
        if (safe) {
          contentText = (contentText ?? '') + safe;
        }
        newState.contentBuffer = remaining.slice(remaining.length - partial);
        remaining = '';
      }
    }
  }

  return { thinkingText, contentText, state: newState };
}

/**
 * Format a thinking block as collapsible markdown.
 */
export function formatThinkingBlock(thinking: string): string {
  return `\n<details><summary>Reasoning</summary>\n\n${thinking.trim()}\n\n</details>\n\n`;
}
