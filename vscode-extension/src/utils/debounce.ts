/**
 * Debounce utility with AbortController support.
 *
 * Like lodash.debounce but cancels the previous invocation's
 * AbortSignal so in-flight HTTP requests get aborted too.
 */

export class DebouncedRequest<T> {
  private timer: NodeJS.Timeout | undefined;
  private controller: AbortController | undefined;

  constructor(
    private fn: (signal: AbortSignal) => Promise<T>,
    private delayMs: number,
  ) {}

  /** Schedule a new invocation, cancelling any pending one. */
  trigger(): Promise<T> {
    // Cancel previous
    this.cancel();

    return new Promise<T>((resolve, reject) => {
      this.controller = new AbortController();
      const signal = this.controller.signal;

      this.timer = setTimeout(async () => {
        try {
          const result = await this.fn(signal);
          if (!signal.aborted) {
            resolve(result);
          }
        } catch (err) {
          if (!signal.aborted) {
            reject(err);
          }
        }
      }, this.delayMs);
    });
  }

  /** Cancel any pending invocation and abort in-flight request. */
  cancel(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    if (this.controller) {
      this.controller.abort();
      this.controller = undefined;
    }
  }

  dispose(): void {
    this.cancel();
  }
}
