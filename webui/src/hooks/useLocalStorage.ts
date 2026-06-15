import { useCallback, useState } from "react";

/**
 * Typed localStorage-backed state hook.
 *
 * Reads the initial value from localStorage on first render (JSON.parse, falling
 * back to `initial` on a missing key or parse error). Writes to localStorage on
 * every set. SSR-safe: returns `initial` when `window` is undefined.
 *
 * The JSON.parse result is cast to `T` inside the try block — this is the single
 * sanctioned type boundary for this hook.
 */
export function useLocalStorage<T>(key: string, initial: T): [T, (value: T) => void] {
  const readInitial = (): T => {
    if (typeof window === "undefined") {
      return initial;
    }
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null) {
        return initial;
      }
      return JSON.parse(raw) as T;
    } catch {
      return initial;
    }
  };

  const [value, setValue] = useState<T>(readInitial);

  const set = useCallback(
    (next: T): void => {
      setValue(next);
      if (typeof window === "undefined") {
        return;
      }
      try {
        window.localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // Ignore write failures (quota exceeded, disabled storage, etc.).
      }
    },
    [key],
  );

  return [value, set];
}
