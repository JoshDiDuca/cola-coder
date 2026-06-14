// Shared, intent-named, type-specific formatters for the web UI.
// This is the ONLY place these live — components import from here and never
// define their own `humanBytes`/`fmtInt`/`fmt`/`fmtValue` helpers.
import type { JsonValue } from './types';

const EM_DASH = '—';

/** Bytes → human-readable, e.g. "1.50 GB", "12 KB". null → "—". */
export function formatBytes(bytes: number | null): string {
  if (bytes === null) return EM_DASH;
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB', 'PB'] as const;
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

/** Integer → locale-grouped, e.g. "1,234". null → "—". */
export function formatInteger(n: number | null): string {
  if (n === null) return EM_DASH;
  return Math.round(n).toLocaleString('en-US');
}

/** Float → fixed decimals (default 2). null → "—". */
export function formatFloat(n: number | null, digits: number = 2): string {
  if (n === null) return EM_DASH;
  return n.toFixed(digits);
}

/** Fraction in 0..1 → percent, e.g. 0.512 → "51.2%" (default 1 dp). null → "—". */
export function formatPercent(fraction: number | null, digits: number = 1): string {
  if (fraction === null) return EM_DASH;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Already-a-percent value → "5.1%" (default 1 dp). null → "—". */
export function formatPercentValue(pct: number | null, digits: number = 1): string {
  if (pct === null) return EM_DASH;
  return `${pct.toFixed(digits)}%`;
}

/** Seconds → "1h 02m" / "3.4s". null → "—". */
export function formatDuration(seconds: number | null): string {
  if (seconds === null) return EM_DASH;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const totalMinutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  if (totalMinutes < 60) {
    return `${totalMinutes}m ${String(remainingSeconds).padStart(2, '0')}s`;
  }
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}h ${String(minutes).padStart(2, '0')}m`;
}

/** Epoch seconds → "2m ago", with an absolute date fallback for older times. null → "—". */
export function formatRelativeTime(epochSeconds: number | null): string {
  if (epochSeconds === null) return EM_DASH;
  const deltaSeconds = Date.now() / 1000 - epochSeconds;
  if (deltaSeconds < 0) return new Date(epochSeconds * 1000).toLocaleString();
  if (deltaSeconds < 60) return `${Math.floor(deltaSeconds)}s ago`;
  if (deltaSeconds < 3600) return `${Math.floor(deltaSeconds / 60)}m ago`;
  if (deltaSeconds < 86400) return `${Math.floor(deltaSeconds / 3600)}h ago`;
  if (deltaSeconds < 7 * 86400) return `${Math.floor(deltaSeconds / 86400)}d ago`;
  return new Date(epochSeconds * 1000).toLocaleString();
}

/** Model parameter count → "124.5M", "1.20B", "999". null → "—". */
export function formatParams(n: number | null): string {
  if (n === null) return EM_DASH;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(Math.round(n));
}

/**
 * The ONE sanctioned JSON renderer. Exhaustive over the `JsonValue` union;
 * replaces every component's local `fmt(value: unknown)`/`fmtValue`.
 */
export function formatJsonValue(value: JsonValue): string {
  if (value === null) return EM_DASH;
  switch (typeof value) {
    case 'string':
      return value;
    case 'number':
      return String(value);
    case 'boolean':
      return value ? 'yes' : 'no';
    case 'object': {
      if (Array.isArray(value)) {
        return `[${value.map(formatJsonValue).join(', ')}]`;
      }
      return JSON.stringify(value);
    }
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}
