/**
 * Shared loading indicator — a CSS-only spinner with an optional label.
 *
 * Replaces the ~40 ad-hoc `loading…` text literals scattered across panels so
 * every async flow shows the same polished feedback. Presentational only.
 */
interface LoadingSpinnerProps {
  /** Text shown beside the spinner. Pass an empty string for spinner-only. */
  label?: string;
  /** Render with no vertical padding (for use inline next to other content). */
  inline?: boolean;
}

export default function LoadingSpinner({
  label = 'Loading…',
  inline = false,
}: LoadingSpinnerProps) {
  return (
    <div
      className={inline ? 'spinner-row spinner-row-inline' : 'spinner-row'}
      role="status"
      aria-live="polite"
    >
      <span className="spinner" aria-hidden="true" />
      {label !== '' && <span className="spinner-label muted">{label}</span>}
    </div>
  );
}
