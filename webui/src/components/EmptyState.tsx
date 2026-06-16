import type { ReactNode } from 'react';

/**
 * Shared empty-state block — a centered icon + title + optional hint and
 * action. Replaces the inconsistent `.muted` one-liners and dashed-border
 * one-offs (`.ds-empty`, `.md-list-empty`) so every "nothing here yet" view
 * reads the same and tells the user what to do next. Presentational only.
 */
interface EmptyStateProps {
  /** Primary line, e.g. "No checkpoints yet". */
  title: string;
  /** Optional secondary line explaining how to populate this view. */
  hint?: string;
  /** Decorative glyph shown above the title. */
  icon?: string;
  /** Optional call-to-action (e.g. a button) rendered below the hint. */
  action?: ReactNode;
}

export default function EmptyState({
  title,
  hint,
  icon = '◇',
  action,
}: EmptyStateProps) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon" aria-hidden="true">
        {icon}
      </span>
      <div className="empty-state-title">{title}</div>
      {hint !== undefined && <div className="empty-state-hint muted">{hint}</div>}
      {action !== undefined && <div className="empty-state-action">{action}</div>}
    </div>
  );
}
