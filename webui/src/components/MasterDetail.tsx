import type { ReactNode } from 'react';
import EmptyState from './EmptyState';

// Reusable master-detail layout: a selectable list on the left, the selected
// item's detail/actions on the right. This is the core screen shape for the app
// (one coherent screen per section) replacing the old grid-of-cards.

export interface MasterItem {
  id: string;
  title: string;
  subtitle?: string;
  meta?: string;
  badge?: ReactNode;
}

interface MasterDetailProps {
  items: MasterItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  detail: ReactNode;
  listLabel?: string;
  listAside?: ReactNode;
  /**
   * Shown when the list is empty. Accepts a ReactNode so callers can pass a
   * polished `<EmptyState>`; a bare string still renders (as the default's
   * title) for backward compatibility.
   */
  emptyList?: ReactNode;
  /** Shown when no item is selected. ReactNode — pass an `<EmptyState>`. */
  emptyDetail?: ReactNode;
}

/**
 * Render empty-slot content: a bare string is wrapped in `<EmptyState>` so even
 * legacy string callers get the polished box; a ReactNode renders as-is.
 */
function renderEmpty(content: ReactNode): ReactNode {
  return typeof content === 'string' ? <EmptyState title={content} /> : content;
}

export default function MasterDetail({
  items,
  selectedId,
  onSelect,
  detail,
  listLabel,
  listAside,
  emptyList = <EmptyState title="Nothing here yet" />,
  emptyDetail = (
    <EmptyState
      title="Nothing selected"
      hint="Choose an item from the list to see its details."
    />
  ),
}: MasterDetailProps) {
  return (
    <div className="md">
      <aside className="md-list">
        {(listLabel || listAside) && (
          <div className="md-list-head">
            {listLabel ? <span className="md-list-label">{listLabel}</span> : <span />}
            {listAside}
          </div>
        )}
        {items.length === 0 ? (
          renderEmpty(emptyList)
        ) : (
          <div className="md-list-items">
            {items.map((it) => (
              <button
                key={it.id}
                type="button"
                className={`md-item${it.id === selectedId ? ' active' : ''}`}
                onClick={() => onSelect(it.id)}
              >
                <div className="md-item-row">
                  <span className="md-item-title">{it.title}</span>
                  {it.badge}
                </div>
                {it.subtitle ? <span className="md-item-sub">{it.subtitle}</span> : null}
                {it.meta ? <span className="md-item-meta">{it.meta}</span> : null}
              </button>
            ))}
          </div>
        )}
      </aside>

      <section className="md-detail">
        {selectedId !== null ? detail : renderEmpty(emptyDetail)}
      </section>
    </div>
  );
}
