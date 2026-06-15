import type { ReactNode } from 'react';

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
  emptyList?: string;
  emptyDetail?: string;
}

export default function MasterDetail({
  items,
  selectedId,
  onSelect,
  detail,
  listLabel,
  listAside,
  emptyList = 'Nothing here yet',
  emptyDetail = 'Select an item to see details',
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
          <div className="md-list-empty">{emptyList}</div>
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
        {selectedId !== null ? detail : <div className="md-detail-empty">{emptyDetail}</div>}
      </section>
    </div>
  );
}
