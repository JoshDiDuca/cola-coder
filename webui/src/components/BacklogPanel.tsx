import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import {
  isApiError,
  type BacklogView,
  type BacklogItem,
  type BacklogAppendRequest,
} from '../types';
import { getBacklog, appendBacklog } from '../api';
import { formatInteger } from '../format';

type Status = 'open' | 'in-progress' | 'done' | 'dropped' | 'unknown';

/** The four real backlog statuses (no synthetic `unknown`) — matches BacklogAppendRequest.status. */
type BacklogStatus = 'open' | 'in-progress' | 'done' | 'dropped';

const APPEND_STATUS_VALUES: readonly BacklogStatus[] = [
  'open',
  'in-progress',
  'done',
  'dropped',
] as const;

/** Typed shape of the "file a backlog item" form. */
interface BacklogAppendForm {
  itemId: string;
  category: string;
  severity: string;
  status: BacklogStatus;
  description: string;
}

const EMPTY_FORM: BacklogAppendForm = {
  itemId: '',
  category: '',
  severity: '',
  status: 'open',
  description: '',
};

const STATUS_VALUES: readonly Status[] = [
  'open',
  'in-progress',
  'done',
  'dropped',
  'unknown',
] as const;

const ALL_STATUSES = '__all__';
const ALL_CATEGORIES = '__all__';

function asStatus(status: string): Status {
  return (STATUS_VALUES as readonly string[]).includes(status) ? (status as Status) : 'unknown';
}

/** Map a status to a `.tag` badge class. Exhaustive switch with a never-check. */
function statusBadgeClass(status: string): string {
  const known = asStatus(status);
  switch (known) {
    case 'done':
      return 'tag done';
    case 'in-progress':
      return 'tag running';
    case 'open':
      return 'tag';
    case 'dropped':
      return 'tag failed';
    case 'unknown':
      return 'tag';
    default: {
      const _exhaustive: never = known;
      return _exhaustive;
    }
  }
}

export default function BacklogPanel() {
  const [view, setView] = useState<BacklogView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>(ALL_STATUSES);
  const [categoryFilter, setCategoryFilter] = useState<string>(ALL_CATEGORIES);
  const [text, setText] = useState<string>('');

  const [form, setForm] = useState<BacklogAppendForm>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [appendError, setAppendError] = useState<string | null>(null);
  const [appendSuccess, setAppendSuccess] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const v = await getBacklog();
      if (isApiError(v)) {
        setView(null);
        setError(v.error);
      } else {
        setView(v);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const categories = useMemo<string[]>(() => {
    if (!view) return [];
    const seen = new Set<string>();
    for (const item of view.items) {
      if (item.category) seen.add(item.category);
    }
    return [...seen].sort();
  }, [view]);

  const rows = useMemo<BacklogItem[]>(() => {
    if (!view) return [];
    const needle = text.trim().toLowerCase();
    return view.items.filter((item) => {
      if (statusFilter !== ALL_STATUSES && item.status !== statusFilter) return false;
      if (categoryFilter !== ALL_CATEGORIES && item.category !== categoryFilter) return false;
      if (needle === '') return true;
      return (
        item.id.toLowerCase().includes(needle) ||
        item.description.toLowerCase().includes(needle) ||
        item.severity.toLowerCase().includes(needle)
      );
    });
  }, [view, statusFilter, categoryFilter, text]);

  const canSubmit: boolean =
    !submitting &&
    form.itemId.trim() !== '' &&
    form.category.trim() !== '' &&
    form.description.trim() !== '';

  const onStatusChange = (e: ChangeEvent<HTMLSelectElement>): void => {
    setForm((prev) => ({ ...prev, status: e.target.value as BacklogStatus }));
  };

  const submit = useCallback(async (): Promise<void> => {
    setAppendError(null);
    setAppendSuccess(null);
    setSubmitting(true);
    const req: BacklogAppendRequest = {
      item_id: form.itemId.trim(),
      category: form.category.trim(),
      description: form.description.trim(),
      severity: form.severity.trim() === '' ? undefined : form.severity.trim(),
      status: form.status,
    };
    try {
      const result = await appendBacklog(req);
      if (isApiError(result)) {
        setAppendError(result.error);
      } else {
        setView(result);
        setForm(EMPTY_FORM);
        setAppendSuccess(`Filed ${req.item_id}.`);
      }
    } catch (e) {
      setAppendError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }, [form]);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Backlog Viewer</div>
        <button className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      <div className="card">
        <div className="card-title">File a backlog item</div>
        <div className="row">
          <input
            className="input"
            value={form.itemId}
            onChange={(e) => setForm((prev) => ({ ...prev, itemId: e.target.value }))}
            placeholder='item_id (e.g. "UI-099")'
            spellCheck={false}
            disabled={submitting}
          />
          <input
            className="input"
            value={form.category}
            onChange={(e) => setForm((prev) => ({ ...prev, category: e.target.value }))}
            placeholder='category (e.g. "ui")'
            spellCheck={false}
            disabled={submitting}
          />
        </div>
        <div className="row">
          <input
            className="input"
            value={form.severity}
            onChange={(e) => setForm((prev) => ({ ...prev, severity: e.target.value }))}
            placeholder='severity (optional, e.g. "low")'
            spellCheck={false}
            disabled={submitting}
          />
          <select
            className="select"
            value={form.status}
            onChange={onStatusChange}
            disabled={submitting}
          >
            {APPEND_STATUS_VALUES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="row">
          <textarea
            className="textarea"
            value={form.description}
            onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
            placeholder="description"
            spellCheck={false}
            disabled={submitting}
          />
        </div>
        <div className="row">
          <button className="btn btn-primary" onClick={() => void submit()} disabled={!canSubmit}>
            {submitting ? 'Filing…' : 'Save'}
          </button>
          {appendSuccess && <span className="muted">{appendSuccess}</span>}
        </div>
        {appendError && <div className="field-error">{appendError}</div>}
      </div>

      {view && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(view.count)} items · {formatInteger(view.open_count)} open ·{' '}
              {formatInteger(view.done_count)} done · {formatInteger(rows.length)} shown
            </span>
            <input
              className="input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="filter by id / description / severity"
              spellCheck={false}
            />
          </div>

          <div className="row">
            <select
              className="input"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value={ALL_STATUSES}>all statuses</option>
              {STATUS_VALUES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <select
              className="input"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
            >
              <option value={ALL_CATEGORIES}>all categories</option>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {rows.length === 0 ? (
            <div className="muted">no backlog items match</div>
          ) : (
            <table className="tbl">
              <tbody>
                {rows.map((item) => (
                  <tr key={item.id}>
                    <td className="mono">{item.id}</td>
                    <td>
                      <span className={statusBadgeClass(item.status)}>{item.status}</span>
                    </td>
                    <td className="muted mono">{item.category}</td>
                    <td className="muted mono">{item.severity || '—'}</td>
                    <td className="muted mono">{item.date ?? '—'}</td>
                    <td className="muted">{item.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}

      {!view && !error && <div className="muted">no backlog</div>}
    </div>
  );
}
