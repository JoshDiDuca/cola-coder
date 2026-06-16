import { useCallback, useEffect, useState } from 'react';
import type { ChangeEvent } from 'react';
import type { SpecialistEntry, SpecialistsView, SpecialistSaveRequest } from '../types';
import { isApiError } from '../types';
import { getSpecialists, saveSpecialist, removeSpecialist } from '../api';
import { formatPercent } from '../format';
import LoadingSpinner from './LoadingSpinner';
import EmptyState from './EmptyState';

// Local controlled-form shape: every field is a string so inputs stay fully
// controlled; we coerce to the typed SpecialistSaveRequest only on submit.
interface SpecialistFormState {
  domain: string;
  checkpoint: string;
  keywords: string;
  config: string;
  confidence_threshold: string;
  description: string;
}

const EMPTY_FORM: SpecialistFormState = {
  domain: '',
  checkpoint: '',
  keywords: '',
  config: '',
  confidence_threshold: '',
  description: '',
};

function entryToForm(entry: SpecialistEntry): SpecialistFormState {
  return {
    domain: entry.domain,
    checkpoint: entry.checkpoint,
    keywords: (entry.keywords ?? []).join(', '),
    config: entry.config ?? '',
    confidence_threshold:
      entry.confidence_threshold != null ? String(entry.confidence_threshold) : '',
    description: entry.description ?? '',
  };
}

// Build the typed request from the raw string form. Blank optional fields → null.
function formToRequest(form: SpecialistFormState): SpecialistSaveRequest {
  const keywords: string[] = form.keywords
    .split(',')
    .map((kw) => kw.trim())
    .filter((kw) => kw.length > 0);

  const config: string | null = form.config.trim().length > 0 ? form.config.trim() : null;
  const description: string | null =
    form.description.trim().length > 0 ? form.description.trim() : null;

  const thresholdRaw: string = form.confidence_threshold.trim();
  const threshold: number | null = thresholdRaw.length > 0 ? Number(thresholdRaw) : null;

  return {
    domain: form.domain.trim(),
    checkpoint: form.checkpoint.trim(),
    keywords,
    config,
    confidence_threshold: threshold,
    description,
  };
}

interface SpecialistRowProps {
  entry: SpecialistEntry;
  busy: boolean;
  onEdit: (entry: SpecialistEntry) => void;
  onRemove: (domain: string) => void;
}

function SpecialistRow({ entry, busy, onEdit, onRemove }: SpecialistRowProps): JSX.Element {
  return (
    <div className="spec-row">
      <div className="spec-head">
        <span className="tag">{entry.domain}</span>
        {entry.confidence_threshold != null && (
          <span className="spec-threshold muted">
            threshold {formatPercent(entry.confidence_threshold)}
          </span>
        )}
      </div>

      <div className="spec-kv">
        <span className="k">checkpoint</span>
        <span className="mono">{entry.checkpoint}</span>
      </div>

      {entry.config != null && (
        <div className="spec-kv">
          <span className="k">config</span>
          <span className="mono">{entry.config}</span>
        </div>
      )}

      {(entry.keywords?.length ?? 0) > 0 && (
        <div className="spec-chips">
          {(entry.keywords ?? []).map((kw, i) => (
            <span key={`${i}-${kw}`} className="spec-chip">
              {kw}
            </span>
          ))}
        </div>
      )}

      {entry.description !== null && <div className="muted spec-desc">{entry.description}</div>}

      <div className="md-toolbar">
        <button className="btn" disabled={busy} onClick={() => onEdit(entry)}>
          use as template
        </button>
        <button className="btn btn-danger" disabled={busy} onClick={() => onRemove(entry.domain)}>
          remove
        </button>
      </div>
    </div>
  );
}

interface SpecialistFormProps {
  form: SpecialistFormState;
  submitting: boolean;
  saveError: string | null;
  onChange: (next: SpecialistFormState) => void;
  onSave: () => void;
  onClear: () => void;
}

function SpecialistForm({
  form,
  submitting,
  saveError,
  onChange,
  onSave,
  onClear,
}: SpecialistFormProps): JSX.Element {
  const setField = (field: keyof SpecialistFormState) => {
    return (e: ChangeEvent<HTMLInputElement>): void => {
      onChange({ ...form, [field]: e.target.value });
    };
  };

  const canSave: boolean =
    !submitting && form.domain.trim().length > 0 && form.checkpoint.trim().length > 0;

  return (
    <div className="spec-form">
      <div className="card-title">Add / update specialist</div>

      <div className="spec-kv">
        <span className="k">domain</span>
        <input
          className="input"
          type="text"
          value={form.domain}
          placeholder="react"
          onChange={setField('domain')}
        />
      </div>

      <div className="spec-kv">
        <span className="k">checkpoint</span>
        <input
          className="input mono"
          type="text"
          value={form.checkpoint}
          placeholder="checkpoints/react_sft/latest"
          onChange={setField('checkpoint')}
        />
      </div>

      <div className="spec-kv">
        <span className="k">keywords</span>
        <input
          className="input"
          type="text"
          value={form.keywords}
          placeholder="comma, separated, keywords"
          onChange={setField('keywords')}
        />
      </div>

      <div className="spec-kv">
        <span className="k">config</span>
        <input
          className="input mono"
          type="text"
          value={form.config}
          placeholder="configs/small.yaml (optional)"
          onChange={setField('config')}
        />
      </div>

      <div className="spec-kv">
        <span className="k">threshold</span>
        <input
          className="input"
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={form.confidence_threshold}
          placeholder="0.50 (optional)"
          onChange={setField('confidence_threshold')}
        />
      </div>

      <div className="spec-kv">
        <span className="k">description</span>
        <input
          className="input"
          type="text"
          value={form.description}
          placeholder="optional notes"
          onChange={setField('description')}
        />
      </div>

      {saveError !== null && <div className="field-error">{saveError}</div>}

      <div className="md-toolbar">
        <button className="btn btn-primary" disabled={!canSave} onClick={onSave}>
          {submitting ? 'saving…' : 'save'}
        </button>
        <button className="btn" disabled={submitting} onClick={onClear}>
          clear
        </button>
      </div>
    </div>
  );
}

export default function SpecialistsPanel(): JSX.Element {
  const [view, setView] = useState<SpecialistsView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const [form, setForm] = useState<SpecialistFormState>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getSpecialists();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setView(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const resp = await getSpecialists();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setView(null);
        } else {
          setView(resp);
        }
      } catch (e) {
        if (active) {
          setError(e instanceof Error ? e.message : String(e));
          setView(null);
        }
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const handleSave = useCallback(async (): Promise<void> => {
    setSubmitting(true);
    setSaveError(null);
    try {
      const resp = await saveSpecialist(formToRequest(form));
      if (isApiError(resp)) {
        setSaveError(resp.error);
      } else {
        setView(resp);
        setForm(EMPTY_FORM);
      }
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }, [form]);

  const handleRemove = useCallback(async (domain: string): Promise<void> => {
    setSubmitting(true);
    setSaveError(null);
    try {
      const resp = await removeSpecialist(domain);
      if (isApiError(resp)) {
        setSaveError(resp.error);
      } else {
        setView(resp);
      }
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }, []);

  const handleEdit = useCallback((entry: SpecialistEntry): void => {
    setSaveError(null);
    setForm(entryToForm(entry));
  }, []);

  const handleClear = useCallback((): void => {
    setSaveError(null);
    setForm(EMPTY_FORM);
  }, []);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Domain specialists</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      <div className="muted spec-help">
        Router&rarr;specialist registry from <span className="mono">configs/specialists.yaml</span>:
        per-domain checkpoints the 125M router can dispatch requests to.
      </div>

      {loading && <LoadingSpinner label="Loading specialists…" />}

      {!loading && error !== null && <div className="err">{error}</div>}

      {!loading && error === null && view !== null && (
        <>
          {!view.exists && (
            <EmptyState
              title="No specialists registry"
              hint={`specialists.yaml not found at ${view.path}. Create it to register per-domain checkpoints.`}
            />
          )}

          {view.exists && view.count === 0 && (
            <EmptyState
              title="No specialists registered yet"
              hint={`Add entries to ${view.path} (domain → checkpoint, keywords, threshold) to route requests to per-domain models.`}
            />
          )}

          {view.exists && view.count > 0 && (
            <div className="spec-list">
              {view.specialists.map((entry) => (
                <SpecialistRow
                  key={entry.domain}
                  entry={entry}
                  busy={submitting}
                  onEdit={handleEdit}
                  onRemove={(domain) => void handleRemove(domain)}
                />
              ))}
            </div>
          )}

          <SpecialistForm
            form={form}
            submitting={submitting}
            saveError={saveError}
            onChange={setForm}
            onSave={() => void handleSave()}
            onClear={handleClear}
          />
        </>
      )}
    </div>
  );
}
