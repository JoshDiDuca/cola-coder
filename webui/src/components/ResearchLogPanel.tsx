import { useCallback, useEffect, useState } from 'react';
import {
  isApiError,
  type ResearchLog,
  type ResearchEntry,
  type ResearchLogAppendRequest,
} from '../types';
import { getResearchLog, appendResearchLog } from '../api';
import { formatInteger } from '../format';

const BODY_PLACEHOLDER = '**Summary:** … **Sources:** https://…';

export default function ResearchLogPanel() {
  const [log, setLog] = useState<ResearchLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [title, setTitle] = useState<string>('');
  const [body, setBody] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [addSuccess, setAddSuccess] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getResearchLog();
      if (isApiError(resp)) {
        setLog(null);
        setError(resp.error);
      } else {
        setLog(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const canSubmit: boolean =
    !submitting && title.trim().length > 0 && body.trim().length > 0;

  const handleSubmit = useCallback(async (): Promise<void> => {
    if (submitting || title.trim().length === 0 || body.trim().length === 0) {
      return;
    }
    setSubmitting(true);
    setAddError(null);
    setAddSuccess(null);
    const req: ResearchLogAppendRequest = { title: title.trim(), body: body.trim() };
    try {
      const resp = await appendResearchLog(req);
      if (isApiError(resp)) {
        setAddError(resp.error);
      } else {
        setLog(resp);
        setError(null);
        setTitle('');
        setBody('');
        setAddSuccess('Research note added.');
      }
    } catch (e) {
      setAddError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }, [submitting, title, body]);

  const entries: ResearchEntry[] = log?.entries ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Research Log</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      <div className="card">
        <div className="card-title">Add research note</div>
        <input
          className="input"
          type="text"
          placeholder="title"
          value={title}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTitle(e.target.value)}
          disabled={submitting}
        />
        <textarea
          className="textarea mono"
          rows={8}
          placeholder={BODY_PLACEHOLDER}
          value={body}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setBody(e.target.value)}
          disabled={submitting}
        />
        <div className="row">
          <span className="muted">markdown is fine</span>
          <button
            className="btn btn-primary"
            onClick={() => void handleSubmit()}
            disabled={!canSubmit}
          >
            {submitting ? 'Adding…' : 'Save'}
          </button>
        </div>
        {addError && <div className="err field-error">{addError}</div>}
        {addSuccess && <div className="muted">{addSuccess}</div>}
      </div>

      {log && entries.length === 0 && !error && (
        <div className="muted">no research-log entries yet</div>
      )}

      {entries.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(log?.count ?? entries.length)} entries · newest first
            </span>
          </div>

          <table className="tbl">
            <thead>
              <tr>
                <th>date</th>
                <th>technique</th>
                <th className="right">sources</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={`${entry.date}:${entry.title}`}>
                  <td className="mono">{entry.date}</td>
                  <td>
                    <div>
                      {entry.title}{' '}
                      {entry.area !== null && <span className="tag">{entry.area}</span>}{' '}
                      {entry.has_original_idea && (
                        <span className="tag done">original idea</span>
                      )}
                    </div>
                    {entry.summary && <div className="muted">{entry.summary}</div>}
                  </td>
                  <td className="right mono">{formatInteger(entry.source_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {!log && !error && <div className="muted">loading research log…</div>}
    </div>
  );
}
