import { useCallback, useEffect, useState } from 'react';
import type { PipelineRun, JsonValue } from '../types';
import { getPipelineRuns, getPipelineRun } from '../api';
import { formatInteger, formatJsonValue } from '../format';

const RUN_MAX_CHARS = 6000;

// Map a pipeline status string to a .tag modifier class; unknown statuses get no extra class.
function tagClass(status: string | null): string {
  switch (status) {
    case 'running':
      return 'tag running';
    case 'completed':
      return 'tag done';
    case 'failed':
      return 'tag failed';
    case 'pending':
    case 'skipped':
      return 'tag';
    default:
      return 'tag';
  }
}

// The run-detail endpoint returns genuinely open JSON; render it through the one
// sanctioned JsonValue formatter, then cap the displayed length.
function formatRun(detail: JsonValue): string {
  const text = formatJsonValue(detail);
  return text.length > RUN_MAX_CHARS
    ? `${text.slice(0, RUN_MAX_CHARS)}\n…(truncated)`
    : text;
}

export default function PipelinePanel() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [runText, setRunText] = useState<string>('');
  const [runLoading, setRunLoading] = useState(false);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getPipelineRuns();
        if (active) setRuns(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (run: PipelineRun) => {
    setSelectedPath(run.path);
    setRunLoading(true);
    setRunText('');

    try {
      const detail = (await getPipelineRun(run.path)) as JsonValue;
      setRunText(formatRun(detail));
    } catch (e) {
      setRunText(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRunLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="card-title">Pipeline runs</div>

      {error && <div className="err">{error}</div>}

      {runs.length === 0 && !error ? (
        <div className="muted">no pipeline runs</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th className="right">stages</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.path}>
                <td className="mono">{run.name}</td>
                <td>
                  <span className={tagClass(run.status)}>{run.status ?? '—'}</span>
                </td>
                <td className="right mono">
                  {run.completed == null || run.num_stages == null
                    ? '—'
                    : `${formatInteger(run.completed)} / ${formatInteger(run.num_stages)}`}
                </td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(run)}>
                    view
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <div className="pre scroll">{runLoading ? 'loading…' : runText}</div>
      )}
    </div>
  );
}
