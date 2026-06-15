import { useCallback, useEffect, useState } from 'react';
import type { PipelineRun, JsonValue } from '../types';
import { getPipelineRuns, getPipelineRun } from '../api';
import { formatInteger, formatJsonValue, formatRelativeTime } from '../format';

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

// Fraction of stages completed in 0..1, or null when stage counts are unknown.
function completionFraction(run: PipelineRun): number | null {
  if (run.completed == null || run.num_stages == null || run.num_stages <= 0) {
    return null;
  }
  return Math.min(1, Math.max(0, run.completed / run.num_stages));
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
        <div className="ds-empty">
          <div className="ds-empty-title">No pipeline runs</div>
          <div className="muted">
            Create a named run in the Pipeline run manager to track the 10-stage build.
          </div>
        </div>
      ) : (
        <div className="pipe-list">
          {runs.map((run) => {
            const fraction = completionFraction(run);
            const isSelected = selectedPath === run.path;
            return (
              <div
                key={run.path}
                className={isSelected ? 'pipe-run pipe-run-active' : 'pipe-run'}
              >
                <div className="pipe-run-main">
                  <div className="pipe-run-id">
                    <span className="pipe-run-name mono">{run.name}</span>
                    <span className="pipe-run-time muted">
                      updated {formatRelativeTime(run.mtime)}
                    </span>
                  </div>
                  <span className={tagClass(run.status)}>{run.status ?? '—'}</span>
                </div>

                <div className="pipe-run-progress">
                  <div className="bar">
                    <div
                      className="fill"
                      style={{ width: `${(fraction ?? 0) * 100}%` }}
                    />
                  </div>
                  <span className="pipe-run-count mono">
                    {run.completed == null || run.num_stages == null
                      ? '—'
                      : `${formatInteger(run.completed)} / ${formatInteger(run.num_stages)}`}
                  </span>
                </div>

                <div className="pipe-run-foot">
                  {run.error != null && run.error !== '' && (
                    <span className="err pipe-run-err">{run.error}</span>
                  )}
                  <button className="btn" onClick={() => void onView(run)}>
                    {isSelected ? 'viewing' : 'view'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {selectedPath !== null && (
        <div className="pre scroll">{runLoading ? 'loading…' : runText}</div>
      )}
    </div>
  );
}
