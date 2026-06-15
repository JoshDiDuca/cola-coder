import { useCallback, useEffect, useMemo, useState } from 'react';
import type { LogFile } from '../types';
import { isApiError } from '../types';
import { formatBytes, formatRelativeTime } from '../format';
import { getLogs, getLog } from '../api';

const LINE_OPTIONS = [100, 200, 500] as const;
const DEFAULT_LINES = 200;

export default function LogsPanel(): JSX.Element {
  const [logs, setLogs] = useState<LogFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [tailText, setTailText] = useState<string>('');
  const [tailLoading, setTailLoading] = useState(false);
  const [lines, setLines] = useState<number>(DEFAULT_LINES);

  const loadList = useCallback(async () => {
    setError(null);
    try {
      setLogs(await getLogs());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getLogs();
        if (active) setLogs(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const loadTail = useCallback(async (path: string, n: number) => {
    setSelectedPath(path);
    setTailLoading(true);
    setTailText('');

    try {
      const t = await getLog(path, n);
      if (isApiError(t)) {
        setTailText(`error: ${t.error}`);
      } else {
        const body = t.lines.join('\n');
        setTailText(t.truncated ? `…(truncated)\n${body}` : body);
      }
    } catch (e) {
      setTailText(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTailLoading(false);
    }
  }, []);

  const onLinesChange = useCallback(
    (n: number) => {
      setLines(n);
      if (selectedPath !== null) void loadTail(selectedPath, n);
    },
    [selectedPath, loadTail],
  );

  const selectedLog = useMemo(
    () => logs.find((lf) => lf.path === selectedPath) ?? null,
    [logs, selectedPath],
  );

  const hasLogs = logs.length > 0;

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Logs</div>
        <button type="button" className="btn" onClick={() => void loadList()}>
          ↻ refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {!hasLogs && !error ? (
        <div className="muted">no log files found</div>
      ) : (
        <div className="log-layout">
          <div className="log-pick">
            {logs.map((lf) => (
              <button
                key={lf.path}
                type="button"
                className={`log-pick-item${lf.path === selectedPath ? ' log-pick-active' : ''}`}
                onClick={() => void loadTail(lf.path, lines)}
                aria-pressed={lf.path === selectedPath}
              >
                <span className="log-pick-name mono">{lf.name}</span>
                <span className="log-pick-meta muted mono">
                  {formatBytes(lf.size_bytes)} · {formatRelativeTime(lf.mtime)}
                </span>
              </button>
            ))}
          </div>

          <div className="log-view">
            {selectedLog === null ? (
              <div className="muted log-view-empty">select a log to view its tail</div>
            ) : (
              <>
                <div className="log-view-head">
                  <span className="log-view-title mono">{selectedLog.name}</span>
                  <label className="log-view-lines">
                    <span className="k">lines</span>
                    <select
                      className="select"
                      value={lines}
                      onChange={(e) => onLinesChange(Number(e.target.value))}
                    >
                      {LINE_OPTIONS.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <pre className="pre scroll">
                  {tailLoading ? 'loading…' : tailText || <span className="muted">empty</span>}
                </pre>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
