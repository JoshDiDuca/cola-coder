import { useCallback, useEffect, useState } from 'react';
import type { LogFile } from '../types';
import { isApiError } from '../types';
import { formatBytes, formatRelativeTime } from '../format';
import { getLogs, getLog } from '../api';

const LINE_OPTIONS = [100, 200, 500] as const;
const DEFAULT_LINES = 200;

export default function LogsPanel() {
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
    [selectedPath, loadTail]
  );

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Logs</div>
        <button className="btn" onClick={() => void loadList()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {logs.length === 0 && !error ? (
        <div className="muted">no log files found</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th className="right">size</th>
              <th className="right">when</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((lf) => (
              <tr key={lf.path}>
                <td className="mono">{lf.name}</td>
                <td className="right mono">{formatBytes(lf.size_bytes)}</td>
                <td className="right mono">{formatRelativeTime(lf.mtime)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void loadTail(lf.path, lines)}>
                    view
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <>
          <div className="row">
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
          </div>
          <pre className="pre scroll">{tailLoading ? 'loading…' : tailText}</pre>
        </>
      )}
    </div>
  );
}
