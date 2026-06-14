import { useCallback, useEffect, useState } from 'react';
import type { ActionDef } from '../types';
import { getActions, runAction } from '../api';

export default function ActionsPanel({ onRan }: { onRan?: () => void }) {
  const [actions, setActions] = useState<ActionDef[]>([]);
  const [args, setArgs] = useState<Record<string, string>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const defs = await getActions();
        if (!active) return;
        setActions(defs);
        const initial: Record<string, string> = {};
        for (const d of defs) initial[d.key] = d.args.join(' ');
        setArgs(initial);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onChange = useCallback((key: string, value: string) => {
    setArgs((prev) => ({ ...prev, [key]: value }));
  }, []);

  const onRun = useCallback(
    async (key: string) => {
      setRunning(key);
      setStarted(null);
      setError(null);
      try {
        const value = (args[key] ?? '').trim();
        const job = await runAction(key, value ? value.split(/\s+/) : []);
        setStarted(`started ${job.name} (${job.id})`);
        onRan?.();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        // The backend returns HTTP 409 to refuse a trainer action while
        // training is already running; surface that clearly instead of a
        // generic failure (j<T> throws an Error whose message includes "409").
        setError(msg.includes('409') ? 'refused: training already running' : msg);
      } finally {
        setRunning(null);
      }
    },
    [args, onRan],
  );

  return (
    <div className="card card-wide">
      <div className="card-title">Run an action</div>

      {error && <div className="err">{error}</div>}
      {started && <div className="muted mono">{started}</div>}

      {actions.length === 0 && !error ? (
        <div className="muted">no actions available</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>action</th>
              <th>args</th>
              <th className="right">run</th>
            </tr>
          </thead>
          <tbody>
            {actions.map((a) => (
              <tr key={a.key}>
                <td>
                  <div>
                    {a.label}
                    {a.trainer && (
                      <span className="tag running" style={{ marginLeft: 8 }}>
                        trainer
                      </span>
                    )}
                  </div>
                  <div className="muted mono">{a.script}</div>
                </td>
                <td>
                  <input
                    className="input"
                    style={{ width: '100%' }}
                    value={args[a.key] ?? ''}
                    onChange={(e) => onChange(a.key, e.target.value)}
                    placeholder="args (space-separated)"
                    spellCheck={false}
                  />
                </td>
                <td className="right">
                  <button
                    className="btn btn-primary"
                    onClick={() => void onRun(a.key)}
                    disabled={running !== null}
                  >
                    {running === a.key ? '…running' : '▶ Run'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
