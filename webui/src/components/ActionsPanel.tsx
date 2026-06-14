import { useCallback, useEffect, useState } from 'react';
import type { ActionDef } from '../types';
import { getActions, runAction } from '../api';

interface ActionsPanelProps {
  onRan?: () => void;
  trainingAlive?: boolean;
}

export default function ActionsPanel({ onRan, trainingAlive }: ActionsPanelProps) {
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
    async (action: ActionDef) => {
      const key = action.key;
      setRunning(key);
      setStarted(null);
      setError(null);
      // Warn before launching a VRAM-heavy action while the live trainer runs:
      // loading a model on the GPU competes for VRAM and may OOM the live run.
      if (action.gpu && trainingAlive === true) {
        const ok = window.confirm(
          `${action.label} loads the model on the GPU. Training is live — running this will compete for VRAM and may OOM. Run anyway?`,
        );
        if (!ok) {
          setRunning(null);
          return;
        }
      }
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
    [args, onRan, trainingAlive],
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
                    {a.gpu && (
                      <span className="tag done" style={{ marginLeft: 8 }}>
                        GPU
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
                    onClick={() => void onRun(a)}
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
