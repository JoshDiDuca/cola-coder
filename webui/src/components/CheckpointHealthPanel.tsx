import { useCallback, useState } from 'react';
import type { Checkpoint, CheckpointHealth } from '../types';
import { isApiError } from '../types';
import { getCheckpointHealth } from '../api';
import { formatInteger, formatFloat } from '../format';

const MAX_ROWS = 12;

function healthBadgeClass(ok: boolean): string {
  return ok ? 'tag done' : 'tag failed';
}

interface CheckpointHealthPanelProps {
  checkpoints: Checkpoint[];
}

export default function CheckpointHealthPanel({ checkpoints }: CheckpointHealthPanelProps) {
  const rows = [...checkpoints].sort((a, b) => b.step - a.step).slice(0, MAX_ROWS);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [health, setHealth] = useState<CheckpointHealth | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSelect = useCallback(async (ckpt: Checkpoint): Promise<void> => {
    setSelectedPath(ckpt.path);
    setHealth(null);
    setError(null);
    setLoading(true);
    try {
      const resp = await getCheckpointHealth(ckpt.model, String(ckpt.step));
      if (isApiError(resp)) {
        setError(resp.error);
      } else {
        setHealth(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="card-title">Checkpoint Health</div>

      {rows.length === 0 ? (
        <div className="muted">no checkpoints</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>model</th>
              <th>step</th>
              <th className="right">loss</th>
              <th className="right">check</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((ckpt) => (
              <tr key={ckpt.path}>
                <td>{ckpt.model}</td>
                <td className="mono">{ckpt.step.toLocaleString()}</td>
                <td className="right mono">{ckpt.loss == null ? '—' : ckpt.loss.toFixed(4)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onSelect(ckpt)}>
                    check
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <>
          <div className="muted mono">{selectedPath}</div>
          {error && <div className="err">{error}</div>}
          {loading && <div className="muted">loading…</div>}

          {health && (
            <>
              <div className="row">
                <span className={healthBadgeClass(health.ok)}>
                  {health.ok ? 'healthy' : 'no weights'}
                </span>
                <span className="v muted">{health.config_stem ?? 'unknown config'}</span>
              </div>

              <div className="row">
                <span className="k">model</span>
                <span className="v mono">{health.model}</span>
              </div>
              <div className="row">
                <span className="k">step</span>
                <span className="v mono">{formatInteger(health.step)}</span>
              </div>
              <div className="row">
                <span className="k">loss</span>
                <span className="v mono">{formatFloat(health.loss, 4)}</span>
              </div>
              <div className="row">
                <span className="k">size (MB)</span>
                <span className="v mono">{formatFloat(health.size_mb, 2)}</span>
              </div>
              <div className="row">
                <span className="k">num_tensors</span>
                <span className="v mono">{formatInteger(health.num_tensors)}</span>
              </div>

              <div className="card-title">files</div>
              {health.files.length === 0 ? (
                <div className="muted">none</div>
              ) : (
                <div className="scroll">
                  {health.files.map((f) => (
                    <div key={f} className="mono">
                      {f}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
