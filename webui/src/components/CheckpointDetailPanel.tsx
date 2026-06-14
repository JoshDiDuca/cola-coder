import { useCallback, useState } from 'react';
import type { Checkpoint, CheckpointDetail } from '../types';
import { isApiError } from '../types';
import { getCheckpointDetail } from '../api';

const MAX_ROWS = 12;

function humanParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export default function CheckpointDetailPanel({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const rows = [...checkpoints].sort((a, b) => b.step - a.step).slice(0, MAX_ROWS);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [detail, setDetail] = useState<CheckpointDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSelect = useCallback(async (ckpt: Checkpoint) => {
    setSelectedPath(ckpt.path);
    setDetail(null);
    setError(null);
    setLoading(true);
    try {
      const resp = await getCheckpointDetail(ckpt.path);
      if (isApiError(resp)) setError(resp.error);
      else setDetail(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="card-title">Checkpoint Detail</div>

      {rows.length === 0 ? (
        <div className="muted">no checkpoints</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>model</th>
              <th>step</th>
              <th className="right">loss</th>
              <th className="right">inspect</th>
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
                    inspect
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

          {detail && (
            <>
              <div className="row">
                <span className="k">num_params</span>
                <span className="v">{humanParams(detail.num_params)}</span>
              </div>
              <div className="row">
                <span className="k">tensor_count</span>
                <span className="v">{detail.tensor_count.toLocaleString()}</span>
              </div>
              <div className="row">
                <span className="k">is_moe</span>
                <span>
                  <span className={`tag ${detail.is_moe ? 'done' : 'failed'}`}>
                    {detail.is_moe ? 'moe' : 'dense'}
                  </span>
                </span>
              </div>
              <div className="row">
                <span className="k">has_training_state</span>
                <span>
                  <span className={`tag ${detail.has_training_state ? 'done' : 'failed'}`}>
                    {detail.has_training_state ? 'yes' : 'no'}
                  </span>
                </span>
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 6, borderBottom: 'none' }}>
                <span className="k">dtypes</span>
                {detail.dtypes.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  detail.dtypes.map((dt) => (
                    <span key={dt} className="tag mono">
                      {dt}
                    </span>
                  ))
                )}
              </div>

              {detail.is_moe && detail.moe_config && (
                <div>
                  <div className="card-title">moe_config</div>
                  <pre className="pre scroll">{JSON.stringify(detail.moe_config, null, 2)}</pre>
                </div>
              )}

              <div className="card-title">files</div>
              {detail.files.length === 0 ? (
                <div className="muted">none</div>
              ) : (
                <div className="scroll">
                  {detail.files.map((f) => (
                    <div key={f} className="mono">
                      {f}
                    </div>
                  ))}
                </div>
              )}

              <div className="card-title">metadata</div>
              {detail.metadata === null ? (
                <div className="muted">none</div>
              ) : (
                <pre className="pre scroll">{JSON.stringify(detail.metadata, null, 2)}</pre>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
