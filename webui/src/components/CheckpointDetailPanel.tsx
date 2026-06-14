import { useCallback, useState } from 'react';
import type { Checkpoint, CheckpointDetail, JsonValue } from '../types';
import { isApiError } from '../types';
import { formatParams, formatInteger, formatJsonValue } from '../format';
import { getCheckpointDetail } from '../api';

const MAX_ROWS = 12;

interface JsonKvTableProps {
  title: string;
  data: Record<string, JsonValue>;
}

function JsonKvTable({ title, data }: JsonKvTableProps) {
  const entries = Object.entries(data);
  return (
    <div>
      <div className="card-title">{title}</div>
      {entries.length === 0 ? (
        <div className="muted">none</div>
      ) : (
        <table className="tbl">
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k}>
                <td className="k">{k}</td>
                <td className="right mono">{formatJsonValue(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

interface CheckpointDetailPanelProps {
  checkpoints: Checkpoint[];
}

export default function CheckpointDetailPanel({ checkpoints }: CheckpointDetailPanelProps) {
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
                <span className="v">{formatParams(detail.num_params)}</span>
              </div>
              <div className="row">
                <span className="k">tensor_count</span>
                <span className="v">{formatInteger(detail.tensor_count)}</span>
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
                <JsonKvTable title="moe_config" data={detail.moe_config} />
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

              {detail.metadata === null ? (
                <div>
                  <div className="card-title">metadata</div>
                  <div className="muted">none</div>
                </div>
              ) : (
                <JsonKvTable title="metadata" data={detail.metadata} />
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
