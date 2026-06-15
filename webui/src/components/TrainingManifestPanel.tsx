import { useCallback, useEffect, useState } from 'react';
import type { TrainingManifest, TrainingManifests } from '../types';
import { isApiError } from '../types';
import { getTrainingManifests } from '../api';
import { formatInteger } from '../format';

// "0.0006" reads better than fixed-2 for LR; show enough significant digits.
function formatLearningRate(lr: number | null): string {
  if (lr === null) return '—';
  if (lr === 0) return '0';
  if (lr >= 0.001) return lr.toFixed(4);
  return lr.toExponential(1);
}

function ManifestRow({ manifest }: { manifest: TrainingManifest }) {
  return (
    <tr>
      <td>{manifest.model}</td>
      <td className="mono">{manifest.config ?? '—'}</td>
      <td className="right mono">{formatInteger(manifest.dim)}</td>
      <td className="right mono">{formatInteger(manifest.n_layers)}</td>
      <td className="right mono">{formatInteger(manifest.n_heads)}</td>
      <td className="right mono">{formatInteger(manifest.seq_len)}</td>
      <td className="right mono">{formatInteger(manifest.batch_size)}</td>
      <td className="right mono">{formatLearningRate(manifest.learning_rate)}</td>
      <td className="right mono">
        {formatInteger(manifest.latest_step)}
        {manifest.max_steps !== null ? ` / ${formatInteger(manifest.max_steps)}` : ''}
      </td>
    </tr>
  );
}

export default function TrainingManifestPanel() {
  const [view, setView] = useState<TrainingManifests | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getTrainingManifests();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setView(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const manifests = view?.manifests ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Training Manifest</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && view === null && <div className="muted">loading…</div>}

      {view !== null &&
        (manifests.length === 0 ? (
          <div className="muted">no training manifests</div>
        ) : (
          <table className="tbl">
            <thead>
              <tr>
                <th>model</th>
                <th>config / tool</th>
                <th className="right">dim</th>
                <th className="right">layers</th>
                <th className="right">heads</th>
                <th className="right">seq</th>
                <th className="right">batch</th>
                <th className="right">lr</th>
                <th className="right">step</th>
              </tr>
            </thead>
            <tbody>
              {manifests.map((m) => (
                <ManifestRow key={m.path} manifest={m} />
              ))}
            </tbody>
          </table>
        ))}

      {view !== null && manifests.length > 0 && (
        <div className="muted">
          {formatInteger(view.count)} model{view.count === 1 ? '' : 's'} with provenance
        </div>
      )}
    </div>
  );
}
