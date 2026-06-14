import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dataset, Preview, ScoreSummary } from '../types';
import { getDatasets, getPreview, getScores } from '../api';

const PREVIEW_N = 12;
const PREVIEW_MAX_CHARS = 4000;

function humanBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatPreview(p: Preview): string {
  if (p.error) return `error: ${p.error}`;
  const text = JSON.stringify(p, null, 2);
  return text.length > PREVIEW_MAX_CHARS ? `${text.slice(0, PREVIEW_MAX_CHARS)}\n…(truncated)` : text;
}

// getScores may resolve to an error-shaped object instead of a ScoreSummary.
function isScoreError(v: unknown): v is { error: string } {
  return typeof v === 'object' && v !== null && typeof (v as { error?: unknown }).error === 'string';
}

export default function DatasetsPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewText, setPreviewText] = useState<string>('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [scores, setScores] = useState<ScoreSummary | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getDatasets();
        if (active) setDatasets(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (ds: Dataset) => {
    setSelectedPath(ds.path);
    setPreviewLoading(true);
    setScores(null);
    setScoreError(null);
    setPreviewText('');

    try {
      const p = await getPreview(ds.path, PREVIEW_N);
      setPreviewText(formatPreview(p));
    } catch (e) {
      setPreviewText(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPreviewLoading(false);
    }

    if (ds.has_weights) {
      try {
        const s: ScoreSummary = await getScores(ds.path.replace(/\.npy$/, '.weights.npy'));
        if (isScoreError(s)) setScoreError(s.error);
        else setScores(s);
      } catch (e) {
        setScoreError(e instanceof Error ? e.message : String(e));
      }
    }
  }, []);

  const histMax = useMemo(() => {
    if (!scores || scores.histogram.length === 0) return 0;
    return Math.max(...scores.histogram);
  }, [scores]);

  return (
    <div className="card card-wide">
      <div className="card-title">Datasets</div>

      {error && <div className="err">{error}</div>}

      {datasets.length === 0 && !error ? (
        <div className="muted">no datasets in data/</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>kind</th>
              <th className="right">samples</th>
              <th className="right">size</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((ds) => (
              <tr key={ds.path}>
                <td>{ds.name}</td>
                <td className="mono">
                  {ds.kind}
                  {ds.has_weights ? ' ⊕' : ''}
                </td>
                <td className="right mono">
                  {ds.num_samples == null ? '—' : ds.num_samples.toLocaleString()}
                </td>
                <td className="right mono">{humanBytes(ds.size_bytes)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(ds)}>
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
          <div className="pre scroll">
            {previewLoading ? 'loading…' : previewText}
          </div>

          {scoreError && <div className="err">{scoreError}</div>}

          {scores && (
            <>
              <div className="hist">
                {scores.histogram.map((count, i) => (
                  <div
                    key={i}
                    className="b"
                    title={`${scores.bins[i] ?? ''}: ${count}`}
                    style={{ height: `${histMax > 0 ? (count / histMax) * 100 : 0}%` }}
                  />
                ))}
              </div>
              <div className="muted mono">
                n {scores.n.toLocaleString()} · mean {scores.mean.toFixed(3)} · [
                {scores.min.toFixed(3)}, {scores.max.toFixed(3)}]
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
