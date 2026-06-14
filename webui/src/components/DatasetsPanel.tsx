import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dataset, Preview, ScoreSummary, JsonValue } from '../types';
import { getDatasets, getPreview, getScores } from '../api';
import { formatBytes, formatFloat, formatInteger, formatJsonValue } from '../format';

const PREVIEW_N = 12;
const PREVIEW_MAX_CHARS = 4000;

function formatPreview(p: Preview): string {
  if (p.error) return `error: ${p.error}`;
  const rows: JsonValue[] = p.preview ?? [];
  const text = rows.map((row) => formatJsonValue(row)).join('\n');
  return text.length > PREVIEW_MAX_CHARS
    ? `${text.slice(0, PREVIEW_MAX_CHARS)}\n…(truncated)`
    : text;
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
        setScores(s);
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
                <td className="right mono">{formatInteger(ds.num_samples)}</td>
                <td className="right mono">{formatBytes(ds.size_bytes)}</td>
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
                n {formatInteger(scores.n)} · mean {formatFloat(scores.mean, 3)} · [
                {formatFloat(scores.min, 3)}, {formatFloat(scores.max, 3)}]
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
