import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dataset, Preview, ScoreSummary, JsonValue } from '../types';
import { getDatasets, getPreview, getScores } from '../api';
import {
  formatBytes,
  formatFloat,
  formatInteger,
  formatJsonValue,
  formatRelativeTime,
} from '../format';

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

  const selected = useMemo(
    () => datasets.find((d) => d.path === selectedPath) ?? null,
    [datasets, selectedPath],
  );

  return (
    <div className="card card-wide">
      <div className="card-title">Datasets</div>

      {error && <div className="err">{error}</div>}

      {datasets.length === 0 && !error ? (
        <div className="ds-empty muted">
          <div className="ds-empty-title">No datasets found</div>
          <div>Nothing under <span className="mono">data/</span> yet — collect or prepare data to populate this view.</div>
        </div>
      ) : (
        <div className="ds-grid">
          {datasets.map((ds) => {
            const active = ds.path === selectedPath;
            return (
              <div
                key={ds.path}
                className={`ds-card${active ? ' ds-card-active' : ''}`}
              >
                <div className="ds-card-head">
                  <span className="ds-name mono" title={ds.path}>
                    {ds.name}
                  </span>
                  <span className="tag">{ds.kind}</span>
                </div>

                <div className="ds-meta">
                  <span className="ds-meta-item">
                    <span className="k">samples</span>
                    <span className="mono">{formatInteger(ds.num_samples)}</span>
                  </span>
                  <span className="ds-meta-item">
                    <span className="k">size</span>
                    <span className="mono">{formatBytes(ds.size_bytes)}</span>
                  </span>
                  <span className="ds-meta-item">
                    <span className="k">modified</span>
                    <span className="mono">{formatRelativeTime(ds.mtime)}</span>
                  </span>
                </div>

                <div className="ds-card-foot">
                  {ds.has_weights ? (
                    <span className="tag done">weighted</span>
                  ) : (
                    <span className="ds-noweights muted">no weights</span>
                  )}
                  <button className="btn" onClick={() => void onView(ds)}>
                    {active ? 'refresh' : 'preview'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {selected !== null && (
        <div className="ds-detail">
          <div className="card-title ds-detail-title">
            Preview · <span className="mono">{selected.name}</span>
          </div>

          <pre className="pre scroll">{previewLoading ? 'loading…' : previewText}</pre>

          {scoreError && <div className="err">{scoreError}</div>}

          {scores && (
            <div className="ds-scores">
              <div className="card-title">Quality weights</div>
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
              <div className="stat-tiles">
                <div className="stat-tile">
                  <div className="stat-tile-label">n</div>
                  <div className="stat-tile-value mono">{formatInteger(scores.n)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">mean</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.mean, 3)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">min</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.min, 3)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">max</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.max, 3)}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
