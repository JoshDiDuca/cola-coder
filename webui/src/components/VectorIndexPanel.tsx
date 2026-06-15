import { useCallback, useEffect, useState } from 'react';
import type { IndexStats } from '../types';
import { isApiError } from '../types';
import { getIndexStats } from '../api';
import { formatBytes, formatInteger } from '../format';

// "—" placeholder for unknown / not-applicable scalar fields.
function orDash(value: string | null): string {
  return value === null || value === '' ? '—' : value;
}

export default function VectorIndexPanel() {
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getIndexStats();
      if (isApiError(resp)) {
        setError(resp.error);
        setStats(null);
      } else {
        setStats(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      setError(null);
      setLoading(true);
      try {
        const resp = await getIndexStats();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setStats(null);
        } else {
          setStats(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setStats(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Vector Index Stats</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !stats && <div className="muted">loading…</div>}

      {stats && !stats.exists && (
        <div className="muted">
          No vector index built. Run “Index Repository” to create one.
        </div>
      )}

      {stats && stats.exists && (
        <>
          <div className="row">
            <span className="tag done">indexed</span>
            <span className="v muted">
              {formatInteger(stats.doc_count)} documents
            </span>
          </div>

          <div className="row">
            <span className="k">documents</span>
            <span className="v mono">{formatInteger(stats.doc_count)}</span>
          </div>
          <div className="row">
            <span className="k">chunks</span>
            <span className="v mono">{formatInteger(stats.chunk_count)}</span>
          </div>
          <div className="row">
            <span className="k">embedding model</span>
            <span className="v mono">{orDash(stats.embedding_model)}</span>
          </div>
          <div className="row">
            <span className="k">embedding dim</span>
            <span className="v mono">{formatInteger(stats.embedding_dim)}</span>
          </div>
          <div className="row">
            <span className="k">index size</span>
            <span className="v mono">{formatBytes(stats.size_bytes)}</span>
          </div>
          <div className="row">
            <span className="k">path</span>
            <span className="v mono">{orDash(stats.path)}</span>
          </div>
          <div className="row">
            <span className="k">last updated</span>
            <span className="v mono">{orDash(stats.last_updated)}</span>
          </div>
        </>
      )}
    </div>
  );
}
