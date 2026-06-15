import { useCallback, useEffect, useState } from 'react';
import type { MemoryStats, MemoryEntry } from '../types';
import { isApiError } from '../types';
import { getMemoryStats } from '../api';
import { formatBytes, formatInteger } from '../format';

function EntryRow({ entry }: { entry: MemoryEntry }) {
  return (
    <div className="row">
      <span className="k">
        <span className="tag">{entry.type}</span>{' '}
        <span className="muted mono">{entry.created_at || '—'}</span>
      </span>
      <span className="v mono">{entry.content_preview}</span>
    </div>
  );
}

export default function ProjectMemoryPanel() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getMemoryStats();
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
        const resp = await getMemoryStats();
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
        <div className="card-title">Project Memory</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !stats && <div className="muted">loading…</div>}

      {stats && (
        <>
          <div className="row">
            <span className="k">total entries</span>
            <span className="v mono">{formatInteger(stats.total_entries)}</span>
          </div>
          <div className="row">
            <span className="k">pinned</span>
            <span className="v mono">{formatInteger(stats.pinned)}</span>
          </div>
          <div className="row">
            <span className="k">size</span>
            <span className="v mono">{formatBytes(stats.size_bytes)}</span>
          </div>
          <div className="row">
            <span className="k">types</span>
            <span className="v mono">
              {stats.types.length === 0 ? 'none' : stats.types.join(', ')}
            </span>
          </div>
          <div className="row">
            <span className="k">oldest</span>
            <span className="v mono">{stats.oldest_at ?? '—'}</span>
          </div>
          <div className="row">
            <span className="k">newest</span>
            <span className="v mono">{stats.newest_at ?? '—'}</span>
          </div>

          <div className="tbl">
            {stats.recent_sample.length === 0 ? (
              <div className="muted">no entries</div>
            ) : (
              stats.recent_sample.map((entry) => <EntryRow key={entry.id} entry={entry} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}
