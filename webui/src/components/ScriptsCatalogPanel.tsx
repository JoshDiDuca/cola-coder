import { useCallback, useEffect, useMemo, useState } from 'react';
import { isApiError, type ScriptsCatalog, type ScriptInfo } from '../types';
import { getScriptsCatalog } from '../api';

export default function ScriptsCatalogPanel() {
  const [catalog, setCatalog] = useState<ScriptsCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('');

  const load = useCallback(async () => {
    setError(null);
    try {
      const c = await getScriptsCatalog();
      if (isApiError(c)) {
        setCatalog(null);
        setError(c.error);
      } else {
        setCatalog(c);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const grouped = useMemo<[string, ScriptInfo[]][]>(() => {
    if (!catalog) return [];
    const needle = filter.trim().toLowerCase();
    const match = (s: ScriptInfo): boolean =>
      needle === '' ||
      s.name.toLowerCase().includes(needle) ||
      s.purpose.toLowerCase().includes(needle);

    return catalog.categories
      .map((cat): [string, ScriptInfo[]] => [
        cat,
        catalog.scripts.filter((s) => s.category === cat && match(s)),
      ])
      .filter(([, scripts]) => scripts.length > 0);
  }, [catalog, filter]);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Scripts Catalog</div>
        <button className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {catalog && (
        <>
          <div className="row">
            <span className="muted mono">
              {catalog.count} scripts · {catalog.on_disk} on disk
            </span>
            <input
              className="input"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="filter by name / purpose"
              spellCheck={false}
            />
          </div>

          {grouped.length === 0 ? (
            <div className="muted">no scripts match</div>
          ) : (
            grouped.map(([category, scripts]) => (
              <div key={category}>
                <div className="card-title">{category}</div>
                <table className="tbl">
                  <tbody>
                    {scripts.map((s) => (
                      <tr key={s.name}>
                        <td style={{ width: 18 }}>
                          <span className={`dot ${s.exists ? 'live' : 'dead'}`} />
                        </td>
                        <td className="mono">{s.name}</td>
                        <td className="muted">{s.purpose}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))
          )}
        </>
      )}

      {!catalog && !error && <div className="muted">no scripts catalog</div>}
    </div>
  );
}
