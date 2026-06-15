import { useCallback, useEffect, useMemo, useState } from 'react';
import { isApiError, type FiltersCatalog, type FilterInfo } from '../types';
import { getFiltersCatalog } from '../api';
import { formatInteger } from '../format';

export default function FiltersCatalogPanel() {
  const [catalog, setCatalog] = useState<FiltersCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('');

  const load = useCallback(async () => {
    setError(null);
    try {
      const c = await getFiltersCatalog();
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

  const grouped = useMemo<[string, FilterInfo[]][]>(() => {
    if (!catalog) return [];
    const needle = filter.trim().toLowerCase();
    const match = (f: FilterInfo): boolean =>
      needle === '' ||
      f.name.toLowerCase().includes(needle) ||
      f.purpose.toLowerCase().includes(needle) ||
      f.module.toLowerCase().includes(needle);

    return catalog.categories
      .map((cat): [string, FilterInfo[]] => [
        cat,
        catalog.filters.filter((f) => f.category === cat && match(f)),
      ])
      .filter(([, filters]) => filters.length > 0);
  }, [catalog, filter]);

  const enabledCount = useMemo<number>(
    () => (catalog ? catalog.filters.filter((f) => f.default_enabled).length : 0),
    [catalog],
  );

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Data Filters Catalog</div>
        <button className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {catalog && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(catalog.count)} filters · {formatInteger(enabledCount)} on by
              default · {formatInteger(catalog.categories.length)} categories
            </span>
            <input
              className="input"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="filter by name / purpose / module"
              spellCheck={false}
            />
          </div>

          {grouped.length === 0 ? (
            <div className="muted">no filters match</div>
          ) : (
            grouped.map(([category, filters]) => (
              <div key={category}>
                <div className="card-title">{category}</div>
                <table className="tbl">
                  <tbody>
                    {filters.map((f) => (
                      <tr key={f.name}>
                        <td style={{ width: 18 }}>
                          <span
                            className={`dot ${f.default_enabled ? 'live' : 'dead'}`}
                            title={f.default_enabled ? 'on by default' : 'opt-in'}
                          />
                        </td>
                        <td className="mono">{f.name}</td>
                        <td className="muted">{f.purpose}</td>
                        <td className="muted mono">{f.module}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))
          )}
        </>
      )}

      {!catalog && !error && <div className="muted">no filters catalog</div>}
    </div>
  );
}
