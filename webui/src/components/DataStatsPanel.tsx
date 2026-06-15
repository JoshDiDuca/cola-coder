import { useCallback, useEffect, useState } from 'react';
import type { DataStats, WeightTier } from '../types';
import { isApiError } from '../types';
import { getDataStats } from '../api';
import { formatFloat, formatInteger } from '../format';

interface StatTile {
  label: string;
  value: string;
}

function buildTiles(stats: DataStats): StatTile[] {
  return [
    { label: 'file size', value: `${formatFloat(stats.file_size_mb, 1)} MB` },
    { label: 'shape', value: stats.shape.join(' × ') },
    { label: 'total tokens', value: formatInteger(stats.total_tokens) },
    {
      label: 'token range',
      value: `${formatInteger(stats.token_min)}–${formatInteger(stats.token_max)}`,
    },
    { label: 'token mean', value: formatFloat(stats.token_mean) },
    { label: 'est. unique', value: formatInteger(stats.est_unique_tokens ?? null) },
  ];
}

export default function DataStatsPanel() {
  const [stats, setStats] = useState<DataStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (active: () => boolean) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await getDataStats();
      if (!active()) return;
      if (isApiError(resp)) {
        setError(resp.error);
        setStats(null);
      } else {
        setStats(resp);
      }
    } catch (e) {
      if (active()) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (active()) setLoading(false);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    void load(() => alive);
    return () => {
      alive = false;
    };
  }, [load]);

  const onRefresh = useCallback(() => {
    void load(() => true);
  }, [load]);

  const tiers: WeightTier[] = stats?.weight_tiers ?? [];
  const hasWeights: boolean = stats?.has_weights ?? false;

  return (
    <div className="card">
      <div className="row" style={{ borderBottom: 'none', padding: 0 }}>
        <span className="card-title">Data Statistics</span>
        <button className="btn" onClick={onRefresh} disabled={loading}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !stats && <div className="muted">loading…</div>}
      {!loading && !stats && !error && (
        <div className="ds-empty muted">
          <div className="ds-empty-title">No prepared data</div>
          <div>Run <span className="mono">prepare_data.py</span> to generate training data, then refresh.</div>
        </div>
      )}

      {stats && (
        <>
          <div className="row">
            <span className="k">data path</span>
            <span className="v mono">{stats.data_path}</span>
          </div>

          <div className="stat-tiles ds-stats-tiles">
            {buildTiles(stats).map((tile) => (
              <div className="stat-tile" key={tile.label}>
                <div className="stat-tile-label">{tile.label}</div>
                <div className="stat-tile-value mono">{tile.value}</div>
              </div>
            ))}
          </div>

          {hasWeights ? (
            <div className="ds-scores">
              <div className="card-title">Quality score distribution</div>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>tier</th>
                    <th className="right">count</th>
                    <th className="right">pct</th>
                    <th>share</th>
                  </tr>
                </thead>
                <tbody>
                  {tiers.map((tier) => (
                    <tr key={tier.label}>
                      <td>{tier.label}</td>
                      <td className="right mono">{formatInteger(tier.count)}</td>
                      <td className="right mono">{formatFloat(tier.pct, 1)}%</td>
                      <td>
                        <div className="bar">
                          <div className="fill" style={{ width: `${tier.pct}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="muted mono">
                mean {formatFloat(stats.weight_mean ?? null, 4)} · std{' '}
                {formatFloat(stats.weight_std ?? null, 4)}
              </div>
            </div>
          ) : (
            <div className="muted">no quality weights (run prepare_data --score)</div>
          )}
        </>
      )}
    </div>
  );
}
