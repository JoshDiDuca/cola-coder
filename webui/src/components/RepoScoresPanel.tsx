import { useCallback, useEffect, useState } from 'react';
import type { RepoScore, RepoScoresResult } from '../types';
import { isApiError } from '../types';
import { getRepoScores } from '../api';
import { formatInteger, formatFloat } from '../format';

function ScoreRow({ rank, repo }: { rank: number; repo: RepoScore }) {
  return (
    <tr>
      <td className="right mono muted">{rank}</td>
      <td className="mono">{repo.repo}</td>
      <td className="right mono">{formatFloat(repo.score, 3)}</td>
      <td className="right mono">{repo.stars === null ? '—' : formatInteger(repo.stars)}</td>
      <td>
        {repo.language ? (
          <span className="tag">{repo.language}</span>
        ) : (
          <span className="muted">—</span>
        )}
      </td>
      <td className="mono muted">{repo.license ?? '—'}</td>
    </tr>
  );
}

export default function RepoScoresPanel() {
  const [view, setView] = useState<RepoScoresResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getRepoScores();
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
    let active = true;
    void (async () => {
      setError(null);
      setLoading(true);
      try {
        const resp = await getRepoScores();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setView(null);
        } else {
          setView(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setView(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const repos = view?.repos ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Repo Scoring</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !view && <div className="muted">loading…</div>}

      {view && repos.length === 0 && !error && (
        <div className="muted">no saved repo-score reports found</div>
      )}

      {repos.length > 0 && (
        <>
          <div className="muted mono">
            {formatInteger(view?.count ?? repos.length)} repos · {view?.path}
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th className="right">#</th>
                <th>repo</th>
                <th className="right">score</th>
                <th className="right">stars</th>
                <th>language</th>
                <th>license</th>
              </tr>
            </thead>
            <tbody>
              {repos.map((repo, index) => (
                <ScoreRow key={repo.repo} rank={index + 1} repo={repo} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
