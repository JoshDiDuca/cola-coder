import { useCallback, useEffect, useState } from 'react';
import { isApiError, type ReasoningProblem, type ReasoningProblemSet } from '../types';
import { getReasoningProblems } from '../api';
import { formatInteger } from '../format';

type WhichSet = 'all' | 'builtin' | 'curriculum';

function difficultyClass(difficulty: string): string {
  switch (difficulty) {
    case 'easy':
      return 'tag live';
    case 'medium':
      return 'tag';
    case 'hard':
      return 'tag dead';
    default:
      return 'tag muted';
  }
}

export default function ReasoningProblemsPanel() {
  const [data, setData] = useState<ReasoningProblemSet | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [which, setWhich] = useState<WhichSet>('all');

  const load = useCallback(async (sel: WhichSet) => {
    setError(null);
    try {
      const result = await getReasoningProblems(sel);
      if (isApiError(result)) {
        setData(null);
        setError(result.error);
      } else {
        setData(result);
      }
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load(which);
  }, [load, which]);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Reasoning Problems</div>
        <div className="row">
          <select
            className="input"
            value={which}
            onChange={(e) => setWhich(e.target.value as WhichSet)}
          >
            <option value="all">all (62)</option>
            <option value="builtin">builtin (20)</option>
            <option value="curriculum">curriculum (easy→hard)</option>
          </select>
          <button className="btn" onClick={() => void load(which)}>
            refresh
          </button>
        </div>
      </div>

      {error && <div className="err">{error}</div>}

      {data && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(data.count)} problems · {data.difficulties.join(' / ') || 'no difficulties'}{' '}
              · {data.languages.join(' / ') || 'no languages'}
            </span>
          </div>

          {data.problems.length === 0 ? (
            <div className="muted">no problems in this set</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>id</th>
                  <th>difficulty</th>
                  <th>language</th>
                  <th>tests</th>
                  <th>prompt</th>
                </tr>
              </thead>
              <tbody>
                {data.problems.map((p: ReasoningProblem) => (
                  <tr key={p.id}>
                    <td className="mono">{p.id}</td>
                    <td>
                      <span className={difficultyClass(p.difficulty)}>{p.difficulty}</span>
                    </td>
                    <td>
                      <span className="tag">{p.language}</span>
                    </td>
                    <td>
                      <span className={`dot ${p.has_tests ? 'live' : 'dead'}`} />
                    </td>
                    <td className="muted">{p.prompt_preview}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}

      {!data && !error && <div className="muted">loading reasoning problems…</div>}
    </div>
  );
}
