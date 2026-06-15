import { useCallback, useEffect, useState } from 'react';
import type { TokenizerInfo } from '../types';
import { isApiError } from '../types';
import { getTokenizer } from '../api';
import { formatInteger } from '../format';

function Flag({ label, on }: { label: string; on: boolean }) {
  return (
    <div className="tok-tile">
      <span className="k">{label}</span>
      <span className={`tag ${on ? 'done' : 'failed'}`}>{on ? 'on' : 'off'}</span>
    </div>
  );
}

export default function TokenizerPanel() {
  const [info, setInfo] = useState<TokenizerInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getTokenizer();
      if (isApiError(resp)) {
        setError(resp.error);
        setInfo(null);
      } else {
        setInfo(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Tokenizer</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {!info && !error && <div className="muted">tokenizer not found</div>}

      {info && (
        <>
          <div className="tok-stat">
            <span className="stat-big">{formatInteger(info.vocab_size)}</span>
            <span className="k">vocab size</span>
          </div>

          <div className="muted mono tok-path">{info.path}</div>

          <div className="tok-tiles">
            <div className="tok-tile">
              <span className="k">n_merges</span>
              <span className="v">{formatInteger(info.n_merges)}</span>
            </div>
            <div className="tok-tile">
              <span className="k">model_type</span>
              <span className="v mono">{info.model_type}</span>
            </div>
            <Flag label="digit_splitting" on={info.digit_splitting} />
            <Flag label="has_fim_tokens" on={info.has_fim_tokens} />
          </div>

          <div className="card-title">special tokens</div>
          {info.special_tokens.length === 0 ? (
            <div className="muted">none</div>
          ) : (
            <div className="tok-chips">
              {info.special_tokens.map((tok) => (
                <span key={tok} className="tag mono">
                  {tok}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
