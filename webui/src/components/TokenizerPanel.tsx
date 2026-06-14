import { useCallback, useEffect, useState } from 'react';
import type { TokenizerInfo } from '../types';
import { isApiError } from '../types';
import { getTokenizer } from '../api';

function Flag({ label, on }: { label: string; on: boolean }) {
  return (
    <div className="row">
      <span className="k">{label}</span>
      <span>
        <span className={`dot ${on ? 'live' : 'dead'}`} />{' '}
        <span className={`tag ${on ? 'done' : 'failed'}`}>{on ? 'on' : 'off'}</span>
      </span>
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
          <div className="muted mono">{info.path}</div>

          <div className="row">
            <span className="k">vocab_size</span>
            <span className="v">{info.vocab_size.toLocaleString()}</span>
          </div>
          <div className="row">
            <span className="k">n_merges</span>
            <span className="v">{info.n_merges.toLocaleString()}</span>
          </div>
          <div className="row">
            <span className="k">model_type</span>
            <span className="v">{info.model_type}</span>
          </div>

          <Flag label="digit_splitting" on={info.digit_splitting} />
          <Flag label="has_fim_tokens" on={info.has_fim_tokens} />

          <div className="card-title">special tokens</div>
          {info.special_tokens.length === 0 ? (
            <div className="muted">none</div>
          ) : (
            <div className="row" style={{ flexWrap: 'wrap', gap: 6, borderBottom: 'none' }}>
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
