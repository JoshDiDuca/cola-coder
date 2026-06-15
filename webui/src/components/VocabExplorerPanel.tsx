import { useCallback, useEffect, useRef, useState } from 'react';
import type { VocabSearchResult, VocabToken } from '../types';
import { isApiError } from '../types';
import { searchVocab } from '../api';
import { formatInteger } from '../format';

// Byte-level BPE pieces carry the 'Ġ' space marker and may embed newlines/tabs.
// Render them visibly so the table stays readable.
function displayPiece(piece: string): string {
  return piece
    .replace(/Ġ/g, '␣')
    .replace(/Ċ/g, '⏎')
    .replace(/\t/g, '⇥')
    .replace(/\n/g, '⏎');
}

function TokenRow({ token }: { token: VocabToken }) {
  return (
    <div className="row">
      <span className="k mono">{token.id}</span>
      <span className="v mono">{displayPiece(token.piece)}</span>
      <span>{token.is_special && <span className="tag done">special</span>}</span>
    </div>
  );
}

export default function VocabExplorerPanel() {
  const [query, setQuery] = useState<string>('');
  const [result, setResult] = useState<VocabSearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const run = useCallback(async (q: string): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const resp = await searchVocab(q);
      if (isApiError(resp)) {
        setError(resp.error);
        setResult(null);
      } else {
        setResult(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load the vocab head once on mount.
  useEffect(() => {
    void run('');
  }, [run]);

  // Debounce search-as-you-type (300ms) so each keystroke doesn't hit the API.
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onChange = useCallback(
    (next: string): void => {
      setQuery(next);
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void run(next);
      }, 300);
    },
    [run],
  );

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Vocab Explorer</div>
        <button className="btn" onClick={() => void run(query)} disabled={loading}>
          {loading ? 'searching…' : 'search'}
        </button>
      </div>

      <input
        className="input"
        type="text"
        placeholder="Search vocabulary (substring, case-sensitive)…"
        value={query}
        onChange={(e) => onChange(e.target.value)}
      />

      {error && <div className="err">{error}</div>}
      {loading && !result && <div className="muted">loading…</div>}

      {result && (
        <>
          <div className="row">
            <span className="k">vocab size</span>
            <span className="v mono">{formatInteger(result.vocab_size)}</span>
          </div>
          <div className="row">
            <span className="k">matches</span>
            <span className="v mono">
              {formatInteger(result.tokens.length)} / {formatInteger(result.total_matches)}
            </span>
          </div>
          {result.truncated && (
            <div className="muted">
              showing first {formatInteger(result.tokens.length)} of{' '}
              {formatInteger(result.total_matches)} matches
            </div>
          )}

          <div className="tbl scroll">
            {result.tokens.length === 0 ? (
              <div className="muted">no matching tokens</div>
            ) : (
              result.tokens.map((token) => <TokenRow key={token.id} token={token} />)
            )}
          </div>

          {result.special_tokens.length > 0 && (
            <>
              <div className="card-title" style={{ marginTop: 12 }}>
                Special tokens ({formatInteger(result.special_tokens.length)})
              </div>
              <div className="tbl scroll">
                {result.special_tokens.map((token) => (
                  <TokenRow key={`sp-${token.id}`} token={token} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
