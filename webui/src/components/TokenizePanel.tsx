import { useCallback, useState } from 'react';
import type { TokenizeResult } from '../types';
import { isApiError } from '../types';
import { postTokenize } from '../api';

// Render whitespace visibly: leading space → '␣', tab → '⇥', newline → '⏎'.
function displayToken(tok: string): string {
  return tok
    .replace(/^ /, '␣')
    .replace(/\t/g, '⇥')
    .replace(/\n/g, '⏎');
}

export default function TokenizePanel() {
  const [text, setText] = useState<string>('');
  const [result, setResult] = useState<TokenizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showIds, setShowIds] = useState<boolean>(false);

  const onTokenize = useCallback(async () => {
    if (text.trim() === '') {
      setError('enter some text to tokenize');
      setResult(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await postTokenize(text);
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
  }, [text]);

  const onClear = useCallback(() => {
    setText('');
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Tokenize</div>
        <button
          className="btn"
          onClick={onClear}
          disabled={loading || (text === '' && result === null && error === null)}
        >
          clear
        </button>
      </div>

      <textarea
        className="input"
        rows={4}
        placeholder="Enter text to tokenize…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="row">
        <button
          className="btn btn-primary"
          onClick={() => void onTokenize()}
          disabled={loading}
        >
          {loading ? 'tokenizing…' : 'tokenize'}
        </button>
        {result && (
          <button className="btn" onClick={() => setShowIds((v) => !v)}>
            {showIds ? 'hide ids' : 'show ids'}
          </button>
        )}
      </div>

      {error && <div className="err">{error}</div>}

      {result && (
        <>
          <div className="row">
            <span className="k">tokens</span>
            <span className="stat-big">{result.count}</span>
          </div>
          {result.truncated && (
            <div className="muted">output truncated to first {result.count} tokens</div>
          )}
          <div className="muted mono">{result.path}</div>

          <div className="scroll" style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {result.tokens.map((tok, i) => (
              <span key={i} className="tag mono" title={`id ${result.ids[i]}`}>
                {displayToken(tok)}
              </span>
            ))}
          </div>

          {showIds && <pre className="pre scroll">[{result.ids.join(', ')}]</pre>}
        </>
      )}
    </div>
  );
}
