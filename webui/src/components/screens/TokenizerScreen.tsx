import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  TokenizerInfo,
  TokenizerHealthReport,
  TokenizerHealthItem,
  TokenizeResult,
  VocabSearchResult,
  VocabToken,
} from '../../types';
import { isApiError } from '../../types';
import { getTokenizer, getTokenizerHealth, postTokenize, searchVocab } from '../../api';
import { formatInteger } from '../../format';

// ── Tab identity ──────────────────────────────────────────────────────────────
// One polished tabbed screen replaces the old four-panel card grid. Each tab
// owns its own fetch + loading/empty/error state and is lazy-loaded on first open
// (Info loads on mount as the default tab).

type TokTab = 'info' | 'health' | 'tokenize' | 'vocab';

interface TabDef {
  id: TokTab;
  label: string;
}

const TABS: readonly TabDef[] = [
  { id: 'info', label: 'Info' },
  { id: 'health', label: 'Health' },
  { id: 'tokenize', label: 'Tokenize' },
  { id: 'vocab', label: 'Vocab' },
];

// ── Shared display helpers (reused from the source panels) ──────────────────────

// Tokenize pieces: render leading space / tab / newline visibly.
function displayToken(tok: string): string {
  return tok.replace(/^ /, '␣').replace(/\t/g, '⇥').replace(/\n/g, '⏎');
}

// Byte-level BPE vocab pieces carry the 'Ġ'/'Ċ' markers; render them visibly.
function displayPiece(piece: string): string {
  return piece.replace(/Ġ/g, '␣').replace(/Ċ/g, '⏎').replace(/\t/g, '⇥').replace(/\n/g, '⏎');
}

// PASS/FAIL rows reuse the existing tag colour classes; exhaustive over ok.
function healthClass(ok: boolean): string {
  switch (ok) {
    case true:
      return 'tag done';
    case false:
      return 'tag failed';
    default: {
      const _exhaustive: never = ok;
      return _exhaustive;
    }
  }
}

// ── Info tab ────────────────────────────────────────────────────────────────────

function Flag({ label, on }: { label: string; on: boolean }): JSX.Element {
  return (
    <div className="tok-tile">
      <span className="k">{label}</span>
      <span className={`tag ${on ? 'done' : 'failed'}`}>{on ? 'on' : 'off'}</span>
    </div>
  );
}

function InfoTab(): JSX.Element {
  const [info, setInfo] = useState<TokenizerInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
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
      setInfo(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="tokscreen-body">
      <div className="md-toolbar">
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !info && <div className="muted">loading…</div>}
      {!loading && !info && !error && <div className="muted">tokenizer not found</div>}

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

// ── Health tab ──────────────────────────────────────────────────────────────────

function CheckRow({ check }: { check: TokenizerHealthItem }): JSX.Element {
  return (
    <div className="row">
      <span className="k">
        <span className={healthClass(check.ok)}>{check.ok ? 'PASS' : 'FAIL'}</span> {check.name}
      </span>
      <span className="v muted mono">{check.detail}</span>
    </div>
  );
}

function HealthTab(): JSX.Element {
  const [report, setReport] = useState<TokenizerHealthReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getTokenizerHealth();
      if (isApiError(resp)) {
        setError(resp.error);
        setReport(null);
      } else {
        setReport(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setReport(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="tokscreen-body">
      <div className="md-toolbar">
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !report && <div className="muted">loading…</div>}

      {report && (
        <>
          <div className="row">
            <span className={healthClass(report.ok)}>{report.ok ? 'healthy' : 'unhealthy'}</span>
            <span className="v muted">
              {formatInteger(report.passed)}/{formatInteger(report.checks.length)} passed
            </span>
          </div>

          <div className="row">
            <span className="k">path</span>
            <span className="v mono">{report.path}</span>
          </div>
          <div className="row">
            <span className="k">vocab size</span>
            <span className="v mono">{formatInteger(report.vocab_size)}</span>
          </div>

          <div className="tbl">
            {report.checks.length === 0 ? (
              <div className="muted">no checks</div>
            ) : (
              report.checks.map((check) => <CheckRow key={check.name} check={check} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ── Tokenize tab ────────────────────────────────────────────────────────────────

function TokenizeTab(): JSX.Element {
  const [text, setText] = useState<string>('');
  const [result, setResult] = useState<TokenizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showIds, setShowIds] = useState<boolean>(false);

  const onTokenize = useCallback(async (): Promise<void> => {
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

  const onClear = useCallback((): void => {
    setText('');
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="tokscreen-body">
      <div className="md-toolbar">
        <button
          className="btn"
          onClick={onClear}
          disabled={loading || (text === '' && result === null && error === null)}
        >
          clear
        </button>
      </div>

      <textarea
        className="textarea tok-area mono"
        rows={4}
        placeholder="Enter text to tokenize…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="row">
        <button className="btn btn-primary" onClick={() => void onTokenize()} disabled={loading}>
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
          <div className="tok-stat">
            <span className="stat-big">{formatInteger(result.count)}</span>
            <span className="k">tokens</span>
          </div>

          {result.truncated && (
            <div className="muted">
              output truncated to first {formatInteger(result.count)} tokens
            </div>
          )}
          <div className="muted mono tok-path">{result.path}</div>

          <div className="tok-chips scroll">
            {result.tokens.map((tok, i) => (
              <span
                key={i}
                className={`tok-chip mono ${i % 2 === 0 ? 'tok-chip-a' : 'tok-chip-b'}`}
                title={`id ${result.ids[i]}`}
              >
                <span className="tok-chip-piece">{displayToken(tok)}</span>
                <sub className="tok-chip-id">{result.ids[i]}</sub>
              </span>
            ))}
          </div>

          {showIds && <pre className="pre scroll">[{result.ids.join(', ')}]</pre>}
        </>
      )}
    </div>
  );
}

// ── Vocab tab ───────────────────────────────────────────────────────────────────

function TokenRow({ token }: { token: VocabToken }): JSX.Element {
  return (
    <div className="row">
      <span className="k mono">{token.id}</span>
      <span className="v mono">{displayPiece(token.piece)}</span>
      <span>{token.is_special && <span className="tag done">special</span>}</span>
    </div>
  );
}

function VocabTab(): JSX.Element {
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

  // Load the vocab head once when the tab first mounts.
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
    <div className="tokscreen-body">
      <div className="md-toolbar">
        <input
          className="input"
          type="text"
          placeholder="Search vocabulary (substring, case-sensitive)…"
          value={query}
          onChange={(e) => onChange(e.target.value)}
        />
        <button className="btn" onClick={() => void run(query)} disabled={loading}>
          {loading ? 'searching…' : 'search'}
        </button>
      </div>

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

// ── Screen ────────────────────────────────────────────────────────────────────

export default function TokenizerScreen(): JSX.Element {
  const [tab, setTab] = useState<TokTab>('info');

  // Lazy-load: only the opened tab's component is mounted, so its fetch fires on
  // first open. Switching back later re-mounts and refetches — cheap, and keeps
  // each tab's data fresh.
  function renderTab(active: TokTab): JSX.Element {
    switch (active) {
      case 'info':
        return <InfoTab />;
      case 'health':
        return <HealthTab />;
      case 'tokenize':
        return <TokenizeTab />;
      case 'vocab':
        return <VocabTab />;
      default: {
        const _exhaustive: never = active;
        return _exhaustive;
      }
    }
  }

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <h1 className="md-detail-title">Tokenizer</h1>
        <div className="md-toolbar tokscreen-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`btn${tab === t.id ? ' btn-primary' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {renderTab(tab)}
    </div>
  );
}
