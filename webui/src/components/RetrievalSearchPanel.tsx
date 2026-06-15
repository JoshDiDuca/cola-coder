import { useCallback, useState } from 'react';
import type { RetrievalHit, RetrievalSearchResult } from '../types';
import { isApiError } from '../types';
import { searchRetrieval } from '../api';
import { formatInteger, formatPercent } from '../format';

const DEFAULT_TOP_K = 10;

// Parse the top-k <input> value safely: empty / non-numeric / non-positive
// falls back to the default. Clamped to a sane upper bound to avoid huge fetches.
function parseTopK(raw: string): number {
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_TOP_K;
  return Math.min(n, 100);
}

// A hit's header is its source path, or the index id when source is absent.
function hitLabel(hit: RetrievalHit): string {
  return hit.source ?? hit.id;
}

interface HitRowProps {
  hit: RetrievalHit;
}

function HitRow({ hit }: HitRowProps): JSX.Element {
  return (
    <div className="rsearch-hit">
      <div className="rsearch-hit-head">
        <span className="tag mono rsearch-hit-source">{hitLabel(hit)}</span>
        <span className="tag rsearch-score">{formatPercent(hit.score)}</span>
      </div>
      <pre className="mono rsearch-snippet">{hit.snippet}</pre>
    </div>
  );
}

export default function RetrievalSearchPanel(): JSX.Element {
  const [query, setQuery] = useState<string>('');
  const [topKRaw, setTopKRaw] = useState<string>(String(DEFAULT_TOP_K));
  const [result, setResult] = useState<RetrievalSearchResult | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const run = useCallback(async (q: string, topK: number): Promise<void> => {
    setLoading(true);
    setError(null);
    setLastQuery(q);
    try {
      const resp = await searchRetrieval(q, topK);
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

  const trimmed = query.trim();
  const canSearch = trimmed.length > 0 && !loading;

  const submit = useCallback((): void => {
    if (trimmed.length === 0 || loading) return;
    void run(trimmed, parseTopK(topKRaw));
  }, [trimmed, loading, topKRaw, run]);

  return (
    <div className="card card-wide">
      <div className="card-title">Code search</div>
      <div className="muted rsearch-help">
        Lexical search over the retrieval index (data/vector_index).
      </div>

      <div className="rsearch-controls">
        <input
          className="input rsearch-query"
          type="text"
          placeholder="Search the indexed code…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
        />
        <input
          className="input rsearch-topk"
          type="number"
          min={1}
          max={100}
          title="Top K results"
          value={topKRaw}
          onChange={(e) => setTopKRaw(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
        />
        <button className="btn btn-primary" onClick={submit} disabled={!canSearch}>
          {loading ? 'searching…' : 'Search'}
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !result && <div className="muted">loading…</div>}

      {!loading && result && !result.exists && (
        <div className="muted rsearch-empty">
          No retrieval index built yet. Build one from the CLI (Retrieval &amp; Search → index a
          repo) to enable code search here.
        </div>
      )}

      {!loading && result && result.exists && result.hits.length === 0 && (
        <div className="muted rsearch-empty">
          No matches for &lsquo;{lastQuery}&rsquo; across {formatInteger(result.total_indexed)}{' '}
          indexed chunks.
        </div>
      )}

      {!loading && result && result.exists && result.hits.length > 0 && (
        <>
          <div className="muted rsearch-meta">
            {formatInteger(result.hits.length)} hits · {formatInteger(result.total_indexed)} indexed
            chunks
          </div>
          <div className="rsearch-results">
            {result.hits.map((hit, i) => (
              <HitRow key={`${hit.id}-${i}`} hit={hit} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
