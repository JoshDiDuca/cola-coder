import { useCallback, useEffect, useState } from 'react';
import type {
  MemoryStats,
  MemoryEntry,
  MemoryExport,
  MemoryFile,
  MemoryAddRequest,
  MemorySearchResult,
  MemoryChunkOut,
  MemoryCompactResult,
} from '../types';
import { isApiError } from '../types';
import {
  getMemoryStats,
  getMemoryExport,
  addMemory,
  searchMemory,
  compactMemory,
} from '../api';
import { formatBytes, formatInteger, formatPercent } from '../format';
import LoadingSpinner from './LoadingSpinner';
import EmptyState from './EmptyState';

// ---------------------------------------------------------------------------
// Local typed helpers
// ---------------------------------------------------------------------------

type MemoryKind = MemoryAddRequest['kind'];

const KIND_OPTIONS: readonly MemoryKind[] = [
  'pattern',
  'error',
  'decision',
  'domain',
  'session',
] as const;

/** Per-kind label + placeholder for the `secondary` field of the add form. */
interface SecondaryFieldHint {
  label: string;
  placeholder: string;
}

function secondaryHint(kind: MemoryKind): SecondaryFieldHint {
  switch (kind) {
    case 'pattern':
      return { label: 'example', placeholder: 'concrete example of the pattern' };
    case 'error':
      return { label: 'fix', placeholder: 'how the error was fixed' };
    case 'decision':
      return { label: 'rationale', placeholder: 'why this decision was made' };
    case 'domain':
      return { label: 'content', placeholder: 'domain knowledge to remember' };
    case 'session':
      return { label: 'domain', placeholder: 'which domain this session covered' };
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

function primaryHint(kind: MemoryKind): SecondaryFieldHint {
  switch (kind) {
    case 'pattern':
      return { label: 'pattern', placeholder: 'the pattern to remember' };
    case 'error':
      return { label: 'error', placeholder: 'the mistake or error encountered' };
    case 'decision':
      return { label: 'decision', placeholder: 'the decision that was made' };
    case 'domain':
      return { label: 'topic', placeholder: 'the domain topic' };
    case 'session':
      return { label: 'summary', placeholder: 'session summary' };
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

/** Convert a thrown value or a resolved ApiError into a display string. */
function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// ---------------------------------------------------------------------------
// Stats sub-section
// ---------------------------------------------------------------------------

function EntryRow({ entry }: { entry: MemoryEntry }) {
  return (
    <div className="row">
      <span className="k">
        <span className="tag">{entry.type}</span>{' '}
        <span className="muted mono">{entry.created_at || '—'}</span>
      </span>
      <span className="v mono">{entry.content_preview}</span>
    </div>
  );
}

function StatsSection({ stats }: { stats: MemoryStats }) {
  return (
    <>
      <div className="row">
        <span className="k">total entries</span>
        <span className="v mono">{formatInteger(stats.total_entries)}</span>
      </div>
      <div className="row">
        <span className="k">pinned</span>
        <span className="v mono">{formatInteger(stats.pinned)}</span>
      </div>
      <div className="row">
        <span className="k">size</span>
        <span className="v mono">{formatBytes(stats.size_bytes)}</span>
      </div>
      <div className="row">
        <span className="k">types</span>
        <span className="v mono">
          {stats.types.length === 0 ? 'none' : stats.types.join(', ')}
        </span>
      </div>
      <div className="row">
        <span className="k">oldest</span>
        <span className="v mono">{stats.oldest_at ?? '—'}</span>
      </div>
      <div className="row">
        <span className="k">newest</span>
        <span className="v mono">{stats.newest_at ?? '—'}</span>
      </div>

      <div className="tbl">
        {stats.recent_sample.length === 0 ? (
          <div className="muted">no entries</div>
        ) : (
          stats.recent_sample.map((entry) => <EntryRow key={entry.id} entry={entry} />)
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Add-entry sub-section
// ---------------------------------------------------------------------------

interface AddEntrySectionProps {
  onAdded: (stats: MemoryStats) => void;
}

function AddEntrySection({ onAdded }: AddEntrySectionProps) {
  const [kind, setKind] = useState<MemoryKind>('pattern');
  const [primary, setPrimary] = useState<string>('');
  const [secondary, setSecondary] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const secHint = secondaryHint(kind);
  const priHint = primaryHint(kind);
  const canSubmit = primary.trim().length > 0 && !submitting;

  const submit = useCallback(async (): Promise<void> => {
    setError(null);
    setSubmitting(true);
    try {
      const trimmedSecondary = secondary.trim();
      const req: MemoryAddRequest = {
        kind,
        primary: primary.trim(),
        ...(trimmedSecondary.length > 0 ? { secondary: trimmedSecondary } : {}),
      };
      const resp = await addMemory(req);
      if (isApiError(resp)) {
        setError(resp.error);
        return;
      }
      setPrimary('');
      setSecondary('');
      onAdded(resp);
    } catch (e) {
      setError(errMessage(e));
    } finally {
      setSubmitting(false);
    }
  }, [kind, primary, secondary, onAdded]);

  return (
    <div className="tbl">
      <div className="card-title">Add entry</div>

      <div className="row">
        <span className="k">kind</span>
        <select
          className="select"
          value={kind}
          disabled={submitting}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
            setKind(e.target.value as MemoryKind)
          }
        >
          {KIND_OPTIONS.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </div>

      <div className="row">
        <span className="k">{priHint.label}</span>
        <input
          className="input"
          type="text"
          value={primary}
          placeholder={priHint.placeholder}
          disabled={submitting}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPrimary(e.target.value)}
        />
      </div>

      <div className="row">
        <span className="k">{secHint.label}</span>
        <input
          className="input"
          type="text"
          value={secondary}
          placeholder={secHint.placeholder}
          disabled={submitting}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSecondary(e.target.value)}
        />
      </div>

      {error && <div className="field-error">{error}</div>}

      <div className="md-toolbar">
        <button
          className="btn btn-primary"
          onClick={() => void submit()}
          disabled={!canSubmit}
        >
          {submitting ? 'saving…' : 'save entry'}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search sub-section
// ---------------------------------------------------------------------------

function SearchResultRow({ chunk }: { chunk: MemoryChunkOut }) {
  return (
    <div className="tbl">
      <div className="row">
        <span className="k">
          <span className="tag">{chunk.section}</span>{' '}
          <span className="muted mono">{chunk.source_file}</span>
        </span>
        <span className="v mono">{formatPercent(chunk.relevance_score)}</span>
      </div>
      <pre className="pre scroll">{chunk.content}</pre>
    </div>
  );
}

function SearchSection() {
  const [query, setQuery] = useState<string>('');
  const [searching, setSearching] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MemorySearchResult | null>(null);

  const canSearch = query.trim().length > 0 && !searching;

  const run = useCallback(async (): Promise<void> => {
    setError(null);
    setSearching(true);
    try {
      const resp = await searchMemory({ query: query.trim() });
      if (isApiError(resp)) {
        setError(resp.error);
        setResult(null);
        return;
      }
      setResult(resp);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setSearching(false);
    }
  }, [query]);

  return (
    <div className="tbl">
      <div className="card-title">Search</div>

      <div className="row">
        <input
          className="input"
          type="text"
          value={query}
          placeholder="search memory…"
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
        />
        <button className="btn btn-primary" onClick={() => void run()} disabled={!canSearch}>
          {searching ? 'searching…' : 'search'}
        </button>
      </div>

      {error && <div className="field-error">{error}</div>}

      {result && (
        <>
          <div className="muted mono">
            {result.results.length} result{result.results.length === 1 ? '' : 's'} for “
            {result.query}”
          </div>
          {result.results.length === 0 ? (
            <div className="muted">no matches</div>
          ) : (
            result.results.map((chunk, i) => (
              <SearchResultRow key={`${chunk.source_file}:${chunk.section}:${i}`} chunk={chunk} />
            ))
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// View (export) sub-section
// ---------------------------------------------------------------------------

function ExportFileBlock({ file }: { file: MemoryFile }) {
  const [open, setOpen] = useState<boolean>(false);
  return (
    <div className="tbl">
      <div className="row">
        <button className="btn" onClick={() => setOpen((v) => !v)}>
          {open ? '▾' : '▸'} {file.type}
        </button>
        <span className="v mono">
          {formatInteger(file.entry_count)} entries
          {file.truncated ? ' (truncated)' : ''}
        </span>
      </div>
      {open && <pre className="pre scroll">{file.content}</pre>}
    </div>
  );
}

function ViewSection() {
  const [data, setData] = useState<MemoryExport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getMemoryExport();
      setData(resp);
    } catch (e) {
      setError(errMessage(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="tbl">
      <div className="row">
        <div className="card-title">View export</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          {loading ? 'loading…' : data ? 'reload' : 'load export'}
        </button>
      </div>

      {error && <div className="field-error">{error}</div>}
      {loading && !data && <LoadingSpinner label="Loading export…" />}

      {data &&
        (!data.initialized ? (
          <EmptyState
            title="Memory is empty"
            hint="The memory store has not been initialised yet. Add an entry above to create it."
          />
        ) : data.files.length === 0 ? (
          <div className="muted">no theme files</div>
        ) : (
          data.files.map((file) => <ExportFileBlock key={file.name} file={file} />)
        ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compact sub-section
// ---------------------------------------------------------------------------

function CompactSection() {
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MemoryCompactResult | null>(null);

  const run = useCallback(async (): Promise<void> => {
    setError(null);
    setRunning(true);
    try {
      const resp = await compactMemory();
      if (isApiError(resp)) {
        setError(resp.error);
        setResult(null);
        return;
      }
      setResult(resp);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setRunning(false);
    }
  }, []);

  return (
    <div className="tbl">
      <div className="row">
        <div className="card-title">Compact</div>
        <button className="btn" onClick={() => void run()} disabled={running}>
          {running ? 'compacting…' : 'compact now'}
        </button>
      </div>

      {error && <div className="field-error">{error}</div>}

      {result && (
        <div className="row">
          <span className="k">removed</span>
          <span className="v mono">
            {formatInteger(result.removed_total)} duplicate
            {result.removed_total === 1 ? '' : 's'}
            {result.removed.length > 0
              ? ` (${result.removed
                  .map((f) => `${f.name}: ${formatInteger(f.removed)}`)
                  .join(', ')})`
              : ''}
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main workbench
// ---------------------------------------------------------------------------

export default function ProjectMemoryPanel() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getMemoryStats();
      if (isApiError(resp)) {
        setError(resp.error);
        setStats(null);
      } else {
        setStats(resp);
      }
    } catch (e) {
      setError(errMessage(e));
      setStats(null);
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
        const resp = await getMemoryStats();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setStats(null);
        } else {
          setStats(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(errMessage(e));
        setStats(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Add-entry returns refreshed stats directly; adopt it (no extra round-trip).
  const handleAdded = useCallback((next: MemoryStats): void => {
    setStats(next);
    setError(null);
  }, []);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Project Memory</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !stats && <LoadingSpinner label="Loading memory…" />}

      {stats && <StatsSection stats={stats} />}

      <AddEntrySection onAdded={handleAdded} />
      <SearchSection />
      <ViewSection />
      <CompactSection />
    </div>
  );
}
