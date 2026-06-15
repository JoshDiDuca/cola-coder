import { useCallback, useEffect, useState } from 'react';
import type { DocFile, DocContent } from '../types';
import { isApiError } from '../types';
import { formatBytes } from '../format';
import { getDocs, getDocContent } from '../api';

// A single rendered line of the lightweight markdown view. Headings are bolded,
// fenced ``` blocks are monospaced; everything else is plain text. A discriminated
// union keeps the renderer exhaustive and free of any/unknown.
type RenderedLine =
  | { kind: 'heading'; level: number; text: string; key: number }
  | { kind: 'code'; text: string; key: number }
  | { kind: 'text'; text: string; key: number };

function renderMarkdown(source: string): RenderedLine[] {
  const out: RenderedLine[] = [];
  let inFence = false;
  source.split('\n').forEach((line, index) => {
    if (line.trimStart().startsWith('```')) {
      inFence = !inFence;
      return; // fence delimiters themselves are not rendered
    }
    if (inFence) {
      out.push({ kind: 'code', text: line, key: index });
      return;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading !== null) {
      out.push({
        kind: 'heading',
        level: heading[1].length,
        text: heading[2],
        key: index,
      });
      return;
    }
    out.push({ kind: 'text', text: line, key: index });
  });
  return out;
}

function MarkdownLine({ line }: { line: RenderedLine }) {
  switch (line.kind) {
    case 'heading':
      return (
        <div className="mono" style={{ fontWeight: 700, marginTop: '0.6em' }}>
          {'#'.repeat(line.level)} {line.text}
        </div>
      );
    case 'code':
      return <div className="mono">{line.text || ' '}</div>;
    case 'text':
      return <div>{line.text || ' '}</div>;
    default: {
      const _exhaustive: never = line;
      return _exhaustive;
    }
  }
}

export default function DocsBrowserPanel() {
  const [docs, setDocs] = useState<DocFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [lines, setLines] = useState<RenderedLine[]>([]);
  const [contentError, setContentError] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadList = useCallback(async () => {
    setError(null);
    try {
      const next = await getDocs();
      if (isApiError(next)) {
        setError(next.error);
        setDocs([]);
      } else {
        setDocs(next.docs);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getDocs();
        if (!active) return;
        if (isApiError(next)) {
          setError(next.error);
        } else {
          setDocs(next.docs);
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (doc: DocFile) => {
    setSelectedPath(doc.path);
    setLoading(true);
    setContentError(null);
    setLines([]);
    setTruncated(false);

    try {
      const c: DocContent | { error: string } = await getDocContent(doc.path);
      if (isApiError(c)) {
        setContentError(c.error);
      } else {
        setLines(renderMarkdown(c.content));
        setTruncated(c.truncated);
      }
    } catch (e) {
      setContentError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Docs Browser</div>
        <button className="btn" onClick={() => void loadList()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {docs.length === 0 && !error ? (
        <div className="muted">no docs found</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>title</th>
              <th>file</th>
              <th className="right">size</th>
              <th className="right">read</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((doc) => (
              <tr key={doc.path}>
                <td>{doc.title}</td>
                <td className="mono">{doc.rel}</td>
                <td className="right mono">{formatBytes(doc.size_bytes)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(doc)}>
                    read
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <>
          {truncated && <div className="muted">…(truncated)</div>}
          <div className="pre scroll">
            {loading ? (
              'loading…'
            ) : contentError !== null ? (
              `error: ${contentError}`
            ) : (
              lines.map((line) => <MarkdownLine key={line.key} line={line} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}
