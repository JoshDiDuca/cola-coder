import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, ConfigContent, JsonValue } from '../types';
import { getConfigs, getConfig } from '../api';
import { formatBytes, formatJsonValue } from '../format';

interface ParsedRow {
  key: string;
  value: JsonValue;
}

/** Flatten the top level of a parsed config object into label/value rows. */
function topLevelRows(parsed: JsonValue): ParsedRow[] | null {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return null;
  }
  return Object.entries(parsed).map(([key, value]) => ({ key, value }));
}

function rawText(c: ConfigContent): string {
  const body = c.content ?? '';
  return c.truncated ? `${body}\n…(truncated)` : body;
}

export default function ConfigsPanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [content, setContent] = useState<ConfigContent | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getConfigs();
        if (active) setConfigs(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (cfg: ConfigFile) => {
    setSelectedPath(cfg.path);
    setContentLoading(true);
    setContent(null);

    try {
      const c = await getConfig(cfg.path);
      setContent(c);
    } catch (e) {
      setContent({ error: e instanceof Error ? e.message : String(e) });
    } finally {
      setContentLoading(false);
    }
  }, []);

  const parsedRows = content && !content.error ? topLevelRows(content.parsed ?? null) : null;

  return (
    <div className="card card-wide">
      <div className="card-title">Configs</div>

      {error && <div className="err">{error}</div>}

      {configs.length === 0 && !error ? (
        <div className="muted">no configs found</div>
      ) : (
        <div className="cfg-layout">
          <div className="cfg-list scroll">
            {configs.map((cfg) => (
              <button
                key={cfg.path}
                className={`cfg-item${cfg.path === selectedPath ? ' active' : ''}`}
                onClick={() => void onView(cfg)}
              >
                <span className="cfg-item-name mono">{cfg.name}</span>
                <span className="cfg-item-meta">
                  <span className="muted mono">{cfg.rel}</span>
                  <span className="muted mono">{formatBytes(cfg.size_bytes)}</span>
                </span>
              </button>
            ))}
          </div>

          <div className="cfg-viewer">
            {selectedPath === null ? (
              <div className="muted">select a config to view</div>
            ) : contentLoading ? (
              <div className="muted">loading…</div>
            ) : content === null ? (
              <div className="muted">no content</div>
            ) : content.error ? (
              <div className="err">{content.error}</div>
            ) : parsedRows !== null ? (
              <table className="tbl">
                <tbody>
                  {parsedRows.map((r) => (
                    <tr key={r.key}>
                      <td className="k mono">{r.key}</td>
                      <td className="v">{formatJsonValue(r.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <pre className="pre scroll">{rawText(content)}</pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
