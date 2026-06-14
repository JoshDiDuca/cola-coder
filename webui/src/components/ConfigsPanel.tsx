import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, ConfigContent } from '../types';
import { getConfigs, getConfig } from '../api';

function humanBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatContent(c: ConfigContent): string {
  if (c.error) return `error: ${c.error}`;
  const body = c.content ?? '';
  return c.truncated ? `${body}\n…(truncated)` : body;
}

export default function ConfigsPanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [contentText, setContentText] = useState<string>('');
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
    setContentText('');

    try {
      const c = await getConfig(cfg.path);
      setContentText(formatContent(c));
    } catch (e) {
      setContentText(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setContentLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="card-title">Configs</div>

      {error && <div className="err">{error}</div>}

      {configs.length === 0 && !error ? (
        <div className="muted">no configs found</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>config</th>
              <th className="right">size</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {configs.map((cfg) => (
              <tr key={cfg.path}>
                <td className="mono">{cfg.rel}</td>
                <td className="right mono">{humanBytes(cfg.size_bytes)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(cfg)}>
                    view
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <div className="pre scroll">
          {contentLoading ? 'loading…' : contentText}
        </div>
      )}
    </div>
  );
}
