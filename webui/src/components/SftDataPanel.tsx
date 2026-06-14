import { useCallback, useEffect, useState } from 'react';
import { isApiError, type SftFile, type SftPreview } from '../types';
import { getSftFiles, getSftPreview } from '../api';

const PREVIEW_N = 10;

function humanBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

function cell(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

export default function SftDataPanel() {
  const [files, setFiles] = useState<SftFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [preview, setPreview] = useState<SftPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const next = await getSftFiles();
      setFiles(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getSftFiles();
        if (active) setFiles(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (f: SftFile) => {
    setSelectedPath(f.path);
    setPreviewLoading(true);
    setPreview(null);
    setPreviewError(null);
    try {
      const p = await getSftPreview(f.path, PREVIEW_N);
      if (isApiError(p)) setPreviewError(p.error);
      else setPreview(p);
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreviewLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">SFT / Instruction Data</div>
        <button className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {files.length === 0 && !error ? (
        <div className="muted">no instruction/SFT datasets yet</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>kind</th>
              <th className="right">records</th>
              <th className="right">size</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {files.map((f) => (
              <tr key={f.path}>
                <td>{f.name}</td>
                <td>
                  <span className="tag">{f.kind}</span>
                </td>
                <td className="right mono">{f.num_records.toLocaleString()}</td>
                <td className="right mono">{humanBytes(f.size_bytes)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(f)}>
                    view
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <>
          {previewError && <div className="err">{previewError}</div>}

          {previewLoading ? (
            <div className="muted">loading…</div>
          ) : (
            preview && (
              <>
                <div className="scroll">
                  <table className="tbl">
                    <thead>
                      <tr>
                        {preview.fields.map((field) => (
                          <th key={field}>{field}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.records.map((rec, i) => (
                        <tr key={i}>
                          {preview.fields.map((field) => (
                            <td key={field} className="mono">
                              {cell(rec[field])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="muted mono">
                  {preview.count.toLocaleString()} records
                  {preview.truncated ? ' · preview truncated' : ''}
                </div>
              </>
            )
          )}
        </>
      )}
    </div>
  );
}
