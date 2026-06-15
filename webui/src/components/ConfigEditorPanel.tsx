import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, ConfigContent, ConfigWriteRequest } from '../types';
import { isApiError } from '../types';
import { getConfigs, getConfig, writeConfig } from '../api';
import { formatBytes } from '../format';

// Config editor — load a YAML config, edit it, "Validate & Save".
// The backend validates YAML + path and writes atomically, so an invalid edit
// is refused server-side (never corrupts a config). All HTTP/validation
// failures surface as either a thrown Error (j() throws on non-ok) OR a
// resolved ApiError — both are handled and shown verbatim.

interface SaveOk {
  rel: string;
  bytes: number;
}

/** Resolve the human-friendly relative label for the currently selected path. */
function relForPath(configs: readonly ConfigFile[], path: string): string {
  const match = configs.find((c) => c.path === path);
  return match ? match.rel : path;
}

export default function ConfigEditorPanel(): JSX.Element {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [listError, setListError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [truncated, setTruncated] = useState<boolean>(false);

  // `baseline` is the last-loaded / last-saved content; `text` is the live edit.
  const [baseline, setBaseline] = useState<string>('');
  const [text, setText] = useState<string>('');

  const [saving, setSaving] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveOk, setSaveOk] = useState<SaveOk | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getConfigs();
        if (active) setConfigs(next);
      } catch (e) {
        if (active) setListError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const loadConfig = useCallback(async (path: string): Promise<void> => {
    setSelectedPath(path);
    setSaveError(null);
    setSaveOk(null);
    setTruncated(false);
    setBaseline('');
    setText('');
    if (path === '') return;

    setLoading(true);
    try {
      const c: ConfigContent = await getConfig(path);
      if (c.error) {
        setSaveError(c.error);
        return;
      }
      const body: string = c.content ?? '';
      setBaseline(body);
      setText(body);
      setTruncated(c.truncated === true);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const dirty: boolean = selectedPath !== '' && text !== baseline;
  const canSave: boolean = selectedPath !== '' && dirty && !truncated && !saving && !loading;

  const onSave = useCallback(async (): Promise<void> => {
    if (!canSave) return;
    setSaving(true);
    setSaveError(null);
    setSaveOk(null);
    try {
      const req: ConfigWriteRequest = { path: selectedPath, content: text };
      const resp = await writeConfig(req);
      if (isApiError(resp)) {
        setSaveError(resp.error);
        return;
      }
      // Success: lock in the saved text as the new baseline (no longer dirty).
      setBaseline(text);
      setSaveOk({ rel: relForPath(configs, resp.path), bytes: resp.bytes_written });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [canSave, selectedPath, text, configs]);

  const onRevert = useCallback((): void => {
    setText(baseline);
    setSaveError(null);
    setSaveOk(null);
  }, [baseline]);

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <h1 className="md-detail-title">Config editor</h1>
        <div className="cfgedit-toolbar">
          <select
            className="select"
            value={selectedPath}
            onChange={(e) => void loadConfig(e.target.value)}
            disabled={saving}
          >
            <option value="">select a config…</option>
            {configs.map((cfg) => (
              <option key={cfg.path} value={cfg.path}>
                {cfg.rel}
              </option>
            ))}
          </select>
          {dirty && <span className="cfgedit-dirty mono">● unsaved</span>}
        </div>
      </div>

      {listError && <div className="cfgedit-err">{listError}</div>}

      <p className="muted cfgedit-help">
        Edits are validated as YAML and written atomically. Editing a config does not affect a
        running training job (it loaded its config at launch).
      </p>

      {selectedPath === '' ? (
        <div className="muted">select a config to edit</div>
      ) : loading ? (
        <div className="muted">loading…</div>
      ) : (
        <>
          {truncated && (
            <div className="cfgedit-warn">
              file truncated for display — saving would overwrite with the truncated text
            </div>
          )}

          <textarea
            className="textarea mono cfgedit-text"
            rows={24}
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="cfgedit-actions">
            <button className="btn btn-primary" onClick={() => void onSave()} disabled={!canSave}>
              {saving ? 'Saving…' : 'Save'}
            </button>
            <button className="btn" onClick={onRevert} disabled={!dirty || saving}>
              Revert
            </button>
          </div>

          {saveOk && (
            <div className="cfgedit-ok">
              Saved {formatBytes(saveOk.bytes)} to <span className="mono">{saveOk.rel}</span>
            </div>
          )}
          {saveError && <div className="cfgedit-err">{saveError}</div>}
        </>
      )}
    </div>
  );
}
