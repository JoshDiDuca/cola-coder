import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ActionDef } from '../types';
import { getActions, runAction } from '../api';

export default function ActionsPanel({ onRan }: { onRan?: () => void }) {
  const [actions, setActions] = useState<ActionDef[]>([]);
  const [selectedKey, setSelectedKey] = useState<string>('');
  const [argsText, setArgsText] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const defs = await getActions();
        if (!active) return;
        setActions(defs);
        if (defs.length > 0) {
          setSelectedKey(defs[0].key);
          setArgsText(defs[0].args.join(' '));
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const selected = useMemo(
    () => actions.find((a) => a.key === selectedKey) ?? null,
    [actions, selectedKey],
  );

  const onSelect = useCallback(
    (key: string) => {
      setSelectedKey(key);
      const def = actions.find((a) => a.key === key);
      setArgsText(def ? def.args.join(' ') : '');
      setError(null);
    },
    [actions],
  );

  const onRun = useCallback(async () => {
    if (!selectedKey) return;
    setRunning(true);
    setError(null);
    try {
      const trimmed = argsText.trim();
      const args = trimmed ? trimmed.split(/\s+/) : undefined;
      await runAction(selectedKey, args);
      onRan?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [selectedKey, argsText, onRan]);

  return (
    <div className="card card-wide">
      <div className="card-title">Run an action</div>

      <select
        className="select"
        value={selectedKey}
        onChange={(e) => onSelect(e.target.value)}
        disabled={actions.length === 0}
      >
        {actions.length === 0 ? (
          <option value="">no actions available</option>
        ) : (
          actions.map((a) => (
            <option key={a.key} value={a.key}>
              {a.label}
            </option>
          ))
        )}
      </select>

      {selected && (
        <div className="muted mono">
          {selected.script}
          {selected.args.length > 0 ? ` ${selected.args.join(' ')}` : ''}
        </div>
      )}

      <input
        className="input"
        value={argsText}
        onChange={(e) => setArgsText(e.target.value)}
        placeholder="args (space-separated)"
        spellCheck={false}
      />

      <button
        className="btn btn-primary"
        onClick={() => void onRun()}
        disabled={running || !selectedKey}
      >
        {running ? '…running' : '▶ Run'}
      </button>

      {error && <div className="err">{error}</div>}
    </div>
  );
}
