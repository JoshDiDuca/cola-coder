import { useCallback, useEffect, useState } from 'react';
import type { ActionDef, Job } from '../types';
import { getActions, runAction } from '../api';

interface ActionLauncherPanelProps {
  /** Whether a training run is currently live. Trainer actions are refused
   *  while this is true (mirrors the backend's HTTP 409). */
  trainingAlive: boolean;
  /** Optional callback fired after a successful launch (e.g. to refresh jobs). */
  onLaunched?: () => void;
}

type LoadState = 'loading' | 'ready' | 'error';

/** Split a free-text args string into a trimmed, empty-dropped argv list. */
function parseArgs(raw: string): string[] {
  return raw
    .split(/\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/**
 * Action Launcher — pick an allow-listed CLI action, edit its arguments, and
 * launch it as a background job. Richer alternative to ActionsPanel: a single
 * focused selector + editable args + an explicit trainer guard.
 */
export default function ActionLauncherPanel({
  trainingAlive,
  onLaunched,
}: ActionLauncherPanelProps) {
  const [actions, setActions] = useState<ActionDef[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [argsText, setArgsText] = useState<string>('');

  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const defs = await getActions();
        if (!active) return;
        setActions(defs);
        setLoadState('ready');
      } catch (e) {
        if (!active) return;
        setLoadError(e instanceof Error ? e.message : String(e));
        setLoadState('error');
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const selectAction = useCallback((action: ActionDef): void => {
    setSelectedKey(action.key);
    setArgsText(action.args.join(' '));
    setLaunched(null);
    setLaunchError(null);
  }, []);

  const selected: ActionDef | null =
    selectedKey === null
      ? null
      : (actions.find((a) => a.key === selectedKey) ?? null);

  const blockedByTraining: boolean =
    selected !== null && selected.trainer === true && trainingAlive;

  const onLaunch = useCallback(async (): Promise<void> => {
    if (selected === null || blockedByTraining) return;
    setPending(true);
    setLaunched(null);
    setLaunchError(null);
    try {
      const job = await runAction(selected.key, parseArgs(argsText));
      setLaunched(job);
      onLaunched?.();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // The backend returns HTTP 409 to refuse a trainer action while training
      // is already running; surface that clearly (j<T> embeds "409" in its msg).
      setLaunchError(
        msg.includes('409') ? 'refused: training already running' : msg,
      );
    } finally {
      setPending(false);
    }
  }, [selected, blockedByTraining, argsText, onLaunched]);

  return (
    <div className="card card-wide">
      <div className="card-title">Action Launcher</div>

      {loadState === 'loading' && <div className="muted">loading actions…</div>}
      {loadState === 'error' && (
        <div className="err">{loadError ?? 'failed to load actions'}</div>
      )}

      {loadState === 'ready' && actions.length === 0 && (
        <div className="muted">no actions available</div>
      )}

      {loadState === 'ready' && actions.length > 0 && (
        <>
          <table className="tbl">
            <thead>
              <tr>
                <th>action</th>
                <th className="right">pick</th>
              </tr>
            </thead>
            <tbody>
              {actions.map((a) => (
                <tr key={a.key}>
                  <td>
                    <div>
                      {a.label}
                      {a.trainer === true && (
                        <span className="tag running" style={{ marginLeft: 8 }}>
                          trainer
                        </span>
                      )}
                      {a.gpu === true && (
                        <span className="tag failed" style={{ marginLeft: 8 }}>
                          GPU
                        </span>
                      )}
                    </div>
                    <div className="muted mono">{a.script}</div>
                  </td>
                  <td className="right">
                    <button
                      className={
                        selectedKey === a.key ? 'btn btn-primary' : 'btn'
                      }
                      onClick={() => selectAction(a)}
                      disabled={pending}
                    >
                      {selectedKey === a.key ? '✓ selected' : 'select'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {selected !== null && (
            <div style={{ marginTop: 12 }}>
              <div className="card-title">
                {selected.label}
                {selected.trainer === true && (
                  <span className="tag running" style={{ marginLeft: 8 }}>
                    trainer
                  </span>
                )}
                {selected.gpu === true && (
                  <span className="tag failed" style={{ marginLeft: 8 }}>
                    GPU
                  </span>
                )}
              </div>
              <div className="muted mono" style={{ marginBottom: 8 }}>
                {selected.script}
              </div>

              {selected.gpu === true && (
                <div className="muted" style={{ marginBottom: 8 }}>
                  ⚠ GPU action — contends with training for VRAM if a run is live.
                </div>
              )}

              <label className="muted" htmlFor="action-args">
                arguments (space-separated)
              </label>
              <textarea
                id="action-args"
                className="input"
                style={{ width: '100%', minHeight: 60, marginTop: 4 }}
                value={argsText}
                onChange={(e) => setArgsText(e.target.value)}
                placeholder="args (space-separated)"
                spellCheck={false}
                disabled={pending}
              />

              {blockedByTraining && (
                <div className="err" style={{ marginTop: 8 }}>
                  training already running — refused
                </div>
              )}

              <div style={{ marginTop: 8 }}>
                <button
                  className="btn btn-primary"
                  onClick={() => void onLaunch()}
                  disabled={pending || blockedByTraining}
                >
                  {pending ? '…launching' : '▶ Launch'}
                </button>
              </div>

              {launched !== null && (
                <div className="muted mono" style={{ marginTop: 8 }}>
                  launched {launched.name} ({launched.id}) — {launched.status}
                </div>
              )}
              {launchError !== null && (
                <div className="err" style={{ marginTop: 8 }}>
                  {launchError}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
