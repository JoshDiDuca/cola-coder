import { useCallback, useMemo, useState } from 'react';
import type { Checkpoint, Job } from '../types';
import { runAction } from '../api';

/**
 * The non-interactive `--action` values accepted by `scripts/export_model.py`.
 * These are the EXACT argparse choices (see the script's `_ACTION_MAP`):
 *   gguf-f16 | gguf-q8 | gguf-q4 | ollama | quantize | benchmark.
 */
type ExportAction =
  | 'gguf-f16'
  | 'gguf-q8'
  | 'gguf-q4'
  | 'ollama'
  | 'quantize'
  | 'benchmark';

interface ExportFormatOption {
  readonly action: ExportAction;
  readonly label: string;
}

/** Selectable export formats, in menu order matching the CLI's interactive menu. */
const EXPORT_FORMATS: readonly ExportFormatOption[] = [
  { action: 'gguf-f16', label: 'GGUF — F16 (full precision)' },
  { action: 'gguf-q8', label: 'GGUF — Q8_0 (8-bit)' },
  { action: 'gguf-q4', label: 'GGUF — Q4_K_M (4-bit)' },
  { action: 'ollama', label: 'Ollama Modelfile (F16 GGUF + Modelfile)' },
  { action: 'quantize', label: 'Quantize — dynamic INT8 (CPU inference)' },
  { action: 'benchmark', label: 'Benchmark — original vs INT8' },
];

type LaunchState = 'ready' | 'launching' | 'launched' | 'error';

function checkpointLabel(ckpt: Checkpoint): string {
  return `${ckpt.model} / ${ckpt.name} @ ${ckpt.step.toLocaleString()}`;
}

/**
 * Derive the YAML config path from a checkpoint's model name. Checkpoints live
 * under `checkpoints/{model}/…` and the project convention is one config per
 * model size at `configs/{model}.yaml` (tiny, small, medium, 4080_max, …). The
 * derived value is shown in an editable field so the user can correct it for
 * non-standard layouts (e.g. an `_sft` checkpoint whose config differs).
 */
function defaultConfigPath(model: string): string {
  // `model` may carry a stage suffix (e.g. "tiny_sft"); the base size config is
  // the closest sane default. Strip a single trailing "_<suffix>" if present.
  const base = model.replace(/_(sft|moe|router|reasoning)$/u, '');
  return `configs/${base}.yaml`;
}

interface ExportModelPanelProps {
  /** All known checkpoints (App passes `snap?.checkpoints ?? []`). */
  checkpoints: Checkpoint[];
}

/**
 * Export Model launcher — pick a checkpoint + export format and launch
 * `scripts/export_model.py` as a background job via the `export_model` action.
 *
 * Loads model weights on CPU (not a GPU trainer), so there is intentionally NO
 * `trainingAlive` guard: exporting can run alongside a live training run. It can
 * take a while for larger checkpoints.
 */
export default function ExportModelPanel({ checkpoints }: ExportModelPanelProps) {
  const rows = useMemo(
    () => [...checkpoints].sort((a, b) => b.step - a.step),
    [checkpoints],
  );

  const [checkpointPath, setCheckpointPath] = useState<string>('');
  const [action, setAction] = useState<ExportAction>('gguf-f16');
  const [configPath, setConfigPath] = useState<string>('');
  const [configTouched, setConfigTouched] = useState<boolean>(false);
  const [outputDir, setOutputDir] = useState<string>('');

  const [state, setState] = useState<LaunchState>('ready');
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selected: Checkpoint | null = useMemo(
    () => rows.find((c) => c.path === checkpointPath) ?? null,
    [rows, checkpointPath],
  );

  // The config field auto-fills from the chosen checkpoint's model until the
  // user edits it (then their value sticks).
  const effectiveConfigPath: string =
    configTouched || configPath !== ''
      ? configPath
      : selected !== null
        ? defaultConfigPath(selected.model)
        : '';

  const onSelectCheckpoint = useCallback(
    (path: string): void => {
      setCheckpointPath(path);
      setState('ready');
      setJob(null);
      setError(null);
      if (!configTouched) {
        const ckpt = rows.find((c) => c.path === path) ?? null;
        setConfigPath(ckpt !== null ? defaultConfigPath(ckpt.model) : '');
      }
    },
    [rows, configTouched],
  );

  const canLaunch: boolean =
    checkpointPath !== '' &&
    effectiveConfigPath.trim() !== '' &&
    state !== 'launching';

  const onLaunch = useCallback(async (): Promise<void> => {
    if (checkpointPath === '' || effectiveConfigPath.trim() === '') return;
    setState('launching');
    setJob(null);
    setError(null);

    // Exact flags from scripts/export_model.py argparse:
    //   --checkpoint <path> --config <yaml> --action <choice> [--output-dir <dir>]
    const args: string[] = [
      '--checkpoint',
      checkpointPath,
      '--config',
      effectiveConfigPath.trim(),
      '--action',
      action,
    ];
    const trimmedOutput = outputDir.trim();
    if (trimmedOutput !== '') {
      args.push('--output-dir', trimmedOutput);
    }

    try {
      const launched = await runAction('export_model', args);
      setJob(launched);
      setState('launched');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState('error');
    }
  }, [checkpointPath, effectiveConfigPath, action, outputDir]);

  return (
    <div className="card card-wide">
      <div className="card-title">Export Model</div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Convert a checkpoint to GGUF / Ollama / quantized format. Runs on CPU
        (no GPU trainer) — exporting larger checkpoints can take a while.
      </div>

      {rows.length === 0 ? (
        <div className="muted">no checkpoints to export</div>
      ) : (
        <>
          <div className="row" style={{ borderBottom: 'none' }}>
            <label className="k" htmlFor="export-checkpoint">
              checkpoint
            </label>
            <select
              id="export-checkpoint"
              className="select"
              value={checkpointPath}
              onChange={(e) => onSelectCheckpoint(e.target.value)}
              style={{ flex: 1, minWidth: 180 }}
              disabled={state === 'launching'}
            >
              <option value="">select checkpoint…</option>
              {rows.map((ckpt) => (
                <option key={ckpt.path} value={ckpt.path}>
                  {checkpointLabel(ckpt)}
                </option>
              ))}
            </select>
          </div>

          <div className="row" style={{ borderBottom: 'none' }}>
            <label className="k" htmlFor="export-format">
              format
            </label>
            <select
              id="export-format"
              className="select"
              value={action}
              onChange={(e) => setAction(e.target.value as ExportAction)}
              style={{ flex: 1, minWidth: 180 }}
              disabled={state === 'launching'}
            >
              {EXPORT_FORMATS.map((f) => (
                <option key={f.action} value={f.action}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>

          <div className="row" style={{ borderBottom: 'none' }}>
            <label className="k" htmlFor="export-config">
              config
            </label>
            <input
              id="export-config"
              className="input mono"
              value={effectiveConfigPath}
              onChange={(e) => {
                setConfigTouched(true);
                setConfigPath(e.target.value);
              }}
              placeholder="configs/tiny.yaml"
              spellCheck={false}
              style={{ flex: 1, minWidth: 180 }}
              disabled={state === 'launching'}
            />
          </div>

          <div className="row" style={{ borderBottom: 'none' }}>
            <label className="k" htmlFor="export-output">
              output dir
            </label>
            <input
              id="export-output"
              className="input mono"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="(default: <checkpoint_parent>/exports/)"
              spellCheck={false}
              style={{ flex: 1, minWidth: 180 }}
              disabled={state === 'launching'}
            />
          </div>

          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-primary"
              onClick={() => void onLaunch()}
              disabled={!canLaunch}
            >
              {state === 'launching' ? '…launching' : '▶ Export'}
            </button>
          </div>

          {state === 'launched' && job !== null && (
            <div className="muted mono" style={{ marginTop: 8 }}>
              launched {job.name} ({job.id}) — {job.status}
            </div>
          )}
          {state === 'error' && error !== null && (
            <div className="err" style={{ marginTop: 8 }}>
              {error}
            </div>
          )}
        </>
      )}
    </div>
  );
}
