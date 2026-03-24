import * as vscode from 'vscode';
import { ColaCoderClient } from '../client/ColaCoderClient';
import { HealthStatus } from '../client/types';

export type ServerState = 'disconnected' | 'starting' | 'ready' | 'generating' | 'error';

export interface HealthMonitorEvents {
  onStateChange: vscode.Event<ServerState>;
  onHealthUpdate: vscode.Event<HealthStatus>;
}

export class HealthMonitor implements vscode.Disposable {
  private _state: ServerState = 'disconnected';
  private _timer: NodeJS.Timeout | undefined;
  private _stateEmitter = new vscode.EventEmitter<ServerState>();
  private _healthEmitter = new vscode.EventEmitter<HealthStatus>();
  private _lastHealth: HealthStatus | undefined;

  readonly onStateChange = this._stateEmitter.event;
  readonly onHealthUpdate = this._healthEmitter.event;

  constructor(private client: ColaCoderClient, private intervalMs = 5000) {}

  get state() { return this._state; }
  get lastHealth() { return this._lastHealth; }

  setState(state: ServerState) {
    if (this._state !== state) {
      this._state = state;
      this._stateEmitter.fire(state);
    }
  }

  start() {
    this.stop();
    this._poll(); // immediate first check
    this._timer = setInterval(() => this._poll(), this.intervalMs);
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = undefined; }
  }

  /** Wait for server to become ready, polling every pollMs, up to timeoutMs */
  async waitForReady(timeoutMs = 60000, pollMs = 1000): Promise<boolean> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      try {
        const health = await this.client.healthCheck();
        if (health.status === 'ok') {
          this.setState('ready');
          this._lastHealth = health;
          this._healthEmitter.fire(health);
          return true;
        }
      } catch { /* server not ready yet */ }
      await new Promise(r => setTimeout(r, pollMs));
    }
    this.setState('error');
    return false;
  }

  private async _poll() {
    try {
      const health = await this.client.healthCheck();
      this._lastHealth = health;
      this._healthEmitter.fire(health);
      if (this._state !== 'generating') {
        this.setState('ready');
      }
    } catch {
      if (this._state !== 'starting') {
        this.setState('disconnected');
      }
    }
  }

  dispose() {
    this.stop();
    this._stateEmitter.dispose();
    this._healthEmitter.dispose();
  }
}
