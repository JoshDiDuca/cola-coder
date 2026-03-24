import * as vscode from 'vscode';
import { ChildProcess, spawn } from 'child_process';
import * as net from 'net';
import { ColaCoderClient } from '../client/ColaCoderClient';
import { HealthMonitor } from './HealthMonitor';
import { getConfig } from '../utils/config';
import { logger } from '../utils/logger';

export class ServerManager implements vscode.Disposable {
  private process: ChildProcess | null = null;
  private port = 0;
  private restartCount = 0;
  private maxRestarts = 3;

  constructor(
    private client: ColaCoderClient,
    private healthMonitor: HealthMonitor,
  ) {}

  get isRunning() { return this.process !== null && !this.process.killed; }

  async start(): Promise<void> {
    const config = getConfig();
    if (config.mode !== 'auto') return;

    // Validate required settings
    if (!config.pythonPath) throw new Error('cola-coder.pythonPath not set');
    if (!config.checkpoint) throw new Error('cola-coder.checkpoint not set');
    if (!config.configFile) throw new Error('cola-coder.configFile not set');

    // Find a free port
    this.port = await this._findFreePort();

    // Determine project root
    const projectRoot = config.projectRoot || this._detectProjectRoot();
    if (!projectRoot) throw new Error('Could not determine cola-coder project root');

    // Build args
    const args = [
      'scripts/serve.py',
      '--checkpoint', config.checkpoint,
      '--config', config.configFile,
      '--port', String(this.port),
      '--cors',
    ];

    // Add optional flags
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (workspaceRoot) {
      args.push('--repo', workspaceRoot);
    }
    args.push('--enable-thinking');

    logger.info(`Starting server: ${config.pythonPath} ${args.join(' ')}`);
    this.healthMonitor.setState('starting');

    // Spawn the process
    this.process = spawn(config.pythonPath, args, {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    // Capture output
    this.process.stdout?.on('data', (data: Buffer) => {
      logger.info(data.toString().trimEnd());
    });
    this.process.stderr?.on('data', (data: Buffer) => {
      logger.error(data.toString().trimEnd());
    });

    // Handle unexpected exit
    this.process.on('exit', (code) => {
      logger.info(`Server process exited with code ${code}`);
      this.process = null;
      if (code !== 0 && code !== null) {
        this._handleCrash();
      } else {
        this.healthMonitor.setState('disconnected');
      }
    });

    // Update client URL
    this.client.setBaseUrl(`http://localhost:${this.port}`);

    // Wait for ready
    const ready = await this.healthMonitor.waitForReady(60000);
    if (!ready) {
      throw new Error('Server failed to start within 60 seconds');
    }

    this.restartCount = 0; // Reset on successful start
    logger.info(`Server ready on port ${this.port}`);
  }

  async stop(): Promise<void> {
    if (!this.process) return;

    logger.info('Stopping server...');
    const proc = this.process;
    this.process = null;

    // Try graceful shutdown first (SIGTERM)
    proc.kill('SIGTERM');

    // Wait up to 5 seconds, then force kill
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        if (!proc.killed) proc.kill('SIGKILL');
        resolve();
      }, 5000);
      proc.on('exit', () => { clearTimeout(timeout); resolve(); });
    });

    this.healthMonitor.setState('disconnected');
  }

  async restart(): Promise<void> {
    await this.stop();
    await this.start();
  }

  private async _handleCrash(): Promise<void> {
    this.restartCount++;
    if (this.restartCount <= this.maxRestarts) {
      logger.warn(`Server crashed. Restarting (${this.restartCount}/${this.maxRestarts})...`);
      this.healthMonitor.setState('starting');
      await new Promise(r => setTimeout(r, 5000));
      try { await this.start(); } catch (e) {
        logger.error(`Restart failed: ${e}`);
        this.healthMonitor.setState('error');
      }
    } else {
      logger.error(`Server crashed ${this.maxRestarts} times. Giving up.`);
      this.healthMonitor.setState('error');
      vscode.window.showErrorMessage(
        'Cola-Coder server crashed repeatedly.',
        'Open Logs', 'Configure'
      ).then(choice => {
        if (choice === 'Open Logs') {
          vscode.commands.executeCommand('cola-coder.showLogs');
        } else if (choice === 'Configure') {
          vscode.commands.executeCommand('workbench.action.openSettings', 'cola-coder');
        }
      });
    }
  }

  private async _findFreePort(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, () => {
        const addr = server.address();
        if (addr && typeof addr !== 'string') {
          const port = addr.port;
          server.close(() => resolve(port));
        } else {
          reject(new Error('Failed to find free port'));
        }
      });
      server.on('error', reject);
    });
  }

  private _detectProjectRoot(): string | undefined {
    // Check if extension is inside the cola-coder project
    const extPath = vscode.extensions.getExtension('cola-coder.cola-coder')?.extensionPath;
    if (extPath) {
      // If extension is in vscode-extension/ subdir, parent is project root
      const parent = require('path').dirname(extPath);
      const fs = require('fs');
      if (fs.existsSync(require('path').join(parent, 'scripts', 'serve.py'))) {
        return parent;
      }
    }
    return undefined;
  }

  dispose() {
    // Fire and forget stop
    this.stop().catch(() => {});
  }
}
