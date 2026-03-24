# VS Code Extension Rules

## Build & Package
- Always run `npx tsc --noEmit` after TypeScript changes
- Always rebuild (`npm run build`) and repackage (`npx vsce package --no-dependencies`) before testing
- Test by installing: `code --install-extension cola-coder-0.1.0.vsix --force`
- Reload VS Code after reinstalling to pick up changes

## Architecture
- The extension MUST activate and register ALL providers even when the server is down
- Never let server connection failures prevent provider registration
- Provider registration happens FIRST, server connection happens in the background
- Use `logger.info/error/warn` for all diagnostic output — never console.log

## Chat Participant
- The `@cola-coder` participant has two modes controlled by `config.baseModelMode`:
  - `true` (default): base code model — no system prompts, raw code completion
  - `false`: instruction-tuned model — system prompts and structured messages
- After instruction-tuning, flip `baseModelMode` to `false` in settings
- The `/generate` command echoes the user's code prefix + wraps output in a code block

## Inline Completions
- Uses `/v1/fim` endpoint (FIM fill-in-the-middle format)
- Requires `client.isConnected() === true` — logs throttled warning when disconnected
- AbortSignal MUST be passed to `client.fim()` to cancel on new keystrokes
- Document selector is built from `config.inlineLanguages` array

## Server Communication
- Default mode is `external` — user starts the FastAPI server manually
- Server must be started with `--cors` flag for the extension
- Client sets `connected = true` only via successful `healthCheck()` calls
- HealthMonitor polls every 5s and auto-detects when server comes online

## Settings
- All settings namespaced under `cola-coder.*`
- Server: `mode`, `serverUrl`, `pythonPath`, `projectRoot`, `checkpoint`, `configFile`
- Inline: `inline.enabled`, `inline.debounceMs`, `inline.maxTokens`, `inline.temperature`, `inline.languages`
- Chat: `chat.temperature`, `chat.maxTokens`, `chat.showThinking`, `chat.baseModelMode`
- Context: `context.enabled`, `context.maxTokens`
