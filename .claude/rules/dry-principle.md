# DRY Principle — Don't Repeat Yourself

## Before Writing Code: Mandatory Checks

Before implementing any new function, class, or logic:

1. **Search first**: grep for similar function names, patterns, or keywords in `src/`
2. **Check shared utilities**: `data/scorers/utils.py`, `data/scorers/language_detect.py` already exist
3. **Check if a protocol/interface exists**: `ScorerProtocol`, `MalwareScannerProtocol`
4. **Ask**: "Does this logic already exist somewhere in a different form?"

## Shared Utilities (MUST reuse, never reimplement)

| Utility | Location | Use For |
|---------|----------|---------|
| `code_hash(code)` | `data/scorers/utils.py` | MD5 hashing code for caching/dedup |
| `ScoreMapper(thresholds)` | `data/scorers/utils.py` | Mapping counts (errors, warnings) to 0.0-1.0 scores |
| `is_typescript(code, meta)` | `data/scorers/language_detect.py` | Detecting TypeScript code |
| `is_js_ts(code, meta)` | `data/scorers/language_detect.py` | Detecting JavaScript or TypeScript |
| `detect_extension(meta)` | `data/scorers/language_detect.py` | Getting file extension from metadata |
| `TYPESCRIPT_EXTENSIONS` | `data/scorers/language_detect.py` | Set of TS file extensions |
| `JS_TS_EXTENSIONS` | `data/scorers/language_detect.py` | Set of JS+TS file extensions |
| `TscRunner` | `reasoning/rewards/tsc_runner.py` | ALL tsc subprocess execution (never call tsc directly) |
| `SandboxedRunner` | `data/scorers/sandbox.py` | ALL external tool execution on untrusted code |
| `create_hardened_tsconfig()` | `data/scorers/tsconfig_factory.py` | Safe tsconfig.json generation |
| `CredentialScanner` | `data/scorers/credential_scanner.py` | Scanning code for secrets before API calls |
| `ScoringAuditLogger` | `data/scorers/audit.py` | Logging scoring operations |
| `CompositeMalwareScanner` | `security/scanner.py` | Running multiple malware scanners |

## Common DRY Violations to Watch For

### 1. Language/Extension Detection
**Wrong**: Writing inline `if lang in ("typescript", "ts", "tsx"):` checks
**Right**: `from cola_coder.data.scorers.language_detect import is_typescript`

### 2. Score Mapping
**Wrong**: Defining `_SCORE_MAP = [(0, 1.0), (2, 0.9), ...]` with a loop
**Right**: `from cola_coder.data.scorers.utils import ScoreMapper`

### 3. Code Hashing
**Wrong**: `hashlib.md5(code.encode("utf-8")).hexdigest()` inline
**Right**: `from cola_coder.data.scorers.utils import code_hash`

### 4. TypeScript Compilation
**Wrong**: Calling `subprocess.run(["tsc", ...])` anywhere
**Right**: `from cola_coder.reasoning.rewards.tsc_runner import TscRunner`

### 5. External Tool Execution on Untrusted Code
**Wrong**: `subprocess.run(cmd, ...)` directly
**Right**: `SandboxedRunner.run(cmd, cwd=isolated_dir)`

### 6. TSConfig Generation
**Wrong**: Writing `{"compilerOptions": {"strict": true, ...}}` inline
**Right**: `from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig`

## When Duplication IS Acceptable

- **Test code**: Test files can duplicate setup code for clarity
- **Scripts vs library**: CLI argument parsing can be duplicated (different concerns)
- **Performance-critical hot paths**: If abstracting would add overhead in inner loops (document why)
- **Cross-package boundaries**: Sometimes importing across packages creates unwanted coupling — document the tradeoff
- **Different responsibilities**: Two functions that happen to look similar but serve fundamentally different purposes (e.g., a filter vs a scorer)

## Code Review Checklist

When reviewing changes, verify:
- [ ] No new inline MD5 hashing (use `code_hash()`)
- [ ] No new language detection logic (use `language_detect.py`)
- [ ] No new tsc subprocess calls (use `TscRunner`)
- [ ] No new score mapping patterns (use `ScoreMapper`)
- [ ] No new tsconfig generation (use `create_hardened_tsconfig()`)
- [ ] No new unsandboxed subprocess calls on untrusted code (use `SandboxedRunner`)
- [ ] Shared constants used (not string literals for extensions/languages)
