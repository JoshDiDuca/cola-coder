# Feature 18: Specialist Registry

**Status:** Optional | **CLI Flag:** `--use-registry` | **Complexity:** Low-Medium

---

## Overview

A config-driven system for registering, discovering, and loading specialist model checkpoints. A YAML registry file maps each domain to a checkpoint path, model configuration, routing keywords, and a confidence threshold. The registry provides a Python API (`registry.get_specialist`, `registry.list_specialists`, `registry.load_specialist`) and supports hot-reload when the YAML file changes. A CLI menu enables interactive management of registered specialists.

---

## Motivation

As Cola-Coder gains multiple specialist models (React, Next.js, GraphQL, etc.), a structured registry prevents hardcoded paths and enables:

- Decoupled specialist model files from source code
- Easy addition/removal of specialists without code changes
- Per-domain confidence thresholds and metadata
- Hot-reload during development without restarting the process
- CLI tooling for humans to inspect and manage registered specialists
- Foundation for features 19, 20, 22, 23 (routing, hot-swap, ensemble)

---

## Architecture / Design

### Registry YAML Format

```yaml
# configs/specialists.yaml
version: "1.0"
default_specialist: "general_ts"
fallback_threshold: 0.7  # global fallback if no specialist confidence exceeds this

specialists:
  react:
    checkpoint: "checkpoints/specialists/react-small-v1.safetensors"
    model_config: "configs/model/small.yaml"
    description: "React component generation specialist"
    routing_keywords:
      - "useState"
      - "useEffect"
      - "React.FC"
      - "JSX"
    confidence_threshold: 0.65
    enabled: true
    tags: ["frontend", "ui"]
    version: "1.0.0"
    trained_on: "2025-12-01"

  nextjs:
    checkpoint: "checkpoints/specialists/nextjs-small-v1.safetensors"
    model_config: "configs/model/small.yaml"
    description: "Next.js App Router specialist"
    routing_keywords:
      - "getServerSideProps"
      - "getStaticProps"
      - "NextPage"
      - "use client"
    confidence_threshold: 0.70
    enabled: true
    tags: ["frontend", "fullstack"]
    version: "1.0.0"
    trained_on: "2025-12-15"

  graphql:
    checkpoint: "checkpoints/specialists/graphql-small-v1.safetensors"
    model_config: "configs/model/small.yaml"
    description: "GraphQL schema and resolver specialist"
    routing_keywords:
      - "gql`"
      - "useQuery"
      - "GraphQLSchema"
    confidence_threshold: 0.60
    enabled: true
    tags: ["api", "data"]
    version: "1.0.0"
    trained_on: "2026-01-10"

  general_ts:
    checkpoint: "checkpoints/base/medium-v2.safetensors"
    model_config: "configs/model/medium.yaml"
    description: "General TypeScript/JavaScript model (fallback)"
    routing_keywords: []
    confidence_threshold: 0.0  # Always accept
    enabled: true
    tags: ["general", "fallback"]
    version: "2.0.0"
    trained_on: "2026-02-01"
```

### Specialist Entry Schema

```python
# cola_coder/registry/schema.py
from dataclasses import dataclass, field
from typing import Optional
from datetime import date

@dataclass
class SpecialistEntry:
    name: str
    checkpoint: str
    model_config: str
    description: str = ""
    routing_keywords: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    trained_on: Optional[str] = None

    def is_available(self) -> bool:
        """Check if checkpoint file exists on disk."""
        import pathlib
        return pathlib.Path(self.checkpoint).exists()


@dataclass
class RegistryConfig:
    version: str = "1.0"
    default_specialist: str = "general_ts"
    fallback_threshold: float = 0.7
    specialists: dict[str, SpecialistEntry] = field(default_factory=dict)
```

---

## Implementation Steps

### Step 1: YAML Loader + Validator

```python
# cola_coder/registry/loader.py
import yaml
import pathlib
from .schema import SpecialistEntry, RegistryConfig

def load_registry(path: str) -> RegistryConfig:
    """Load and validate the specialists.yaml registry file."""
    data = yaml.safe_load(pathlib.Path(path).read_text())
    specialists = {}
    for name, spec_data in data.get("specialists", {}).items():
        specialists[name] = SpecialistEntry(
            name=name,
            checkpoint=spec_data["checkpoint"],
            model_config=spec_data["model_config"],
            description=spec_data.get("description", ""),
            routing_keywords=spec_data.get("routing_keywords", []),
            confidence_threshold=spec_data.get("confidence_threshold", 0.7),
            enabled=spec_data.get("enabled", True),
            tags=spec_data.get("tags", []),
            version=spec_data.get("version", "1.0.0"),
            trained_on=spec_data.get("trained_on"),
        )
    return RegistryConfig(
        version=data.get("version", "1.0"),
        default_specialist=data.get("default_specialist", "general_ts"),
        fallback_threshold=data.get("fallback_threshold", 0.7),
        specialists=specialists,
    )
```

### Step 2: Registry API Class

```python
# cola_coder/registry/registry.py
import threading
import time
import pathlib
from typing import Optional, Callable
from .loader import load_registry
from .schema import SpecialistEntry, RegistryConfig

class SpecialistRegistry:
    def __init__(
        self,
        registry_path: str,
        auto_reload: bool = True,
        reload_interval: float = 5.0,
    ):
        self._path = registry_path
        self._config: RegistryConfig = load_registry(registry_path)
        self._loaded_models: dict = {}
        self._auto_reload = auto_reload
        self._reload_interval = reload_interval
        self._last_mtime: float = pathlib.Path(registry_path).stat().st_mtime
        self._lock = threading.RLock()
        self._reload_callbacks: list[Callable] = []

        if auto_reload:
            self._start_watcher()

    # --- Core API ---

    def get_specialist(self, domain: str) -> Optional[SpecialistEntry]:
        """Return metadata for a domain, or None if not registered/disabled."""
        with self._lock:
            entry = self._config.specialists.get(domain)
            if entry and entry.enabled:
                return entry
            return None

    def list_specialists(self) -> list[SpecialistEntry]:
        """Return all enabled specialists."""
        with self._lock:
            return [e for e in self._config.specialists.values() if e.enabled]

    def list_all(self) -> list[SpecialistEntry]:
        """Return all specialists including disabled ones."""
        with self._lock:
            return list(self._config.specialists.values())

    def load_specialist(self, domain: str, device: str = "cuda"):
        """Load specialist model into memory and return it."""
        from cola_coder.model.loader import load_model_from_checkpoint
        with self._lock:
            if domain in self._loaded_models:
                return self._loaded_models[domain]
            entry = self.get_specialist(domain)
            if entry is None:
                raise ValueError(f"Specialist '{domain}' not found or disabled")
            if not entry.is_available():
                raise FileNotFoundError(f"Checkpoint not found: {entry.checkpoint}")
            model = load_model_from_checkpoint(entry.checkpoint, entry.model_config, device)
            self._loaded_models[domain] = model
            return model

    def unload_specialist(self, domain: str):
        """Remove specialist from memory."""
        with self._lock:
            if domain in self._loaded_models:
                del self._loaded_models[domain]
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()

    def get_default(self) -> SpecialistEntry:
        """Return the default (fallback) specialist."""
        return self.get_specialist(self._config.default_specialist)

    def get_fallback_threshold(self) -> float:
        return self._config.fallback_threshold

    def register_reload_callback(self, fn: Callable):
        """Called whenever the registry file is reloaded."""
        self._reload_callbacks.append(fn)

    # --- Hot-Reload ---

    def _start_watcher(self):
        def watch():
            while self._auto_reload:
                time.sleep(self._reload_interval)
                self._check_reload()
        t = threading.Thread(target=watch, daemon=True)
        t.start()

    def _check_reload(self):
        try:
            mtime = pathlib.Path(self._path).stat().st_mtime
            if mtime != self._last_mtime:
                self.reload()
                self._last_mtime = mtime
        except OSError:
            pass

    def reload(self):
        """Force reload the registry from disk."""
        with self._lock:
            new_config = load_registry(self._path)
            self._config = new_config
            # Unload any models whose config changed
            for domain in list(self._loaded_models.keys()):
                if domain not in new_config.specialists:
                    self.unload_specialist(domain)
            for cb in self._reload_callbacks:
                cb(new_config)
        print(f"[registry] Reloaded from {self._path}")
```

### Step 3: CLI Menu

```python
# cola_coder/cli.py additions
from rich.table import Table
from rich.prompt import Confirm, Prompt

@app.command()
def registry_status(
    registry_path: str = typer.Option("configs/specialists.yaml", "--registry"),
):
    """Show the current state of the specialist registry."""
    from cola_coder.registry import SpecialistRegistry
    reg = SpecialistRegistry(registry_path, auto_reload=False)
    table = Table(title="Specialist Registry")
    table.add_column("Domain", style="cyan")
    table.add_column("Version")
    table.add_column("Checkpoint")
    table.add_column("Threshold")
    table.add_column("Available", style="green")
    table.add_column("Enabled")

    for entry in reg.list_all():
        available = "[green]YES[/green]" if entry.is_available() else "[red]NO[/red]"
        enabled = "[green]YES[/green]" if entry.enabled else "[dim]NO[/dim]"
        table.add_row(
            entry.name,
            entry.version,
            entry.checkpoint[-40:],  # Truncate path
            str(entry.confidence_threshold),
            available,
            enabled,
        )
    console.print(table)


@app.command()
def registry_add(
    domain: str = typer.Argument(...),
    checkpoint: str = typer.Argument(...),
    model_config: str = typer.Option("configs/model/small.yaml"),
    registry_path: str = typer.Option("configs/specialists.yaml"),
):
    """Register a new specialist checkpoint."""
    import yaml
    path = pathlib.Path(registry_path)
    data = yaml.safe_load(path.read_text())
    data["specialists"][domain] = {
        "checkpoint": checkpoint,
        "model_config": model_config,
        "confidence_threshold": 0.7,
        "enabled": True,
    }
    path.write_text(yaml.dump(data, default_flow_style=False))
    console.print(f"[green]Registered specialist '{domain}'[/green]")
```

---

## Key Files to Modify

- `cola_coder/registry/__init__.py` — new package, exports `SpecialistRegistry`
- `cola_coder/registry/schema.py` — dataclasses for SpecialistEntry, RegistryConfig
- `cola_coder/registry/loader.py` — YAML loading and validation
- `cola_coder/registry/registry.py` — main SpecialistRegistry class
- `cola_coder/cli.py` — `registry-status`, `registry-add`, `registry-remove` commands
- `configs/specialists.yaml` — the registry file itself
- `cola_coder/model/loader.py` — `load_model_from_checkpoint` helper (may already exist)

---

## Testing Strategy

```python
# tests/test_specialist_registry.py
import yaml, tempfile, pathlib

def make_temp_registry(specialists: dict) -> str:
    data = {
        "version": "1.0",
        "default_specialist": "general_ts",
        "fallback_threshold": 0.7,
        "specialists": specialists,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        return f.name

def test_list_specialists():
    path = make_temp_registry({
        "react": {"checkpoint": "ckpt.safetensors", "model_config": "cfg.yaml",
                  "enabled": True, "confidence_threshold": 0.7},
        "prisma": {"checkpoint": "ckpt2.safetensors", "model_config": "cfg.yaml",
                   "enabled": False, "confidence_threshold": 0.7},
    })
    from cola_coder.registry import SpecialistRegistry
    reg = SpecialistRegistry(path, auto_reload=False)
    enabled = reg.list_specialists()
    assert len(enabled) == 1
    assert enabled[0].name == "react"

def test_get_nonexistent_specialist():
    path = make_temp_registry({})
    reg = SpecialistRegistry(path, auto_reload=False)
    assert reg.get_specialist("react") is None

def test_hot_reload(tmp_path):
    # Write initial registry
    reg_path = tmp_path / "specialists.yaml"
    reg_path.write_text(yaml.dump({"version": "1.0", "default_specialist": "general_ts",
                                    "fallback_threshold": 0.7, "specialists": {}}))
    reg = SpecialistRegistry(str(reg_path), auto_reload=False)
    assert len(reg.list_specialists()) == 0
    # Update file
    reg_path.write_text(yaml.dump({"version": "1.0", "default_specialist": "general_ts",
                                    "fallback_threshold": 0.7,
                                    "specialists": {"react": {"checkpoint": "x.safetensors",
                                    "model_config": "c.yaml", "enabled": True,
                                    "confidence_threshold": 0.7}}}))
    reg.reload()
    assert len(reg.list_specialists()) == 1
```

---

## Performance Considerations

- **Lazy loading:** Don't load any model weights at registry initialization — only load on first `load_specialist()` call.
- **Thread safety:** All mutation of `_loaded_models` and `_config` is protected by `threading.RLock`.
- **File watching:** Poll with `stat().st_mtime` rather than `watchdog` library to avoid a dependency. 5-second poll interval is sufficient for development.
- **Cache invalidation:** When a checkpoint path changes in the YAML, automatically unload the old model from `_loaded_models`.
- **VRAM tracking:** Integrate with Feature 22 (hot-swap) to track which models are loaded and enforce VRAM limits.

---

## Dependencies

- `pyyaml` (already standard in Python ML projects)
- `threading` (stdlib)
- Cola-Coder model loader (`load_model_from_checkpoint`)
- Feature 16 (router model) — uses registry to dispatch after routing
- Feature 22 (hot-swap) — extends registry with VRAM-aware loading

---

## Estimated Complexity

| Task                        | Effort   |
|-----------------------------|----------|
| Schema dataclasses          | 0.5h     |
| YAML loader + validation    | 1h       |
| SpecialistRegistry class    | 3h       |
| Hot-reload watcher          | 1h       |
| CLI commands                | 1.5h     |
| Tests                       | 1.5h     |
| specialists.yaml template   | 0.5h     |
| **Total**                   | **~9h**  |

Overall complexity: **Low-Medium** (config management + threading, no ML components)

---

## 2026 Best Practices

- **Pydantic v2 validation:** Consider replacing manual dataclasses with `pydantic.BaseModel` for automatic YAML schema validation with clear error messages.
- **Immutable config snapshots:** Return copies of `SpecialistEntry` objects rather than references; prevents callers from mutating registry state.
- **Structured logging:** Use Python `logging` module (not print) for reload events; integrates with CLI log level settings.
- **Registry versioning:** Track `version` field in YAML; warn if loaded registry version doesn't match expected schema version.
- **Environment variable overrides:** Support `COLA_CODER_REGISTRY_PATH` env var so CI/CD environments can point to different registries without code changes.
- **Checkpoint integrity:** On load, verify safetensors file hash against an optional `sha256` field in the registry entry for tamper detection.
