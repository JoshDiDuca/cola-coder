# Feature 39: Prompt Templates

## Overview

Pre-built prompt templates provide structured, reusable prompts for common code generation
tasks. Templates define a fixed instruction structure with named placeholders that users
fill in. This produces better, more consistent model outputs than freeform prompting —
the model has seen thousands of examples matching the template structure during training.

Templates are defined in YAML, loaded at startup, and accessible via the CLI menu and
HTTP API. Users can also create and save custom templates.

Status: OPTIONAL — enable via `--feature prompt-templates` or CLI menu toggle.

---

## Motivation

- Well-structured prompts consistently outperform vague ones. A template ensures the
  model always receives the right context format.
- Reduces friction: users select a template, fill the blanks, generate. No need to craft
  prompts manually.
- Templates can be shared and versioned alongside the model.
- Custom templates let users encode domain-specific patterns (e.g., "Generate a REST
  endpoint for {{resource}} using FastAPI").

---

## Architecture / Design

### Template Format (YAML)

```yaml
# templates/builtin/write_function.yaml
id: write_function
name: "Write a function"
description: "Generate a complete function implementation"
version: "1.0"
variables:
  - name: description
    type: string
    required: true
    prompt: "What should the function do?"
    example: "compute the factorial of a non-negative integer"
  - name: language
    type: string
    required: false
    default: "Python"
    prompt: "Programming language"
    choices: ["Python", "TypeScript", "JavaScript", "Go"]
  - name: signature
    type: string
    required: false
    prompt: "Function signature (optional)"
    example: "def factorial(n: int) -> int:"
template: |
  Write a {{language}} function that {{description}}.
  {% if signature %}
  Function signature: {{signature}}
  {% endif %}
  Implement the function with proper error handling and docstring.

  ```{{language | lower}}
generation:
  max_new_tokens: 256
  temperature: 0.4
  top_k: 40
  top_p: 0.9
  stop_tokens: ["```"]
```

### All Built-in Templates

```yaml
# templates/builtin/ — list of template IDs

- write_function     # Write a function that...
- fix_bug            # Fix this bug:
- add_types          # Add type annotations to:
- write_tests        # Write tests for:
- explain_code       # Explain this code:
- refactor           # Refactor this code:
- convert_ts         # Convert to TypeScript:
- add_docstring      # Add docstring to:
- optimize_perf      # Optimize this for performance:
- code_review        # Review this code for issues:
```

### Template Engine

```python
# cola_coder/templates/engine.py

import yaml
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemplateVariable:
    name: str
    type: str = "string"
    required: bool = True
    default: Any = None
    prompt_text: str = ""
    example: str = ""
    choices: list[str] = field(default_factory=list)


@dataclass
class Template:
    id: str
    name: str
    description: str
    version: str
    variables: list[TemplateVariable]
    template_str: str
    generation_params: dict = field(default_factory=dict)

    def render(self, variables: dict[str, Any]) -> str:
        """Render template with variable substitution."""
        # Fill defaults for missing optional vars
        filled = {}
        for var in self.variables:
            val = variables.get(var.name, var.default)
            if val is None and var.required:
                raise ValueError(f"Required variable '{var.name}' not provided")
            filled[var.name] = val or ""

        result = self.template_str
        # Simple {{variable}} substitution
        for key, val in filled.items():
            result = result.replace(f"{{{{{key}}}}}", str(val))

        # Handle {% if variable %} ... {% endif %} blocks
        result = self._process_conditionals(result, filled)

        # Handle filters like {{ language | lower }}
        result = self._process_filters(result)

        return result.strip()

    def _process_conditionals(self, text: str, vars: dict) -> str:
        """Process simple {% if var %} ... {% endif %} blocks."""
        pattern = r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}"
        def replace_block(m):
            var_name = m.group(1)
            block = m.group(2)
            return block.strip() if vars.get(var_name) else ""
        return re.sub(pattern, replace_block, text, flags=re.DOTALL)

    def _process_filters(self, text: str) -> str:
        """Process {{ var | filter }} expressions."""
        def apply_filter(m):
            parts = m.group(1).split("|")
            val = parts[0].strip()
            for f in parts[1:]:
                f = f.strip()
                if f == "lower":
                    val = val.lower()
                elif f == "upper":
                    val = val.upper()
                elif f == "title":
                    val = val.title()
        return text  # filters handled in render pass; placeholder for extension


class TemplateRegistry:
    def __init__(self, builtin_dir: Path, user_dir: Path | None = None):
        self.templates: dict[str, Template] = {}
        self._load_directory(builtin_dir)
        if user_dir and user_dir.exists():
            self._load_directory(user_dir)

    def _load_directory(self, directory: Path) -> None:
        for yaml_file in directory.glob("*.yaml"):
            try:
                tmpl = self._parse_yaml(yaml_file)
                self.templates[tmpl.id] = tmpl
            except Exception as e:
                print(f"Warning: failed to load template {yaml_file}: {e}")

    def _parse_yaml(self, path: Path) -> Template:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        variables = [
            TemplateVariable(
                name=v["name"],
                type=v.get("type", "string"),
                required=v.get("required", True),
                default=v.get("default"),
                prompt_text=v.get("prompt", v["name"]),
                example=v.get("example", ""),
                choices=v.get("choices", []),
            )
            for v in data.get("variables", [])
        ]
        return Template(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            variables=variables,
            template_str=data["template"],
            generation_params=data.get("generation", {}),
        )

    def get(self, template_id: str) -> Template | None:
        return self.templates.get(template_id)

    def list_templates(self) -> list[dict]:
        return [
            {"id": t.id, "name": t.name, "description": t.description}
            for t in self.templates.values()
        ]

    def save_custom(self, template: Template, user_dir: Path) -> None:
        user_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "variables": [
                {"name": v.name, "type": v.type, "required": v.required,
                 "default": v.default, "prompt": v.prompt_text}
                for v in template.variables
            ],
            "template": template.template_str,
            "generation": template.generation_params,
        }
        out = user_dir / f"{template.id}.yaml"
        out.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        self.templates[template.id] = template
        print(f"Saved template to {out}")
```

### CLI Integration

```python
# cola_coder/cli/template_menu.py

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from .engine import TemplateRegistry, Template

console = Console()


def run_template_menu(registry: TemplateRegistry, generator) -> None:
    """Interactive template selection and fill-in-the-blanks CLI."""
    templates = registry.list_templates()

    # Display template table
    table = Table(title="Available Templates", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("ID", style="green")
    table.add_column("Name")
    table.add_column("Description")

    for i, t in enumerate(templates, 1):
        table.add_row(str(i), t["id"], t["name"], t["description"])

    console.print(table)

    choice = Prompt.ask("Select template number (or 'q' to quit)")
    if choice.lower() == "q":
        return

    tmpl_data = templates[int(choice) - 1]
    tmpl = registry.get(tmpl_data["id"])
    if not tmpl:
        console.print("[red]Template not found[/red]")
        return

    # Fill variables interactively
    var_values = {}
    for var in tmpl.variables:
        if var.choices:
            console.print(f"  Options: {', '.join(var.choices)}")
        prompt_str = f"  {var.prompt_text}"
        if var.example:
            prompt_str += f" [dim](e.g. {var.example})[/dim]"
        if not var.required:
            prompt_str += f" [dim](optional, default: {var.default or 'none'})[/dim]"

        val = Prompt.ask(prompt_str, default=str(var.default) if var.default else "")
        if val:
            var_values[var.name] = val
        elif not var.required:
            var_values[var.name] = var.default

    try:
        prompt = tmpl.render(var_values)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print("\n[bold]Rendered prompt:[/bold]")
    console.print(f"[dim]{prompt}[/dim]\n")

    if not Confirm.ask("Generate with this prompt?"):
        return

    # Merge template generation params with defaults
    gen_params = {"max_new_tokens": 256, "temperature": 0.7}
    gen_params.update(tmpl.generation_params)

    result = generator.generate(prompt, **gen_params)
    console.print("\n[bold green]Output:[/bold green]")
    console.print(result)
```

### HTTP API Endpoint

```python
# cola_coder/server.py  (template endpoints)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class TemplateRenderRequest(BaseModel):
    template_id: str
    variables: dict[str, str]
    generation_params: dict = {}

class TemplateSaveRequest(BaseModel):
    id: str
    name: str
    description: str
    variables: list[dict]
    template: str
    generation: dict = {}

@app.get("/templates")
def list_templates():
    return app.state.registry.list_templates()

@app.get("/templates/{template_id}")
def get_template(template_id: str):
    tmpl = app.state.registry.get(template_id)
    if not tmpl:
        raise HTTPException(404, f"Template '{template_id}' not found")
    return {
        "id": tmpl.id,
        "name": tmpl.name,
        "variables": [vars(v) for v in tmpl.variables],
        "template": tmpl.template_str,
    }

@app.post("/templates/render")
def render_and_generate(req: TemplateRenderRequest):
    tmpl = app.state.registry.get(req.template_id)
    if not tmpl:
        raise HTTPException(404, f"Template '{req.template_id}' not found")
    try:
        prompt = tmpl.render(req.variables)
    except ValueError as e:
        raise HTTPException(422, str(e))

    params = {**tmpl.generation_params, **req.generation_params}
    result = app.state.generator.generate(prompt, **params)
    return {"prompt": prompt, "completion": result}

@app.post("/templates")
def save_custom_template(req: TemplateSaveRequest):
    from .templates.engine import Template, TemplateVariable
    tmpl = Template(
        id=req.id, name=req.name, description=req.description,
        version="1.0", variables=[], template_str=req.template,
        generation_params=req.generation,
    )
    app.state.registry.save_custom(tmpl, Path("templates/user"))
    return {"status": "saved", "id": req.id}
```

---

## Implementation Steps

1. **Create `templates/builtin/` directory** with 10 YAML template files (one per task).

2. **Create `cola_coder/templates/` package**: `__init__.py`, `engine.py`.

3. **Add `TemplateRegistry` instantiation** in `server.py` startup and CLI startup.

4. **Implement CLI template menu** in `cli/template_menu.py`.

5. **Add HTTP endpoints** to `server.py`: list, get, render, save custom.

6. **User template storage**: `~/.cola_coder/templates/` for user-specific templates.

7. **Template validation**: check for undefined variables, circular references, malformed
   YAML on load. Fail gracefully with a warning rather than crashing.

8. **CLI menu integration**: add "Use prompt template" option that launches `run_template_menu`.

---

## Key Files to Modify

| File | Change |
|---|---|
| `server.py` | Add template endpoints, registry init |
| `cli/menu.py` | Add "Use template" option |
| `config.py` | Add `TemplatesConfig` with directory paths |
| `templates/builtin/*.yaml` | New directory with 10 templates |
| `cola_coder/templates/engine.py` | New file |
| `cli/template_menu.py` | New file |

---

## Testing Strategy

```python
# tests/test_templates.py

def test_template_render_basic():
    tmpl = Template(
        id="test", name="test", description="", version="1.0",
        variables=[TemplateVariable("language", default="Python"),
                   TemplateVariable("description")],
        template_str="Write a {{language}} function that {{description}}.",
    )
    result = tmpl.render({"language": "TypeScript", "description": "sorts an array"})
    assert result == "Write a TypeScript function that sorts an array."

def test_template_render_conditional():
    tmpl = Template(
        id="t", name="t", description="", version="1.0",
        variables=[TemplateVariable("sig", required=False)],
        template_str="Make a function.{% if sig %}\nSignature: {{sig}}{% endif %}",
    )
    with_sig = tmpl.render({"sig": "def f():"})
    assert "Signature" in with_sig
    without_sig = tmpl.render({})
    assert "Signature" not in without_sig

def test_template_missing_required_raises():
    tmpl = Template(
        id="t", name="t", description="", version="1.0",
        variables=[TemplateVariable("code", required=True)],
        template_str="{{code}}",
    )
    import pytest
    with pytest.raises(ValueError, match="Required variable"):
        tmpl.render({})

def test_registry_loads_builtin_templates():
    registry = TemplateRegistry(Path("templates/builtin"))
    templates = registry.list_templates()
    assert len(templates) >= 10
    assert any(t["id"] == "write_function" for t in templates)

def test_custom_template_save_load(tmp_path):
    registry = TemplateRegistry(Path("templates/builtin"), user_dir=tmp_path)
    tmpl = Template(
        id="custom_1", name="My Template", description="test",
        version="1.0", variables=[], template_str="Hello {{name}}",
    )
    registry.save_custom(tmpl, tmp_path)
    loaded = registry.get("custom_1")
    assert loaded is not None
    assert loaded.template_str == "Hello {{name}}"
```

---

## Performance Considerations

- **Template rendering** is pure string manipulation — negligible overhead vs generation.
- **YAML loading** happens once at startup. Cache the registry in memory.
- **Variable validation** on the critical path: skip heavy validation (regex checks)
  during generation requests; only validate on save/load.
- **Template file watching**: optionally use `watchfiles` to auto-reload templates when
  they change on disk during development.

---

## Dependencies

```
pyyaml>=6.0.0     # YAML template parsing (likely already installed)
rich>=13.0.0      # CLI display (already required)
pydantic>=2.0.0   # HTTP request validation (already required by FastAPI)
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Template YAML format + 10 built-in templates | 3 hours |
| TemplateRegistry (load, parse, save) | 3 hours |
| Template rendering engine | 2 hours |
| CLI interactive menu | 2 hours |
| HTTP API endpoints | 2 hours |
| Custom template creation flow | 2 hours |
| Tests | 2 hours |
| **Total** | **~16 hours** |

Complexity rating: **Low-Medium** — well-understood problem (template engines), main
work is building the YAML schema and CLI UX.

---

## 2026 Best Practices

- **Prompt chaining**: allow templates to reference other templates as sub-steps
  (e.g., "explain code" template automatically feeds into "write tests" template).
- **Template versioning with semantic versions**: track which model version each
  template was optimized for. Templates tuned for one checkpoint may not work well
  on another.
- **Few-shot examples in templates**: add an optional `examples` section to templates
  that injects 1–3 example completions as in-context demonstrations.
- **System prompt vs user prompt**: for instruction-tuned models, separate the template
  into `system_prompt` and `user_prompt` sections. System prompts can encode persistent
  context ("You are a senior Python developer").
- **Mustache compatibility**: the `{{variable}}` syntax is Mustache-compatible. Using
  Chevron (Python Mustache) instead of a custom engine avoids reinventing the wheel and
  enables sharing templates with other Mustache-based tools.
