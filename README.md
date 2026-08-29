# LLM Config Manager

A lightweight Python package for loading YAML configuration files, expanding `${macro}` placeholders, and providing structured access to model/ service configurations with runtime environment variable injection.

## Overview

ConfigManager performs a two-stage expansion:

1. **Boot-time**: Expands all `${macro}` placeholders defined in the `macros` section across the entire config.
2. **Runtime**: Domain subclasses (e.g. ``ModelConfigManager``) resolve remaining placeholders (e.g., ``${PORT}``) against a provided ``env_vars`` dict at call time via ``_traverse()``.

All string values are automatically whitespace-normalized — newlines, tabs, and multiple spaces collapse to single spaces.

## Installation

```bash
pip install llm-config-manager
```

## Usage

```python
from llm_config_manager import ConfigManager

cm = ConfigManager("config.yaml")
full_config = cm.get_dict()

# Path-based access (``/`` separator)
m1 = cm["models/my-model"]              # → nested dict lookup
rocm = cm["backends/rocm"]              # → backend section
cm["models/m2"] = {"checkpoint": "..."} # creates entry in-place

# Iteration and containment
for key in cm: print(key)
if "models" in cm: …
len(cm)  # number of top-level sections

# Set nested keys (auto-creates intermediate dicts)
cm["backends/my-backend"] = {"args": "-ngl 99"}
cm["new/section/key"] = 42

# Delete top-level keys
del cm["macros"]

# Export the full config to a file
cm.export("full_dump.yml")                       # YAML (default)
cm.export("full_dump.json", fmt='json')          # JSON
```

### CLI

```bash
# Export config as YAML
llm-config-manager --input config.yaml --export dump.yml

# Export config as JSON
llm-config-manager --input config.yaml --export models.json
```

## API Reference

| Method | Description |
|--------|-------------|
| `cm.export(path, fmt='yaml')` | Write config to file (`'json'` or `'yaml'`). |
| `cm.get_dict() → dict` | Complete macro-expanded config. |
| `cm.merge(update: dict) → dict` | Deep-merge *update* into ``self.data`` (recursive). |
| `cm["key"]` / `cm["path/to/key"]` | Get config by path (nested traversal via ``/``). |
| `cm["key"] = value` | Set config by path (creates intermediate dicts). |
| `del cm["key"]` | Remove a top-level key. |
| `for k in cm:` | Iterate over top-level keys. |
| `len(cm)` | Number of top-level sections. |
| `"key" in cm` | Check existence of a top-level key. |

## Configuration Format (YAML)

```yaml
models-start-port: 18000

macros:
  llama-args: --port ${PORT} -ngl 999
  ctx-default: 4096

backends:
  default: process
  process:
    args: ${llama-args}

models:
  my-model:
    checkpoint: model/path/to/model.gguf
    args: --ctx-size ${ctx-default} --temp 0.7
```

Backend `args` templates are resolved against the config at runtime.

## Strict Expansion Mode

By default (`strict_expansion=True`), unresolved `${KEY}` placeholders during traversal raise ``ValueError``. Set ``strict_expansion=False`` to let unknown keys pass through as-is.

## Domain Subclasses

Path-based access is generic — it works on any nested dict in the config. For model-specific operations (lookup + runtime env-var resolution), extend ConfigManager:

```python
from llm_config_manager import ConfigManager

class ModelConfigManager(ConfigManager):
    def get_model(self, name, env_vars=None):
        models = self.data.get("models") or {}
        model = models.get(name)
        if model is None:
            return None
        if env_vars is not None:
            return self._traverse(model, env_vars, strict=self.strict_expansion)
        return self._traverse(model, {}, strict=False)

    def get_models(self):
        models = self.data.get("models") or {}
        return list(models.keys())

cm = ModelConfigManager("config.yaml")
names = cm.get_models()          # → ["my-model", …]
model = cm.get_model("my-model")  # macro-expanded, ${PORT} literal
model = cm.get_model("my-model", env_vars={"PORT": "8081"})  # resolved
```

Subclasses have full access to ``_traverse()`` and ``_resolve_string()`` for their own traversal logic.
