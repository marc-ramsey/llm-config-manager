# LLM Config Manager

A lightweight Python package for loading YAML configuration files, expanding `${macro}` placeholders, and providing structured access to model/ service configurations with runtime environment variable injection.

## Overview

ConfigManager performs a two-stage expansion:

1. **Boot-time**: Expands all `${macro}` placeholders defined in the `macros` section across the entire config.
2. **Runtime**: `get_model()` resolves remaining placeholders (e.g., `${PORT}`) against a provided `env_vars` dict at call time.

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
if "macros" in cm: …
len(cm)  # number of top-level sections

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
| `cm["key"]` / `cm["path/to/key"]` | Access or set config by ``/``-separated path. |
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

By default (`strict_expansion=True`), unresolved `${KEY}` placeholders during `get_model(env_vars=...)` raise `ValueError`. Set `strict_expansion=False` to let unknown keys pass through as-is:

```python
cm = ConfigManager("config.yaml", strict_expansion=False)
model = cm.get_model("my-model")  # ${PORT} survives as literal text
```
