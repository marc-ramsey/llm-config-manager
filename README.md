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

# Look up a single model (raw — unresolved ${PORT} etc.)
model_cfg = cm.get_model("my-model")

# Runtime env var injection — resolves ${PORT}, ${TOKEN}, etc.
model = cm.get_model("my-model", env_vars={"PORT": "8081", "TOKEN": "abc"})

# List all model names
names = cm.get_models()  # → ["my-model", "other-model"]

# Get backend definition
backend = cm.get_backend("process")

# Assemble command arguments from the model's backend config
result = cm.assemble_command(
    "my-model",
    env_vars={"PORT": "8081"}
)
# result → (['--port', '8081', '--ctx-size', '4096'], "--port 8081 --ctx-size 4096")

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
| `cm.get_vector(key) → dict \| None` | Get a top-level section (e.g. `macros`, `models`). |
| `cm.get_model(name, env_vars=None) → dict \| None` | Resolved model config dict. |
| `cm.get_models() → list[str]` | All available model names. |
| `cm.assemble_command(name, env_vars=None) → (list, str)` | Assembled arguments for a model's backend. |

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
