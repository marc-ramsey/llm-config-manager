import tempfile
from pathlib import Path

import pytest


# ── Minimal valid config used across most tests ──────────────────────────────

_VALID_YAML = """\
macros:
  llama-cmd: /usr/bin/llama-vulkan.sh --port ${PORT} -ngl 999
  reasoning-off: --temp 1.0 --top-k 64
  ctx-size: 4096          # integer macro value
models:
  test-model:
    cmd: |
      ${llama-cmd} ${reasoning-off}
      --ctx ${ctx-size}
"""


def _write_config(tmp_path, content):
    """Write *content* to a temporary YAML file and return the path."""
    p = tmp_path / "test_config.yaml"
    p.write_text(content)
    return str(p)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def config_dir(tmp_path):
    """Provide a writable temporary directory and return it."""
    return tmp_path


@pytest.fixture()
def valid_config(config_dir):
    """Path to a minimal, valid YAML config with macros + one model."""
    return _write_config(config_dir, _VALID_YAML)


@pytest.fixture()
def config_manager(valid_config):
    """ConfigManager instance loaded from the valid fixture config."""
    from llm_config_manager.config_manager import ConfigManager
    return ConfigManager(valid_config)
