"""Tests for ConfigManager — macro expansion, strict mode, model lookup, and exports."""

import json
from pathlib import Path
import yaml

import pytest

from llm_config_manager.config_manager import ConfigManager


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write(tmp_dir, name, content):
    p = tmp_dir / name
    p.write_text(content)
    return str(p)


NO_MODELS_YAML = """\
models: {}
"""

CIRCULAR_YAML = """\
macros:
  a: "${b}"
  b: "${a}"
models: {}
"""

CHAIN_YAML = """\
macros:
  foo: bar
  baz: "prefix-${foo}-suffix"
models: {}
"""


# ── __init__ / bootstrap ────────────────────────────────────────────────────

class TestInit:
    def test_loads_valid_config(self, config_manager):
        data = config_manager.get_dict()
        assert "macros" in data
        assert "models" in data

    def test_missing_file_raises(self, tmp_path):
        bad = str(tmp_path / "nope.yaml")
        with pytest.raises(RuntimeError, match="Config file not found"):
            ConfigManager(bad)

    def test_invalid_yaml_raises(self, tmp_path):
        p = _write(tmp_path, "bad.yaml", ":\n  :\n    {{{invalid>")
        with pytest.raises(RuntimeError, match="Invalid YAML"):
            ConfigManager(p)

    def test_empty_file_becomes_dict(self, tmp_path):
        p = _write(tmp_path, "empty.yaml", "")
        cm = ConfigManager(p)
        assert cm.get_dict() == {}

    def test_no_macros_section(self, config_dir):
        p = _write(config_dir, "nomac.yaml", NO_MODELS_YAML)
        cm = ConfigManager(p)
        assert cm.data.get("macros") is None


# ── Strict expansion (fail-fast) ────────────────────────────────────────────

# NOTE: Model-specific resolution tests moved to model_arkestra's
#       ModelConfigManager. These remain for the boot-time strict mode.

class TestStrictExpansion:
    def test_explicitly_disable_strict(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  x: foo
models:
  m1:
    cmd: "${x} ${MISSING}"
""")
        cm = ConfigManager(p, strict_expansion=False)
        model = cm._traverse(cm.data["models"]["m1"], {}, strict=False)
        assert "${MISSING}" in model["cmd"]


# ── lenient (non-strict) expansion ──────────────────────────────────────────

# NOTE: Runtime env-var resolution moved to model_arkestra's ModelConfigManager.


# ── Circular macro detection ────────────────────────────────────────────────

class TestCycles:
    def test_circular_macros_raise(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", CIRCULAR_YAML)
        with pytest.raises(RuntimeError, match="Circular macro reference"):
            ConfigManager(p)


# ── Macro chaining ──────────────────────────────────────────────────────────

class TestChainedMacros:
    def test_two_level_chain(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", CHAIN_YAML)
        cm = ConfigManager(p)
        # baz → ${foo} → bar, so baz resolves to "prefix-bar-suffix"
        assert cm.data["macros"]["baz"] == "prefix-bar-suffix"

    def test_non_string_macro_value(self, config_manager):
        """Integer macros (ctx-size: 4096) survive macro expansion as int."""
        assert isinstance(config_manager.data["macros"]["ctx-size"], int)
        assert config_manager.data["macros"]["ctx-size"] == 4096

    def test_integer_macro_in_boot_expansion(self, tmp_path):
        """Integer macro values survive boot-time expansion."""
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  max-conn: 99
  prefix: start
models:
  m1:
    cmd: "${prefix} --max ${max-conn}"
""")
        cm = ConfigManager(p)
        assert cm.data["macros"]["max-conn"] == 99


# ── Whitespace normalization (via _traverse) ────────────────────────

class TestWhitespaceNormalization:
    def test_newlines_collapsed(self, tmp_path):
        yaml_content = """\
macros: {}
models:
  m1:
    cmd: |
      line1
      line2
      line3
"""
        p = _write(tmp_path, "cfg.yaml", yaml_content)
        cm = ConfigManager(p, strict_expansion=False)
        result = cm._traverse(cm.data["models"]["m1"], {}, strict=False)
        assert "\n" not in result["cmd"]
        assert "line1 line2 line3" == result["cmd"]

    def test_tabs_collapsed(self, tmp_path):
        yaml_content = """\
macros: {}
models:
  m1:
    cmd: |
      \t\tpart1\tpart2
"""
        p = _write(tmp_path, "cfg.yaml", yaml_content)
        cm = ConfigManager(p, strict_expansion=False)
        result = cm._traverse(cm.data["models"]["m1"], {}, strict=False)
        assert "\t" not in result["cmd"]
        assert "part1 part2" == result["cmd"]

    def test_multiple_spaces_collapsed(self, tmp_path):
        yaml_content = """\
macros: {}
models:
  m1:
    cmd: "hello     world"
"""
        p = _write(tmp_path, "cfg.yaml", yaml_content)
        cm = ConfigManager(p, strict_expansion=False)
        result = cm._traverse(cm.data["models"]["m1"], {}, strict=False)
        assert "  " not in result["cmd"]
        assert "hello world" == result["cmd"]


# ── get_dict / path-based access ──────────────────────────────────────

class TestAccessors:
    def test_get_dict_returns_full_config(self, config_manager):
        d = config_manager.get_dict()
        assert isinstance(d, dict)
        assert "macros" in d
        assert "models" in d

    def test_path_access_existing_key(self, config_manager):
        m = config_manager["macros"]
        assert isinstance(m, dict)

    def test_path_access_missing_key_raises(self, config_manager):
        with pytest.raises(KeyError):
            _ = config_manager["nonexistent"]

    def test_data_get_missing_returns_none(self, config_manager):
        assert config_manager.data.get("nonexistent") is None


# ── Export helpers ──────────────────────────────────────────────────────────

class TestExports:
    def test_export_yaml_default(self, config_manager, tmp_path):
        out = str(tmp_path / "out.yml")
        config_manager.export(out)
        loaded = yaml.safe_load(Path(out).read_text())
        assert "macros" in loaded
        assert "models" in loaded

    def test_export_empty_models_ok(self, tmp_path):
        cfg = _write(tmp_path, "cfg.yaml", NO_MODELS_YAML)
        cm = ConfigManager(cfg)
        out = str(tmp_path / "out.json")
        cm.export(out)  # shouldn't raise

    def test_export_yaml(self, config_manager, tmp_path):
        out = str(tmp_path / "config.yaml")
        config_manager.export(out)
        loaded = yaml.safe_load(Path(out).read_text())
        assert loaded == config_manager.data

    def test_export_json(self, config_manager, tmp_path):
        out = str(tmp_path / "config.json")
        config_manager.export(out, fmt='json')
        loaded = json.loads(Path(out).read_text())
        assert loaded == config_manager.data

