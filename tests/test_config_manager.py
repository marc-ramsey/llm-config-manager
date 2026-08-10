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
        assert cm.get_vector("macros") is None


# ── Strict expansion (fail-fast) ────────────────────────────────────────────

class TestStrictExpansion:
    def test_unresolved_placeholder_during_model_get(self, config_manager):
        """${PORT} is a boot-time placeholder that survives as literal text;
        passing env_vars without it should raise in strict mode."""
        with pytest.raises(ValueError, match="Unresolved placeholder.*\\$PORT"):
            config_manager.get_model("test-model", {"OTHER": "val"})

    def test_explicitly_disable_strict(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  x: foo
models:
  m1:
    cmd: "${x} ${MISSING}"
""")
        cm = ConfigManager(p, strict_expansion=False)
        # Should NOT raise; unresolved placeholder passes through.
        model = cm.get_model("m1", {})
        assert "${MISSING}" in model["cmd"]


# ── lenient (non-strict) expansion ──────────────────────────────────────────

class TestLenientExpansion:
    def test_unresolved_passes_through(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
macros: {}
models:
  m1:
    cmd: "${MISSING}"
""")
        cm = ConfigManager(p, strict_expansion=False)
        model = cm.get_model("m1")
        assert "${MISSING}" in model["cmd"]

    def test_env_vars_resolve_at_runtime(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
models:
  m1:
    cmd: "/run/llama --port ${PORT} --ctx ${CTX_SIZE}"
""")
        cm = ConfigManager(p, strict_expansion=False)
        model = cm.get_model("m1", {"PORT": "8080", "CTX_SIZE": "4096"})
        assert "--port 8080" in model["cmd"]
        assert "--ctx 4096" in model["cmd"]


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
        """Integer macros (ctx-size: 4096) interpolate into strings via str()."""
        model = config_manager.get_model("test-model")
        # ${ctx-size} (int 4096) must appear as "4096" in the expanded cmd.
        assert "--ctx 4096" in model["cmd"]

    def test_integer_macro_in_boot_expansion(self, tmp_path):
        """Integer macro values survive boot-time expansion and are accessible."""
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  max-conn: 99
  prefix: start
models:
  m1:
    cmd: "${prefix} --max ${max-conn}"
""")
        cm = ConfigManager(p)
        # Integer 99 → str(99) = "99" during interpolation.
        assert cm.data["macros"]["max-conn"] == 99
        model = cm.get_model("m1")
        assert "start --max 99" in model["cmd"]


# ── Model lookup ────────────────────────────────────────────────────────────

class TestModelLookup:
    def test_get_model_found(self, config_manager):
        model = config_manager.get_model("test-model")
        assert model is not None
        assert isinstance(model["cmd"], str)

    def test_get_model_missing_returns_none(self, config_manager):
        assert config_manager.get_model("nonexistent") is None

    def test_empty_env_vars_dict_triggers_traversal(self, config_manager):
        """An explicit empty dict should trigger the traversal pass.
        In strict mode this raises because ${PORT} can't be resolved."""
        with pytest.raises(ValueError, match="Unresolved placeholder"):
            config_manager.get_model("test-model", {})

    def test_none_env_vars_skips_traversal(self, config_manager):
        """No env_vars → raw cached model (no resolution pass)."""
        model = config_manager.get_model("test-model")
        # In strict mode but no traversal, unresolved placeholders survive.
        assert "${PORT}" in model["cmd"]

    def test_get_models_returns_all_names(self, config_manager):
        names = config_manager.get_models()
        assert isinstance(names, list)
        assert "test-model" in names

    def test_get_models_empty_when_no_models_section(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  x: foo
""")
        cm = ConfigManager(p)
        assert cm.get_models() == []


# ── Whitespace normalization ────────────────────────────────────────────────

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
        model = cm.get_model("m1")
        assert "\n" not in model["cmd"]
        assert "line1 line2 line3" == model["cmd"]

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
        model = cm.get_model("m1")
        assert "\t" not in model["cmd"]
        assert "part1 part2" == model["cmd"]

    def test_multiple_spaces_collapsed(self, tmp_path):
        yaml_content = """\
macros: {}
models:
  m1:
    cmd: "hello     world"
"""
        p = _write(tmp_path, "cfg.yaml", yaml_content)
        cm = ConfigManager(p, strict_expansion=False)
        model = cm.get_model("m1")
        assert "  " not in model["cmd"]
        assert "hello world" == model["cmd"]


# ── get_dict / get_vector ──────────────────────────────────────────────────

class TestAccessors:
    def test_get_dict_returns_full_config(self, config_manager):
        d = config_manager.get_dict()
        assert isinstance(d, dict)
        assert "macros" in d
        assert "models" in d

    def test_get_vector_existing_key(self, config_manager):
        m = config_manager.get_vector("macros")
        assert isinstance(m, dict)

    def test_get_vector_missing_key(self, config_manager):
        assert config_manager.get_vector("nonexistent") is None


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

