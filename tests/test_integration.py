"""Integration tests — load real-world-ish config, resolve macros + env_vars, export."""

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def test_data_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def full_config(test_data_dir):
    """Load the real test config and expand macros at boot time."""
    from llm_config_manager.config_manager import ConfigManager

    cfg_path = test_data_dir / "test_config.yaml"
    assert cfg_path.exists(), f"Fixture missing: {cfg_path}"
    return ConfigManager(str(cfg_path))


# ── Boot-time macro expansion ─────────────────────────────────────────────

class TestBootExpansion:
    def test_models_section_exists(self, full_config):
        models = full_config.data.get("models", {})
        assert len(models) == 5

    def test_model_cmds_resolved_macros(self, full_config):
        """Internal macro refs are expanded; ${PORT} survives by design."""
        for name, cfg in full_config.data.get("models", {}).items():
            cmd = cfg.get("cmd", "")
            assert "${llama-cmd}" not in cmd, f"'{name}' has unresolved llama-cmd"
            assert "${ctx-default}" not in cmd




# ── Runtime env_vars resolution ───────────────────────────────────────────

# (assemble_command tests removed — command assembly moved to model-arkestra)


# ── Whitespace normalization ──────────────────────────────────────────────


# ── Model listing ─────────────────────────────────────────────────────────

class TestModelListing:
    def test_get_models(self, full_config):
        names = full_config.get_models()
        assert len(names) == 5
        expected = {"llama-3.2-1b-instruct", "llama-3.2-1b-think",
                     "qwen-2.5-1.5b-instruct", "qwen-2.5-1.5b-think",
                     "nested-params"}
        assert set(names) == expected


# ── Export formats ────────────────────────────────────────────────────────

class TestExports:
    def test_export_json(self, full_config, tmp_path):
        out = str(tmp_path / "full.json")
        full_config.export(out, fmt='json')
        loaded = json.loads(Path(out).read_text())
        assert loaded == full_config.data

    def test_export_yaml(self, full_config, tmp_path):
        out = str(tmp_path / "full.yml")
        full_config.export(out)
        loaded = yaml.safe_load(Path(out).read_text())
        assert "macros" in loaded
        assert "models" in loaded

    def test_nested_model_env_vars(self, full_config):
        """env_vars resolve placeholders at arbitrary nesting depth."""
        env = {"PORT": "5555", "TOKEN": "abc123"}
        model = full_config.get_model("nested-params", env_vars=env)
        assert model["extra"]["endpoint_url"] == "http://localhost:5555/v1"
        assert model["extra"]["auth_header"] == "Bearer token-abc123"
        assert "5555-instance" in model["extra"]["tags"]
        assert "gpu" in model["extra"]["tags"]


# ── CLI entry point ────────────────────────────────────────────────────────

class TestCLI:
    def test_export_unsupported_format(self, config_manager, tmp_path):
        out = str(tmp_path / "out.xml")
        with pytest.raises(ValueError, match="Unsupported format"):
            config_manager.export(out, fmt='xml')

    def test_cli_export(self, full_config, tmp_path):
        """Invoke main() with --export and verify file written."""
        out = str(tmp_path / "config.json")

        from llm_config_manager.config_manager import main
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "config_manager",
                "--input", full_config.config_path,
                "--export", out,
            ]
            main()
        finally:
            sys.argv = old_argv

        loaded = json.loads(Path(out).read_text())
        assert loaded == full_config.data
