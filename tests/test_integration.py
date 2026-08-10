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

class TestRuntimeResolution:
    @pytest.mark.parametrize(
        "model_name", ["llama-3.2-1b-instruct", "qwen-2.5-1.5b-think"]
    )
    def test_env_vars_resolve(self, full_config, model_name):
        result = full_config.assemble_command(
            model_name, env_vars={"PORT": "9999"}
        )
        assert result is not None
        arg_list, cmd_str = result
        assert "--port" in arg_list and "9999" in arg_list
        assert "--port 9999" in cmd_str

    def test_missing_env_uses_config_default(self, full_config):
        """When env_vars is empty or missing, PORT defaults to models-start-port from config."""
        result = full_config.assemble_command(
            "llama-3.2-1b-instruct", env_vars={}
        )
        assert result is not None
        arg_list = result[0]
        idx = arg_list.index("--port")
        assert arg_list[idx + 1] == "18000"

    def test_assemble_command_with_env(self, full_config):
        """assemble_command() resolves PORT and returns (arg_list, cmd_str)."""
        result = full_config.assemble_command(
            "qwen-2.5-1.5b-instruct", env_vars={"PORT": "8080"}
        )
        assert result is not None
        arg_list, cmd_str = result
        assert "--port" in arg_list
        assert "8080" in arg_list
        assert "--port 8080" in cmd_str

    def test_assemble_command_missing_model(self, full_config):
        """Missing model returns None."""
        assert full_config.assemble_command("does-not-exist") is None


# ── Whitespace normalization ──────────────────────────────────────────────

class TestWhitespace:
    def test_cmd_is_single_line(self, full_config):
        result = full_config.assemble_command(
            "llama-3.2-1b-think", env_vars={"PORT": "7000"}
        )
        assert result is not None
        _, cmd_str = result
        assert "\n" not in cmd_str

    def test_no_consecutive_spaces(self, full_config):
        result = full_config.assemble_command(
            "qwen-2.5-1.5b-instruct", env_vars={"PORT": "7000"}
        )
        assert result is not None
        _, cmd_str = result
        assert "  " not in cmd_str


# ── Model listing ─────────────────────────────────────────────────────────

class TestModelListing:
    def test_get_models(self, full_config):
        names = full_config.get_models()
        assert len(names) == 5
        expected = {"llama-3.2-1b-instruct", "llama-3.2-1b-think",
                     "qwen-2.5-1.5b-instruct", "qwen-2.5-1.5b-think",
                     "nested-params"}
        assert set(names) == expected

    def test_reasoning_flags_present(self, full_config):
        result = full_config.assemble_command(
            "llama-3.2-1b-instruct", env_vars={"PORT": "7000"}
        )
        assert result is not None
        arg_list = result[0]
        idx = arg_list.index("--reasoning-budget")
        assert arg_list[idx + 1] == "0"

    def test_reasoning_flags_think(self, full_config):
        result = full_config.assemble_command(
            "llama-3.2-1b-think", env_vars={"PORT": "7000"}
        )
        assert result is not None
        arg_list = result[0]
        idx = arg_list.index("--reasoning-budget")
        assert arg_list[idx + 1] == "-1"


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
        result = full_config.assemble_command(
            "nested-params", env_vars=env
        )
        assert result is not None
        arg_list, cmd_str = result
        # --port resolves to 5555 in the assembled command.
        assert "--port" in arg_list and "5555" in arg_list
        assert "--port 5555" in cmd_str
        # Nested dicts/lists: get_model traverses and resolves them too.
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
