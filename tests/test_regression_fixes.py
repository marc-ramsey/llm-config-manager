"""Regression tests for targeted fixes to ConfigManager.

Covers:
- strict-mode error message shows the full ``${KEY}`` placeholder
- ``merge()`` does not alias nested dicts from the caller's update
- cycle messages list macros in reference order (deterministic)
- path accessors tolerate a null (non-dict) intermediate node
"""

import pytest

from llm_config_manager.config_manager import ConfigManager


def _write(tmp_dir, name, content):
    p = tmp_dir / name
    p.write_text(content)
    return str(p)


# ── Strict error message ────────────────────────────────────────────────

class TestStrictErrorMessage:
    def test_unresolved_placeholder_shows_braces(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
models: {}
""")
        cm = ConfigManager(p)
        with pytest.raises(ValueError) as exc:
            cm._resolve_string("x ${MISSING}", {}, strict=True)
        assert "${MISSING}" in str(exc.value)


# ── merge() aliasing ────────────────────────────────────────────────────

class TestMergeNoAlias:
    def test_nested_update_not_aliased(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
models: {}
backends:
  cpu:
    runner: process
""")
        cm = ConfigManager(p)
        update = {"backends": {"cpu": {"args": {"ngl": 4}}}}
        cm.merge(update)
        # Mutating the caller's dict must not mutate the config.
        update["backends"]["cpu"]["args"]["ngl"] = 999
        assert cm.data["backends"]["cpu"]["args"]["ngl"] == 4


# ── Cycle message determinism ───────────────────────────────────────────

class TestCycleMessageOrder:
    def test_cycle_listed_in_reference_order(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
macros:
  a: "${b}"
  b: "${c}"
  c: "${a}"
models: {}
""")
        with pytest.raises(RuntimeError) as exc:
            ConfigManager(p)
        # The cycle is reported in the order the references were followed,
        # not in set-iteration order.
        msg = str(exc.value)
        assert "a -> b -> c -> a" in msg


# ── Path access with null intermediates ─────────────────────────────────

class TestPathAccessNullNode:
    def test_get_returns_default_on_null_node(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
backends:
models: {}
""")
        cm = ConfigManager(p)
        assert cm.get("backends/default") is None

    def test_setitem_clobbers_null_node(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
backends:
models: {}
""")
        cm = ConfigManager(p)
        cm["backends/cpu"] = {"runner": "process"}
        assert cm.data["backends"] == {"cpu": {"runner": "process"}}

    def test_getitem_raises_on_null_node(self, tmp_path):
        p = _write(tmp_path, "cfg.yaml", """\
backends:
models: {}
""")
        cm = ConfigManager(p)
        with pytest.raises(KeyError):
            _ = cm["backends/default"]
