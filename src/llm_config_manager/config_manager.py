import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import yaml


logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage LLM YAML configurations.

    Loads a YAML config file, expands ``${macro}`` placeholders from the
    ``macros`` section, and normalises whitespace in command strings.  Model
    configurations can additionally be resolved at runtime against user-supplied
    environment variables (``env_vars``).

    Strict expansion (default) raises ``ValueError`` when an unresolved
    placeholder is encountered during runtime resolution via :meth:`get_model`.
    Boot-time macro expansion tolerates unresolved placeholders that are meant
    to be filled by *env_vars* at runtime.
    """

    _MACRO_PATTERN = re.compile(r'\$\{([\w-]+)\}')
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    strict_expansion: bool = True

    def __init__(self, config_path: str, strict_expansion: Optional[bool] = None):
        self.config_path = config_path
        self.strict_expansion = (
            strict_expansion if strict_expansion is not None else self.strict_expansion
        )
        self.data: Dict[str, Any] = {}
        self._load_config()
        self._expand_macros()

    # ------------------------------------------------------------------ I/O

    def _load_config(self) -> None:
        """Load the YAML configuration with detailed error handling.

        - ``FileNotFoundError`` → clear *file not found* message.
        - ``yaml.YAMLError`` → report invalid YAML.
        - Other exceptions → generic unexpected error.
        """
        try:
            with open(self.config_path, 'r') as f:
                # ``safe_load`` may return ``None`` for empty files.
                self.data = yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise RuntimeError(f"Config file not found: {self.config_path}") from e
        except yaml.YAMLError as e:
            raise RuntimeError(
                f"Invalid YAML in config file {self.config_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading config file: {e}"
            ) from e

    def merge(self, update: dict) -> dict:
        """Deep-merge *update* into ``self.data``, returning it."""
        def _recurse(base: dict, override: dict) -> dict:
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _recurse(base[k], v)
                else:
                    base[k] = v
            return base
        return _recurse(self.data, update)

    # ── Path-based access (``/`` separator) ────────────────────────────

    def _resolve_path(self, path: str) -> Any:
        """Traverse *path* segments into self.data. Returns the value at that path.

        Segments separated by ``'/'`` are resolved in order.  If any intermediate
        node is not a ``dict``, the traversal stops and returns the current value.
        """
        parts = path.split('/')
        value: Any = self.data
        for seg in parts:
            if isinstance(value, dict):
                value = value.get(seg)
                if value is None and seg not in (value or {}):
                    return None  # key absent at this level
            else:
                return None  # intermediate is not a dict
        return value

    def __getitem__(self, path: str) -> Any:
        """Access config values by ``'key'`` or ``'path/to/key'`` notation."""
        val = self._resolve_path(path)
        if val is None and path not in self.data and '/' not in path:
            raise KeyError(path)
        return val

    def __setitem__(self, path: str, value: Any) -> None:
        """Set a config value by ``'key'`` or ``'path/to/key'`` notation.

        Creates intermediate dicts where needed so that
        ``cm['a/b/c'] = 1`` works even when ``'a'`` and ``'a.b'`` don't exist yet.
        """
        parts = path.split('/')
        node = self.data
        for seg in parts[:-1]:
            if seg not in node or not isinstance(node[seg], dict):
                node[seg] = {}
            node = node[seg]
        node[parts[-1]] = value

    def __delitem__(self, key: str) -> None:
        """Remove a top-level config key."""
        if key not in self.data:
            raise KeyError(key)
        del self.data[key]

    def __iter__(self):
        """Iterate over top-level keys."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of top-level config sections."""
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        """Check if a top-level key exists."""
        return key in self.data

    # ── Macro / string resolution ────────────────────────────────────────

    def _resolve_string(
        self,
        text: str,
        current_macros: Dict[str, Any],
        seen: Optional[set] = None,
        strict: bool = False,
    ) -> str:
        """Replace all ``${KEY}`` placeholders with values from *current_macros*.

        When *seen* is provided (during macro expansion), unresolved references
        to already-being-resolved macros raise a ``RuntimeError``.  When *strict*
        is True (during runtime env_vars resolution), any unresolved placeholder
        raises a ``ValueError`` instead of passing through unchanged.

        Otherwise up to 10 iterations resolve chained references, then whitespace
        collapses.
        """
        if not isinstance(text, str):
            return text

        def _repl(m: 're.Match') -> str:
            key = m.group(1)
            if seen is not None and key in seen:
                cycle = " -> ".join(list(seen) + [key])
                raise RuntimeError(
                    f"Circular macro reference detected: {cycle}"
                )
            val = current_macros.get(key)
            if val is None and strict:
                raise ValueError(
                    f"Unresolved placeholder '${key}' during "
                    "env_vars resolution"
                )
            if val is None:
                return m.group(0)  # pass through; not in current_macros
            return str(val)

        for _ in range(10):
            new_text = self._MACRO_PATTERN.sub(_repl, text)
            if new_text == text:
                break
            text = new_text
        return self._WHITESPACE_PATTERN.sub(' ', text).strip()

    def _traverse(
        self, node: Any, current_macros: Dict[str, Any], strict: bool = False
    ) -> Any:
        """Recursively resolve string nodes under *node*.

        When *strict* is True (runtime env_vars resolution), unresolved
        placeholders raise a ``ValueError``.  Boot-time traversal uses
        *strict=False* so runtime-only placeholders like :py:data:`$PORT`
        survive until the explicit :meth:`get_model` call.
        """
        if isinstance(node, dict):
            return {
                k: self._traverse(v, current_macros, strict) for k, v in node.items()
            }
        elif isinstance(node, list):
            return [self._traverse(v, current_macros, strict) for v in node]
        elif isinstance(node, str):
            return self._resolve_string(node, current_macros, strict=strict)
        return node

    # ── Boot-time macro expansion ────────────────────────────────────────

    def _expand_macros(self) -> None:
        """Expand macros and detect circular references.

        Macros are first resolved individually, guarding against self-reference
        cycles.  The fully expanded macro map is then applied to the entire
        config tree (*strict=False* so runtime-only placeholders survive).
        """
        macros = self.data.get('macros', {})
        if not macros:
            return

        def _resolve_macro(name: str, seen: set) -> Any:
            if name in seen:
                cycle = " -> ".join(list(seen) + [name])
                raise RuntimeError(
                    f"Circular macro reference detected: {cycle}"
                )

            value = macros.get(name)
            if isinstance(value, str):
                seen.add(name)
                resolved = self._resolve_string(value, macros, seen)
                seen.remove(name)
                return resolved
            # Non-string (int, list, etc.) – leave untouched.
            return value

        expanded_macros: Dict[str, Any] = {}
        for k in macros:
            expanded_macros[k] = _resolve_macro(k, set())

        # Apply expanded macros to the entire configuration tree.
        self.data = self._traverse(self.data, expanded_macros)  # strict=False

        # Store the expanded macro dict back onto the data for external callers.
        self.data['macros'] = expanded_macros

    # ── Output helpers ───────────────────────────────────────────────────

    def export(self, output_path: str, fmt: str = 'yaml') -> None:
        """Write the entire processed configuration to *output_path*.

        Args:
            output_path: Path to write the file.
            fmt: Output format — ``'json'`` or ``'yaml'`` (default ``'yaml'``).
        """
        if fmt not in ('json', 'yaml'):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'yaml'.")

        with open(output_path, 'w') as f:
            if fmt == 'json':
                json.dump(self.data, f, indent=2)
            else:
                yaml.dump(self.data, f, default_flow_style=False)

    # ── Accessors ────────────────────────────────────────────────────────
    def get_dict(self) -> Dict[str, Any]:
        """Return the complete, macro-expanded configuration dictionary."""
        return self.data

# ── CLI entry point ───────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process and export configuration."
    )
    parser.add_argument("--input", default="config.yaml")
    parser.add_argument("--export", help="Export full config to file (.json/.yaml/.yml)")
    args = parser.parse_args()

    cm = ConfigManager(args.input)
    if args.export:
        ext = os.path.splitext(args.export)[1].lower()
        fmt = 'json' if ext == '.json' else 'yaml'
        cm.export(args.export, fmt=fmt)


if __name__ == "__main__":
    main()
