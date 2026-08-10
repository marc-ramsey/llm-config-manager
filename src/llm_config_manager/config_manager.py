import argparse
import json
import logging
import os
import re
import shlex
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
    def get_vector(self, key: str) -> Union[Dict[str, Any], None]:
        """Retrieve a top-level config section (e.g. ``'macros'``, ``'models'``)."""
        return self.data.get(key)
    def get_models(self) -> list[str]:
        """Return a list of all available model names."""
        models = self.get_vector('models')
        return list(models.keys()) if models else []

    def get_model(
        self, model_name: str, env_vars: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], None]:
        """Return the config dict for a named model.

        If *env_vars* is provided the model's string values are resolved
        against them (*strict=True*).  An empty dict ``{}`` still triggers
        traversal (previously it was silently skipped because ``{}`` is falsy).
        """
        models = self.get_vector('models')
        model = models.get(model_name) if models else None
        if model is None:
            return None
        if env_vars is not None:
            return self._traverse(model, env_vars, strict=self.strict_expansion)
        # Always normalize whitespace, but without strict mode so unresolved
        # placeholders (e.g. $PORT) survive for later runtime resolution.
        return self._traverse(model, {}, strict=False)

    def get_backend(
        self, backend_id: str
    ) -> Union[Dict[str, Any], None]:
        """Return the backend dict for *backend_id*.

        Returns a new dictionary containing all fields from the config.yaml
        ``backends`` entry.  Returns **None** if the backend does not exist.
        """
        backends = self.data.get('backends')
        if not backends or not isinstance(backends, dict):
            return None
        be = backends.get(backend_id)
        if be is None:
            return None
        # Return a copy so callers can inspect without mutating the original.
        result = dict(be) if isinstance(be, dict) else {}
        return result

    def _resolve_backend_for_model(
        self,
        model: Dict[str, Any],
        model_name: str,
        env_vars: Optional[Dict[str, Any]],
        override_backend: Optional[str] = None,
    ) -> str:
        """Determine which backend ID to use for a given model.

        Resolution order (first match wins):
        1. *override_backend* argument (runtime call-time override)
        2. ``model[backend]`` key (per-model YAML default)
        3. ``backends.default`` (global default)

        Returns the resolved backend ID string.
        """
        if override_backend:
            return override_backend

        model_backend = model.get('backend')
        if model_backend:
            return str(model_backend)

        global_default = self.data.get('backends', {})
        if isinstance(global_default, dict):
            default_id = global_default.get('default')
            if default_id:
                return str(default_id)

        raise RuntimeError(
            f"No backend specified for model '{model_name}' "
            "(no override, no per-model 'backend' key, no 'backends.default')"
        )

    def assemble_command(
        self,
        model_name: str,
        env_vars: Optional[Dict[str, Any]] = None,
        override_backend: Optional[str] = None,
    ) -> Optional[tuple[List[str], str]]:
        """Assemble command arguments for *model_name* using the backend config.

        Resolves the effective backend, fills template placeholders
        (``${CHECKPOINT}``, ``${PORT}``), and concatenates backend arguments
        with model-specific arguments.

        Args:
            model_name:   Model name as defined in the config file.
            env_vars:     Runtime environment variables for resolving
                          ``${PORT}`` and other placeholders.  If omitted,
                          unresolved placeholders are left as-is (strict mode).
            override_backend: Optional backend ID that overrides whichever
                              backend the model would normally use.

        Returns:
            A tuple of ``(arg_list, cmd_str)`` where ``arg_list`` is a list
            of individual arguments ready for ``subprocess_exec``,
            and ``cmd_str`` is the space-joined string representation.
        """
        models = self.get_vector('models')
        if not models:
            return None
        model = models.get(model_name)
        if model is None:
            return None

        # 1. Resolve which backend to use
        backend_id = self._resolve_backend_for_model(model, model_name, env_vars, override_backend)

        # 2. Get backend definition
        backend = self.get_backend(backend_id)
        if backend is None:
            raise RuntimeError(
                f"Backend '{backend_id}' not found in backends config"
            )

        # 3. Resolve CHECKPOINT and PORT
        checkpoint = model.get('checkpoint', '')
        port = env_vars.get('PORT') if env_vars is not None else None
        if port is None:
            port = str(self.data.get('models-start-port', 18000))

        # Build a temporary macro map for this resolution pass
        resolve_macros = dict(self.data.get('macros', {}))
        resolve_macros['CHECKPOINT'] = checkpoint
        resolve_macros['PORT'] = port

        # 4. Resolve backend args template against this context
        backend_args_template = str(backend.get('args', ''))
        backend_args_resolved = self._resolve_string(backend_args_template, resolve_macros, strict=False)

        # 5. Get model args (already whitespace-normalized by _traverse at boot time)
        model_args = model.get('args', '')
        if isinstance(model_args, str):
            model_args = ' '.join(model_args.split())  # ensure single-space
        else:
            model_args = str(model_args) if model_args else ''

        # 6. Build command args: resolved backend args + model args
        combined = shlex.split(backend_args_resolved) if backend_args_resolved.strip() else []
        combined += shlex.split(model_args) if model_args.strip() else []

        return (combined, ' '.join(combined))

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
