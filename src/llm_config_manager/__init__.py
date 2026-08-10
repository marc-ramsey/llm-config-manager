import importlib.metadata
from .config_manager import ConfigManager

try:
    __version__ = importlib.metadata.version("llm-config-manager")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable/development installs

__all__ = ["ConfigManager"]
