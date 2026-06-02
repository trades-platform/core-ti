"""
_config.py — Global backend configuration for core_ti.

Priority chain for default backend resolution:
  IndicatorEngine(backend=...) > _backend param > set_default_backend()
  > CORE_TI_BACKEND env var > "pandas" hardcoded default
"""
from __future__ import annotations

import os

_DEFAULT_BACKEND: str | None = None


def set_default_backend(name: str) -> None:
    """Set the global default backend for new IndicatorEngine instances.

    Parameters
    ----------
    name:
        Backend name (e.g. "pandas", "talib", "pandas_ta", "tulipy").
        Validation is deferred to IndicatorEngine construction.
    """
    global _DEFAULT_BACKEND  # noqa: PLW0603
    _DEFAULT_BACKEND = name


def get_default_backend() -> str | None:
    """Return the current global default backend name, or None if unset."""
    return _DEFAULT_BACKEND


def reset_default_backend() -> None:
    """Reset the global default backend to None (use env var or "pandas")."""
    global _DEFAULT_BACKEND  # noqa: PLW0603
    _DEFAULT_BACKEND = None


def _resolve_default_backend() -> str:
    """Resolve the default backend name using the priority chain.

    Returns
    -------
    str
        Backend name to use.
    """
    if _DEFAULT_BACKEND is not None:
        return _DEFAULT_BACKEND
    env_val = os.environ.get("CORE_TI_BACKEND", "").strip()
    if env_val:
        return env_val
    return "pandas"
