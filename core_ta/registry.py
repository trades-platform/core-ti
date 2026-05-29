"""
registry.py — Indicator registration, discovery, and decorator API.
"""
from __future__ import annotations

import inspect
import typing
from typing import Any, Callable

from .schema import Column, IndicatorMeta, Param


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IndicatorNotFoundError(KeyError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Indicator '{name}' is not registered")


class MissingColumnError(ValueError):
    def __init__(self, column: str, indicator: str) -> None:
        self.column = column
        self.indicator = indicator
        super().__init__(
            f"Column '{column}' required by indicator '{indicator}' "
            f"is not present in the DataFrame and cannot be resolved"
        )


class CircularDependencyError(ValueError):
    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' → '.join(cycle)}")


class IndicatorNotImplementedError(NotImplementedError):
    def __init__(self, indicator: str, backend: str) -> None:
        super().__init__(
            f"No implementation for indicator '{indicator}' in backend "
            f"'{backend}' or any fallback backend"
        )


# ---------------------------------------------------------------------------
# Helpers — auto-extract metadata from function/class
# ---------------------------------------------------------------------------

def _extract_params(fn: Callable, skip: set[str]) -> dict[str, Param]:
    """Infer Param declarations from a function's type-annotated signature."""
    sig = inspect.signature(fn)
    # Use get_type_hints() to resolve string annotations (from __future__ import annotations)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = getattr(fn, "__annotations__", {})

    params: dict[str, Param] = {}
    for pname, pval in sig.parameters.items():
        if pname in skip or pname.startswith("_"):
            continue
        annotation = hints.get(pname, None)
        # Only turn primitive-typed params into Param objects
        if annotation not in (int, float, str, bool, None):
            continue
        default = pval.default if pval.default is not inspect.Parameter.empty else None
        params[pname] = Param(
            type=annotation if annotation is not None else type(default) if default is not None else str,
            default=default,
        )
    return params


def _extract_doc(obj: Any) -> str:
    return (inspect.getdoc(obj) or "").strip()


def _detect_source(obj: Any) -> str:
    module = getattr(obj, "__module__", "") or ""
    if "core_ta.indicators" in module:
        return "builtin"
    return "user"


# ---------------------------------------------------------------------------
# IndicatorRegistry
# ---------------------------------------------------------------------------

class IndicatorRegistry:
    """Central store for all registered indicators."""

    def __init__(self) -> None:
        self._store: dict[str, IndicatorMeta] = {}

    # --- mutation ---

    def register(self, meta: IndicatorMeta) -> None:
        self._store[meta.name] = meta

    def add_backend(self, indicator_name: str, backend_name: str, fn: Callable) -> None:
        """Add an additional backend implementation to an existing indicator."""
        if indicator_name not in self._store:
            raise IndicatorNotFoundError(indicator_name)
        self._store[indicator_name].backends[backend_name] = fn

    # --- lookup ---

    def get(self, name: str) -> IndicatorMeta:
        try:
            return self._store[name]
        except KeyError:
            raise IndicatorNotFoundError(name)

    def has(self, name: str) -> bool:
        return name in self._store

    # --- discovery ---

    def list_all(
        self,
        *,
        type: str | None = None,
        backend: str | None = None,
        requires_subset: list[str] | None = None,
    ) -> list[IndicatorMeta]:
        results = list(self._store.values())

        if type is not None:
            results = [m for m in results if m.indicator_type == type]

        if backend is not None:
            results = [m for m in results if backend in m.backends]

        if requires_subset is not None:
            subset = set(requires_subset)
            results = [
                m for m in results
                if all(c.name in subset for c in m.requires if not c.optional)
            ]

        return results


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------

_registry = IndicatorRegistry()


def get_registry() -> IndicatorRegistry:
    return _registry


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

class _RegisterNamespace:
    """Provides @register.column() and @register.scalar() decorators."""

    def column(
        self,
        name: str,
        requires: list[Column] | None = None,
        outputs: list[str] | None = None,
        source: str | None = None,
    ) -> Callable:
        """Register a function or class as a column indicator."""
        def decorator(obj: Any) -> Any:
            self._register(
                obj,
                name=name,
                indicator_type="column",
                requires=requires or [],
                outputs=outputs or [name],
                source=source,
            )
            return obj
        return decorator

    def scalar(
        self,
        name: str,
        requires: list[Column] | None = None,
        source: str | None = None,
    ) -> Callable:
        """Register a function or class as a scalar indicator."""
        def decorator(obj: Any) -> Any:
            self._register(
                obj,
                name=name,
                indicator_type="scalar",
                requires=requires or [],
                outputs=[],
                source=source,
            )
            return obj
        return decorator

    def _register(
        self,
        obj: Any,
        *,
        name: str,
        indicator_type: str,
        requires: list[Column],
        outputs: list[str],
        source: str | None,
    ) -> None:
        # Determine compute function and check for dynamic requires
        dynamic_requires: Callable | None = None

        if inspect.isclass(obj):
            instance = obj()
            compute_fn = instance.compute
            if hasattr(instance, "requires") and callable(instance.requires):
                dynamic_requires = instance.requires
            params = _extract_params(instance.compute, skip=set())
            doc = _extract_doc(obj)
        else:
            compute_fn = obj
            params = _extract_params(obj, skip={"df"})
            doc = _extract_doc(obj)

        resolved_source = source or _detect_source(obj)

        meta = IndicatorMeta(
            name=name,
            indicator_type=indicator_type,
            requires=requires,
            params=params,
            outputs=outputs,
            doc=doc,
            backends={"_fn": compute_fn},   # stored under "_fn"; dispatched by engine
            source=resolved_source,
            dynamic_requires=dynamic_requires,
        )
        _registry.register(meta)


register = _RegisterNamespace()
