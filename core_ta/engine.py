"""
engine.py — IndicatorEngine: Facade + Pipe API, backend selection, fallback.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .backends.base import BackendProtocol, nan_safe_compute
from .backends.pandas_backend import PandasBackend
from .dag import DependencyGraph
from .registry import (
    IndicatorNotFoundError,
    IndicatorNotImplementedError,
    MissingColumnError,
    get_registry,
)
from .schema import IndicatorMeta, resolve_output_names

_BACKEND_REGISTRY: dict[str, type] = {}


def _register_builtin_backends() -> dict[str, BackendProtocol]:
    from .backends.pandas_backend import PandasBackend
    from .backends.talib_backend import TALibBackend
    from .backends.pandas_ta_backend import PandasTABackend
    from .backends.tulipy_backend import TulipyBackend
    return {
        "pandas": PandasBackend(),
        "talib": TALibBackend(),
        "pandas_ta": PandasTABackend(),
        "tulipy": TulipyBackend(),
    }


# ---------------------------------------------------------------------------
# BackendFallbackWarning
# ---------------------------------------------------------------------------

class BackendFallbackWarning(UserWarning):
    pass


# ---------------------------------------------------------------------------
# ExecutionError — wraps errors from individual indicator calls
# ---------------------------------------------------------------------------

@dataclass
class IndicatorError:
    indicator_name: str
    exception: Exception


# ---------------------------------------------------------------------------
# IndicatorEngine
# ---------------------------------------------------------------------------

class IndicatorEngine:
    """Unified entry point for indicator computation.

    Parameters
    ----------
    backend:
        Name of the preferred backend ("pandas", "talib", "pandas_ta")
        or a custom BackendProtocol instance.
    on_error:
        Default error strategy for Pipe mode ("raise" or "skip").
    """

    def __init__(
        self,
        backend: str | BackendProtocol = "pandas",
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        self._all_backends = _register_builtin_backends()
        self._on_error = on_error
        self._last_report: list[IndicatorError] = []

        if isinstance(backend, str):
            if backend not in self._all_backends:
                raise ValueError(
                    f"Unknown backend '{backend}'. "
                    f"Available: {list(self._all_backends)}"
                )
            self._primary_backend: BackendProtocol = self._all_backends[backend]
        else:
            self._primary_backend = backend

        self._default_backend: BackendProtocol = self._all_backends["pandas"]

    # --- backend resolution ---

    def _resolve_backend(self, indicator_name: str) -> BackendProtocol:
        """Return the best available backend for this indicator, with fallback."""
        if self._primary_backend.supports(indicator_name):
            return self._primary_backend

        # Fallback to pandas
        if self._default_backend.supports(indicator_name):
            warnings.warn(
                f"Backend '{self._primary_backend.name}' does not support "
                f"'{indicator_name}'; falling back to "
                f"'{self._default_backend.name}'",
                BackendFallbackWarning,
                stacklevel=3,
            )
            return self._default_backend

        raise IndicatorNotImplementedError(
            indicator_name, self._primary_backend.name
        )

    # --- core execution ---

    def _execute_node(
        self,
        indicator_name: str,
        params: dict[str, Any],
        output_names: list[str],
        df: pd.DataFrame,
    ) -> None:
        """Compute one indicator and write results into df in-place."""
        meta = get_registry().get(indicator_name)

        # Resolve requires (static or dynamic)
        if meta.dynamic_requires is not None:
            requires = meta.dynamic_requires(**params)
        else:
            requires = meta.requires

        # Gather inputs
        inputs: dict[str, pd.Series] = {}
        for col in requires:
            if col.name in df.columns:
                inputs[col.name] = df[col.name]
            elif col.optional:
                inputs[col.name] = None  # type: ignore[assignment]
            else:
                raise MissingColumnError(col.name, indicator_name)

        # Also provide the compute function's own compute_fn if user-registered
        # (backed stored under "_fn"); use it when neither primary nor default supports
        # User-defined indicators (registered via decorator) use _fn directly
        if "_fn" in meta.backends and not self._primary_backend.supports(indicator_name) \
                and not self._default_backend.supports(indicator_name):
            result = meta.backends["_fn"](**{
                c.name: inputs[c.name] for c in requires
            }, **params)
        else:
            backend = self._resolve_backend(indicator_name)
            result = nan_safe_compute(
                backend, indicator_name, inputs, params, output_names, len(df)
            )

        # Write result columns into df
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                # Map computed column names to declared output_names if needed
                df[col] = result[col]
        elif isinstance(result, pd.Series):
            df[output_names[0]] = result
        else:
            # scalar — nothing to write to df (handled separately)
            pass

    def _execute_scalar(
        self,
        indicator_name: str,
        params: dict[str, Any],
        df: pd.DataFrame,
    ) -> float:
        meta = get_registry().get(indicator_name)

        # Resolve dependencies first
        self._run_dag(indicator_name, params, df, on_error="raise")

        requires = (
            meta.dynamic_requires(**params) if meta.dynamic_requires else meta.requires
        )
        inputs: dict[str, pd.Series] = {}
        for col in requires:
            if col.name in df.columns:
                inputs[col.name] = df[col.name]
            elif col.optional:
                inputs[col.name] = None  # type: ignore[assignment]
            else:
                raise MissingColumnError(col.name, indicator_name)

        if "_fn" in meta.backends and not self._primary_backend.supports(indicator_name) \
                and not self._default_backend.supports(indicator_name):
            return meta.backends["_fn"](**{
                c.name: inputs[c.name] for c in requires
            }, **params)
        backend = self._resolve_backend(indicator_name)
        return backend.compute(indicator_name, inputs, params)  # type: ignore[return-value]

    def _run_dag(
        self,
        indicator_name: str,
        params: dict[str, Any],
        df: pd.DataFrame,
        on_error: Literal["raise", "skip"],
    ) -> list[IndicatorError]:
        dag = DependencyGraph(get_registry())
        output_names = dag.add(indicator_name, dict(params))
        plan = dag.build_execution_plan(df)

        errors: list[IndicatorError] = []
        skipped: set[str] = set()

        for node in plan:
            # Skip if any of this node's required columns are in the skipped set
            if any(c.name in skipped for c in node.requires if c.indicator):
                skipped.update(node.output_names)
                continue

            try:
                self._execute_node(
                    node.indicator_name, node.params, node.output_names, df
                )
            except Exception as exc:
                if on_error == "raise":
                    raise
                errors.append(IndicatorError(node.indicator_name, exc))
                skipped.update(node.output_names)

        return errors

    # --- Facade API ---

    def __getattr__(self, name: str) -> Any:
        # Only intercept names that are registered indicators
        registry = get_registry()
        if not registry.has(name):
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )

        def _call(df: pd.DataFrame, **kwargs: Any) -> Any:
            meta = registry.get(name)
            if meta.indicator_type == "scalar":
                return self._execute_scalar(name, kwargs, df)
            # Column indicator
            errors = self._run_dag(name, kwargs, df, on_error=self._on_error)
            self._last_report = errors
            return df

        return _call

    def pipe(
        self,
        df: pd.DataFrame,
        on_error: Literal["raise", "skip"] | None = None,
    ) -> "Pipe":
        return Pipe(df, self, on_error=on_error or self._on_error)

    # --- Discovery ---

    def list_indicators(
        self,
        *,
        type: str | None = None,
        backend: str | None = None,
        requires_subset: list[str] | None = None,
    ) -> list[IndicatorMeta]:
        return get_registry().list_all(
            type=type, backend=backend, requires_subset=requires_subset
        )

    def get_indicator(self, name: str) -> IndicatorMeta:
        return get_registry().get(name)

    def last_report(self) -> list[IndicatorError]:
        return list(self._last_report)


# ---------------------------------------------------------------------------
# Pipe — chained column indicator calls
# ---------------------------------------------------------------------------

class Pipe:
    """Fluent interface for chaining column indicator calls."""

    def __init__(
        self,
        df: pd.DataFrame,
        engine: IndicatorEngine,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        self._df = df
        self._engine = engine
        self._on_error = on_error
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._errors: list[IndicatorError] = []

    def __getattr__(self, name: str) -> Any:
        registry = get_registry()
        if not registry.has(name):
            raise AttributeError(f"Pipe has no indicator '{name}'")
        meta = registry.get(name)
        if meta.indicator_type == "scalar":
            raise TypeError(
                f"Scalar indicator '{name}' cannot be used in Pipe. "
                f"Use engine.{name}(df, ...) instead."
            )

        def _chain(**kwargs: Any) -> "Pipe":
            self._pending.append((name, kwargs))
            return self

        return _chain

    def result(self) -> pd.DataFrame:
        """Execute all pending indicator calls and return the mutated df."""
        # Build a single unified DAG across all pending calls
        dag = DependencyGraph(get_registry())
        for ind_name, params in self._pending:
            dag.add(ind_name, dict(params))

        plan = dag.build_execution_plan(self._df)
        skipped: set[str] = set()

        for node in plan:
            if any(c.name in skipped for c in node.requires if c.indicator):
                skipped.update(node.output_names)
                continue
            try:
                self._engine._execute_node(
                    node.indicator_name, node.params, node.output_names, self._df
                )
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                self._errors.append(IndicatorError(node.indicator_name, exc))
                skipped.update(node.output_names)

        self._engine._last_report = self._errors
        return self._df
