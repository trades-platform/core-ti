"""
schema.py — Core type declarations for the indicator framework.

Provides: Column, Param, IndicatorMeta, and output template interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

@dataclass
class Column:
    """Declares one input column required by an indicator.

    For standard OHLCV columns set `indicator=None`.
    For columns produced by another indicator, set `indicator` to its name
    and `params` to the parameters that should be forwarded to it.
    """

    name: str
    dtype: Literal["float64", "int64", "bool"] = "float64"
    optional: bool = False
    indicator: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Param
# ---------------------------------------------------------------------------

class ParamValidationError(ValueError):
    """Raised when a parameter value violates declared constraints."""

    def __init__(self, param_name: str, value: Any, reason: str) -> None:
        self.param_name = param_name
        self.value = value
        self.reason = reason
        super().__init__(f"Parameter '{param_name}={value!r}': {reason}")


@dataclass
class Param:
    """Declares one parameter accepted by an indicator."""

    type: type
    default: Any = None          # None means required (no default)
    min: float | None = None
    max: float | None = None
    choices: list | None = None

    def validate(self, name: str, value: Any) -> None:
        """Validate *value* against declared constraints; raise on violation."""
        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except (TypeError, ValueError):
                raise ParamValidationError(
                    name, value,
                    f"expected {self.type.__name__}, got {type(value).__name__}"
                )
        if self.min is not None and value < self.min:
            raise ParamValidationError(
                name, value, f"must be >= {self.min}"
            )
        if self.max is not None and value > self.max:
            raise ParamValidationError(
                name, value, f"must be <= {self.max}"
            )
        if self.choices is not None and value not in self.choices:
            raise ParamValidationError(
                name, value, f"must be one of {self.choices}"
            )


# ---------------------------------------------------------------------------
# IndicatorMeta
# ---------------------------------------------------------------------------

@dataclass
class IndicatorMeta:
    """Complete metadata for a registered indicator."""

    name: str
    indicator_type: Literal["column", "scalar"]
    requires: list[Column]
    params: dict[str, Param]
    outputs: list[str]                          # output name templates
    doc: str
    backends: dict[str, Callable]               # backend_name → compute_fn
    source: Literal["builtin", "user", "plugin"]

    # For indicators with parameter-dependent requires
    dynamic_requires: Callable | None = None    # fn(**params) -> list[Column]


# ---------------------------------------------------------------------------
# Output naming template interpolation
# ---------------------------------------------------------------------------

def resolve_output_names(
    outputs: list[str],
    params: dict[str, Any],
    override: str | dict[str, str] | None = None,
) -> list[str]:
    """Interpolate ``{param_name}`` placeholders in output name templates.

    Parameters
    ----------
    outputs:
        List of template strings, e.g. ``["sma_{period}"]``.
    params:
        Resolved parameter values used for interpolation.
    override:
        ``_output`` value supplied by the user.
        - ``str`` → applies only when ``len(outputs) == 1``
        - ``dict`` → maps template-resolved names to custom names

    Returns
    -------
    list[str]
        Concrete column names after interpolation and optional override.
    """
    # Format each template with param values; unknown placeholders are left as-is
    resolved = []
    for template in outputs:
        try:
            resolved.append(template.format(**params))
        except KeyError:
            resolved.append(template)

    if override is None:
        return resolved

    if isinstance(override, str):
        if len(resolved) != 1:
            raise ValueError(
                f"_output as str is only valid for single-output indicators "
                f"(this indicator has {len(resolved)} outputs)"
            )
        return [override]

    if isinstance(override, dict):
        return [override.get(name, name) for name in resolved]

    raise TypeError(f"_output must be str or dict, got {type(override).__name__}")
