"""
dag.py — Dependency graph construction, topological sort, and cycle detection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .schema import Column, resolve_output_names

if TYPE_CHECKING:
    from .registry import IndicatorRegistry
    from .schema import IndicatorMeta


# ---------------------------------------------------------------------------
# Node in the execution plan
# ---------------------------------------------------------------------------

class _Node:
    """Represents one indicator call to be executed."""

    def __init__(
        self,
        indicator_name: str,
        params: dict[str, Any],
        output_names: list[str],
        requires: list[Column],
    ) -> None:
        self.indicator_name = indicator_name
        self.params = params
        self.output_names = output_names
        self.requires = requires          # resolved at construction time

    def __repr__(self) -> str:
        return f"<Node {self.indicator_name}({self.params}) → {self.output_names}>"


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """Builds and orders an execution plan for a set of indicator calls."""

    def __init__(self, registry: IndicatorRegistry) -> None:
        self._registry = registry
        self._nodes: list[_Node] = []               # in insertion order
        self._node_index: dict[str, _Node] = {}     # output_col → node

    def add(self, indicator_name: str, params: dict[str, Any]) -> list[str]:
        """Add one indicator call; returns the resolved output column names."""
        meta = self._registry.get(indicator_name)

        # Resolve params with defaults
        resolved_params = self._resolve_params(meta, params)

        # Compute output column names
        output_names = resolve_output_names(
            meta.outputs, resolved_params,
            override=resolved_params.pop("_output", None),
        )

        # Get requires (static or dynamic)
        if meta.dynamic_requires is not None:
            requires = meta.dynamic_requires(**resolved_params)
        else:
            requires = meta.requires

        # Recursively add dependency indicators
        for col in requires:
            if col.indicator is not None:
                dep_params = dict(col.params)
                self.add(col.indicator, dep_params)

        node = _Node(indicator_name, resolved_params, output_names, requires)
        self._nodes.append(node)
        for name in output_names:
            self._node_index[name] = node

        return output_names

    def _resolve_params(
        self, meta: IndicatorMeta, user_params: dict[str, Any]
    ) -> dict[str, Any]:
        from .schema import ParamValidationError  # avoid circular at module level

        resolved: dict[str, Any] = {}
        # Pass through _output if present
        if "_output" in user_params:
            resolved["_output"] = user_params.pop("_output")

        for pname, param_spec in meta.params.items():
            if pname in user_params:
                value = user_params[pname]
            elif param_spec.default is not None:
                value = param_spec.default
            else:
                raise ValueError(
                    f"Parameter '{pname}' is required for indicator '{meta.name}'"
                )
            param_spec.validate(pname, value)
            resolved[pname] = value

        # Pass through extra kwargs not in param spec (e.g. user-supplied)
        for k, v in user_params.items():
            if k not in resolved:
                resolved[k] = v

        return resolved

    # --- topological sort with cycle detection ---

    def build_execution_plan(self, df: pd.DataFrame) -> list[_Node]:
        """Return nodes in topological order, skipping already-cached ones."""
        order = self._toposort()
        # Filter: if ALL output columns already exist in df, skip
        return [
            node for node in order
            if not self._all_outputs_present(node, df)
        ]

    def _all_outputs_present(self, node: _Node, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in node.output_names)

    def _toposort(self) -> list[_Node]:
        """Kahn's algorithm on the node list."""
        # Build adjacency: output_col → nodes that need it
        # We need: for each node, which nodes must run before it
        visited: list[_Node] = []
        seen: set[int] = set()

        def visit(node: _Node) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            # Visit dependencies first
            for col in node.requires:
                if col.indicator is not None:
                    dep_col = resolve_output_names(
                        self._registry.get(col.indicator).outputs,
                        col.params,
                    )
                    for dc in dep_col:
                        dep_node = self._node_index.get(dc)
                        if dep_node is not None:
                            visit(dep_node)
            visited.append(node)

        # Cycle detection with DFS colouring (white=0, grey=1, black=2)
        colour: dict[int, int] = {}
        path: list[str] = []

        def dfs(node: _Node) -> None:
            nid = id(node)
            if colour.get(nid) == 2:
                return
            if colour.get(nid) == 1:
                # Found a cycle — build the cycle path
                cycle_start = node.indicator_name
                idx = next(
                    (i for i, n in enumerate(path) if n == cycle_start), 0
                )
                from .registry import CircularDependencyError
                raise CircularDependencyError(path[idx:] + [cycle_start])

            colour[nid] = 1
            path.append(node.indicator_name)

            for col in node.requires:
                if col.indicator is not None:
                    dep_cols = resolve_output_names(
                        self._registry.get(col.indicator).outputs,
                        col.params,
                    )
                    for dc in dep_cols:
                        dep_node = self._node_index.get(dc)
                        if dep_node is not None:
                            dfs(dep_node)

            path.pop()
            colour[nid] = 2

        for node in self._nodes:
            dfs(node)

        # Now build ordered list without duplicates via DFS post-order
        for node in self._nodes:
            visit(node)

        # Deduplicate preserving order
        deduped: list[_Node] = []
        seen_ids: set[int] = set()
        for n in visited:
            if id(n) not in seen_ids:
                seen_ids.add(id(n))
                deduped.append(n)
        return deduped
