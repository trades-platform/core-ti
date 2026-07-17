"""Project-wide pytest configuration.

Quarantines a pandas-ta side effect: ``pandas_ta/core.py`` sets
``pd_options.mode.copy_on_write = True`` at *import* time, flipping pandas
Copy-on-Write on for the whole process. The ``IndicatorEngine`` imports the
pandas_ta backend, so once any test triggers that path, CoW stays on for every
later test in the same run — silently changing pandas semantics (e.g.
``Series.to_numpy()`` then returns a read-only view).

The autouse fixture below snapshots the CoW setting at session start and
restores it around every test, so a test (or an import it triggers) cannot leak
the change to sibling tests. Tests should still avoid mutating ``to_numpy()``
output in place, but this removes the cross-test landmine.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Captured at conftest import — before any test runs or lazily imports pandas_ta
# (all such imports are function-local), so this is the clean session baseline.
_COW_BASELINE = pd.get_option("mode.copy_on_write")


@pytest.fixture(autouse=True)
def _isolate_copy_on_write():
    """Restore ``mode.copy_on_write`` to its session baseline around each test."""
    pd.set_option("mode.copy_on_write", _COW_BASELINE)
    yield
    pd.set_option("mode.copy_on_write", _COW_BASELINE)
