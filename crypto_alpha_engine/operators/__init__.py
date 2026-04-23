"""Operator library — pure-function kernels registered by AST-level name.

Importing this package triggers registration of every built-in operator
via the submodule imports below. External operators (contributed factor
primitives) register from user code via
:func:`crypto_alpha_engine.operators.registry.register_operator`.
"""

from __future__ import annotations

# Submodule imports trigger @register_operator side-effects.
from crypto_alpha_engine.operators import (  # noqa: F401
    conditional,
    crypto,
    math,
    timeseries,
)
