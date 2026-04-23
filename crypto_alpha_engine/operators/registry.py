"""Operator registry — AST-level name → pure-function kernel mapping.

Every operator the factor compiler can invoke lives here. The key is the
name as it appears inside a factor AST (e.g. ``"ts_mean"``,
``"funding_z"``); the value is a pure-function kernel that takes Series
and numeric parameters and returns a Series.

Separation of concerns
----------------------

* This registry is **separate** from the :mod:`data.registry`
  (DataSource registry). Different vocabularies: sources produce data
  at the edge of the engine; operators transform data inside it.
* The Phase 4 compiler walks a ``FactorNode`` tree, resolves each
  string-typed argument (e.g. ``"BTC/USD"``) to a Series via the
  features dict the engine passes in, and calls the kernel. Phase 3
  doesn't implement that compiler — it just ships the kernels.

Registration
------------

Operators register via the :func:`register_operator` decorator at
import time::

    @register_operator("ts_mean")
    def ts_mean(x: pd.Series, window: int) -> pd.Series:
        ...

Importing :mod:`crypto_alpha_engine.operators` triggers registration of
every built-in operator via that package's ``__init__.py``. External
operators (contributed factor primitives) register from user code the
same way.

Duplicate names raise :class:`ConfigError`. There's no concept of
"override" — operator semantics are part of the public contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crypto_alpha_engine.exceptions import ConfigError

Operator = Callable[..., Any]
"""Type alias for the operator-kernel signature.

Operators have varying parameter shapes (unary, binary, ternary,
window-based, etc.), so we don't pin a Callable[[Series, int], Series]
here — that would refuse to register, say, ``if_else`` (ternary).
"""

_OPERATORS: dict[str, Operator] = {}


def register_operator(name: str) -> Callable[[Operator], Operator]:
    """Decorator: register ``fn`` under ``name`` in the operator registry.

    Args:
        name: AST-level operator name (e.g. ``"ts_mean"``,
            ``"funding_z"``). Must be unique.

    Returns:
        The original function, unchanged. The decorator's only
        side effect is registration.

    Raises:
        ConfigError: If ``name`` is already registered. Re-registration
            is refused rather than silently overwriting; operator
            semantics are part of the engine's public contract.

    Example:
        >>> @register_operator("demo_op")
        ... def demo(x, window):
        ...     return x
        >>> get_operator("demo_op") is demo
        True
    """

    def decorator(fn: Operator) -> Operator:
        if name in _OPERATORS:
            existing = _OPERATORS[name]
            raise ConfigError(
                f"cannot register operator {name!r}: another kernel is "
                f"already registered ({existing.__module__}."
                f"{existing.__qualname__})"
            )
        _OPERATORS[name] = fn
        return fn

    return decorator


def get_operator(name: str) -> Operator:
    """Look up a registered operator kernel by name.

    Raises:
        ConfigError: If no operator is registered under that name.
    """
    try:
        return _OPERATORS[name]
    except KeyError as err:
        raise ConfigError(f"unknown operator {name!r}; registered: {sorted(_OPERATORS)!r}") from err


def list_operators() -> list[str]:
    """Return all registered operator names, sorted."""
    return sorted(_OPERATORS)


def has_operator(name: str) -> bool:
    """``True`` if an operator is registered under ``name``."""
    return name in _OPERATORS


def _reset_for_tests() -> None:
    """Clear the registry. Tests only."""
    _OPERATORS.clear()


def _snapshot_for_tests() -> dict[str, Operator]:
    """Return a copy of the current registry. Tests only."""
    return dict(_OPERATORS)


def _restore_for_tests(snapshot: dict[str, Operator]) -> None:
    """Replace the registry with ``snapshot``. Tests only."""
    _OPERATORS.clear()
    _OPERATORS.update(snapshot)
