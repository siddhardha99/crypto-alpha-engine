"""Operator registry — AST-level name → pure-function kernel mapping.

Every operator the factor compiler can invoke lives here. Each entry
records:

* The AST-level name (``"ts_mean"``, ``"funding_z"``, ...) — the key.
* The pure-function kernel that implements it.
* A tuple of **AST-level argument types** describing what each
  positional argument in the AST may contain.

The third field is what the Phase 4 compiler consults when walking a
``FactorNode``. It distinguishes "this position expects a Series
(possibly via a feature-name string that resolves to one)" from "this
position expects a literal int". See SPEC §5.1 / §7 and the design
rationale recorded in ``docs/methodology.md``.

AST-level type vocabulary
-------------------------

* ``"series"`` — the arg evaluates to a ``pd.Series``. At the AST
  level this may be a sub-:class:`FactorNode` (another computed
  expression) OR a string naming a feature that the compiler resolves
  via the engine's features dict. It is NEVER a literal numeric.
* ``"int"`` — literal integer (windows, lags, half-lives).
* ``"float"`` — literal float (quantile positions, thresholds).
* ``"bool"`` — literal boolean.
* ``"series_or_scalar"`` — either a ``"series"`` or a literal numeric.
  Used by the binary arithmetic and comparison operators so a factor
  can write ``greater_than("close", 100)``.

Separation of concerns
----------------------

This registry is **separate** from the :mod:`data.registry`: the data
registry produces data at the edge of the engine; the operator
registry transforms data inside it.
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

ArgType = str
"""One of the strings in :data:`VALID_ARG_TYPES`."""

VALID_ARG_TYPES: frozenset[ArgType] = frozenset(
    {"series", "int", "float", "bool", "series_or_scalar"}
)
"""The closed set of AST-level argument type tags."""


class OperatorSpec:
    """What the registry knows about one operator."""

    __slots__ = ("name", "fn", "arg_types", "causal_safe")

    def __init__(
        self,
        name: str,
        fn: Operator,
        arg_types: tuple[ArgType, ...],
        causal_safe: bool,
    ) -> None:
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.causal_safe = causal_safe

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OperatorSpec(name={self.name!r}, arg_types={self.arg_types!r}, "
            f"causal_safe={self.causal_safe!r})"
        )


_OPERATORS: dict[str, OperatorSpec] = {}


def register_operator(
    name: str,
    *,
    arg_types: tuple[ArgType, ...],
    causal_safe: bool,
) -> Callable[[Operator], Operator]:
    """Decorator: register ``fn`` under ``name`` with the given AST arg types.

    ``causal_safe`` is **required**, no default. Every registration
    must state explicitly whether the operator respects Principle 2
    (no lookahead bias). Engine's Layer 1 causality check reads this
    attribute off the registry to reject any factor that uses an
    operator declared unsafe. Requiring explicit declaration means
    adding a future look-ahead-using operator (e.g., for retrospective
    research) is a deliberate, visible act — not a silent default.

    A canary test in ``tests/unit/test_engine.py`` iterates the
    registry after import and asserts every spec has ``causal_safe``
    set to True. If a future operator is registered with
    ``causal_safe=False``, the canary fires — the registration is
    allowed (for research use cases), but nobody can silently slip
    one in.

    Args:
        name: AST-level operator name. Must be unique.
        arg_types: Tuple describing the AST-level type of each
            positional argument. Every element must be in
            :data:`VALID_ARG_TYPES`.
        causal_safe: Whether the operator's output at time ``t``
            depends only on inputs at times ``<= t``. True for every
            operator in Phase 3-5; False is reserved for deliberately
            acausal operators added for research purposes.

    Returns:
        The original function, unchanged.

    Raises:
        ConfigError: If ``name`` is already registered, or if any
            entry in ``arg_types`` is not a known tag.

    Example:
        >>> @register_operator("demo_op", arg_types=("series", "int"), causal_safe=True)
        ... def demo(x, window):
        ...     return x
        >>> get_operator_arg_types("demo_op")
        ('series', 'int')
    """
    normalised: tuple[ArgType, ...] = tuple(arg_types)
    for tag in normalised:
        if tag not in VALID_ARG_TYPES:
            raise ConfigError(
                f"register_operator {name!r}: unknown arg type {tag!r}; "
                f"valid: {sorted(VALID_ARG_TYPES)}"
            )

    def decorator(fn: Operator) -> Operator:
        if name in _OPERATORS:
            existing = _OPERATORS[name].fn
            raise ConfigError(
                f"cannot register operator {name!r}: another kernel is "
                f"already registered ({existing.__module__}."
                f"{existing.__qualname__})"
            )
        _OPERATORS[name] = OperatorSpec(
            name=name, fn=fn, arg_types=normalised, causal_safe=causal_safe
        )
        return fn

    return decorator


def get_operator_causal_safe(name: str) -> bool:
    """Return the ``causal_safe`` flag for the operator ``name``.

    The engine's Layer 1 causality check uses this to reject factors
    that reference unsafe operators before any simulation runs.

    Raises:
        ConfigError: If no operator is registered under that name.
    """
    return _get_spec(name).causal_safe


def get_operator(name: str) -> Operator:
    """Return the kernel registered under ``name``.

    Raises:
        ConfigError: If no operator is registered under that name.
    """
    return _get_spec(name).fn


def get_operator_arg_types(name: str) -> tuple[ArgType, ...]:
    """Return the AST-level argument types for the operator ``name``.

    The Phase 4 compiler uses this to decide how to transform each
    AST argument before calling the kernel:

    * ``"series"`` args are either recursively evaluated (if the AST
      gives a :class:`FactorNode`) or resolved via the features dict
      (if the AST gives a string).
    * Scalar-typed args are passed through literally.

    Raises:
        ConfigError: If no operator is registered under that name.
    """
    return _get_spec(name).arg_types


def _get_spec(name: str) -> OperatorSpec:
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


def _snapshot_for_tests() -> dict[str, OperatorSpec]:
    """Return a copy of the current registry. Tests only."""
    return dict(_OPERATORS)


def _restore_for_tests(snapshot: dict[str, OperatorSpec]) -> None:
    """Replace the registry with ``snapshot``. Tests only."""
    _OPERATORS.clear()
    _OPERATORS.update(snapshot)
