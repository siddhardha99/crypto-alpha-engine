"""Conditional / boolean operators.

Element-wise and trivially causal. Comparison operators return a
boolean Series (matched to pandas' float-based NaN semantics: a
comparison involving NaN yields ``False``, matching pandas' default).

7 operators land here, matching SPEC §6.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_engine.operators.registry import register_operator

# ---------------------------------------------------------------------------
# Ternary select
# ---------------------------------------------------------------------------


@register_operator("if_else")
def if_else(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise: ``a`` where ``cond`` is true, else ``b``.

    All three arguments must share the same index. ``NaN`` in ``cond``
    propagates as ``NaN`` in the output (the branch is undefined).
    """
    # pd.where keeps self when cond is True, else uses other. We want `a`
    # where cond, else `b` — so flip.
    return a.where(cond, other=b)


# ---------------------------------------------------------------------------
# Comparisons (return Boolean Series)
# ---------------------------------------------------------------------------


@register_operator("greater_than")
def greater_than(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """Element-wise ``x > y``."""
    return x > y


@register_operator("less_than")
def less_than(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """Element-wise ``x < y``."""
    return x < y


@register_operator("equal")
def equal(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """Element-wise ``x == y``. Watch out for float precision — prefer ``abs(x-y) < eps``."""
    return x == y


# ---------------------------------------------------------------------------
# Logical combinators
# ---------------------------------------------------------------------------


@register_operator("and_")
def and_(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise logical AND. Inputs are cast to bool dtype."""
    return a.astype(bool) & b.astype(bool)


@register_operator("or_")
def or_(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise logical OR."""
    return a.astype(bool) | b.astype(bool)


@register_operator("not_")
def not_(a: pd.Series) -> pd.Series:
    """Element-wise logical NOT."""
    return ~a.astype(bool)
