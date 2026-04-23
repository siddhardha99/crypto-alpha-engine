"""Elementwise math operators — arithmetic and unary functions.

Every operator is a pure function: takes one or two ``pd.Series``
(or scalar floats, where the math is well-defined) and returns a
``pd.Series``. Element-wise; no windows, no state, trivially causal.

Domain failures (``log`` of a non-positive, ``div`` by zero, ``sqrt``
of a negative) return ``NaN`` rather than raising — the Pandera
schemas at the ingestion edge reject malformed data; by the time math
operators run, the caller has accepted that NaN is the failure mode.

13 operators land here, matching SPEC §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha_engine.operators.registry import register_operator

Scalar = float | int
SeriesOrScalar = pd.Series | Scalar


# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------


@register_operator("add")
def add(a: SeriesOrScalar, b: SeriesOrScalar) -> pd.Series:
    """Element-wise ``a + b``. At least one argument should be a Series."""
    return _to_series(a) + b


@register_operator("sub")
def sub(a: SeriesOrScalar, b: SeriesOrScalar) -> pd.Series:
    """Element-wise ``a - b``."""
    return _to_series(a) - b


@register_operator("mul")
def mul(a: SeriesOrScalar, b: SeriesOrScalar) -> pd.Series:
    """Element-wise ``a * b``."""
    return _to_series(a) * b


@register_operator("div")
def div(a: SeriesOrScalar, b: SeriesOrScalar) -> pd.Series:
    """Element-wise ``a / b``. Division by zero returns ``NaN``, not ``inf``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = _to_series(a) / b
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Unary transforms
# ---------------------------------------------------------------------------


@register_operator("log")
def log(x: pd.Series) -> pd.Series:
    """Natural log. Non-positive inputs → ``NaN``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # numpy emits -inf on log(0), nan on log(<0); collapse both to NaN.
        arr = np.log(x.to_numpy(dtype=float))
    return pd.Series(arr, index=x.index, name=x.name).replace([np.inf, -np.inf], np.nan)


@register_operator("exp")
def exp(x: pd.Series) -> pd.Series:
    """``e ** x``. Overflow clips to ``inf`` (numpy's default)."""
    with np.errstate(over="ignore"):
        arr = np.exp(x.to_numpy(dtype=float))
    return pd.Series(arr, index=x.index, name=x.name)


@register_operator("abs")
def abs_(x: pd.Series) -> pd.Series:
    """Element-wise absolute value. Registered as ``abs``."""
    return x.abs()


@register_operator("sign")
def sign(x: pd.Series) -> pd.Series:
    """Element-wise sign: +1, 0, or -1. ``NaN`` stays ``NaN``."""
    return pd.Series(np.sign(x.to_numpy(dtype=float)), index=x.index, name=x.name)


@register_operator("sqrt")
def sqrt(x: pd.Series) -> pd.Series:
    """Square root. Negative inputs → ``NaN``."""
    with np.errstate(invalid="ignore"):
        arr = np.sqrt(x.to_numpy(dtype=float))
    return pd.Series(arr, index=x.index, name=x.name)


@register_operator("power")
def power(x: pd.Series, p: float) -> pd.Series:
    """``x ** p``. Results that would be complex (e.g. ``(-1) ** 0.5``) are ``NaN``."""
    with np.errstate(invalid="ignore"):
        arr = np.power(x.to_numpy(dtype=float), float(p))
    return pd.Series(arr, index=x.index, name=x.name)


@register_operator("tanh")
def tanh(x: pd.Series) -> pd.Series:
    """Hyperbolic tangent. Saturates at ``±1``."""
    return pd.Series(np.tanh(x.to_numpy(dtype=float)), index=x.index, name=x.name)


@register_operator("sigmoid")
def sigmoid(x: pd.Series) -> pd.Series:
    """Logistic sigmoid: ``1 / (1 + exp(-x))``. Output is in ``(0, 1)``."""
    arr = 1.0 / (1.0 + np.exp(-x.to_numpy(dtype=float)))
    return pd.Series(arr, index=x.index, name=x.name)


@register_operator("clip")
def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip values to ``[lo, hi]``. If ``lo > hi``, the operator raises."""
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        raise ValueError(f"clip: lo ({lo_f}) must be <= hi ({hi_f})")
    return x.clip(lower=lo_f, upper=hi_f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_series(x: SeriesOrScalar) -> pd.Series:
    """Wrap a scalar as a single-element Series for the arithmetic binops."""
    if isinstance(x, pd.Series):
        return x
    return pd.Series([float(x)])
