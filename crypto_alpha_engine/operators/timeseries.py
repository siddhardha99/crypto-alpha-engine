"""Time-series operators — rolling-window and lag-based primitives.

Every operator here is **pure** (no I/O, no state) and **causal**
(output at time ``t`` depends only on input at times ``<= t``). The
implementation rule that guarantees causality:

* Use ``.rolling(window=w)`` — it looks at ``[t-w+1, t]`` inclusive.
* Use ``.shift(positive_int)`` only — shifts values from the past
  forward. Negative shifts look into the future and are forbidden
  (SPEC §6, CLAUDE.md Pitfall 2).
* Use ``.ewm(...)`` without ``adjust=False`` center-style options
  that would bring in future data.

Per SPEC §6, every operator is registered at import time under its
AST-level name (``ts_mean``, ``ts_std``, ...). The registration is the
only mechanism by which the Phase 4 compiler finds a kernel given a
:class:`~crypto_alpha_engine.types.FactorNode`.

17 operators land here, matching SPEC §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.operators.registry import register_operator


def _require_positive_window(window: int, name: str) -> None:
    if not isinstance(window, int) or window <= 0:
        raise ConfigError(
            f"{name}: window must be a positive int, got {window!r} "
            f"(type {type(window).__name__})"
        )


def _require_positive_lag(lag: int, name: str) -> None:
    if not isinstance(lag, int) or lag <= 0:
        raise ConfigError(f"{name}: lag must be a positive int, got {lag!r}")


# ---------------------------------------------------------------------------
# Central-tendency and dispersion
# ---------------------------------------------------------------------------


@register_operator("ts_mean", arg_types=("series", "int"))
def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the last ``window`` bars.

    Args:
        x: Input series.
        window: Number of bars (positive).

    Returns:
        Series of the same length as ``x``. The first ``window-1``
        entries are ``NaN`` (insufficient history).

    Raises:
        ConfigError: If ``window <= 0``.

    Example:
        >>> ts_mean(pd.Series([1.0, 2, 3, 4, 5]), 3).iloc[-1]
        4.0
    """
    _require_positive_window(window, "ts_mean")
    return x.rolling(window=window).mean()


@register_operator("ts_std", arg_types=("series", "int"))
def ts_std(x: pd.Series, window: int) -> pd.Series:
    """Rolling sample standard deviation (ddof=1)."""
    _require_positive_window(window, "ts_std")
    return x.rolling(window=window).std()


@register_operator("ts_min", arg_types=("series", "int"))
def ts_min(x: pd.Series, window: int) -> pd.Series:
    """Rolling minimum over the last ``window`` bars."""
    _require_positive_window(window, "ts_min")
    return x.rolling(window=window).min()


@register_operator("ts_max", arg_types=("series", "int"))
def ts_max(x: pd.Series, window: int) -> pd.Series:
    """Rolling maximum over the last ``window`` bars."""
    _require_positive_window(window, "ts_max")
    return x.rolling(window=window).max()


@register_operator("ts_rank", arg_types=("series", "int"))
def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """Rank of the current value within the rolling window, ``method="average"``.

    Output is in ``[1, window]`` — rank 1 = smallest in the window, rank
    ``window`` = largest. The first ``window-1`` entries are ``NaN``.
    """
    _require_positive_window(window, "ts_rank")
    return x.rolling(window=window).rank()


@register_operator("ts_zscore", arg_types=("series", "int"))
def ts_zscore(x: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: ``(x - rolling_mean) / rolling_std``.

    If the rolling std is zero (constant window), the output is ``NaN``
    rather than ``inf``.
    """
    _require_positive_window(window, "ts_zscore")
    rolling = x.rolling(window=window)
    mean = rolling.mean()
    std = rolling.std()
    # Avoid inf from div-by-zero: mask std==0 as NaN so downstream code
    # treats a constant window as "no signal" rather than "infinite signal".
    std_safe = std.where(std != 0, other=np.nan)
    return (x - mean) / std_safe


# ---------------------------------------------------------------------------
# Lag-based change
# ---------------------------------------------------------------------------


@register_operator("ts_diff", arg_types=("series", "int"))
def ts_diff(x: pd.Series, lag: int) -> pd.Series:
    """First difference at the given positive ``lag``: ``x[t] - x[t - lag]``."""
    _require_positive_lag(lag, "ts_diff")
    return x - x.shift(lag)


@register_operator("ts_pct_change", arg_types=("series", "int"))
def ts_pct_change(x: pd.Series, lag: int) -> pd.Series:
    """Percent change at ``lag``: ``(x[t] / x[t - lag]) - 1``."""
    _require_positive_lag(lag, "ts_pct_change")
    return x.pct_change(periods=lag)


# ---------------------------------------------------------------------------
# Higher moments and quantiles
# ---------------------------------------------------------------------------


@register_operator("ts_skew", arg_types=("series", "int"))
def ts_skew(x: pd.Series, window: int) -> pd.Series:
    """Rolling sample skewness (Fisher-Pearson, bias-corrected)."""
    _require_positive_window(window, "ts_skew")
    return x.rolling(window=window).skew()


@register_operator("ts_kurt", arg_types=("series", "int"))
def ts_kurt(x: pd.Series, window: int) -> pd.Series:
    """Rolling sample excess kurtosis (zero for a normal distribution)."""
    _require_positive_window(window, "ts_kurt")
    return x.rolling(window=window).kurt()


@register_operator("ts_quantile", arg_types=("series", "int", "float"))
def ts_quantile(x: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling ``q``-th quantile (``0 <= q <= 1``) via linear interpolation.

    Raises:
        ConfigError: If ``window <= 0`` or ``q`` is outside ``[0, 1]``.
    """
    _require_positive_window(window, "ts_quantile")
    if not (0.0 <= float(q) <= 1.0):
        raise ConfigError(f"ts_quantile: q must be in [0, 1], got {q!r}")
    return x.rolling(window=window).quantile(q)


# ---------------------------------------------------------------------------
# Cross-series rolling statistics
# ---------------------------------------------------------------------------


@register_operator("ts_corr", arg_types=("series", "series", "int"))
def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between ``x`` and ``y`` over ``window`` bars."""
    _require_positive_window(window, "ts_corr")
    return x.rolling(window=window).corr(y)


@register_operator("ts_cov", arg_types=("series", "series", "int"))
def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling sample covariance between ``x`` and ``y`` over ``window`` bars."""
    _require_positive_window(window, "ts_cov")
    return x.rolling(window=window).cov(y)


# ---------------------------------------------------------------------------
# Argmax / argmin in a rolling window
# ---------------------------------------------------------------------------


@register_operator("ts_argmax", arg_types=("series", "int"))
def ts_argmax(x: pd.Series, window: int) -> pd.Series:
    """Bars-ago of the maximum within the rolling window.

    Returns 0 if the maximum is the current bar, ``window-1`` if the
    oldest bar in the window is the maximum.
    """
    _require_positive_window(window, "ts_argmax")

    def _argmax_offset(arr: np.ndarray) -> float:
        # arr[-1] is the current bar; arr[0] is the oldest in the window.
        return float(len(arr) - 1 - int(np.argmax(arr)))

    return x.rolling(window=window).apply(_argmax_offset, raw=True)


@register_operator("ts_argmin", arg_types=("series", "int"))
def ts_argmin(x: pd.Series, window: int) -> pd.Series:
    """Bars-ago of the minimum within the rolling window (see :func:`ts_argmax`)."""
    _require_positive_window(window, "ts_argmin")

    def _argmin_offset(arr: np.ndarray) -> float:
        return float(len(arr) - 1 - int(np.argmin(arr)))

    return x.rolling(window=window).apply(_argmin_offset, raw=True)


# ---------------------------------------------------------------------------
# Weighted smoothers
# ---------------------------------------------------------------------------


@register_operator("ts_decay_linear", arg_types=("series", "int"))
def ts_decay_linear(x: pd.Series, window: int) -> pd.Series:
    """Linearly-weighted rolling average.

    The most recent bar carries weight ``window``; the oldest bar in
    the window carries weight ``1``. Weights are normalised to sum to 1
    so the output is on the same scale as ``x``.
    """
    _require_positive_window(window, "ts_decay_linear")
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _weighted(arr: np.ndarray) -> float:
        return float(np.dot(arr, weights))

    return x.rolling(window=window).apply(_weighted, raw=True)


@register_operator("ts_ema", arg_types=("series", "int"))
def ts_ema(x: pd.Series, halflife: int) -> pd.Series:
    """Exponentially-weighted moving average with the given integer half-life.

    Pandas' ``ewm`` is causal by default (``adjust=True`` still only
    uses past observations).
    """
    if not isinstance(halflife, int) or halflife <= 0:
        raise ConfigError(f"ts_ema: halflife must be a positive int, got {halflife!r}")
    return x.ewm(halflife=halflife, adjust=True).mean()
