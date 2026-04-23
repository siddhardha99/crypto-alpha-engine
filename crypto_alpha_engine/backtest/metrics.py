"""Pure-function performance metrics on a returns series.

Every function in this module takes a ``pd.Series`` of periodic
returns (or two series for IC metrics) and returns a ``float`` scalar.
No I/O, no state, no side effects.

periods_per_year
----------------

Annualised metrics (Sharpe, Sortino, Calmar, annualized_return) need
to know how many periods per year the returns series represents. The
default is **8760** (hours in a year) because crypto trades 24/7 at
1-hour bars. Callers using other bar sizes should pass explicitly:

=====  ======================
freq    periods_per_year
=====  ======================
1h      8760
4h      2190
1d      365        (crypto: 365; NOT 252 — equities use 252 trading
                    days, but crypto markets trade every day of the year)
1w      52
=====  ======================

Passing the equity-market default of 252 for crypto data is the most
common footgun in quant-finance code. This docstring is the single
source of truth for the right value.

NaN convention
--------------

All metrics return ``float('nan')`` on degenerate input rather than
raising:

* Empty or all-NaN input → NaN for all metrics.
* Single data point → NaN for anything involving std (Sharpe, Sortino,
  skew, kurt). ``mean_return`` returns the one value; ``hit_rate``
  returns 1.0 if that return is strictly positive, else 0.0.
* Zero-volatility series → NaN for Sharpe, Sortino, Calmar
  (division by zero on the std denominator; we treat ``std < 1e-15``
  as zero to absorb pandas' floating-point noise on constant series).
* All-positive returns → ``profit_factor`` returns ``float('inf')``;
  Phase 7's JSON ledger serializer needs to handle inf → null/string,
  flagged here as a downstream concern.
* Hit rate on a series of all-zero returns → ``0.0`` (not 0.5).
  A zero return under non-zero costs is a loss, not a tie.

Programmer errors (wrong types, non-positive ``periods_per_year``)
raise :class:`ConfigError`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from crypto_alpha_engine.exceptions import ConfigError

DEFAULT_PERIODS_PER_YEAR: float = 8760.0  # 1-hour bars; the crypto-research default

_PERIODS_PER_YEAR_LOOKUP: dict[str, float] = {
    "1h": 8760.0,
    "4h": 2190.0,
    "1d": 365.0,
    "1w": 52.0,
}

_ZERO_STD_TOL: float = 1e-15  # below this, treat std as exactly zero


def periods_per_year_for(freq: Literal["1h", "4h", "1d", "1w"]) -> float:
    """Return the correct ``periods_per_year`` for a bar frequency.

    Convenience over remembering the table in the module docstring.
    """
    try:
        return _PERIODS_PER_YEAR_LOOKUP[freq]
    except KeyError as err:
        raise ConfigError(
            f"unknown freq {freq!r}; known: {sorted(_PERIODS_PER_YEAR_LOOKUP)}"
        ) from err


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def mean_return(returns: pd.Series) -> float:
    """Arithmetic mean of returns. NaN for empty/all-NaN input."""
    arr = _clean_array(returns)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def return_std(returns: pd.Series) -> float:
    """Sample standard deviation (ddof=1) of returns.

    NaN for empty, all-NaN, or single-point input.
    """
    arr = _clean_array(returns)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1))


def skew(returns: pd.Series) -> float:
    """Sample skewness (third standardised moment). NaN for <3 observations.

    Uses the bias-corrected form (``scipy.stats.skew(..., bias=False)``
    matches pandas' default).
    """
    arr = _clean_array(returns)
    if arr.size < 3:
        return float("nan")
    return float(stats.skew(arr, bias=False))


def kurt(returns: pd.Series) -> float:
    """Sample *excess* kurtosis (0 for a normal distribution).

    NaN for <4 observations. Note: Deflated Sharpe uses the raw
    (Pearson) kurtosis, which is ``kurt(r) + 3``.
    """
    arr = _clean_array(returns)
    if arr.size < 4:
        return float("nan")
    return float(stats.kurtosis(arr, fisher=True, bias=False))


# ---------------------------------------------------------------------------
# Risk-adjusted and cumulative
# ---------------------------------------------------------------------------


def sharpe(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio: ``mean/std * sqrt(periods_per_year)``.

    Risk-free rate is assumed zero (crypto convention). NaN for
    zero-volatility input.
    """
    _require_positive_ppy(periods_per_year)
    arr = _clean_array(returns)
    if arr.size < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if math.isnan(std) or std < _ZERO_STD_TOL:
        return float("nan")
    return float(arr.mean()) / std * math.sqrt(periods_per_year)


def sortino(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio — uses downside-deviation in the denominator.

    Downside deviation = std of the negative-returns-only subset. NaN
    when fewer than two negative returns exist (Sortino is degenerate).
    """
    _require_positive_ppy(periods_per_year)
    arr = _clean_array(returns)
    if arr.size < 2:
        return float("nan")
    downside = arr[arr < 0]
    if downside.size < 2:
        return float("nan")
    dd = float(downside.std(ddof=1))
    if math.isnan(dd) or dd < _ZERO_STD_TOL:
        return float("nan")
    return float(arr.mean()) / dd * math.sqrt(periods_per_year)


def total_return(returns: pd.Series) -> float:
    """Compound return: ``prod(1 + r) - 1``.

    Zero for empty/all-NaN input.
    """
    arr = _clean_array(returns)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def annualized_return(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Compound-annualised return: ``(1 + total_return)^(ppy/n) - 1``.

    NaN for empty input. NaN if total return ≤ -100% (wipe-out).
    """
    _require_positive_ppy(periods_per_year)
    arr = _clean_array(returns)
    if arr.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + arr))
    if total <= 0:
        return float("nan")
    return float(total ** (periods_per_year / arr.size) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the equity curve, as a negative number.

    0.0 for empty or strictly monotone-up input. Follows SPEC §8: the
    return value is ``<= 0``.
    """
    arr = _clean_array(returns)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calmar(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualised return divided by absolute max drawdown.

    NaN when max drawdown is zero (no risk to amortise return over).
    """
    ar = annualized_return(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0 or math.isnan(mdd) or math.isnan(ar):
        return float("nan")
    return ar / abs(mdd)


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive return.

    ``0.0`` for empty input, all-zero returns, or all-negative returns.
    A zero return counts as a loss, not a tie — crypto strategies carry
    cost drag every period (fees / slippage / funding), so a headline
    return of 0 is cost-adjusted negative.
    """
    arr = _clean_array(returns)
    if arr.size == 0:
        return 0.0
    wins: int = int(np.sum(arr > 0))
    return wins / arr.size


def profit_factor(returns: pd.Series) -> float:
    """Sum of wins / |sum of losses|.

    Returns ``float('inf')`` for all-positive returns (no losses). NaN
    for empty or all-zero input. The Phase 7 ledger serializer must
    coerce inf → a JSON-representable sentinel (null or the string
    ``"inf"``) — flagged here as a downstream concern, not a
    responsibility of this function.
    """
    arr = _clean_array(returns)
    if arr.size == 0:
        return float("nan")
    gains = float(arr[arr > 0].sum())
    losses_abs = float((-arr[arr < 0]).sum())
    if gains == 0 and losses_abs == 0:
        return float("nan")
    if losses_abs == 0:
        return float("inf")
    return gains / losses_abs


# ---------------------------------------------------------------------------
# Information coefficient (two-series)
# ---------------------------------------------------------------------------


def _ic_series(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    method: str = "spearman",
    window: int = 20,
) -> pd.Series:
    """Rolling IC series between factor and forward returns.

    Pairs are aligned on the shared index; rows where either is NaN
    are dropped BEFORE the rolling correlation runs, so NaN in either
    series is non-destructive. Rank-based methods (Spearman, default)
    are computed by globally ranking both series first and then using
    rolling Pearson on the ranks.
    """
    idx = factor_values.index.intersection(forward_returns.index)
    a = factor_values.loc[idx]
    b = forward_returns.loc[idx]
    mask = ~(a.isna() | b.isna())
    a = a[mask]
    b = b[mask]
    if len(a) < window:
        return pd.Series(dtype=float)
    if method == "spearman":
        a = a.rank()
        b = b.rank()
    elif method != "pearson":
        raise ConfigError(f"IC method must be 'spearman' or 'pearson', got {method!r}")
    return a.rolling(window=window).corr(b)


def ic_mean(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    window: int = 20,
    method: str = "spearman",
) -> float:
    """Mean of the rolling IC series. NaN when IC is undefined over the span."""
    ic = _ic_series(factor_values, forward_returns, method=method, window=window)
    if ic.empty:
        return float("nan")
    val = float(ic.mean())
    return val if not math.isnan(val) else float("nan")


def ic_std(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    window: int = 20,
    method: str = "spearman",
) -> float:
    """Std of the rolling IC series."""
    ic = _ic_series(factor_values, forward_returns, method=method, window=window)
    if len(ic.dropna()) < 2:
        return float("nan")
    val = float(ic.std(ddof=1))
    return val if not math.isnan(val) else float("nan")


def ic_ir(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    *,
    window: int = 20,
    method: str = "spearman",
) -> float:
    """IC information ratio — ``ic_mean / ic_std``. NaN on zero-variance IC."""
    m = ic_mean(factor_values, forward_returns, window=window, method=method)
    s = ic_std(factor_values, forward_returns, window=window, method=method)
    if math.isnan(m) or math.isnan(s) or s == 0:
        return float("nan")
    return m / s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_array(returns: pd.Series) -> np.ndarray:
    """Drop NaN and return a float64 numpy array.

    Converting to ``np.ndarray`` early buys us clean types downstream
    — pandas-stubs over-broadens return types on Series arithmetic,
    and reducing to numpy early avoids the noise.
    """
    if not isinstance(returns, pd.Series):
        raise ConfigError(f"metric expected pd.Series, got {type(returns).__name__}")
    return returns.dropna().to_numpy(dtype=np.float64)


def _require_positive_ppy(periods_per_year: float) -> None:
    if periods_per_year <= 0:
        raise ConfigError(
            f"periods_per_year must be positive, got {periods_per_year!r} "
            f"(see metrics.py docstring for the frequency table)"
        )
