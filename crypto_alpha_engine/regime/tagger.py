"""Causal regime taggers.

Per SPEC §9, every timestamp is tagged along three independent
regime dimensions. This module ships one function per dimension,
each a pure causal transform from a data series to a label series:

* :func:`tag_trend` — ``bull`` / ``bear`` / ``crab`` based on
  BTC daily close vs its 200-day SMA plus the 30-day SMA slope.
* :func:`tag_volatility` — ``low_vol`` / ``normal_vol`` /
  ``high_vol`` based on 30-day rolling annualised volatility.
* :func:`tag_funding` — ``euphoric`` / ``fearful`` / ``neutral``
  based on the trailing 7-day average funding rate.

Causality is non-negotiable
---------------------------

Every tagger uses only ``.rolling(window=w)`` with positive
windows (trailing, not centered). Never ``.shift(-n)``, never
``.rolling(..., center=True)``. The walk-forward engine in
Phase 6 consumes these labels; a lookahead here would poison
every downstream regime-segmented metric.

The ``test_regime_tagger.py`` file contains a causality test for
each of the three functions in the same shape as the Phase-3
operator tests: modify input values in the future half, assert
past labels are unchanged.

Not using @register_operator
----------------------------

Regime taggers return a ``pd.Series`` of string labels, not floats.
The Phase-3 operator ``arg_types`` vocabulary (series → series of
numerics) can't express that shape cleanly. Rather than expanding
the vocabulary with a ``label_series`` tag (which we'd use exactly
three times), taggers live as standalone module-level functions
and get the same causality-test discipline via explicit tests
instead of the operator canary.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError


class TrendLabel(StrEnum):
    """Trend-regime labels from SPEC §9."""

    BULL = "bull"
    BEAR = "bear"
    CRAB = "crab"


class VolatilityLabel(StrEnum):
    """Volatility-regime labels from SPEC §9."""

    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"


class FundingLabel(StrEnum):
    """Funding-regime labels from SPEC §9."""

    EUPHORIC = "euphoric"
    FEARFUL = "fearful"
    NEUTRAL = "neutral"


# SPEC §9 thresholds — exposed as module constants so Phase 6+ can
# reason about them without re-reading the implementation.
TREND_SMA_WINDOW: int = 200  # days
TREND_SLOPE_WINDOW: int = 30  # days; slope of the 200d SMA over this window
VOL_WINDOW: int = 30  # days; rolling realised vol window
VOL_LOW_THRESHOLD: float = 0.40  # annualised
VOL_HIGH_THRESHOLD: float = 0.80
FUNDING_AVG_WINDOW: int = 7  # days; rolling average of funding rate
FUNDING_EUPHORIC_THRESHOLD: float = 0.0005  # 0.05% per period (8h)
FUNDING_FEARFUL_THRESHOLD: float = -0.0002  # -0.02% per period (8h)


# ---------------------------------------------------------------------------
# Trend regime
# ---------------------------------------------------------------------------


def tag_trend(
    close: pd.Series,
    *,
    sma_window: int = TREND_SMA_WINDOW,
    slope_window: int = TREND_SLOPE_WINDOW,
) -> pd.Series:
    """Classify each bar as ``bull`` / ``bear`` / ``crab`` (SPEC §9).

    * ``bull``: ``close > 200d SMA`` AND ``SMA slope over last 30d > 0``.
    * ``bear``: ``close < 200d SMA`` AND ``SMA slope over last 30d < 0``.
    * ``crab``: everything else.

    Input is expected to be a daily close series. For hourly data,
    resample to daily first. NaN is emitted as the label for bars
    before the warmup window is filled (``sma_window - 1 + slope_window``).

    Causality: uses only ``.rolling(window=w)`` (trailing) and
    ``.shift(slope_window)`` with positive ``slope_window``. No lookahead.

    Raises:
        ConfigError: If ``sma_window`` or ``slope_window`` is non-positive.
    """
    _require_positive(sma_window, "sma_window")
    _require_positive(slope_window, "slope_window")

    sma = close.rolling(window=sma_window).mean()
    # Slope over slope_window days = (SMA_today - SMA_slope_window_ago) / slope_window.
    # .shift(+slope_window) pulls past values forward — purely causal.
    slope = (sma - sma.shift(slope_window)) / float(slope_window)

    above = close > sma
    rising = slope > 0
    below = close < sma
    falling = slope < 0

    labels = pd.Series(index=close.index, dtype=object)
    labels[:] = None  # unfilled positions

    # Where both SMA and slope are valid, pick a concrete label.
    valid = ~(sma.isna() | slope.isna())
    is_bull = valid & above & rising
    is_bear = valid & below & falling
    is_crab = valid & ~is_bull & ~is_bear

    labels[is_bull] = TrendLabel.BULL.value
    labels[is_bear] = TrendLabel.BEAR.value
    labels[is_crab] = TrendLabel.CRAB.value
    return labels


# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------


def tag_volatility(
    close: pd.Series,
    *,
    window: int = VOL_WINDOW,
    low_threshold: float = VOL_LOW_THRESHOLD,
    high_threshold: float = VOL_HIGH_THRESHOLD,
) -> pd.Series:
    """Classify each bar's volatility regime (SPEC §9).

    Volatility = rolling std of daily log-returns, annualised with
    ``√365`` (crypto convention — markets trade 24/7 every day of the
    year, so 365 not 252).

    * ``low_vol``: < 40% annualised
    * ``high_vol``: > 80% annualised
    * ``normal_vol``: in between

    Input is expected to be a daily close series.

    Raises:
        ConfigError: On bad window or thresholds.
    """
    _require_positive(window, "window")
    if not (0 < low_threshold < high_threshold):
        raise ConfigError(
            f"vol thresholds must satisfy 0 < low ({low_threshold}) " f"< high ({high_threshold})"
        )

    # Log-returns: ln(close_t / close_{t-1}) — causal (one-period shift).
    ratio = close / close.shift(1)
    log_ret = pd.Series(np.log(ratio.to_numpy()), index=close.index)
    vol = log_ret.rolling(window=window).std() * math.sqrt(365.0)

    labels = pd.Series(index=close.index, dtype=object)
    labels[:] = None
    valid = ~vol.isna()
    is_low = valid & (vol < low_threshold)
    is_high = valid & (vol > high_threshold)
    is_normal = valid & ~is_low & ~is_high
    labels[is_low] = VolatilityLabel.LOW.value
    labels[is_high] = VolatilityLabel.HIGH.value
    labels[is_normal] = VolatilityLabel.NORMAL.value
    return labels


# ---------------------------------------------------------------------------
# Funding regime
# ---------------------------------------------------------------------------


def tag_funding(
    funding_rate: pd.Series,
    *,
    avg_window: int = FUNDING_AVG_WINDOW,
    euphoric_threshold: float = FUNDING_EUPHORIC_THRESHOLD,
    fearful_threshold: float = FUNDING_FEARFUL_THRESHOLD,
) -> pd.Series:
    """Classify each bar's funding regime (SPEC §9).

    Uses the rolling-mean of the funding-rate series. Bars where the
    rolling mean exceeds ``euphoric_threshold`` are ``euphoric``;
    below ``fearful_threshold``, ``fearful``; otherwise ``neutral``.

    ``avg_window`` is in units of the funding series itself — for a
    Bybit/BitMEX 8-hour funding series, a 7-day window is ``7 * 3 = 21``
    periods. Callers compute this mapping; SPEC §9 specifies 7 days
    but this function works on the series' native cadence.

    Raises:
        ConfigError: On bad window or inverted thresholds.
    """
    _require_positive(avg_window, "avg_window")
    if fearful_threshold >= euphoric_threshold:
        raise ConfigError(
            f"fearful_threshold ({fearful_threshold}) must be strictly "
            f"less than euphoric_threshold ({euphoric_threshold})"
        )

    rolling_mean = funding_rate.rolling(window=avg_window).mean()

    labels = pd.Series(index=funding_rate.index, dtype=object)
    labels[:] = None
    valid = ~rolling_mean.isna()
    is_euphoric = valid & (rolling_mean > euphoric_threshold)
    is_fearful = valid & (rolling_mean < fearful_threshold)
    is_neutral = valid & ~is_euphoric & ~is_fearful
    labels[is_euphoric] = FundingLabel.EUPHORIC.value
    labels[is_fearful] = FundingLabel.FEARFUL.value
    labels[is_neutral] = FundingLabel.NEUTRAL.value
    return labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_positive(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive int, got {value!r}")
