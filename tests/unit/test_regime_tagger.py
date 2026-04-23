"""Regime-tagger tests — labels, causality, thresholds.

Each of the three taggers (trend, volatility, funding) gets an
**explicit causality test** in the shape of the Phase-3 operator
tests: modify the input at indices >= cutoff, assert labels at
indices < cutoff are unchanged. This is the lookahead canary the
user asked for in the Phase-5 scope note — regime labels going
into the walk-forward engine is exactly the place where lookahead
bias sneaks in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.regime.tagger import (
    FundingLabel,
    TrendLabel,
    VolatilityLabel,
    tag_funding,
    tag_trend,
    tag_volatility,
)


def _price_series(n: int = 300, *, start: float = 100.0, drift: float = 0.001) -> pd.Series:
    """Log-normal-ish daily price path, deterministic via seed."""
    rng = np.random.default_rng(0)
    returns = rng.normal(drift, 0.02, size=n)
    prices = start * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(prices, index=idx, name="close")


def _funding_series(
    n: int = 300, *, loc: float = 0.0001, scale: float = 0.0002, seed: int = 1
) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(loc, scale, size=n)
    idx = pd.date_range("2023-01-01", periods=n, freq="8h", tz="UTC")
    return pd.Series(values, index=idx, name="funding_rate")


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class TestTagTrend:
    def test_labels_are_in_expected_set(self) -> None:
        close = _price_series(n=400, drift=0.002)
        labels = tag_trend(close, sma_window=50, slope_window=10)
        non_null = labels.dropna().unique().tolist()
        assert set(non_null) <= {"bull", "bear", "crab"}

    def test_warmup_is_nan(self) -> None:
        close = _price_series(n=100)
        labels = tag_trend(close, sma_window=50, slope_window=10)
        # First sma_window-1 + slope_window values have no label.
        assert labels.iloc[:50].isna().all()

    def test_strong_uptrend_produces_bull(self) -> None:
        """A strongly monotone-up series eventually labels bull."""
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        close = pd.Series(np.linspace(100.0, 400.0, 300), index=idx)
        labels = tag_trend(close, sma_window=50, slope_window=10)
        # Late-series labels should be bull.
        tail_labels = labels.iloc[-50:].dropna()
        assert (tail_labels == TrendLabel.BULL.value).mean() > 0.9

    def test_strong_downtrend_produces_bear(self) -> None:
        idx = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")
        close = pd.Series(np.linspace(400.0, 100.0, 300), index=idx)
        labels = tag_trend(close, sma_window=50, slope_window=10)
        tail_labels = labels.iloc[-50:].dropna()
        assert (tail_labels == TrendLabel.BEAR.value).mean() > 0.9

    @pytest.mark.parametrize("cutoff", [150, 250, 350])
    def test_causality_future_does_not_affect_past(self, cutoff: int) -> None:
        """The locked-in regime causality test.

        Parametrized over three cutoffs to prove the invariant holds at
        multiple temporal locations — a subtle indexing bug could cheat
        the check at one cutoff while failing at another.
        """
        close = _price_series(n=400)
        labels_full = tag_trend(close, sma_window=50, slope_window=10)

        clobbered = close.copy()
        clobbered.iloc[cutoff:] = 9999.0  # absurd future prices
        labels_mod = tag_trend(clobbered, sma_window=50, slope_window=10)

        pd.testing.assert_series_equal(
            labels_full.iloc[:cutoff],
            labels_mod.iloc[:cutoff],
            check_names=False,
        )

    def test_rejects_non_positive_window(self) -> None:
        close = _price_series(n=50)
        with pytest.raises(ConfigError, match="sma_window"):
            tag_trend(close, sma_window=0)
        with pytest.raises(ConfigError, match="slope_window"):
            tag_trend(close, slope_window=-1)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestTagVolatility:
    def test_labels_are_in_expected_set(self) -> None:
        close = _price_series(n=400)
        labels = tag_volatility(close, window=30)
        non_null = labels.dropna().unique().tolist()
        assert set(non_null) <= {"low_vol", "normal_vol", "high_vol"}

    def test_low_vol_regime_on_calm_series(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
        # Very low vol returns → low_vol label.
        returns = rng.normal(0.0, 0.005, size=400)
        close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)
        labels = tag_volatility(close, window=30)
        # Tail (after warmup) should be mostly low_vol.
        tail = labels.iloc[-50:].dropna()
        assert (tail == VolatilityLabel.LOW.value).mean() > 0.8

    def test_high_vol_regime_on_wild_series(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
        returns = rng.normal(0.0, 0.08, size=400)  # very high daily vol
        close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)
        labels = tag_volatility(close, window=30)
        tail = labels.iloc[-50:].dropna()
        assert (tail == VolatilityLabel.HIGH.value).mean() > 0.8

    @pytest.mark.parametrize("cutoff", [100, 200, 300])
    def test_causality_future_does_not_affect_past(self, cutoff: int) -> None:
        close = _price_series(n=400)
        labels_full = tag_volatility(close, window=30)

        clobbered = close.copy()
        clobbered.iloc[cutoff:] = 9999.0
        labels_mod = tag_volatility(clobbered, window=30)

        pd.testing.assert_series_equal(
            labels_full.iloc[:cutoff],
            labels_mod.iloc[:cutoff],
            check_names=False,
        )

    def test_rejects_inverted_thresholds(self) -> None:
        close = _price_series(n=100)
        with pytest.raises(ConfigError, match="thresholds"):
            tag_volatility(close, low_threshold=0.8, high_threshold=0.4)


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


class TestTagFunding:
    def test_labels_are_in_expected_set(self) -> None:
        funding = _funding_series(n=300)
        labels = tag_funding(funding, avg_window=21)  # 7 days at 8h cadence
        non_null = labels.dropna().unique().tolist()
        assert set(non_null) <= {"euphoric", "fearful", "neutral"}

    def test_high_positive_funding_produces_euphoric(self) -> None:
        """Average funding well above 0.05% per 8h → euphoric."""
        funding = _funding_series(n=300, loc=0.002, scale=0.0001)
        labels = tag_funding(funding, avg_window=10)
        tail = labels.iloc[-50:].dropna()
        assert (tail == FundingLabel.EUPHORIC.value).mean() > 0.9

    def test_high_negative_funding_produces_fearful(self) -> None:
        funding = _funding_series(n=300, loc=-0.002, scale=0.0001)
        labels = tag_funding(funding, avg_window=10)
        tail = labels.iloc[-50:].dropna()
        assert (tail == FundingLabel.FEARFUL.value).mean() > 0.9

    @pytest.mark.parametrize("cutoff", [100, 200, 275])
    def test_causality_future_does_not_affect_past(self, cutoff: int) -> None:
        funding = _funding_series(n=300)
        labels_full = tag_funding(funding, avg_window=21)

        clobbered = funding.copy()
        clobbered.iloc[cutoff:] = 9.999  # absurd future funding rates
        labels_mod = tag_funding(clobbered, avg_window=21)

        pd.testing.assert_series_equal(
            labels_full.iloc[:cutoff],
            labels_mod.iloc[:cutoff],
            check_names=False,
        )

    def test_rejects_inverted_thresholds(self) -> None:
        funding = _funding_series(n=50)
        with pytest.raises(ConfigError, match="fearful_threshold"):
            tag_funding(
                funding,
                euphoric_threshold=-0.001,
                fearful_threshold=0.001,
            )
