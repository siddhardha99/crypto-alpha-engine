"""Integration tests that run operators against real Phase-2 fixtures.

Unit tests use synthetic data (``np.random.default_rng``). These tests
use the real market-data parquets committed under ``tests/fixtures/``
— January 2022 slices of BTC/USD + ETH/USD hourly OHLCV and BTC perp
funding. They validate that operator kernels produce sensible output
on production-shaped inputs with real noise, not just on textbook
series.

The tests deliberately make *structural* assertions (output length,
warmup NaN count, value range bounds) rather than specific-value
ones, so fixture refreshes don't require test changes unless the
underlying data shape actually changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_alpha_engine.operators.crypto import (
    dxy_change,
    funding_z,
    oi_change,
)
from crypto_alpha_engine.operators.timeseries import (
    ts_corr,
    ts_mean,
    ts_pct_change,
    ts_zscore,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# funding_z on BTC perp funding fixture (January 2022)
# ---------------------------------------------------------------------------


def test_funding_z_on_real_bitmex_funding_january_2022() -> None:
    df = pd.read_parquet(FIXTURES / "btc_funding_8h_bitmex_perp.parquet")
    funding = df.set_index("timestamp")["funding_rate"]

    window = 10
    z = funding_z(funding, window=window)

    # Structural assertions on the output shape.
    assert len(z) == len(funding)
    assert z.iloc[: window - 1].isna().all(), "first window-1 outputs must be NaN (warmup)"
    assert not z.iloc[window - 1 :].isna().any(), "no NaN after warmup on real data"

    # Delegation: funding_z ≡ ts_zscore on the same series + window.
    pd.testing.assert_series_equal(z, ts_zscore(funding, window=window))

    # Sanity range: real funding z-scores sit well within ±10.
    assert z.abs().max() < 10.0, "z-score of real funding is implausibly extreme"


# ---------------------------------------------------------------------------
# Returns + oi_change semantics using BTC OHLCV fixture
# ---------------------------------------------------------------------------


def test_pct_change_on_btc_close_matches_manual_return() -> None:
    df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
    close = df.set_index("timestamp")["close"]
    # Hourly returns: compare ts_pct_change at lag=1 vs hand-computed.
    hourly_return = ts_pct_change(close, lag=1)
    manual = close / close.shift(1) - 1
    pd.testing.assert_series_equal(hourly_return, manual, check_names=False)


def test_oi_change_is_pct_change_alias_on_a_real_series() -> None:
    """oi_change is a pure AST-vocabulary alias for ts_pct_change."""
    df = pd.read_parquet(FIXTURES / "btc_funding_8h_bitmex_perp.parquet")
    # We don't have a fixture-level OI series; funding_rate works as a
    # stand-in — we're verifying the ALIAS relationship, not semantics.
    series = df.set_index("timestamp")["funding_rate"]
    pd.testing.assert_series_equal(oi_change(series, 3), ts_pct_change(series, lag=3))


# ---------------------------------------------------------------------------
# Causality check on real data — the modify-future-past-unchanged rule
# holds on actual noisy market data, not just synthetic.
# ---------------------------------------------------------------------------


def test_ts_mean_causality_on_real_btc_close() -> None:
    df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
    close = df.set_index("timestamp")["close"]

    window = 24  # 24-hour rolling mean
    full = ts_mean(close, window=window)

    cutoff = len(close) // 2
    clobbered = close.copy()
    clobbered.iloc[cutoff:] = 1e9  # absurd future values
    modified = ts_mean(clobbered, window=window)

    pd.testing.assert_series_equal(full.iloc[:cutoff], modified.iloc[:cutoff])


# ---------------------------------------------------------------------------
# Cross-series: ts_corr across BTC + ETH real closes
# ---------------------------------------------------------------------------


def test_rolling_btc_eth_correlation_is_mostly_positive_in_jan_2022() -> None:
    btc_df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
    eth_df = pd.read_parquet(FIXTURES / "eth_usd_1h_coinbase_spot.parquet")

    # Align on timestamp.
    btc = btc_df.set_index("timestamp")["close"]
    eth = eth_df.set_index("timestamp")["close"]
    common = btc.index.intersection(eth.index)
    btc = btc.loc[common]
    eth = eth.loc[common]

    rho = ts_corr(btc, eth, window=24).dropna()
    # BTC and ETH were highly correlated during Jan 2022 — not a
    # perfect physical law, but consistent enough that a mostly-
    # positive median is a sensible assertion.
    assert rho.median() > 0.3


# ---------------------------------------------------------------------------
# dxy_change on synthetic near-flat series — sanity check the alias.
# ---------------------------------------------------------------------------


def test_dxy_change_alias_on_constructed_series() -> None:
    """Belt-and-suspenders: the dxy_change → ts_pct_change alias holds."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    # A linearly-growing series so pct_change is non-trivial and deterministic.
    series = pd.Series(np.linspace(100.0, 200.0, 100), index=idx, name="dxy")
    pd.testing.assert_series_equal(dxy_change(series, 5), ts_pct_change(series, lag=5))
