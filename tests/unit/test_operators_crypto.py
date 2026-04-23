"""Contract tests for the crypto-specific operators.

Most crypto operators are thin semantic wrappers over generic
timeseries kernels (``funding_z → ts_zscore``, ``*_change →
ts_pct_change``, ``spy_correlation → ts_corr``). The tests here verify
**the wrapping** — that the AST-level name delegates to the right
math — rather than re-testing the underlying math, which is covered
by ``test_operators_timeseries.py``.

Causality / determinism / registration are exercised via the
registry-walker canary in ``test_operators_causality.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.operators.crypto import (
    active_addresses_change,
    btc_dominance_change,
    dxy_change,
    fear_greed,
    funding_z,
    hashrate_change,
    oi_change,
    spy_correlation,
    stablecoin_mcap_change,
)
from crypto_alpha_engine.operators.timeseries import (
    ts_corr,
    ts_mean,
    ts_pct_change,
    ts_zscore,
)


def _sample(n: int = 50, *, seed: int = 1, mean: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(mean, 1.0, size=n), index=idx, name="x")


# ---------------------------------------------------------------------------
# funding_z
# ---------------------------------------------------------------------------


class TestFundingZ:
    def test_delegates_to_ts_zscore(self) -> None:
        funding = _sample(n=50, seed=1)
        pd.testing.assert_series_equal(
            funding_z(funding, window=10),
            ts_zscore(funding, window=10),
        )

    def test_mean_near_zero_for_stationary_input(self) -> None:
        funding = _sample(n=200, seed=7, mean=0.0001)
        z = funding_z(funding, window=20).dropna()
        assert abs(z.mean()) < 0.5  # loose bound — it's a z-score after all


# ---------------------------------------------------------------------------
# oi_change, *_change family — all delegate to ts_pct_change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        oi_change,
        btc_dominance_change,
        stablecoin_mcap_change,
        active_addresses_change,
        hashrate_change,
        dxy_change,
    ],
)
def test_change_family_delegates_to_ts_pct_change(fn: object) -> None:
    x = _sample(n=30, seed=42, mean=1000.0)
    direct = ts_pct_change(x, lag=5)
    via_alias = fn(x, 5)  # type: ignore[operator]
    pd.testing.assert_series_equal(via_alias, direct)


# ---------------------------------------------------------------------------
# fear_greed
# ---------------------------------------------------------------------------


class TestFearGreed:
    def test_is_rolling_mean(self) -> None:
        fg = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        pd.testing.assert_series_equal(
            fear_greed(fg, window=3),
            ts_mean(fg, window=3),
        )


# ---------------------------------------------------------------------------
# spy_correlation
# ---------------------------------------------------------------------------


class TestSpyCorrelation:
    def test_delegates_to_ts_corr(self) -> None:
        a = _sample(n=50, seed=1)
        b = _sample(n=50, seed=2)
        pd.testing.assert_series_equal(
            spy_correlation(a, b, window=10),
            ts_corr(a, b, window=10),
        )

    def test_correlation_with_self_is_one(self) -> None:
        a = _sample(n=50, seed=1)
        out = spy_correlation(a, a, window=10).dropna()
        assert (out > 0.999).all()
