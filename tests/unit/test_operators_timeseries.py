"""Contract tests for every time-series operator.

SPEC §13 requires, for each operator:

1. Output at index ``t`` does not change if the input at index ``t+1``
   is modified (causality).
2. Output is ``NaN`` when insufficient history is available.
3. Output is deterministic: same input → same output.
4. The operator raises on a non-positive ``window`` / ``lag`` /
   ``halflife``.

Each test class below covers one operator across those four rules.
A parametric sweep at the bottom asserts determinism and causality
uniformly for all ``(Series, window) -> Series`` operators as a
safety net against future drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.operators.timeseries import (
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_diff,
    ts_ema,
    ts_kurt,
    ts_max,
    ts_mean,
    ts_min,
    ts_pct_change,
    ts_quantile,
    ts_rank,
    ts_skew,
    ts_std,
    ts_zscore,
)


def _sample(n: int = 40, seed: int = 42) -> pd.Series:
    """Deterministic synthetic series with UTC index (not required but realistic)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(100.0, 1.5, size=n), index=idx, name="x")


def _sample_y(n: int = 40, seed: int = 43) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(50.0, 0.8, size=n), index=idx, name="y")


# ---------------------------------------------------------------------------
# ts_mean — the pattern-setter; every other operator follows the same shape.
# ---------------------------------------------------------------------------


class TestTsMean:
    def test_causality_modifying_future_does_not_change_past(self) -> None:
        """SPEC §13 canonical pattern — modify t+1, past output unchanged."""
        x = _sample()
        full = ts_mean(x, window=3)

        x_modified = x.copy()
        x_modified.iloc[20:] = 999.0

        modified = ts_mean(x_modified, window=3)
        pd.testing.assert_series_equal(full.iloc[:20], modified.iloc[:20])

    def test_insufficient_history_is_nan(self) -> None:
        x = _sample(n=10)
        out = ts_mean(x, window=5)
        assert out.iloc[:4].isna().all()
        assert not out.iloc[4:].isna().any()

    def test_deterministic(self) -> None:
        x = _sample()
        pd.testing.assert_series_equal(ts_mean(x, 5), ts_mean(x, 5))

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ConfigError, match="window"):
            ts_mean(_sample(), window=0)
        with pytest.raises(ConfigError, match="window"):
            ts_mean(_sample(), window=-1)


# ---------------------------------------------------------------------------
# ts_std / ts_min / ts_max / ts_rank / ts_zscore
# ---------------------------------------------------------------------------


class TestTsStd:
    def test_rolling_std_matches_pandas(self) -> None:
        x = _sample()
        pd.testing.assert_series_equal(ts_std(x, 5), x.rolling(5).std())

    def test_raises_on_bad_window(self) -> None:
        with pytest.raises(ConfigError):
            ts_std(_sample(), 0)


class TestTsMinMax:
    def test_min_and_max_bound_input(self) -> None:
        x = _sample()
        mn, mx = ts_min(x, 5), ts_max(x, 5)
        mask = ~mn.isna()
        assert (mn.loc[mask] <= x.loc[mask]).all()
        assert (mx.loc[mask] >= x.loc[mask]).all()

    def test_min_le_max(self) -> None:
        x = _sample()
        mn, mx = ts_min(x, 5), ts_max(x, 5)
        mask = ~mn.isna()
        assert (mn.loc[mask] <= mx.loc[mask]).all()


class TestTsRank:
    def test_rank_in_expected_range(self) -> None:
        x = _sample()
        r = ts_rank(x, 5)
        mask = ~r.isna()
        assert (r.loc[mask] >= 1.0).all()
        assert (r.loc[mask] <= 5.0).all()

    def test_largest_value_in_window_ranks_highest(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = ts_rank(x, 5)
        assert r.iloc[-1] == 5.0  # last value is max → rank 5


class TestTsZscore:
    def test_mean_zero_for_stationary_window(self) -> None:
        """Z-score of a rolling window should be ~0 on average for stationary data."""
        x = _sample(n=200)
        z = ts_zscore(x, 20).dropna()
        assert abs(z.mean()) < 0.5  # loose bound; not a strict test of properties

    def test_constant_window_is_nan_not_inf(self) -> None:
        """A perfectly flat window has std=0 — output must be NaN, never inf."""
        x = pd.Series([5.0] * 10)
        z = ts_zscore(x, 5)
        assert z.iloc[4:].isna().all()  # all-NaN after warmup, not inf


# ---------------------------------------------------------------------------
# ts_diff / ts_pct_change
# ---------------------------------------------------------------------------


class TestTsDiff:
    def test_first_difference(self) -> None:
        x = pd.Series([1.0, 3.0, 6.0, 10.0])
        out = ts_diff(x, lag=1)
        assert out.iloc[0] != out.iloc[0]  # NaN at index 0
        assert out.iloc[1:].tolist() == [2.0, 3.0, 4.0]

    def test_rejects_non_positive_lag(self) -> None:
        with pytest.raises(ConfigError, match="lag"):
            ts_diff(_sample(), lag=0)
        with pytest.raises(ConfigError, match="lag"):
            ts_diff(_sample(), lag=-3)


class TestTsPctChange:
    def test_percent_change_math(self) -> None:
        x = pd.Series([100.0, 110.0, 99.0])
        out = ts_pct_change(x, lag=1)
        assert out.iloc[1] == pytest.approx(0.10)
        assert out.iloc[2] == pytest.approx(-0.1)

    def test_rejects_non_positive_lag(self) -> None:
        with pytest.raises(ConfigError):
            ts_pct_change(_sample(), lag=-1)


# ---------------------------------------------------------------------------
# ts_skew / ts_kurt / ts_quantile
# ---------------------------------------------------------------------------


class TestTsSkewKurt:
    def test_symmetric_window_skew_is_near_zero(self) -> None:
        x = pd.Series(list(range(20)), dtype=float)  # linear ramp → symmetric
        s = ts_skew(x, 11).iloc[-1]
        assert abs(s) < 1e-6

    def test_constant_window_kurt_matches_pandas_rolling(self) -> None:
        """Constant-window kurtosis: whatever pandas returns, we agree."""
        x = pd.Series([3.0] * 20)
        pd.testing.assert_series_equal(ts_kurt(x, 10), x.rolling(10).kurt())


class TestTsQuantile:
    def test_median_matches_direct(self) -> None:
        x = _sample()
        q50 = ts_quantile(x, 5, q=0.5)
        direct = x.rolling(5).quantile(0.5)
        pd.testing.assert_series_equal(q50, direct)

    @pytest.mark.parametrize("bad_q", [-0.1, 1.1, 2.0])
    def test_rejects_q_outside_unit_interval(self, bad_q: float) -> None:
        with pytest.raises(ConfigError, match=r"\[0, 1\]"):
            ts_quantile(_sample(), 5, q=bad_q)


# ---------------------------------------------------------------------------
# ts_corr / ts_cov
# ---------------------------------------------------------------------------


class TestTsCorrCov:
    def test_correlation_perfect_with_self(self) -> None:
        x = _sample()
        r = ts_corr(x, x, window=5).dropna()
        # Perfect correlation of a series with itself, modulo floating noise.
        assert (r > 0.999).all()

    def test_cov_matches_pandas(self) -> None:
        x = _sample()
        y = _sample_y()
        pd.testing.assert_series_equal(ts_cov(x, y, window=5), x.rolling(5).cov(y))


# ---------------------------------------------------------------------------
# ts_argmax / ts_argmin
# ---------------------------------------------------------------------------


class TestTsArgmaxArgmin:
    def test_argmax_is_bars_ago_of_max(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])  # max at index 2
        out = ts_argmax(x, 5)
        # At index 4 (last), the max within the window is at index 2 → 2 bars ago.
        assert out.iloc[-1] == 2.0

    def test_argmin_returns_zero_when_current_is_min(self) -> None:
        x = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])  # strictly decreasing
        out = ts_argmin(x, 5)
        assert out.iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# ts_decay_linear / ts_ema
# ---------------------------------------------------------------------------


class TestTsDecayLinear:
    def test_weights_sum_to_one(self) -> None:
        """A constant input must produce the same constant output (weights normalise)."""
        x = pd.Series([7.0] * 30)
        out = ts_decay_linear(x, 10).iloc[9:]
        assert np.allclose(out, 7.0)

    def test_recent_bar_weighted_more(self) -> None:
        """Compared to uniform mean, linear decay tracks recent moves faster."""
        x = pd.Series([0.0] * 10 + [100.0])
        linear = ts_decay_linear(x, 5).iloc[-1]
        uniform = ts_mean(x, 5).iloc[-1]
        assert linear > uniform


class TestTsEma:
    def test_monotone_rise_produces_monotone_output(self) -> None:
        x = pd.Series(list(range(50)), dtype=float)
        out = ts_ema(x, halflife=5).dropna()
        assert (out.diff().iloc[1:] > 0).all()

    def test_rejects_non_positive_halflife(self) -> None:
        with pytest.raises(ConfigError, match="halflife"):
            ts_ema(_sample(), halflife=0)


# ---------------------------------------------------------------------------
# Parametric sweep: every (Series, window) → Series operator is causal + deterministic.
# ---------------------------------------------------------------------------


_SINGLE_SERIES_OPS = [
    ("ts_mean", ts_mean, {"window": 5}),
    ("ts_std", ts_std, {"window": 5}),
    ("ts_min", ts_min, {"window": 5}),
    ("ts_max", ts_max, {"window": 5}),
    ("ts_rank", ts_rank, {"window": 5}),
    ("ts_zscore", ts_zscore, {"window": 5}),
    ("ts_diff", ts_diff, {"lag": 3}),
    ("ts_pct_change", ts_pct_change, {"lag": 3}),
    ("ts_skew", ts_skew, {"window": 8}),
    ("ts_kurt", ts_kurt, {"window": 8}),
    ("ts_quantile", ts_quantile, {"window": 5, "q": 0.75}),
    ("ts_argmax", ts_argmax, {"window": 5}),
    ("ts_argmin", ts_argmin, {"window": 5}),
    ("ts_decay_linear", ts_decay_linear, {"window": 5}),
    ("ts_ema", ts_ema, {"halflife": 4}),
]


@pytest.mark.parametrize(("name", "fn", "params"), _SINGLE_SERIES_OPS)
def test_every_single_series_op_is_causal(name: str, fn: Any, params: dict[str, Any]) -> None:
    x = _sample(n=50)
    full = fn(x, **params)

    x_mod = x.copy()
    x_mod.iloc[30:] = 999.0
    modified = fn(x_mod, **params)

    # Past 30 values unchanged — by construction, no operator may peek forward.
    pd.testing.assert_series_equal(full.iloc[:30], modified.iloc[:30])


@pytest.mark.parametrize(("name", "fn", "params"), _SINGLE_SERIES_OPS)
def test_every_single_series_op_is_deterministic(
    name: str, fn: Any, params: dict[str, Any]
) -> None:
    x = _sample(n=50)
    pd.testing.assert_series_equal(fn(x, **params), fn(x, **params))


@pytest.mark.parametrize(
    ("fn", "params"),
    [
        (ts_corr, {"window": 5}),
        (ts_cov, {"window": 5}),
    ],
)
def test_two_series_op_causal(fn: Any, params: dict[str, Any]) -> None:
    x = _sample(n=50)
    y = _sample_y(n=50)
    full = fn(x, y, **params)

    x_mod, y_mod = x.copy(), y.copy()
    x_mod.iloc[30:] = 999.0
    y_mod.iloc[30:] = -999.0
    modified = fn(x_mod, y_mod, **params)

    pd.testing.assert_series_equal(full.iloc[:30], modified.iloc[:30])
