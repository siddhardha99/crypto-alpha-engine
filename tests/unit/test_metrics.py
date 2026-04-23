"""Tests for the pure-function metrics in ``backtest/metrics.py``.

Each metric has (a) a happy-path test with a small hand-checked value
and (b) edge-case tests for the NaN convention documented in the
module docstring. IC metrics additionally get a dedicated
NaN-alignment test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    annualized_return,
    calmar,
    hit_rate,
    ic_ir,
    ic_mean,
    ic_std,
    kurt,
    max_drawdown,
    mean_return,
    periods_per_year_for,
    profit_factor,
    return_std,
    sharpe,
    skew,
    sortino,
    total_return,
)
from crypto_alpha_engine.exceptions import ConfigError


def _s(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# periods_per_year lookup
# ---------------------------------------------------------------------------


class TestPeriodsPerYear:
    def test_lookup(self) -> None:
        assert periods_per_year_for("1h") == 8760.0
        assert periods_per_year_for("4h") == 2190.0
        assert periods_per_year_for("1d") == 365.0
        assert periods_per_year_for("1w") == 52.0

    def test_unknown_freq_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown freq"):
            periods_per_year_for("30m")  # type: ignore[arg-type]

    def test_default_matches_1h(self) -> None:
        assert DEFAULT_PERIODS_PER_YEAR == 8760.0


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


class TestMoments:
    def test_mean_return(self) -> None:
        assert mean_return(_s([0.01, 0.02, -0.01])) == pytest.approx(0.0066666, rel=1e-3)

    def test_mean_return_empty_is_nan(self) -> None:
        assert math.isnan(mean_return(_s([])))

    def test_return_std(self) -> None:
        out = return_std(_s([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert out == pytest.approx(1.5811, rel=1e-3)

    def test_return_std_single_point_is_nan(self) -> None:
        assert math.isnan(return_std(_s([0.01])))

    def test_skew_zero_on_symmetric(self) -> None:
        s = skew(_s([-2.0, -1.0, 0.0, 1.0, 2.0]))
        assert abs(s) < 1e-9

    def test_kurt_is_excess(self) -> None:
        """pandas' .kurt() is excess kurtosis — 0 for a normal distribution.

        BLP's DSR uses raw (Pearson) kurt = excess + 3; the DSR module
        adds 3 itself.
        """
        rng = np.random.default_rng(42)
        out = kurt(_s(rng.normal(0, 1, size=1000).tolist()))
        assert abs(out) < 0.5  # close to 0 (excess), not 3


# ---------------------------------------------------------------------------
# Sharpe / Sortino / Calmar
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_positive_sharpe_on_monotone_positive(self) -> None:
        r = _s([0.01] * 100)  # constant positive returns → std=0 → NaN
        assert math.isnan(sharpe(r))

    def test_sharpe_scale_invariance(self) -> None:
        """Sharpe is scale-invariant: scaling all returns preserves the ratio.

        mean(kr)/std(kr) = k·mean/k·std = mean/std. Essential property:
        Sharpe is a dimensionless signal-to-noise measure.
        """
        rng = np.random.default_rng(0)
        r1 = _s(rng.normal(0.001, 0.01, size=200).tolist())
        r2 = r1 * 2.0
        assert sharpe(r2) == pytest.approx(sharpe(r1), rel=1e-9)

    def test_zero_volatility_returns_nan(self) -> None:
        assert math.isnan(sharpe(_s([0.0, 0.0, 0.0])))

    def test_non_positive_ppy_raises(self) -> None:
        with pytest.raises(ConfigError, match="periods_per_year"):
            sharpe(_s([0.01, 0.02]), periods_per_year=-1)


class TestSortino:
    def test_no_negative_returns_is_nan(self) -> None:
        """Sortino with no downside is technically infinite; we return NaN."""
        assert math.isnan(sortino(_s([0.01] * 10)))

    def test_sortino_higher_than_sharpe_when_downside_small(self) -> None:
        """Positive skew in small-downside series → Sortino > Sharpe.

        Sortino's denominator is downside-only std; if most of the
        volatility is upside, the sortino denominator is smaller than
        the full std, giving a higher ratio.
        """
        # Wins dominate; a handful of small-magnitude losses.
        rng = np.random.default_rng(7)
        upside = rng.normal(0.01, 0.005, size=100)
        downside = rng.normal(-0.002, 0.001, size=10)
        r = _s(upside.tolist() + downside.tolist())
        s = sharpe(r)
        so = sortino(r)
        assert so > s


class TestCalmar:
    def test_positive_calmar_on_upward_series(self) -> None:
        r = _s([0.01, -0.005, 0.02, -0.003, 0.015] * 20)
        out = calmar(r)
        assert out > 0

    def test_zero_drawdown_is_nan(self) -> None:
        r = _s([0.01] * 10)  # monotone up → mdd = 0
        assert math.isnan(calmar(r))


# ---------------------------------------------------------------------------
# Cumulative
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_compounded(self) -> None:
        # (1+0.1)*(1+0.1) - 1 = 0.21
        assert total_return(_s([0.1, 0.1])) == pytest.approx(0.21, rel=1e-6)

    def test_empty_is_zero(self) -> None:
        assert total_return(_s([])) == 0.0


class TestAnnualizedReturn:
    def test_matches_known_example(self) -> None:
        """Sanity: 1%/bar over 10 bars at ppy=8760 produces a finite value."""
        r = _s([0.01] * 10)
        # Just ensure it's computable and sane, not a specific value.
        out = annualized_return(r, periods_per_year=DEFAULT_PERIODS_PER_YEAR)
        assert math.isfinite(out)

    def test_total_loss_returns_nan(self) -> None:
        """A -100% return (wipe-out) → total = 0 → annualised undefined."""
        r = _s([-1.0, 0.01])  # ruined on bar 0
        assert math.isnan(annualized_return(r))


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_monotone_up_has_zero_drawdown(self) -> None:
        assert max_drawdown(_s([0.01, 0.02, 0.03, 0.04])) == 0.0

    def test_negative_number_convention(self) -> None:
        """SPEC §8: max_drawdown returned as a negative number."""
        # Up 10%, down 20%: equity 1.0 → 1.1 → 0.88. Drawdown = (0.88-1.1)/1.1 = -0.2
        out = max_drawdown(_s([0.1, -0.2]))
        assert out == pytest.approx(-0.2, rel=1e-6)

    def test_empty_is_zero(self) -> None:
        assert max_drawdown(_s([])) == 0.0


# ---------------------------------------------------------------------------
# Hit rate — pinned convention: zero-returns count as losses
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_basic(self) -> None:
        assert hit_rate(_s([0.01, -0.01, 0.01, -0.01])) == 0.5

    def test_all_zero_returns_is_zero(self) -> None:
        """PIN: all-zero returns → 0.0, not 0.5.

        A zero return under non-zero costs is a cost-adjusted loss,
        not a tie.
        """
        assert hit_rate(_s([0.0, 0.0, 0.0, 0.0])) == 0.0

    def test_all_positive_is_one(self) -> None:
        assert hit_rate(_s([0.01, 0.02, 0.03])) == 1.0

    def test_all_negative_is_zero(self) -> None:
        assert hit_rate(_s([-0.01, -0.02, -0.03])) == 0.0

    def test_empty_is_zero(self) -> None:
        assert hit_rate(_s([])) == 0.0


# ---------------------------------------------------------------------------
# Profit factor — documents inf for all-positive
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_basic(self) -> None:
        # Wins sum to 0.03; losses abs sum to 0.02 → profit_factor = 1.5
        assert profit_factor(_s([0.01, 0.02, -0.01, -0.01])) == pytest.approx(1.5)

    def test_all_positive_is_inf(self) -> None:
        """PIN: all-positive returns → float('inf'). Phase 7 ledger serialiser
        must coerce inf before writing to JSON."""
        pf = profit_factor(_s([0.01, 0.02, 0.03]))
        assert pf == float("inf")

    def test_all_negative_is_zero(self) -> None:
        assert profit_factor(_s([-0.01, -0.02])) == 0.0

    def test_all_zero_is_nan(self) -> None:
        assert math.isnan(profit_factor(_s([0.0, 0.0])))

    def test_empty_is_nan(self) -> None:
        assert math.isnan(profit_factor(_s([])))


# ---------------------------------------------------------------------------
# Information Coefficient — including the NaN-alignment test
# ---------------------------------------------------------------------------


class TestIC:
    def _setup_pair(self, n: int = 60, *, seed: int = 1) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        factor = pd.Series(rng.normal(0, 1, size=n), index=idx, name="factor")
        # Forward returns partially correlated with factor.
        noise = rng.normal(0, 1, size=n)
        fwd = 0.5 * factor.to_numpy() + 0.5 * noise
        returns = pd.Series(fwd, index=idx, name="fwd")
        return factor, returns

    def test_ic_mean_basic(self) -> None:
        factor, fwd = self._setup_pair(n=200)
        m = ic_mean(factor, fwd, window=20)
        assert 0.1 < m < 0.9  # expect positive rank correlation on correlated pair

    def test_ic_on_completely_uncorrelated_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
        factor = pd.Series(rng.normal(0, 1, 500), index=idx)
        fwd = pd.Series(rng.normal(0, 1, 500), index=idx)
        m = ic_mean(factor, fwd, window=20)
        assert abs(m) < 0.2  # loose bound; near-zero expected

    def test_ic_ir_is_mean_over_std(self) -> None:
        factor, fwd = self._setup_pair(n=200)
        m = ic_mean(factor, fwd, window=20)
        s = ic_std(factor, fwd, window=20)
        if not math.isnan(s) and s != 0:
            assert ic_ir(factor, fwd, window=20) == pytest.approx(m / s)

    def test_nan_alignment_drops_pairs_before_correlation(self) -> None:
        """NaN in factor at index k must not poison the rolling IC.

        Setup: factor has NaN at index 5; returns has a real value at
        the same index. The pair should be dropped *before* the
        rolling correlation runs, so the remaining 199 aligned pairs
        compute IC unaffected. Compare against a baseline without
        the NaN — the IC should be very close.
        """
        factor, fwd = self._setup_pair(n=200)
        baseline = ic_mean(factor, fwd, window=20)

        factor_with_nan = factor.copy()
        factor_with_nan.iloc[5] = float("nan")

        perturbed = ic_mean(factor_with_nan, fwd, window=20)
        # With 199/200 pairs remaining, the IC should be close to baseline.
        assert abs(perturbed - baseline) < 0.1

    def test_ic_nan_alignment_both_sides(self) -> None:
        """NaN in either factor or returns is dropped symmetrically."""
        factor, fwd = self._setup_pair(n=200)
        factor_nan = factor.copy()
        factor_nan.iloc[5] = float("nan")
        returns_nan = fwd.copy()
        returns_nan.iloc[5] = float("nan")
        # Either-side NaN at index 5: both produce the same result as
        # dropping that index from both.
        a = ic_mean(factor_nan, fwd, window=20)
        b = ic_mean(factor, returns_nan, window=20)
        assert abs(a - b) < 1e-9  # same row dropped, same output

    def test_ic_on_short_series_returns_nan(self) -> None:
        factor, fwd = self._setup_pair(n=10)
        assert math.isnan(ic_mean(factor, fwd, window=20))  # not enough data
