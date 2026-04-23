"""Tests for per-regime metric aggregation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.metrics import mean_return, sharpe
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.regime.breakdown import breakdown_by_regime


def _returns(n: int = 200, *, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(0.001, 0.01, size=n), index=idx)


class TestSingleDimension:
    def test_splits_correctly_by_label(self) -> None:
        r = _returns(n=6)
        labels = pd.Series(
            ["bull", "bull", "bear", "bear", "crab", "crab"],
            index=r.index,
            dtype=object,
        )
        out = breakdown_by_regime(r, labels, mean_return)
        assert set(out.keys()) == {"bull", "bear", "crab"}
        # Each label picked up 2 rows.
        assert out["bull"] == pytest.approx((r.iloc[0] + r.iloc[1]) / 2)
        assert out["bear"] == pytest.approx((r.iloc[2] + r.iloc[3]) / 2)
        assert out["crab"] == pytest.approx((r.iloc[4] + r.iloc[5]) / 2)

    def test_works_with_sharpe_metric(self) -> None:
        r = _returns(n=200)
        labels = pd.Series(
            ["up"] * 100 + ["down"] * 100,
            index=r.index,
            dtype=object,
        )
        out = breakdown_by_regime(r, labels, sharpe)
        assert set(out.keys()) == {"up", "down"}
        assert all(math.isfinite(v) or math.isnan(v) for v in out.values())

    def test_drops_none_labels(self) -> None:
        """Warmup-period None labels are excluded from breakdown."""
        r = _returns(n=10)
        labels = pd.Series(
            [None, None, "bull", "bull", "bull", "bear", "bear", "bear", None, None],
            index=r.index,
            dtype=object,
        )
        out = breakdown_by_regime(r, labels, mean_return)
        assert set(out.keys()) == {"bull", "bear"}


class TestTupleCartesianProduct:
    def test_two_dimensions_produces_composite_labels(self) -> None:
        """The documented multi-dimensional extension — labels are
        joined with ' × '. Phase 6+ can use this to break Sharpe down
        by (trend, vol) jointly without breaking the function signature."""
        r = _returns(n=8)
        trend = pd.Series(
            ["bull", "bull", "bull", "bull", "bear", "bear", "bear", "bear"],
            index=r.index,
            dtype=object,
        )
        vol = pd.Series(
            ["low", "low", "high", "high", "low", "low", "high", "high"],
            index=r.index,
            dtype=object,
        )
        out = breakdown_by_regime(r, (trend, vol), mean_return)
        assert set(out.keys()) == {
            "bull × low",
            "bull × high",
            "bear × low",
            "bear × high",
        }

    def test_single_tuple_element_equivalent_to_single_series(self) -> None:
        r = _returns(n=6)
        labels = pd.Series(["a", "a", "b", "b", "c", "c"], index=r.index, dtype=object)
        single = breakdown_by_regime(r, labels, mean_return)
        tup = breakdown_by_regime(r, (labels,), mean_return)
        assert single == tup

    def test_empty_tuple_raises(self) -> None:
        r = _returns(n=3)
        with pytest.raises(ConfigError, match="at least one Series"):
            breakdown_by_regime(r, (), mean_return)

    def test_row_with_any_none_dimension_is_excluded(self) -> None:
        """A composite label is formed only where ALL dimensions are non-null."""
        r = _returns(n=4)
        a = pd.Series([None, "X", "X", "Y"], index=r.index, dtype=object)
        b = pd.Series(["P", None, "Q", "Q"], index=r.index, dtype=object)
        out = breakdown_by_regime(r, (a, b), mean_return)
        # Only rows 2 and 3 have both dimensions: ("X", "Q") and ("Y", "Q").
        assert set(out.keys()) == {"X × Q", "Y × Q"}


class TestIndexAlignment:
    def test_returns_and_labels_on_disjoint_indices(self) -> None:
        """When returns and labels indices don't fully overlap, the
        intersection is used silently."""
        r_idx = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
        l_idx = pd.date_range("2023-01-02", periods=4, freq="D", tz="UTC")
        r = pd.Series([0.01, 0.02, 0.03, 0.04], index=r_idx)
        labels = pd.Series(["a", "a", "b", "b"], index=l_idx, dtype=object)
        out = breakdown_by_regime(r, labels, mean_return)
        # 3-day intersection: rows 2, 3, 4. "a" at 2-3 (r values 0.02, 0.03); "b" at 4 (0.04).
        assert out["a"] == pytest.approx((0.02 + 0.03) / 2)
        assert out["b"] == pytest.approx(0.04)
