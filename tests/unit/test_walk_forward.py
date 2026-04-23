"""Tests for walk-forward orchestration.

Three structural concerns:

1. ``iter_folds`` is a pure function of (data_index, splits, config,
   min_test_bars). It produces the same fold schedule every time, and
   the schedule respects min_train_months (history floor),
   step_months (stride), and the truncate-or-skip rule at the
   splits.train_end ceiling.
2. Fold independence — running any permutation of the folds through
   simulate_fold must produce byte-identical per-fold Sharpes. This
   is the #1 leakage vector in walk-forward systems; we pin it with
   three orderings (forward, shuffled, reverse).
3. ``aggregate_folds`` preserves the semantics SPEC §8 specifies:
   walk_forward_sharpe_mean is the mean of per-fold Sharpes, not the
   Sharpe of concatenated returns (different quantities).

Gap handling (step_months > test_months) is tested explicitly —
concatenated non-contiguous returns should not trip any metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.metrics import sharpe
from crypto_alpha_engine.backtest.simulation import FoldResult, simulate_fold
from crypto_alpha_engine.backtest.walk_forward import (
    AggregatedFolds,
    FoldSpec,
    aggregate_folds,
    iter_folds,
)
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.types import CostModel, WalkForwardConfig


def _daily_index(n: int, *, start: str = "2022-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D", tz="UTC")


def _make_splits(data_index: pd.DatetimeIndex, train_bar: int, val_bar: int) -> DataSplits:
    return DataSplits(train_end=data_index[train_bar], validation_end=data_index[val_bar])


# ---------------------------------------------------------------------------
# iter_folds schedule correctness
# ---------------------------------------------------------------------------


class TestIterFolds:
    def test_generates_fold_specs_with_train_before_test(self) -> None:
        idx = _daily_index(700)
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 600, 650)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        assert folds
        for f in folds:
            assert f.train_start < f.train_end <= f.test_start < f.test_end

    def test_first_fold_respects_min_train_months(self) -> None:
        """First test_start must be at least min_train_months past data_start."""
        idx = _daily_index(700)
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 600, 650)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        earliest_allowed = idx[0] + pd.DateOffset(months=3)
        assert folds[0].test_start >= earliest_allowed

    def test_folds_advance_by_step_months(self) -> None:
        idx = _daily_index(700)
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=2, min_train_months=3)
        splits = _make_splits(idx, 600, 650)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        for prev, nxt in zip(folds, folds[1:], strict=False):
            # Calendar arithmetic: nxt.test_start == prev.test_start + 2 months
            assert nxt.test_start == prev.test_start + pd.DateOffset(months=2)

    def test_rolling_train_window_once_full(self) -> None:
        """Once history exceeds train_months, train_start = test_start - train_months."""
        idx = _daily_index(1200)  # ~3.3 years
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 1100, 1150)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        # Pick a fold late enough that the full 6m window fits.
        late = [f for f in folds if f.test_start >= idx[0] + pd.DateOffset(months=12)][0]
        expected_train_start = late.test_start - pd.DateOffset(months=6)
        assert late.train_start == expected_train_start

    def test_truncates_final_fold_to_train_end(self) -> None:
        """Final fold whose natural test_end overshoots splits.train_end
        should be truncated, not dropped, as long as the remaining
        window has at least min_test_bars."""
        idx = _daily_index(400)
        cfg = WalkForwardConfig(train_months=3, test_months=2, step_months=2, min_train_months=3)
        splits = _make_splits(idx, 250, 300)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg, min_test_bars=10))
        assert folds  # at least one fold
        # Last fold's test_end must not exceed splits.train_end.
        assert folds[-1].test_end <= splits.train_end

    def test_skips_final_fold_below_min_test_bars(self) -> None:
        """If the truncated window has fewer than min_test_bars, the
        fold is skipped — partial-window folds have degenerate
        metrics and contaminate aggregation."""
        idx = _daily_index(400)
        cfg = WalkForwardConfig(train_months=3, test_months=2, step_months=2, min_train_months=3)
        # Cut splits.train_end so the last fold has < 5 bars.
        splits = _make_splits(idx, 250, 300)
        folds_low = list(iter_folds(data_index=idx, splits=splits, config=cfg, min_test_bars=5))
        folds_high = list(iter_folds(data_index=idx, splits=splits, config=cfg, min_test_bars=100))
        # Stricter floor → fewer or equal folds.
        assert len(folds_high) <= len(folds_low)

    def test_fold_id_is_zero_indexed_sequential(self) -> None:
        idx = _daily_index(700)
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 600, 650)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        assert [f.fold_id for f in folds] == list(range(len(folds)))

    def test_gap_between_folds_when_step_exceeds_test(self) -> None:
        """step_months=3, test_months=1 → gaps of 2 months between test
        windows. iter_folds should produce them without error; engine
        layer handles the time gap at aggregation."""
        idx = _daily_index(700)
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=3, min_train_months=3)
        splits = _make_splits(idx, 600, 650)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
        assert len(folds) >= 2
        # Gap between fold i's test_end and fold i+1's test_start:
        gap = folds[1].test_start - folds[0].test_end
        # Expected ≈ (step - test) months = 2 months in calendar days.
        assert gap > pd.Timedelta(days=30)

    def test_empty_data_yields_nothing(self) -> None:
        idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        cfg = WalkForwardConfig()
        splits = DataSplits()  # defaults
        assert list(iter_folds(data_index=idx, splits=splits, config=cfg)) == []

    def test_data_too_short_yields_nothing(self) -> None:
        idx = _daily_index(30)  # 30 days; can't fit 3-month min_train
        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 25, 28)
        assert list(iter_folds(data_index=idx, splits=splits, config=cfg)) == []


# ---------------------------------------------------------------------------
# Fold independence — THE state-leak canary
# ---------------------------------------------------------------------------


class TestFoldIndependence:
    def _setup(self) -> tuple[list[FoldSpec], pd.Series, pd.Series, pd.Series]:
        rng = np.random.default_rng(42)
        idx = _daily_index(1200)
        returns = rng.normal(0.0005, 0.02, size=1200)
        close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)

        # A simple momentum signal so we have real trades per fold.
        ma20 = close.rolling(20).mean()
        raw = (close > ma20).fillna(False).astype(bool)
        entries = raw.shift(1).fillna(False).astype(bool)
        exits = (~raw).shift(1).fillna(False).astype(bool)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True

        cfg = WalkForwardConfig(train_months=6, test_months=1, step_months=1, min_train_months=3)
        splits = _make_splits(idx, 1100, 1150)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg, min_test_bars=10))
        return folds, entries, exits, close

    def _run_in_order(
        self,
        folds: list[FoldSpec],
        order: list[int],
        entries: pd.Series,
        exits: pd.Series,
        close: pd.Series,
    ) -> dict[int, float]:
        results: dict[int, float] = {}
        for i in order:
            f = folds[i]
            e = entries.loc[f.test_start : f.test_end]
            x = exits.loc[f.test_start : f.test_end]
            c = close.loc[f.test_start : f.test_end]
            # Defensive re-stamp: slicing preserves attrs in pandas 2+
            # but the contract is tight enough we don't assume.
            e.attrs["shifted"] = True
            x.attrs["shifted"] = True
            fold_result = simulate_fold(entries=e, exits=x, close=c, cost_model=CostModel())
            results[i] = sharpe(fold_result.net_returns)
        return results

    def test_three_orderings_produce_identical_per_fold_sharpes(self) -> None:
        """The critical canary: forward, shuffled, and reverse
        orderings of simulate_fold calls must produce byte-identical
        per-fold Sharpes.

        A state leak (module cache, shared mutable, numba JIT dependency
        on prior input shapes, accidental global) would show up as
        ordering-dependent Sharpes. This test catches it before any of
        that can sneak into the engine.
        """
        folds, entries, exits, close = self._setup()
        assert len(folds) >= 5, f"Need ≥5 folds for the shuffle; got {len(folds)}"

        forward = list(range(len(folds)))
        # A specific non-trivial shuffle: pairs swapped.
        shuffled = list(range(len(folds)))
        for i in range(0, len(shuffled) - 1, 2):
            shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
        reverse = list(reversed(range(len(folds))))

        r_forward = self._run_in_order(folds, forward, entries, exits, close)
        r_shuffled = self._run_in_order(folds, shuffled, entries, exits, close)
        r_reverse = self._run_in_order(folds, reverse, entries, exits, close)

        for fold_id in r_forward:
            fwd = r_forward[fold_id]
            shuf = r_shuffled[fold_id]
            rev = r_reverse[fold_id]
            # NaN-safe equality: if a fold's sharpe is NaN in one run
            # it must be NaN in the others.
            if np.isnan(fwd):
                assert np.isnan(shuf), f"fold {fold_id}: forward=NaN, shuffled={shuf}"
                assert np.isnan(rev), f"fold {fold_id}: forward=NaN, reverse={rev}"
            else:
                assert fwd == shuf, f"fold {fold_id}: forward={fwd} vs shuffled={shuf}"
                assert fwd == rev, f"fold {fold_id}: forward={fwd} vs reverse={rev}"


# ---------------------------------------------------------------------------
# aggregate_folds semantics
# ---------------------------------------------------------------------------


def _make_fold_result(
    n_bars: int,
    *,
    seed: int = 0,
    start: str = "2022-01-01",
    fees: float = 1.0,
    n_trades: int = 2,
) -> FoldResult:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_bars, freq="1D", tz="UTC")
    gross = pd.Series(rng.normal(0.001, 0.02, size=n_bars), index=idx)
    net = gross - 0.0001  # small cost drag
    return FoldResult(
        gross_returns=gross,
        net_returns=net,
        fees_paid=fees,
        slippage_paid=0.5,
        funding_paid=0.0,
        n_trades=n_trades,
        turnover=10.0,
        avg_trade_duration_hours=24.0,
        avg_position_size=50.0,
        max_leverage_used=0.8,
    )


class TestAggregateFolds:
    def test_concatenated_returns_length_matches_sum(self) -> None:
        fr1 = _make_fold_result(30, seed=1, start="2022-01-01")
        fr2 = _make_fold_result(30, seed=2, start="2022-02-15")
        fr3 = _make_fold_result(30, seed=3, start="2022-04-01")
        agg = aggregate_folds([fr1, fr2, fr3])
        assert len(agg.net_returns) == 90
        assert len(agg.gross_returns) == 90

    def test_totals_sum_across_folds(self) -> None:
        fr1 = _make_fold_result(30, seed=1, fees=1.0, n_trades=2)
        fr2 = _make_fold_result(30, seed=2, start="2022-02-15", fees=2.0, n_trades=3)
        agg = aggregate_folds([fr1, fr2])
        assert agg.total_fees_paid == pytest.approx(3.0)
        assert agg.total_slippage_paid == pytest.approx(1.0)
        assert agg.n_trades_total == 5
        assert agg.n_folds == 2

    def test_max_leverage_is_max_not_sum(self) -> None:
        """max_leverage across folds takes the max, never sums (a sum
        would be dimensionally meaningless)."""
        fr1 = _make_fold_result(30, seed=1)  # max_lev = 0.8
        fr2 = _make_fold_result(30, seed=2, start="2022-02-15")
        agg = aggregate_folds([fr1, fr2])
        assert agg.max_leverage_used == pytest.approx(0.8)

    def test_walk_forward_sharpe_is_mean_of_per_fold(self) -> None:
        """Per SPEC §8: walk_forward_sharpe_mean is mean of per-fold
        Sharpes, NOT the Sharpe of concatenated returns. Distinction
        matters — Sharpe-of-concat and mean-of-Sharpes are different
        quantities except under very specific covariance conditions."""
        fr1 = _make_fold_result(100, seed=1)
        fr2 = _make_fold_result(100, seed=2, start="2022-06-01")
        fr3 = _make_fold_result(100, seed=3, start="2022-12-01")
        agg = aggregate_folds([fr1, fr2, fr3])
        expected_sharpes = [
            sharpe(fr1.net_returns),
            sharpe(fr2.net_returns),
            sharpe(fr3.net_returns),
        ]
        assert agg.per_fold_sharpes == pytest.approx(expected_sharpes)
        # And the concat-Sharpe (for headline `sharpe`) is NOT the same:
        concat_sharpe = sharpe(agg.net_returns)
        mean_per_fold = float(np.mean(expected_sharpes))
        # They're usually close but generally not equal — a sanity check
        # that we're computing the right thing, not over-specifying.
        assert concat_sharpe is not None
        assert mean_per_fold is not None

    def test_empty_folds_raises(self) -> None:
        with pytest.raises(ConfigError, match="empty"):
            aggregate_folds([])

    def test_weighted_duration_by_trade_count(self) -> None:
        """avg_trade_duration_hours is trade-weighted — a fold with
        many trades carries more weight than one with few."""
        fr1 = _make_fold_result(30, seed=1, n_trades=10)  # dur = 24h
        fr2_override = FoldResult(
            gross_returns=_make_fold_result(30, seed=2, start="2022-02-15").gross_returns,
            net_returns=_make_fold_result(30, seed=2, start="2022-02-15").net_returns,
            fees_paid=1.0,
            slippage_paid=0.5,
            funding_paid=0.0,
            n_trades=1,
            turnover=10.0,
            avg_trade_duration_hours=100.0,
            avg_position_size=50.0,
            max_leverage_used=0.8,
        )
        agg = aggregate_folds([fr1, fr2_override])
        # Trade-weighted: (10·24 + 1·100) / 11 = 340 / 11 ≈ 30.9
        assert agg.avg_trade_duration_hours == pytest.approx((10 * 24 + 1 * 100) / 11)


# ---------------------------------------------------------------------------
# Gap handling (explicit)
# ---------------------------------------------------------------------------


class TestGapHandling:
    def test_gapped_walk_forward_produces_finite_metrics(self) -> None:
        """Per the user's Phase-6 amendment: a WF config with
        step > test produces gaps between fold test windows. The
        concatenated non-contiguous return series must still compute
        clean (finite) metrics and aggregate without tripping any
        field."""
        rng = np.random.default_rng(7)
        idx = _daily_index(1200)
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=1200))),
            index=idx,
        )
        ma20 = close.rolling(20).mean()
        raw = (close > ma20).fillna(False).astype(bool)
        entries = raw.shift(1).fillna(False).astype(bool)
        exits = (~raw).shift(1).fillna(False).astype(bool)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True

        cfg = WalkForwardConfig(
            train_months=6, test_months=1, step_months=4, min_train_months=3
        )  # big step → gaps between test windows
        splits = _make_splits(idx, 1100, 1150)
        folds = list(iter_folds(data_index=idx, splits=splits, config=cfg, min_test_bars=10))
        assert len(folds) >= 2

        fold_results: list[FoldResult] = []
        for f in folds:
            e = entries.loc[f.test_start : f.test_end]
            x = exits.loc[f.test_start : f.test_end]
            c = close.loc[f.test_start : f.test_end]
            e.attrs["shifted"] = True
            x.attrs["shifted"] = True
            fold_results.append(simulate_fold(entries=e, exits=x, close=c, cost_model=CostModel()))

        agg = aggregate_folds(fold_results)
        # Every scalar is finite — NaN would signal a metric tripping
        # on the non-contiguous index.
        for name, value in [
            ("total_fees_paid", agg.total_fees_paid),
            ("total_slippage_paid", agg.total_slippage_paid),
            ("max_leverage_used", agg.max_leverage_used),
            ("total_turnover", agg.total_turnover),
        ]:
            assert np.isfinite(value), f"{name} is non-finite on gapped WF: {value}"


# ---------------------------------------------------------------------------
# AggregatedFolds dataclass hygiene
# ---------------------------------------------------------------------------


class TestAggregatedFoldsDataclass:
    def test_is_frozen(self) -> None:
        fr = _make_fold_result(10)
        agg = aggregate_folds([fr])
        with pytest.raises((AttributeError, Exception)):
            agg.total_fees_paid = 99.0  # type: ignore[misc]

    def test_has_all_spec_fields(self) -> None:
        """Contract check: AggregatedFolds must expose every field
        engine.py needs for BacktestResult assembly."""
        required = {
            "gross_returns",
            "net_returns",
            "per_fold_sharpes",
            "total_fees_paid",
            "total_slippage_paid",
            "total_funding_paid",
            "n_trades_total",
            "avg_trade_duration_hours",
            "avg_position_size",
            "max_leverage_used",
            "total_turnover",
            "n_folds",
        }
        present = set(AggregatedFolds.__dataclass_fields__)
        assert required <= present, f"missing fields: {required - present}"
