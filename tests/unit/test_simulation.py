"""Tests for the single-fold simulation layer — the vectorbt boundary.

Two concerns structurally enforced here:

1. **Causality** via ``_assert_signals_shifted`` — signals passed to
   ``simulate_fold`` must carry an explicit ``attrs["shifted"] = True``
   marker set by the engine after its +1-bar shift. No marker, no sim.
2. **Cost application** via ``CostModel`` being a required positional —
   there is no overload that skips it. A fee-notional-vs-PnL test pins
   the dimensional convention (fees are on notional, not PnL).

Plus the import-scope canary: ``simulate_fold`` is the only module in
``crypto_alpha_engine/`` allowed to touch vectorbt. A grep-based scan
over the package source (not tests/ — tests are free to verify
vectorbt behavior directly) fails if any other module reaches in.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.costs import CostModelSaturation
from crypto_alpha_engine.backtest.simulation import (
    FoldResult,
    simulate_fold,
)
from crypto_alpha_engine.exceptions import ConfigError, LookAheadDetected
from crypto_alpha_engine.types import CostModel


def _make_signals(
    n: int,
    entry_idx: int,
    exit_idx: int,
    *,
    mark_shifted: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Helper: build (entries, exits, close) with one entry and one exit."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(np.full(n, 10.0), index=idx)
    entries = pd.Series([i == entry_idx for i in range(n)], index=idx)
    exits = pd.Series([i == exit_idx for i in range(n)], index=idx)
    if mark_shifted:
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True
    return entries, exits, close


# ---------------------------------------------------------------------------
# Signal-shift guard
# ---------------------------------------------------------------------------


class TestAssertSignalsShifted:
    def test_missing_marker_on_entries_raises(self) -> None:
        entries, exits, close = _make_signals(20, 1, 5)
        del entries.attrs["shifted"]
        with pytest.raises(LookAheadDetected, match="shifted"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())

    def test_missing_marker_on_exits_raises(self) -> None:
        entries, exits, close = _make_signals(20, 1, 5)
        del exits.attrs["shifted"]
        with pytest.raises(LookAheadDetected, match="shifted"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())

    def test_marker_value_must_be_true(self) -> None:
        """Any value other than literal ``True`` (e.g., truthy string, 1)
        fails — we require the exact sentinel the engine sets."""
        entries, exits, close = _make_signals(20, 1, 5)
        entries.attrs["shifted"] = "yes"  # truthy but not True
        with pytest.raises(LookAheadDetected, match="shifted"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())

    def test_index_mismatch_raises(self) -> None:
        entries, exits, close = _make_signals(20, 1, 5)
        # Re-label entries' index so it doesn't match close
        offset_idx = pd.date_range("2025-01-01", periods=20, freq="1h", tz="UTC")
        entries = pd.Series(entries.to_numpy(), index=offset_idx)
        entries.attrs["shifted"] = True
        with pytest.raises(ConfigError, match="index"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())

    def test_non_bool_dtype_raises(self) -> None:
        entries, exits, close = _make_signals(20, 1, 5)
        entries = entries.astype(int)  # bool → int
        entries.attrs["shifted"] = True
        with pytest.raises(ConfigError, match="bool"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())

    def test_nan_signals_raises(self) -> None:
        """Shifted signals should have been ``fillna(False)``'d by the
        caller. A NaN slipping through would be silently converted to
        False by some pandas paths and True by others — fail loud."""
        entries, exits, close = _make_signals(20, 1, 5)
        # Force NaN via object dtype
        entries = pd.Series([None] + [False] * 19, index=close.index, dtype=object)
        entries.attrs["shifted"] = True
        with pytest.raises(ConfigError, match="bool"):
            simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())


# ---------------------------------------------------------------------------
# Fee application: notional vs PnL
# ---------------------------------------------------------------------------


class TestFeeApplication:
    def test_fees_applied_to_notional_not_pnl(self) -> None:
        """The dimensional pin from the Phase-6 kickoff review.

        Scenario: one round-trip trade at constant price 10 →  PnL is
        exactly zero. But fees are charged on each side's notional:

        * init_cash = 100, size = 100%
        * taker_bps = 50 → rate 0.005
        * Entry notional ≈ 100 → entry fee ≈ 100 · 0.005 = 0.5
        * Exit notional ≈ 100 → exit fee ≈ 100 · 0.005 = 0.5
        * Total fees ≈ 1.0

        If fees were mistakenly on PnL, total would be 0 · 0.005 = 0.
        The assertion ``fees_paid > 0`` trips instantly on that bug.
        Slippage is kept at default (0.05% floor); fees_paid isolates
        to the fee component specifically.
        """
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        close = pd.Series(np.full(10, 10.0), index=idx)  # flat price → zero PnL
        entries = pd.Series([False, True] + [False] * 8, index=idx)
        exits = pd.Series([False, False, False, True] + [False] * 6, index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostModelSaturation)
            result = simulate_fold(
                entries=entries,
                exits=exits,
                close=close,
                cost_model=CostModel(taker_bps=50.0),
                initial_cash=100.0,
            )

        # Two-sided fees on ~100 notional at 0.5% rate ≈ 1.0.
        assert result.fees_paid == pytest.approx(1.0, rel=0.02)
        assert result.fees_paid > 0  # guards against fees-on-PnL regression

    def test_fees_proportional_to_notional_under_profitable_trade(self) -> None:
        """Complement to the flat-price pin: a profitable round-trip.

        Setup: price runs from 10 → 12 (20% up). init_cash=100, so
        gross PnL is ~20. Fee rate is 50bps. If fees were mistakenly
        on PnL, fees_paid ≈ 2·20·0.005 = 0.20. If fees are on
        NOTIONAL, fees_paid ≈ (100 + ~120)·0.005 = ~1.10.

        The observed value must match the notional figure — the 5.5x
        gap between the two is too big for noise to hide. A
        fees-on-PnL bug would pass the flat-price test (PnL = 0
        collapses the bug's output) but fail spectacularly here.
        """
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(10.0, 12.0, 10), index=idx)  # +20% over fold
        entries = pd.Series([False, True] + [False] * 8, index=idx)
        exits = pd.Series([False] * 7 + [True] + [False] * 2, index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostModelSaturation)
            result = simulate_fold(
                entries=entries,
                exits=exits,
                close=close,
                cost_model=CostModel(taker_bps=50.0),
                initial_cash=100.0,
            )

        # Notional hypothesis: entry notional ~100, exit notional ~120
        # (position grew with price), fees ≈ (100 + 120) · 0.005 ≈ 1.10.
        # PnL hypothesis: PnL ~20, fees ≈ 20 · 0.005 ≈ 0.10.
        # The assertion admits the notional answer and rejects the PnL one.
        assert result.fees_paid == pytest.approx(1.10, rel=0.15)
        assert result.fees_paid > 0.5  # kills the fees-on-PnL branch

    def test_zero_trades_zero_fees(self) -> None:
        """No signals → no trades → zero fees. Sanity-check the
        aggregation doesn't pick up phantom fees from nothing."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        close = pd.Series(np.full(10, 10.0), index=idx)
        entries = pd.Series([False] * 10, index=idx)
        exits = pd.Series([False] * 10, index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True
        result = simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())
        assert result.n_trades == 0
        assert result.fees_paid == 0.0
        assert result.slippage_paid == 0.0


# ---------------------------------------------------------------------------
# FoldResult shape + end-to-end simulation
# ---------------------------------------------------------------------------


class TestFoldResultShape:
    def test_foldresult_is_frozen(self) -> None:
        result = FoldResult(
            gross_returns=pd.Series(dtype=float),
            net_returns=pd.Series(dtype=float),
            fees_paid=0.0,
            slippage_paid=0.0,
            funding_paid=0.0,
            n_trades=0,
            turnover=0.0,
            avg_trade_duration_hours=0.0,
            avg_position_size=0.0,
            max_leverage_used=0.0,
        )
        with pytest.raises((AttributeError, Exception)):
            result.fees_paid = 99.0  # type: ignore[misc]

    def test_gross_and_net_returns_are_series(self) -> None:
        entries, exits, close = _make_signals(20, 1, 5)
        result = simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())
        assert isinstance(result.gross_returns, pd.Series)
        assert isinstance(result.net_returns, pd.Series)
        assert result.gross_returns.index.equals(close.index)
        assert result.net_returns.index.equals(close.index)

    def test_net_returns_weakly_worse_than_gross_under_costs(self) -> None:
        """Total return with costs should be ≤ total return without.
        Fees and slippage can't add money back."""
        idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
        # Trending-up price with a round-trip in the middle.
        close = pd.Series(np.linspace(10.0, 12.0, 30), index=idx)
        entries = pd.Series([i == 2 for i in range(30)], index=idx)
        exits = pd.Series([i == 20 for i in range(30)], index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True
        result = simulate_fold(
            entries=entries,
            exits=exits,
            close=close,
            cost_model=CostModel(taker_bps=50.0),
            initial_cash=100.0,
        )
        gross_total = float(np.prod(1.0 + result.gross_returns.to_numpy())) - 1.0
        net_total = float(np.prod(1.0 + result.net_returns.to_numpy())) - 1.0
        assert net_total <= gross_total
        # And: the difference should be roughly fees + slippage as a
        # fraction of init_cash.
        expected_cost_drag = (result.fees_paid + result.slippage_paid) / 100.0
        assert (gross_total - net_total) == pytest.approx(expected_cost_drag, rel=0.1)

    def test_n_trades_counted_correctly(self) -> None:
        entries, exits, close = _make_signals(30, 2, 10)
        result = simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())
        assert result.n_trades == 1

    def test_avg_trade_duration_from_bar_count(self) -> None:
        """Entry at bar 2, exit at bar 12 on 1h bars → ~10 hours duration."""
        entries, exits, close = _make_signals(30, 2, 12)
        result = simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())
        assert result.avg_trade_duration_hours == pytest.approx(10.0, abs=1.0)

    def test_max_leverage_never_exceeds_one_longonly(self) -> None:
        """With size=100% cash and direction=longonly, leverage ∈ [0, 1]."""
        entries, exits, close = _make_signals(30, 2, 10)
        result = simulate_fold(entries=entries, exits=exits, close=close, cost_model=CostModel())
        assert 0.0 <= result.max_leverage_used <= 1.0

    def test_funding_applied_when_provided(self) -> None:
        """With funding_rate supplied and CostModel.funding_applied=True,
        the funding_paid field reflects the integral of position · rate."""
        idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
        close = pd.Series(np.full(30, 10.0), index=idx)
        entries = pd.Series([i == 2 for i in range(30)], index=idx)
        exits = pd.Series([i == 20 for i in range(30)], index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True
        funding = pd.Series(np.full(30, 0.0001), index=idx)  # 1 bp per bar, always
        result = simulate_fold(
            entries=entries,
            exits=exits,
            close=close,
            cost_model=CostModel(),
            funding_rate=funding,
            initial_cash=100.0,
        )
        # Position held for ~18 bars at notional ~100 → funding ≈ 100·18·0.0001 = 0.18.
        assert result.funding_paid > 0
        assert result.funding_paid == pytest.approx(0.18, rel=0.2)

    def test_funding_skipped_when_disabled(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
        close = pd.Series(np.full(30, 10.0), index=idx)
        entries = pd.Series([i == 2 for i in range(30)], index=idx)
        exits = pd.Series([i == 20 for i in range(30)], index=idx)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True
        funding = pd.Series(np.full(30, 0.0001), index=idx)
        result = simulate_fold(
            entries=entries,
            exits=exits,
            close=close,
            cost_model=CostModel(funding_applied=False),
            funding_rate=funding,
            initial_cash=100.0,
        )
        assert result.funding_paid == 0.0


# ---------------------------------------------------------------------------
# Import-scope canary — THE structural guarantee
# ---------------------------------------------------------------------------


def test_only_simulation_imports_vectorbt() -> None:
    """Canary: the ONLY module in ``crypto_alpha_engine/`` allowed to
    import vectorbt is ``backtest/simulation.py``.

    Why this matters: vectorbt's abstraction brings same-bar execution
    conventions and other equity-market defaults that can silently
    introduce lookahead bias. Keeping vectorbt behind a single
    boundary means any future change that widens the surface has to
    add a new import and immediately trip this canary. Review catches
    it before the diff lands.

    Uses string grep rather than AST parse because a string search is
    harder to bypass accidentally — an obfuscated ``__import__("vec" +
    "torbt")`` call would dodge an ``ast.parse`` walker, but source-text
    grep gets a human reviewer's attention either way.

    Scanning tests/ is explicitly out of scope — test files are free
    to import vectorbt for independent verification.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "crypto_alpha_engine"
    allowed = {"backtest/simulation.py"}
    patterns = ("import vectorbt", "from vectorbt", "import vbt")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        rel = path.relative_to(root).as_posix()
        for pattern in patterns:
            if pattern in text and rel not in allowed:
                violations.append(f"{rel}: contains '{pattern}'")
    assert not violations, (
        "Vectorbt must only be imported in backtest/simulation.py. "
        "Violations:\n  " + "\n  ".join(violations)
    )
