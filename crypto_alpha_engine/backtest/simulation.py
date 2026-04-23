"""Single-fold simulation layer — the vectorbt boundary.

This module is the **only** place in ``crypto_alpha_engine/`` allowed
to import vectorbt. A canary test in ``tests/unit/test_simulation.py``
fails if any other package file reaches in. Justification: vectorbt's
API carries equity-market defaults (same-bar fills, implicit
close-at-end, position stacking) that can silently reintroduce the
lookahead bias our operator layer works to prevent. Concentrating the
dependency in one file keeps the surface small — any new import
shows up in code review immediately.

Causality pinch-point
---------------------

:func:`simulate_fold` never computes a signal; it consumes already-
shifted signals from the engine. The contract is enforced via a
``Series.attrs["shifted"] = True`` marker the engine must set after
its +1-bar shift. ``_assert_signals_shifted`` rejects any signal
missing the marker. Two-layer defense:

1. Marker check — forgetting the shift is caught even if indices
   happen to align.
2. Index/dtype/NaN checks — catch mechanical errors the marker
   wouldn't (e.g., caller stamped the marker without doing the shift).

Cost pinch-point
----------------

:class:`CostModel` is a required positional arg. There is no overload
that skips fees. ``FoldResult`` separates ``gross_returns`` from
``net_returns`` as distinct series, so downstream code can't
accidentally report gross as net — the field names force the
distinction. Fees and slippage are extracted from vectorbt's
per-trade records and reported as structural fields, not deduced.

Phase-6 cost model scope
------------------------

Slippage is applied as a scalar equal to the SPEC §8 floor (0.05%)
regardless of trade-size-to-volume ratio. The volume-based quadratic
in :mod:`costs` is not threaded into the simulation yet — doing so
requires knowing trade notional at sim-time, which vectorbt resolves
during execution, not before. Phase 7+ will pre-compute a per-bar
slippage array from rolling volume estimates; for Phase 6 the floor
is a conservative-enough approximation for realistic ``init_cash``
on liquid crypto pairs (where a $10K trade is < 1% of daily volume
on BTC/ETH).

Funding is applied post-vectorbt as
``sum(asset_value_per_bar · funding_rate)`` when ``funding_rate`` is
supplied and ``CostModel.funding_applied`` is True.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from crypto_alpha_engine.backtest.costs import (
    _SLIPPAGE_FLOOR,
    compute_funding_charge,
    fee_rate,
)
from crypto_alpha_engine.exceptions import ConfigError, LookAheadDetected
from crypto_alpha_engine.types import CostModel

_SHIFTED_MARKER: str = "shifted"


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold's aggregated simulation output.

    Internal to the backtest layer — :class:`BacktestResult` aggregates
    across folds before exposing anything to the user. Per Principle 1,
    FoldResult fields may carry bar-level series, but these never
    propagate to the public API; ``engine.py`` consumes them to compute
    the aggregate fields on ``BacktestResult`` and then drops them.

    Attributes:
        gross_returns: Pre-cost bar-by-bar returns (from a zero-cost
            shadow vectorbt sim). Used to compute ``gross_sharpe``.
        net_returns: Post-cost bar-by-bar returns. Used for every
            headline metric (``sharpe``, ``sortino``, etc.) and for
            the regime breakdown.
        fees_paid: Sum of entry + exit fees across every trade in the
            fold, in cash units. From vectorbt's trade records — not
            a deduction, a direct extract.
        slippage_paid: Sum of slippage cost (notional · slip_rate) per
            trade. Always non-negative.
        funding_paid: Net funding charge. Signed: positive = paid out,
            negative = received. Zero when ``CostModel.funding_applied``
            is False or no ``funding_rate`` was supplied.
        n_trades: Number of closed round-trip trades in this fold.
        turnover: Sum of absolute changes in asset_value across bars
            (cash units). Used for turnover_annual aggregation.
        avg_trade_duration_hours: Mean of closed-trade durations, in
            hours. Zero when n_trades == 0.
        avg_position_size: Mean of |asset_value| across all bars (cash
            units).
        max_leverage_used: Maximum of ``|asset_value / total_value|``
            across bars. Long-only sim: ∈ [0, 1].
    """

    gross_returns: pd.Series
    net_returns: pd.Series
    fees_paid: float
    slippage_paid: float
    funding_paid: float
    n_trades: int
    turnover: float
    avg_trade_duration_hours: float
    avg_position_size: float
    max_leverage_used: float


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate_fold(
    *,
    entries: pd.Series,
    exits: pd.Series,
    close: pd.Series,
    cost_model: CostModel,
    funding_rate: pd.Series | None = None,
    initial_cash: float = 10_000.0,
    freq: str = "1h",
) -> FoldResult:
    """Simulate one fold's entries/exits against a close series.

    Every vectorbt parameter is set explicitly below — no reliance on
    library defaults for position sizing, conflict resolution, or fill
    pricing. If vectorbt changes a default in a future release, this
    function's behavior does not.

    Args:
        entries: Boolean series of entry signals, **already shifted by
            +1 bar** by the engine. Must carry
            ``entries.attrs["shifted"] = True`` or a
            :class:`LookAheadDetected` is raised.
        exits: Boolean series of exit signals, same shift contract.
        close: Close-price series. Index must equal both signal indices.
        cost_model: Required. Drives taker fee (``is_taker=True``),
            slippage floor, funding toggle, borrow rate.
        funding_rate: Optional per-bar funding rate series, aligned to
            ``close.index``. If omitted, funding_paid = 0.0.
        initial_cash: Starting portfolio cash. Defaults to $10,000
            (keeps trade notional well below 1% of daily volume on
            major crypto pairs, staying in the slippage-floor regime).
        freq: Bar frequency string (``"1h"``, ``"4h"``, ``"1d"``).
            Passed to vectorbt for annualization.

    Returns:
        :class:`FoldResult` with pre-/post-cost returns series and
        aggregate trade stats.

    Raises:
        LookAheadDetected: If either signal series is missing the
            ``"shifted"`` attrs marker.
        ConfigError: On index mismatch, non-bool signal dtype, or NaN
            in signals.

    Example:
        >>> # Engine-side code would do:
        >>> entries = raw_signals.shift(1).fillna(False)
        >>> entries.attrs["shifted"] = True
        >>> result = simulate_fold(
        ...     entries=entries, exits=exits, close=close,
        ...     cost_model=CostModel(),
        ... )
    """
    _assert_signals_shifted(entries=entries, exits=exits, close=close)

    taker = fee_rate(cost_model, is_taker=True)
    slip = _SLIPPAGE_FLOOR

    # Common parameters — every knob set explicitly. See module docstring
    # for the reasoning on why defaults are not good enough.
    common = {
        "close": close,
        "entries": entries,
        "exits": exits,
        "freq": freq,
        "init_cash": initial_cash,
        "size": 1.0,
        "size_type": "percent",
        "accumulate": False,
        "direction": "longonly",
    }

    # Two sims: one with costs (for net_returns + trade records),
    # one zero-cost shadow (for gross_returns). Running twice is
    # ~2x sim cost per fold; compared to vectorbt's compiled Numba
    # kernels, still fast enough to beat the SPEC §15 budget.
    pf_net = vbt.Portfolio.from_signals(fees=taker, slippage=slip, **common)
    pf_gross = vbt.Portfolio.from_signals(fees=0.0, slippage=0.0, **common)

    gross_returns: pd.Series = pf_gross.returns()
    net_returns: pd.Series = pf_net.returns()

    trades = pf_net.trades.records_readable
    n_trades = int(len(trades))

    if n_trades > 0:
        fees_paid = float(trades["Entry Fees"].sum() + trades["Exit Fees"].sum())
        # Slippage isn't line-itemed by vectorbt — reconstruct from the
        # recorded Avg Entry/Exit Prices (already slip-adjusted) times
        # size. The result matches size · reference_price · slip_rate
        # to within rounding.
        entry_notional = (trades["Size"] * trades["Avg Entry Price"]).sum()
        exit_notional = (trades["Size"] * trades["Avg Exit Price"]).sum()
        slippage_paid = float((entry_notional + exit_notional) * slip)

        durations = trades["Exit Timestamp"] - trades["Entry Timestamp"]
        avg_trade_duration_hours = float(durations.dt.total_seconds().mean() / 3600.0)
    else:
        fees_paid = 0.0
        slippage_paid = 0.0
        avg_trade_duration_hours = 0.0

    # Funding charge — post-hoc, applied to the position's cash value.
    if funding_rate is not None:
        asset_value = pf_gross.asset_value()
        if not funding_rate.index.equals(asset_value.index):
            raise ConfigError("simulate_fold: funding_rate.index must equal close.index")
        funding_paid = compute_funding_charge(
            position=asset_value, funding_rate=funding_rate, cost_model=cost_model
        )
    else:
        funding_paid = 0.0

    # Position statistics — from the zero-cost sim (position is
    # identical whether costs are charged or not, so either works).
    asset_value = pf_gross.asset_value()
    total_value = pf_gross.value()
    leverage = (asset_value / total_value).abs()
    max_leverage_used = float(leverage.max())
    avg_position_size = float(asset_value.abs().mean())
    turnover = float(asset_value.diff().abs().sum())

    return FoldResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
        funding_paid=funding_paid,
        n_trades=n_trades,
        turnover=turnover,
        avg_trade_duration_hours=avg_trade_duration_hours,
        avg_position_size=avg_position_size,
        max_leverage_used=max_leverage_used,
    )


# ---------------------------------------------------------------------------
# Signal-shift enforcement
# ---------------------------------------------------------------------------


def _assert_signals_shifted(
    *,
    entries: pd.Series,
    exits: pd.Series,
    close: pd.Series,
) -> None:
    """Defensive gate — rejects any signal the engine didn't shift.

    The marker is an explicit sentinel set by the engine's shift step:
    ``series.attrs["shifted"] = True``. We require **literal** ``True``
    (not truthy), so an accidental string or int won't get waved through.

    In addition to the marker, we check:

    * Signal indices equal close.index (no silent reindex misbehavior).
    * Signal dtypes are bool (not int, not object) — vectorbt handles
      bool cleanly; other dtypes have surprising conversion rules.
    * No NaN in signals — the engine is expected to have applied
      ``fillna(False)`` after the shift.
    """
    for name, sig in (("entries", entries), ("exits", exits)):
        if sig.attrs.get(_SHIFTED_MARKER) is not True:
            raise LookAheadDetected(
                f"simulate_fold: {name!s} is missing the shifted-marker "
                f"(attrs[{_SHIFTED_MARKER!r}] != True). The engine must "
                f"apply a +1-bar shift and stamp the marker before calling "
                f"simulate_fold. This is the causality pinch-point; do not "
                f"bypass by stamping without shifting."
            )
        if not sig.index.equals(close.index):
            raise ConfigError(f"simulate_fold: {name!s}.index must equal close.index")
        # Accept numpy bool_ as well as Python bool.
        if sig.dtype != np.dtype("bool"):
            raise ConfigError(f"simulate_fold: {name!s} must be bool dtype, got {sig.dtype!r}")
        if sig.isna().any():
            raise ConfigError(
                f"simulate_fold: {name!s} contains NaN; caller must "
                f"fillna(False) after shifting"
            )
