"""Cost-rate calculators: fees, slippage, borrow, funding.

Principle 5 ("costs are mandatory") is enforced structurally here,
not conventionally:

* Every rate-returning function requires a :class:`CostModel`
  positional argument — there is no overload that omits cost inputs.
* ``CostModel``'s ``__post_init__`` rejects zero or negative bps at
  construction, so any CostModel that reaches this module is already
  guaranteed to produce strictly positive fee / borrow rates.
* The only cost component that can legitimately return zero is
  funding, and only when the caller explicitly disables it via
  ``CostModel(funding_applied=False)`` — that branch is tested
  explicitly so the opt-out stays visible in the diff if it ever
  changes.

This module is a leaf — it imports only :mod:`types` (for
``CostModel``) and :mod:`exceptions`. The simulation layer (next
commit) composes these rate calculators into trade-by-trade
application on a returns series. Splitting the two keeps the pure
cost math independently testable without standing up a vectorbt
Portfolio.
"""

from __future__ import annotations

import warnings

import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.types import CostModel

# ---------------------------------------------------------------------------
# Slippage model constants (SPEC §8 "volume_based")
# ---------------------------------------------------------------------------

_SLIPPAGE_FLOOR: float = 0.0005  # 0.05% — applied when trade < 1% of daily volume
_SLIPPAGE_CAP: float = 0.01  # 1%   — reached at 10% of daily volume
_VOLUME_RATIO_FLOOR: float = 0.01  # 1% of volume
_VOLUME_RATIO_CAP: float = 0.10  # 10% of volume


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------


def fee_rate(cost_model: CostModel, *, is_taker: bool = True) -> float:
    """Return the per-side fee as a fractional rate.

    Args:
        cost_model: The backtest's cost configuration. Must carry
            strictly positive ``taker_bps`` and ``maker_bps`` (enforced
            at ``CostModel`` construction).
        is_taker: Whether to charge taker (aggressive) pricing. Defaults
            to True — the conservative choice, since most factor-driven
            execution is taker-side unless the caller has modeled a
            maker strategy.

    Returns:
        Fractional fee: ``10 bps → 0.001``. Always strictly positive
        for any valid ``CostModel``.

    Example:
        >>> fee_rate(CostModel())
        0.001
        >>> fee_rate(CostModel(), is_taker=False)
        0.0002
    """
    bps = cost_model.taker_bps if is_taker else cost_model.maker_bps
    return bps / 10_000.0


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


def slippage_rate(
    trade_notional: float,
    daily_volume: float,
    cost_model: CostModel,
) -> float:
    """Return the slippage as a fractional rate for a single trade.

    Implements the SPEC §8 volume-based model:

    * ``trade_notional / daily_volume < 1%`` → flat 0.05% floor.
    * ``1% ≤ ratio ≤ 10%`` → quadratic interpolation from 0.05% to 1%.
    * ``ratio > 10%`` → saturated at 1% with a ``UserWarning`` noting
      the trade exceeds the modeled regime.

    The saturation warning is deliberate: silently capping would
    under-report cost for oversized trades, which is exactly the
    optimistic-bias failure mode we exist to prevent. Logging + cap
    lets the backtest run through but leaves an audit trail.

    Args:
        trade_notional: Absolute notional value of the trade (same
            currency as ``daily_volume``). Must be strictly positive —
            a zero trade is not a call site that should reach this
            function.
        daily_volume: Daily traded volume of the instrument at the
            trade's timestamp. Must be strictly positive.
        cost_model: Must carry ``slippage_model == "volume_based"``;
            any other value raises :class:`ConfigError`. Future models
            (e.g., spread-based) will add elif branches here.

    Returns:
        Fractional slippage rate (``0.01 == 1%``).

    Raises:
        ConfigError: On non-positive inputs or unknown slippage_model.

    Example:
        >>> slippage_rate(500.0, 100_000.0, CostModel())  # 0.5% of vol
        0.0005
        >>> slippage_rate(10_000.0, 100_000.0, CostModel())  # 10% of vol
        0.01
    """
    if trade_notional <= 0:
        raise ConfigError(f"trade_notional must be positive, got {trade_notional!r}")
    if daily_volume <= 0:
        raise ConfigError(f"daily_volume must be positive, got {daily_volume!r}")
    if cost_model.slippage_model != "volume_based":
        raise ConfigError(
            f"unknown slippage_model {cost_model.slippage_model!r}; "
            f"only 'volume_based' is implemented in Phase 6"
        )

    ratio = trade_notional / daily_volume
    if ratio < _VOLUME_RATIO_FLOOR:
        return _SLIPPAGE_FLOOR
    if ratio <= _VOLUME_RATIO_CAP:
        # Quadratic interpolation from (floor_ratio, floor_rate) to
        # (cap_ratio, cap_rate). At ratio = floor_ratio: returns floor.
        # At ratio = cap_ratio: returns cap.
        span = _VOLUME_RATIO_CAP - _VOLUME_RATIO_FLOOR
        normalized = (ratio - _VOLUME_RATIO_FLOOR) / span
        return _SLIPPAGE_FLOOR + (_SLIPPAGE_CAP - _SLIPPAGE_FLOOR) * (normalized**2)
    warnings.warn(
        f"slippage_rate: trade is {ratio:.1%} of daily volume, saturating at "
        f"{_SLIPPAGE_CAP:.2%} (SPEC §8 volume_based model only defined up to "
        f"{_VOLUME_RATIO_CAP:.0%}). Consider splitting the order or modeling "
        f"market impact explicitly.",
        stacklevel=2,
    )
    return _SLIPPAGE_CAP


# ---------------------------------------------------------------------------
# Borrow (short-spot financing)
# ---------------------------------------------------------------------------


def borrow_rate_per_period(
    cost_model: CostModel,
    *,
    periods_per_year: float,
) -> float:
    """Convert the annual borrow-bps rate to a per-bar fraction.

    The borrow cost is charged every bar against a short spot position.
    The caller — typically the simulation layer — multiplies this
    per-period rate by ``|position|`` and subtracts it from the bar's
    gross return.

    Args:
        cost_model: Must carry ``borrow_rate_bps > 0``.
        periods_per_year: Number of bars per year. Crypto convention:
            8760 for hourly, 365 for daily (see ``metrics.py`` for
            the full table).

    Returns:
        Fractional per-period rate.

    Raises:
        ConfigError: On non-positive ``periods_per_year``.

    Example:
        >>> borrow_rate_per_period(CostModel(), periods_per_year=8760.0)
        ... # ≈ 2.28e-7 per hourly bar at 20 bps annual
    """
    if periods_per_year <= 0:
        raise ConfigError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return (cost_model.borrow_rate_bps / 10_000.0) / periods_per_year


# ---------------------------------------------------------------------------
# Funding (perp holding cost)
# ---------------------------------------------------------------------------


def compute_funding_charge(
    position: pd.Series,
    funding_rate: pd.Series,
    cost_model: CostModel,
) -> float:
    """Total funding charge over the period, as a cost (positive = paid).

    Per-bar charge = ``position · funding_rate``. A long paying positive
    funding produces a positive charge (cost out); a short paying
    positive funding produces a negative charge (cash in). Summed
    across the period.

    Args:
        position: Signed position size per bar (positive long, negative
            short). Index must be identical to ``funding_rate.index``.
        funding_rate: Funding rate per bar (8-hour native cadence on
            BitMEX/Bybit; the caller resamples if needed). Must not
            contain NaN — a NaN here indicates upstream data corruption
            and is raised explicitly to avoid silently zeroing charges.
        cost_model: If ``cost_model.funding_applied`` is False, returns
            ``0.0`` immediately. This is the opt-out path for spot-only
            strategies.

    Returns:
        Summed funding cost. Can be negative (receipt) or positive
        (payment).

    Raises:
        ConfigError: On index mismatch or NaN in ``funding_rate``.

    Example:
        >>> # Long 1 BTC across 3 bars with +0.01% funding each:
        >>> pos = pd.Series([1.0, 1.0, 1.0], index=idx)
        >>> fr = pd.Series([0.0001, 0.0001, 0.0001], index=idx)
        >>> compute_funding_charge(pos, fr, CostModel())
        0.00030000000000000003
    """
    if not cost_model.funding_applied:
        return 0.0
    if not position.index.equals(funding_rate.index):
        raise ConfigError("compute_funding_charge: position and funding_rate must share index")
    if funding_rate.isna().any():
        raise ConfigError(
            "compute_funding_charge: funding_rate contains NaN; upstream "
            "data layer must ship complete funding series (Phase 2 contract)"
        )
    return float((position * funding_rate).sum())
