"""Walk-forward orchestration: fold schedule + per-fold aggregation.

Two public surfaces:

* :func:`iter_folds` — pure function from
  ``(data_index, splits, config, min_test_bars)`` to a sequence of
  :class:`FoldSpec`. Deterministic, no side effects, no state.
* :func:`aggregate_folds` — collapses a list of :class:`FoldResult`
  into an :class:`AggregatedFolds` bundle that engine.py maps to
  :class:`BacktestResult`.

Isolation contract
------------------

Both functions are pure. The engine's per-fold simulation loop calls
``simulate_fold`` once per fold; each call takes only slices of the
input features/prices, carries no shared mutable state, and produces
a :class:`FoldResult` by value. Running folds in any permutation
yields byte-identical per-fold results — enforced by the
fold-independence canary in ``tests/unit/test_walk_forward.py``
across three orderings (forward, shuffled, reverse).

Fold semantics
--------------

The walk-forward loop explores ``[data_start, splits.train_end)`` —
the validation and test blocks held by :class:`DataSplits` stay
untouched for final OOS evaluation. Each fold has:

* ``test_start = data_start + (min_train_months + i · step_months)``
* ``test_end = test_start + test_months``
* ``train_end = test_start`` (half-open)
* ``train_start = max(data_start, test_start − train_months)`` —
  rolling window once history is long enough; expanding until then.

The ``min_train_months`` floor gates the first test_start; the
rolling ``train_months`` window starts contributing as soon as the
test_start has enough history behind it. Both at once keeps the
early folds from running against a sub-minimum window.

Truncation at the walk-forward ceiling
--------------------------------------

The final fold whose natural test_end overshoots ``splits.train_end``
is truncated to end at the ceiling — not dropped — provided the
remaining bars meet ``min_test_bars`` (default 24). Below that
threshold, the partial fold is skipped: metrics on a sub-minimum
window are too unstable to contribute honestly to the aggregate.

Walk-forward Sharpe semantics
-----------------------------

``per_fold_sharpes`` in :class:`AggregatedFolds` is the list of Sharpes
computed per fold. Engine.py takes ``mean`` and ``std`` of this list
to populate the SPEC §8 ``walk_forward_sharpe_mean`` and
``walk_forward_sharpe_std`` fields. The headline ``sharpe`` on
``BacktestResult`` is computed separately on the **concatenated**
returns across folds — a different quantity. Mean-of-Sharpes vs
Sharpe-of-concat differ under almost any real covariance structure;
SPEC §8 asks for both on purpose.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_alpha_engine.backtest.metrics import sharpe
from crypto_alpha_engine.backtest.simulation import FoldResult
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.types import WalkForwardConfig

# ---------------------------------------------------------------------------
# Fold spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    """One fold's time bounds — consumed by engine.py for slicing.

    All timestamps are UTC-aware (DataSplits enforces this upstream).
    Bounds are **half-open on the upper end**: a bar with timestamp
    equal to ``test_end`` is NOT in the test window. This matches
    pandas ``.loc[a:b]`` behavior under ``closed="left"`` when the
    caller passes an open-right window explicitly, and matches the
    DataSplits convention (``train_end`` is exclusive).

    Attributes:
        train_start: Earliest bar (inclusive) in the fold's train
            window. Always ``>= data_index[0]``.
        train_end: End of train window (exclusive). Equal to
            ``test_start`` by construction — the train/test boundary
            is a single timestamp.
        test_start: Earliest bar (inclusive) in the fold's test
            window.
        test_end: End of test window (exclusive). Always
            ``<= splits.train_end`` — the walk-forward loop never
            peeks into validation or test splits.
        fold_id: Zero-based index, assigned in schedule order.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold_id: int


# ---------------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedFolds:
    """Walk-forward aggregation of per-fold :class:`FoldResult`s.

    Intermediate bundle — engine.py consumes this to fill the
    corresponding slots on :class:`BacktestResult`. Not returned to
    user code; Principle 1 keeps the bar-level series behind the
    sealed engine boundary.

    Attributes:
        gross_returns: Pre-cost returns concatenated across folds.
            Feeds ``gross_sharpe``.
        net_returns: Post-cost returns concatenated across folds.
            Feeds all headline metrics (``sharpe``, ``sortino``, etc.)
            and the regime breakdown.
        per_fold_sharpes: One Sharpe per fold, computed on that
            fold's ``net_returns`` alone. ``mean`` / ``std`` of this
            list are the SPEC §8 walk-forward robustness fields.
        total_fees_paid: Sum across folds.
        total_slippage_paid: Sum across folds.
        total_funding_paid: Sum across folds (signed).
        n_trades_total: Sum across folds.
        avg_trade_duration_hours: Trade-weighted mean — a 100-trade
            fold contributes 100x a 1-trade fold.
        avg_position_size: Bar-weighted mean across folds.
        max_leverage_used: Max across folds (never sum — dimensionally
            meaningless).
        total_turnover: Sum across folds.
        n_folds: Number of folds aggregated.
    """

    gross_returns: pd.Series
    net_returns: pd.Series
    per_fold_sharpes: list[float]
    total_fees_paid: float
    total_slippage_paid: float
    total_funding_paid: float
    n_trades_total: int
    avg_trade_duration_hours: float
    avg_position_size: float
    max_leverage_used: float
    total_turnover: float
    n_folds: int


# ---------------------------------------------------------------------------
# iter_folds
# ---------------------------------------------------------------------------


def iter_folds(
    *,
    data_index: pd.DatetimeIndex,
    splits: DataSplits,
    config: WalkForwardConfig,
    min_test_bars: int = 24,
) -> Iterator[FoldSpec]:
    """Yield :class:`FoldSpec` in schedule order.

    Pure generator — no side effects. Same inputs always yield the
    same fold sequence.

    Args:
        data_index: The features/prices DatetimeIndex. Empty index
            yields no folds.
        splits: Gives the walk-forward ceiling via ``splits.train_end``.
            No fold's test_end ever exceeds this.
        config: Window/step geometry. All values in calendar months.
        min_test_bars: Minimum bars required in a (possibly-truncated)
            test window before the fold is yielded. Below this, the
            partial fold is skipped. Default 24.

    Yields:
        FoldSpec instances with zero-based ``fold_id`` in schedule order.

    Example:
        >>> idx = pd.date_range("2022-01-01", periods=700, freq="1d", tz="UTC")
        >>> splits = DataSplits(train_end=idx[600], validation_end=idx[650])
        >>> cfg = WalkForwardConfig(train_months=6, test_months=1,
        ...                         step_months=1, min_train_months=3)
        >>> folds = list(iter_folds(data_index=idx, splits=splits, config=cfg))
    """
    if len(data_index) == 0:
        return
    data_start = data_index[0]
    if data_start >= splits.train_end:
        return

    first_test_start = data_start + pd.DateOffset(months=config.min_train_months)

    i = 0
    while True:
        test_start = first_test_start + pd.DateOffset(months=i * config.step_months)
        if test_start >= splits.train_end:
            return  # whole remainder is past the WF ceiling

        natural_test_end = test_start + pd.DateOffset(months=config.test_months)
        effective_test_end = min(natural_test_end, splits.train_end)

        # Count bars in [test_start, effective_test_end).
        mask = (data_index >= test_start) & (data_index < effective_test_end)
        test_bar_count = int(mask.sum())
        if test_bar_count < min_test_bars:
            return  # natural end is beyond the ceiling AND partial window too small

        train_start_candidate = test_start - pd.DateOffset(months=config.train_months)
        train_start = max(data_start, train_start_candidate)

        yield FoldSpec(
            train_start=train_start,
            train_end=test_start,  # exclusive
            test_start=test_start,
            test_end=effective_test_end,
            fold_id=i,
        )
        i += 1


# ---------------------------------------------------------------------------
# aggregate_folds
# ---------------------------------------------------------------------------


def aggregate_folds(fold_results: list[FoldResult]) -> AggregatedFolds:
    """Collapse a list of :class:`FoldResult` into one :class:`AggregatedFolds`.

    Aggregation rules:

    * Cost totals: sum across folds.
    * n_trades_total: sum.
    * avg_trade_duration_hours: mean weighted by per-fold n_trades.
      Rationale — a fold with many trades contributes proportionally
      more to the average trade's typical duration.
    * avg_position_size: mean weighted by per-fold bar count.
    * max_leverage_used: max across folds (never sum).
    * total_turnover: sum.
    * per_fold_sharpes: list of ``sharpe(fold.net_returns)`` values,
      one per fold. Engine.py takes ``mean`` / ``std`` of this for
      the walk-forward robustness fields.
    * gross_returns / net_returns: pd.concat preserving timestamps.
      Non-contiguous (gapped) folds produce non-contiguous series —
      downstream metrics (Sharpe, drawdown) don't care about gaps.

    Args:
        fold_results: One FoldResult per fold, in any order (order
            doesn't affect aggregation by construction — all
            operations are order-independent).

    Returns:
        :class:`AggregatedFolds`.

    Raises:
        ConfigError: On empty ``fold_results``.
    """
    if not fold_results:
        raise ConfigError("aggregate_folds: empty fold_results list")

    gross_returns = pd.concat([r.gross_returns for r in fold_results])
    net_returns = pd.concat([r.net_returns for r in fold_results])

    per_fold_sharpes = [sharpe(r.net_returns) for r in fold_results]

    total_fees = float(sum(r.fees_paid for r in fold_results))
    total_slippage = float(sum(r.slippage_paid for r in fold_results))
    total_funding = float(sum(r.funding_paid for r in fold_results))
    n_trades_total = int(sum(r.n_trades for r in fold_results))
    total_turnover = float(sum(r.turnover for r in fold_results))
    max_leverage = float(max(r.max_leverage_used for r in fold_results))

    if n_trades_total > 0:
        weighted_duration = sum(r.avg_trade_duration_hours * r.n_trades for r in fold_results)
        avg_duration = float(weighted_duration / n_trades_total)
    else:
        avg_duration = 0.0

    total_bars = sum(len(r.net_returns) for r in fold_results)
    if total_bars > 0:
        weighted_position = sum(r.avg_position_size * len(r.net_returns) for r in fold_results)
        avg_position = float(weighted_position / total_bars)
    else:
        avg_position = 0.0

    # Guard against non-finite leakage into downstream aggregation.
    # A NaN in any extracted scalar upstream would poison the ledger;
    # surface loudly if it happens, don't let it through.
    for name, value in (
        ("total_fees_paid", total_fees),
        ("total_slippage_paid", total_slippage),
        ("max_leverage_used", max_leverage),
        ("total_turnover", total_turnover),
    ):
        if not np.isfinite(value):
            raise ConfigError(
                f"aggregate_folds: {name} is non-finite ({value!r}); "
                f"upstream FoldResult contains NaN or inf"
            )

    return AggregatedFolds(
        gross_returns=gross_returns,
        net_returns=net_returns,
        per_fold_sharpes=per_fold_sharpes,
        total_fees_paid=total_fees,
        total_slippage_paid=total_slippage,
        total_funding_paid=total_funding,
        n_trades_total=n_trades_total,
        avg_trade_duration_hours=avg_duration,
        avg_position_size=avg_position,
        max_leverage_used=max_leverage,
        total_turnover=total_turnover,
        n_folds=len(fold_results),
    )
