"""Frozen data contracts for crypto-alpha-engine.

Every type that crosses a module boundary is defined here as a
``@dataclass(frozen=True)``. This makes the shape of data moving through the
engine a matter of static record — tests can pin on it, tools can introspect
it, and mypy --strict verifies callers respect it.

The types cluster into four groups:

1. **Factor AST** — :class:`FactorNode`, :class:`Factor`. Live serializable
   representation of a strategy. Full compiler/parser lands in Phase 4.
2. **Engine configuration** — :class:`WalkForwardConfig`, :class:`CostModel`.
   Constructor-time invariants live in ``__post_init__``.
3. **Engine output** — :class:`BacktestResult`. 38 fields of aggregate
   metrics, per SPEC §8. Deliberately lacks any trade-level, equity-curve,
   or timestamp-series field (Principle 1: sealed engine).
4. (No other groups in Phase 1.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crypto_alpha_engine.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Factor AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorNode:
    """A single node in a factor's AST.

    A factor is a tree of operator applications. A leaf node is an operator
    taking only primitive args (strings naming raw data columns, numeric
    constants, etc.); an internal node's ``args`` contains one or more other
    :class:`FactorNode` children.

    Args:
        operator: Name of a registered operator (e.g. ``"ts_mean"``,
            ``"add"``). Resolved against
            :mod:`crypto_alpha_engine.operators.registry` at compile time
            (Phase 4). The operator name is NOT validated here; that belongs
            in the parser/compiler.
        args: Positional arguments to the operator. Each element is either a
            primitive (``str``, ``int``, ``float``, ``bool``) or a child
            :class:`FactorNode`. Ordered.
        kwargs: Keyword arguments to the operator (e.g. ``{"window": 20}``).
            Defaults to an empty dict.

    Note:
        ``frozen=True`` gives attribute-level immutability. Since ``kwargs``
        is a plain ``dict``, its *contents* are still mutable — don't rely
        on deep immutability. Deep freezing will be revisited in Phase 4 if
        the similarity/hashing code needs it.

    Example:
        >>> # ts_mean(close, window=20)
        >>> node = FactorNode(
        ...     operator="ts_mean",
        ...     args=("close",),
        ...     kwargs={"window": 20},
        ... )
        >>> node.operator
        'ts_mean'
    """

    operator: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Factor:
    """A named, hypothesis-backed factor built from a :class:`FactorNode` tree.

    The metadata distinguishes a *factor* (the entity submitted to the
    engine and logged in the ledger) from a bare AST. ``hypothesis`` is
    required so every stored factor carries the question it is trying to
    answer — a guard against random curve-fitting.

    Args:
        name: Human-readable identifier, snake_case by convention
            (e.g. ``"funding_zscore_24h"``).
        description: One- or two-sentence description.
        hypothesis: The economic or behavioral claim the factor tests.
            Required — an empty string is allowed by the type but
            discouraged by review.
        root: The AST root :class:`FactorNode`.
        metadata: Free-form tags (``{"author": "...", "submitted": "..."}``).
            Defaults to an empty dict.

    Example:
        >>> root = FactorNode(operator="ts_mean", args=("close",),
        ...                   kwargs={"window": 20})
        >>> f = Factor(
        ...     name="close_ma_20h",
        ...     description="20-hour moving average of close price",
        ...     hypothesis="Short-term mean reversion around 20h MA",
        ...     root=root,
        ... )
        >>> f.name
        'close_ma_20h'
    """

    name: str
    description: str
    hypothesis: str
    root: FactorNode
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward split geometry for :class:`BacktestEngine.run`.

    Per SPEC §8 the defaults describe a rolling 24-month train / 3-month test
    window stepped monthly, with at least 12 months of history before the
    first prediction. All values are in calendar months.

    Args:
        train_months: Rolling train window length. Must be positive.
        test_months: Rolling test window length. Must be positive.
        step_months: How far each window advances per step. Must be positive.
        min_train_months: Minimum history required before the first
            prediction is made. Must be positive and ``<= train_months``.

    Raises:
        ConfigError: If any window is zero/negative, or if
            ``min_train_months > train_months``.

    Example:
        >>> cfg = WalkForwardConfig()
        >>> cfg.train_months, cfg.test_months, cfg.step_months
        (24, 3, 1)
    """

    train_months: int = 24
    test_months: int = 3
    step_months: int = 1
    min_train_months: int = 12

    def __post_init__(self) -> None:
        for name, value in (
            ("train_months", self.train_months),
            ("test_months", self.test_months),
            ("step_months", self.step_months),
            ("min_train_months", self.min_train_months),
        ):
            if value <= 0:
                raise ConfigError(f"WalkForwardConfig.{name} must be positive, got {value!r}")
        if self.min_train_months > self.train_months:
            raise ConfigError(
                "WalkForwardConfig.min_train_months "
                f"({self.min_train_months}) must be <= train_months "
                f"({self.train_months})"
            )


@dataclass(frozen=True)
class CostModel:
    """Fees, slippage, and funding costs applied to every backtest.

    Principle 5 of SPEC §2: costs are mandatory. Zero or negative values for
    any cost component are rejected at construction. The defaults are taken
    from SPEC §8 and reflect realistic Binance-tier fees for a mid-volume
    taker.

    Args:
        taker_bps: Taker fee per side, in basis points. Must be positive.
        maker_bps: Maker fee per side, in basis points. Must be positive.
        slippage_model: Identifier for the slippage function to apply.
            ``"volume_based"`` is the Phase 6 default; other models may be
            added by the backtest module.
        funding_applied: Whether perpetual-futures funding is charged to the
            position over its holding period.
        borrow_rate_bps: Annualized borrow cost for short spot positions,
            in basis points. Must be positive.

    Raises:
        ConfigError: If any bps value is zero or negative.

    Example:
        >>> cm = CostModel()
        >>> cm.taker_bps, cm.maker_bps
        (10.0, 2.0)

        >>> from crypto_alpha_engine.exceptions import ConfigError
        >>> try:
        ...     CostModel(taker_bps=0)
        ... except ConfigError as err:
        ...     "taker_bps" in str(err)
        True
    """

    taker_bps: float = 10.0
    maker_bps: float = 2.0
    slippage_model: str = "volume_based"
    funding_applied: bool = True
    borrow_rate_bps: float = 20.0

    def __post_init__(self) -> None:
        for name, value in (
            ("taker_bps", self.taker_bps),
            ("maker_bps", self.maker_bps),
            ("borrow_rate_bps", self.borrow_rate_bps),
        ):
            if value <= 0:
                raise ConfigError(
                    f"CostModel.{name} must be positive "
                    f"(Principle 5: costs are mandatory), got {value!r}"
                )


# ---------------------------------------------------------------------------
# Engine output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """The single, sealed output of :class:`BacktestEngine.run`.

    Every field is an aggregate metric or identifier. No field exposes
    trade-level, equity-curve, or per-timestamp data — this is the
    structural enforcement of Principle 1 (sealed engine). If any future
    change tries to add such a field, the canary test in
    ``tests/unit/test_types.py::TestBacktestResult.test_does_not_expose_trade_level_data``
    will fail.

    Field groups match SPEC §8:

    - **Identity** — who ran what, when, against which data.
    - **Headline** — top-line performance metrics, all post-cost.
    - **Predictive power** — information coefficient and hit rate.
    - **Cost breakdown** — gross vs net, turnover, fees, slippage.
    - **Regime breakdown** — Sharpe per trend/volatility/funding regime.
    - **Robustness** — walk-forward stats, in-sample vs out-of-sample,
      deflated Sharpe, experiment count.
    - **Factor properties** — AST complexity and zoo similarity.
    - **Trade statistics** — aggregate trade-count / duration / size.

    Raises:
        ConfigError: If ``run_timestamp`` is not timezone-aware
            (CLAUDE.md Pitfall 3: never accept naive timestamps).

    Example:
        Instances are constructed by the engine, not by user code. Tests
        use the ``backtest_result_factory`` fixture in ``tests/conftest.py``.
    """

    # --- Identity ---
    factor_id: str
    factor_name: str
    run_timestamp: datetime
    data_version: str

    # --- Headline ---
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    annualized_return: float
    total_return: float

    # --- Predictive power ---
    ic_mean: float
    ic_std: float
    ic_ir: float
    hit_rate: float

    # --- Cost breakdown ---
    gross_sharpe: float
    net_sharpe: float
    turnover_annual: float
    total_fees_paid: float
    total_slippage_paid: float

    # --- Regime breakdown ---
    bull_sharpe: float
    bear_sharpe: float
    crab_sharpe: float
    high_vol_sharpe: float
    low_vol_sharpe: float
    normal_vol_sharpe: float
    euphoric_funding_sharpe: float
    fearful_funding_sharpe: float
    neutral_funding_sharpe: float

    # --- Robustness ---
    walk_forward_sharpe_mean: float
    walk_forward_sharpe_std: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    deflated_sharpe_ratio: float
    n_experiments_in_ledger: int

    # --- Factor properties ---
    factor_ast_depth: int
    factor_node_count: int
    factor_max_similarity_to_zoo: float

    # --- Trade statistics (aggregate only) ---
    n_trades: int
    avg_trade_duration_hours: float
    avg_position_size: float
    max_leverage_used: float

    def __post_init__(self) -> None:
        if self.run_timestamp.tzinfo is None:
            raise ConfigError(
                "BacktestResult.run_timestamp must be timezone-aware " "(UTC); got a naive datetime"
            )
