"""Contract tests for the frozen dataclasses in ``crypto_alpha_engine.types``.

These tests verify structural properties — immutability, defaults, equality,
construction — without depending on any business logic. Validation rules that
encode architectural principles (Principle 5: costs are mandatory) get
dedicated tests here because they're type-intrinsic, not engine-intrinsic.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

import pytest

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.types import (
    BacktestResult,
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)

# ---------------------------------------------------------------------------
# FactorNode
# ---------------------------------------------------------------------------


class TestFactorNode:
    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(FactorNode)
        node = FactorNode(operator="ts_mean", args=(), kwargs={"window": 20})
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.operator = "ts_std"  # type: ignore[misc]

    def test_construct_with_primitives(self) -> None:
        node = FactorNode(operator="add", args=(1, 2), kwargs={})
        assert node.operator == "add"
        assert node.args == (1, 2)
        assert node.kwargs == {}

    def test_construct_with_nested_children(self) -> None:
        child = FactorNode(operator="ts_mean", args=("close",), kwargs={"window": 20})
        parent = FactorNode(operator="sub", args=("close", child), kwargs={})
        assert parent.args[1] is child

    def test_kwargs_default_is_empty_dict_not_shared(self) -> None:
        # Mutable default aliasing is the classic dataclass footgun. Each
        # instance must get its own dict.
        a = FactorNode(operator="x", args=())
        b = FactorNode(operator="y", args=())
        assert a.kwargs == {}
        assert b.kwargs == {}
        assert a.kwargs is not b.kwargs

    def test_equality_by_value(self) -> None:
        a = FactorNode(operator="ts_mean", args=("close",), kwargs={"window": 20})
        b = FactorNode(operator="ts_mean", args=("close",), kwargs={"window": 20})
        c = FactorNode(operator="ts_mean", args=("close",), kwargs={"window": 21})
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------


class TestFactor:
    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Factor)
        root = FactorNode(operator="noop", args=())
        f = Factor(
            name="f1",
            description="test factor",
            hypothesis="random walk",
            root=root,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.name = "renamed"  # type: ignore[misc]

    def test_metadata_defaults_to_empty(self) -> None:
        root = FactorNode(operator="noop", args=())
        f = Factor(name="f1", description="d", hypothesis="h", root=root)
        assert f.metadata == {}

    def test_metadata_per_instance(self) -> None:
        root = FactorNode(operator="noop", args=())
        a = Factor(name="a", description="d", hypothesis="h", root=root)
        b = Factor(name="b", description="d", hypothesis="h", root=root)
        assert a.metadata is not b.metadata


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    def test_defaults_match_spec(self) -> None:
        """SPEC §8: train=24, test=3, step=1, min_train=12."""
        cfg = WalkForwardConfig()
        assert cfg.train_months == 24
        assert cfg.test_months == 3
        assert cfg.step_months == 1
        assert cfg.min_train_months == 12

    def test_is_frozen(self) -> None:
        cfg = WalkForwardConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.train_months = 36  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("train_months", 0),
            ("train_months", -1),
            ("test_months", 0),
            ("step_months", 0),
            ("min_train_months", 0),
        ],
    )
    def test_rejects_non_positive_windows(self, field: str, bad_value: int) -> None:
        kwargs: dict[str, Any] = {field: bad_value}
        with pytest.raises(ConfigError, match=field):
            WalkForwardConfig(**kwargs)

    def test_rejects_min_train_exceeding_train(self) -> None:
        with pytest.raises(ConfigError, match="min_train_months"):
            WalkForwardConfig(train_months=12, min_train_months=24)


# ---------------------------------------------------------------------------
# CostModel — Principle 5 (costs are mandatory)
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_defaults_match_spec(self) -> None:
        cm = CostModel()
        assert cm.taker_bps == 10.0
        assert cm.maker_bps == 2.0
        assert cm.slippage_model == "volume_based"
        assert cm.funding_applied is True
        assert cm.borrow_rate_bps == 20.0

    def test_is_frozen(self) -> None:
        cm = CostModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cm.taker_bps = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("taker_bps", 0.0),
            ("taker_bps", -1.0),
            ("maker_bps", 0.0),
            ("maker_bps", -0.5),
            ("borrow_rate_bps", 0.0),
            ("borrow_rate_bps", -5.0),
        ],
    )
    def test_rejects_zero_or_negative_costs(self, field: str, bad_value: float) -> None:
        """Principle 5: costs cannot be set to zero."""
        kwargs: dict[str, Any] = {field: bad_value}
        with pytest.raises(ConfigError, match=field):
            CostModel(**kwargs)

    def test_accepts_custom_positive_values(self) -> None:
        cm = CostModel(taker_bps=7.5, maker_bps=1.5, borrow_rate_bps=15.0)
        assert cm.taker_bps == 7.5


# ---------------------------------------------------------------------------
# BacktestResult — shape only; values are computed by the engine.
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_is_frozen(self, backtest_result_sample: BacktestResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            backtest_result_sample.sharpe = 99.0  # type: ignore[misc]

    def test_all_spec_fields_present(self, backtest_result_sample: BacktestResult) -> None:
        """Every field listed in SPEC §8 is defined on the dataclass."""
        spec_fields = {
            # Identity
            "factor_id",
            "factor_name",
            "run_timestamp",
            "data_version",
            # Headline
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "annualized_return",
            "total_return",
            # Predictive power
            "ic_mean",
            "ic_std",
            "ic_ir",
            "hit_rate",
            # Cost breakdown
            "gross_sharpe",
            "net_sharpe",
            "turnover_annual",
            "total_fees_paid",
            "total_slippage_paid",
            # Regime breakdown
            "bull_sharpe",
            "bear_sharpe",
            "crab_sharpe",
            "high_vol_sharpe",
            "low_vol_sharpe",
            "normal_vol_sharpe",
            "euphoric_funding_sharpe",
            "fearful_funding_sharpe",
            "neutral_funding_sharpe",
            # Robustness
            "walk_forward_sharpe_mean",
            "walk_forward_sharpe_std",
            "in_sample_sharpe",
            "out_of_sample_sharpe",
            "deflated_sharpe_ratio",
            "n_experiments_in_ledger",
            # Factor properties
            "factor_ast_depth",
            "factor_node_count",
            "factor_max_similarity_to_zoo",
            # Trade statistics
            "n_trades",
            "avg_trade_duration_hours",
            "avg_position_size",
            "max_leverage_used",
        }
        actual_fields = {f.name for f in dataclasses.fields(backtest_result_sample)}
        missing = spec_fields - actual_fields
        assert not missing, f"BacktestResult missing SPEC §8 fields: {sorted(missing)}"

    def test_does_not_expose_trade_level_data(self, backtest_result_sample: BacktestResult) -> None:
        """Principle 1 (sealed engine): trade-level data must NEVER be on the result.

        This test is a canary. If anyone adds ``equity_curve`` or similar to
        BacktestResult, it fires and blocks the commit.
        """
        forbidden = {
            "equity_curve",
            "daily_returns",
            "trade_timestamps",
            "per_trade_pnl",
            "drawdown_dates",
            "returns_series",
            "positions_series",
        }
        actual_fields = {f.name for f in dataclasses.fields(backtest_result_sample)}
        leaks = forbidden & actual_fields
        assert not leaks, f"sealed-engine violation: {sorted(leaks)}"

    def test_run_timestamp_is_tz_aware(self, backtest_result_sample: BacktestResult) -> None:
        """CLAUDE.md Pitfall 3: never accept naive timestamps."""
        ts = backtest_result_sample.run_timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_equality_by_value(
        self,
        backtest_result_factory: Any,
    ) -> None:
        a = backtest_result_factory()
        b = backtest_result_factory()
        # Two results with identical field values are equal.
        assert a == b

    def test_rejects_naive_run_timestamp(self, backtest_result_factory: Any) -> None:
        with pytest.raises(ConfigError, match="timezone"):
            backtest_result_factory(run_timestamp=datetime(2026, 1, 1))  # naive


# ---------------------------------------------------------------------------
# Cross-cutting: no dataclass accidentally leaves frozen=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [FactorNode, Factor, WalkForwardConfig, CostModel, BacktestResult],
)
def test_every_type_is_frozen(cls: type) -> None:
    """SPEC §3 mandates frozen dataclasses for immutable data contracts."""
    assert dataclasses.is_dataclass(cls)
    # dataclasses.fields works for frozen and non-frozen; we check the marker
    # via the generated ``__dataclass_params__``.
    assert cls.__dataclass_params__.frozen is True, (  # type: ignore[attr-defined]
        f"{cls.__name__} must be @dataclass(frozen=True)"
    )
