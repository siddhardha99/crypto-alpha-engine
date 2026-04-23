"""Shared pytest fixtures.

Phase 1 fixtures are deliberately minimal — just what the types/exceptions
tests need. Real-data fixtures (BTC/ETH parquet slices) land in Phase 2 via
``scripts/generate_test_fixtures.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from crypto_alpha_engine.types import BacktestResult


def _default_backtest_result_kwargs() -> dict[str, Any]:
    """Sentinel values for every BacktestResult field.

    The values are deliberately non-zero so tests that filter on "populated"
    fields don't false-negative. Any test that cares about a specific field
    overrides it explicitly.
    """
    return {
        # Identity
        "factor_id": "f_test_00000000",
        "factor_name": "test_factor",
        "run_timestamp": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        "data_version": "sha256:deadbeef",
        # Headline
        "sharpe": 1.23,
        "sortino": 1.45,
        "calmar": 0.87,
        "max_drawdown": -0.18,
        "annualized_return": 0.22,
        "total_return": 0.67,
        # Predictive power
        "ic_mean": 0.03,
        "ic_std": 0.12,
        "ic_ir": 0.25,
        "hit_rate": 0.54,
        # Cost breakdown
        "gross_sharpe": 1.55,
        "net_sharpe": 1.23,
        "turnover_annual": 8.4,
        "total_fees_paid": 1234.5,
        "total_slippage_paid": 678.9,
        # Regime breakdown
        "bull_sharpe": 1.8,
        "bear_sharpe": 0.4,
        "crab_sharpe": 0.9,
        "high_vol_sharpe": 1.1,
        "low_vol_sharpe": 1.4,
        "normal_vol_sharpe": 1.3,
        "euphoric_funding_sharpe": 0.7,
        "fearful_funding_sharpe": 1.6,
        "neutral_funding_sharpe": 1.2,
        # Robustness
        "walk_forward_sharpe_mean": 1.15,
        "walk_forward_sharpe_std": 0.30,
        "in_sample_sharpe": 1.40,
        "out_of_sample_sharpe": 1.05,
        "deflated_sharpe_ratio": 0.78,
        "n_experiments_in_ledger": 42,
        # Factor properties
        "factor_ast_depth": 4,
        "factor_node_count": 11,
        "factor_max_similarity_to_zoo": 0.33,
        # Trade statistics
        "n_trades": 523,
        "avg_trade_duration_hours": 14.2,
        "avg_position_size": 0.25,
        "max_leverage_used": 1.5,
    }


@pytest.fixture
def backtest_result_sample() -> BacktestResult:
    """A single valid BacktestResult with sentinel field values.

    Use this fixture for tests that need an instance but don't care about
    specific values. For tests that need to vary one field, use
    ``backtest_result_factory`` instead.
    """
    return BacktestResult(**_default_backtest_result_kwargs())


@pytest.fixture
def backtest_result_factory() -> Callable[..., BacktestResult]:
    """Factory for constructing BacktestResult instances with overrides.

    Returns a callable ``factory(**overrides) -> BacktestResult`` that starts
    from the sentinel defaults and applies any provided overrides. Lets a
    test vary a single field (e.g. ``run_timestamp``) without spelling out
    the other 37.
    """

    def factory(**overrides: Any) -> BacktestResult:
        kwargs = _default_backtest_result_kwargs()
        kwargs.update(overrides)
        return BacktestResult(**kwargs)

    return factory
