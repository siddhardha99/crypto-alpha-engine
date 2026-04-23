"""Tests for the engine's top-level run_backtest entry point.

Scope (commit 4 of Phase 6):

* Every registered operator declares ``causal_safe`` — canary.
* Layer 1 (AST whitelist) rejects factors using unsafe operators.
* Layer 2 (runtime perturbation) is wired and runs; a dedicated
  end-to-end evil-factor test lands in commit 5.
* data_version is a deterministic byte-stable hash.
* Signal generation is pluggable via ``signal_rule``; the default is
  sign-based.
* End-to-end ``run_backtest`` populates every BacktestResult field.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.engine import (
    _compute_data_version,
    _layer_1_causality_check,
    _shift_signals_by_one_bar,
    default_signal_rule,
    run_backtest,
)
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import LookAheadDetected
from crypto_alpha_engine.operators import registry as op_registry
from crypto_alpha_engine.regime import build_default_labels
from crypto_alpha_engine.types import (
    BacktestResult,
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)


def _build_features_and_prices(
    n: int = 1200,
    *,
    seed: int = 42,
) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """Synthetic 1200-day dataset: features dict + prices + funding."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    returns = rng.normal(0.0005, 0.02, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx, name="close")
    features = {"BTC/USD|close": close}
    funding = pd.Series(rng.normal(0.0001, 0.0001, size=n), index=idx, name="funding_rate")
    return features, close, funding


def _simple_factor() -> Factor:
    """Factor: ts_mean(close, 20) — causal, trivial."""
    root = FactorNode(operator="ts_mean", args=("BTC/USD|close", 20), kwargs={})
    return Factor(
        name="ma_20d",
        description="20-day moving average of close",
        hypothesis="Prices mean-revert around the 20d MA",
        root=root,
    )


# ---------------------------------------------------------------------------
# Registry canary: every operator declares causal_safe
# ---------------------------------------------------------------------------


def test_every_registered_operator_has_causal_safe_annotation() -> None:
    """Every registered operator must carry a ``causal_safe`` attribute.

    Required because Layer 1's AST check reads this flag off every
    operator the factor uses. A missed annotation means Layer 1 is
    silently useless for any factor referencing that operator.
    """
    # Force operator modules to load (they auto-register on import).
    import crypto_alpha_engine.operators  # noqa: F401

    ops = op_registry.list_operators()
    assert len(ops) >= 46, f"expected >= 46 operators, got {len(ops)}"
    for name in ops:
        spec = op_registry._get_spec(name)
        assert hasattr(spec, "causal_safe"), f"operator {name!r} missing causal_safe"
        assert spec.causal_safe is True, (
            f"operator {name!r} registered with causal_safe={spec.causal_safe}; "
            f"canary fires — either flip the flag to True or document the "
            f"research-purpose exception"
        )


# ---------------------------------------------------------------------------
# Layer 1: AST whitelist
# ---------------------------------------------------------------------------


class TestLayer1Causality:
    def test_safe_factor_passes(self) -> None:
        """A factor using only causal_safe=True operators passes Layer 1."""
        _layer_1_causality_check(_simple_factor().root)  # should not raise

    def test_unsafe_operator_trips_layer_1(self) -> None:
        """A factor referencing an operator registered with
        causal_safe=False must be rejected by Layer 1 *before* any
        simulation runs. This validates the Layer 1 check is actually
        wired — commit 5 validates Layer 2 catches runtime leakage."""
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator(
                "fake_unsafe_op", arg_types=("series",), causal_safe=False
            )
            def _evil(x: pd.Series) -> pd.Series:
                return x

            root = FactorNode(operator="fake_unsafe_op", args=("BTC/USD|close",))
            with pytest.raises(LookAheadDetected, match="causal_safe"):
                _layer_1_causality_check(root)
        finally:
            op_registry._restore_for_tests(snapshot)

    def test_unsafe_operator_deep_in_tree_trips(self) -> None:
        """Layer 1 walks the whole AST — an unsafe operator nested
        under safe parents still trips."""
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator("deep_unsafe", arg_types=("series",), causal_safe=False)
            def _deep(x: pd.Series) -> pd.Series:
                return x

            # Nested: ts_mean(deep_unsafe(close), 20) — the unsafe is
            # a grand-child of the root.
            inner = FactorNode(operator="deep_unsafe", args=("BTC/USD|close",))
            root = FactorNode(operator="ts_mean", args=(inner, 20), kwargs={})
            with pytest.raises(LookAheadDetected, match="deep_unsafe"):
                _layer_1_causality_check(root)
        finally:
            op_registry._restore_for_tests(snapshot)


# ---------------------------------------------------------------------------
# Signal generation + shift
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    def test_default_signal_rule_sign_based(self) -> None:
        """Default: entry when factor > 0, exit when factor < 0."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        values = pd.Series([1.0, -1.0, 0.0, 2.0, -3.0], index=idx)
        entries, exits = default_signal_rule(values)
        assert entries.tolist() == [True, False, False, True, False]
        assert exits.tolist() == [False, True, False, False, True]

    def test_default_handles_nan_as_false(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        values = pd.Series([float("nan"), 1.0, -1.0], index=idx)
        entries, exits = default_signal_rule(values)
        assert entries.tolist() == [False, True, False]
        assert exits.tolist() == [False, False, True]

    def test_shift_by_one_bar_marks_attrs(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        raw_e = pd.Series([True, False, True, False, True], index=idx)
        raw_x = pd.Series([False, True, False, True, False], index=idx)
        e, x = _shift_signals_by_one_bar(raw_e, raw_x)
        # First bar must be False post-shift (nothing to execute at t=0).
        assert e.iloc[0] is False or e.iloc[0] == np.False_
        assert x.iloc[0] is False or x.iloc[0] == np.False_
        # Remaining bars = original shifted forward.
        assert e.iloc[1:].tolist() == raw_e.iloc[:-1].tolist()
        assert x.iloc[1:].tolist() == raw_x.iloc[:-1].tolist()
        # Marker stamped.
        assert e.attrs["shifted"] is True
        assert x.attrs["shifted"] is True


# ---------------------------------------------------------------------------
# data_version determinism
# ---------------------------------------------------------------------------


class TestDataVersion:
    def test_same_inputs_yield_byte_identical_strings(self) -> None:
        """Determinism canary: identical features must produce byte-equal
        data_version strings. Silent drift here poisons the ledger."""
        features, _, _ = _build_features_and_prices(n=200)
        v1 = _compute_data_version(
            features, feature_source_names={"BTC/USD|close": "coinbase_spot"}
        )
        v2 = _compute_data_version(
            features, feature_source_names={"BTC/USD|close": "coinbase_spot"}
        )
        assert v1 == v2
        assert v1.startswith("sha256:")

    def test_different_values_yield_different_versions(self) -> None:
        features_a, _, _ = _build_features_and_prices(n=200, seed=1)
        features_b, _, _ = _build_features_and_prices(n=200, seed=2)
        v_a = _compute_data_version(features_a, feature_source_names=None)
        v_b = _compute_data_version(features_b, feature_source_names=None)
        assert v_a != v_b

    def test_different_source_names_yield_different_versions(self) -> None:
        """Two datasets with identical values but different source
        provenance get distinct version strings. This is how the ledger
        distinguishes runs against ``my_sentiment_v2`` vs ``v3`` (SPEC §5.1)."""
        features, _, _ = _build_features_and_prices(n=200)
        v1 = _compute_data_version(features, feature_source_names={"BTC/USD|close": "coinbase"})
        v2 = _compute_data_version(features, feature_source_names={"BTC/USD|close": "bitmex"})
        assert v1 != v2

    def test_key_ordering_does_not_affect_version(self) -> None:
        """Dict iteration order must not matter — data_version sorts
        keys internally."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
        s1 = pd.Series(rng.normal(size=100), index=idx)
        s2 = pd.Series(rng.normal(size=100), index=idx)
        d_fwd = {"a": s1, "b": s2}
        d_rev = {"b": s2, "a": s1}
        v_fwd = _compute_data_version(d_fwd, feature_source_names=None)
        v_rev = _compute_data_version(d_rev, feature_source_names=None)
        assert v_fwd == v_rev


# ---------------------------------------------------------------------------
# End-to-end run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktestEndToEnd:
    def test_populates_every_backtest_result_field(self) -> None:
        """Every field on BacktestResult must be populated by
        run_backtest. Gaps here would turn into ledger corruption in
        Phase 7 — diagnose them now, not six months from now.
        """
        features, close, funding = _build_features_and_prices(n=1200)
        regime_labels = build_default_labels(close_for_trend=close, funding_rate=funding)

        result = run_backtest(
            factor=_simple_factor(),
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=DataSplits(
                train_end=close.index[1100],
                validation_end=close.index[1150],
            ),
            cost_model=CostModel(),
            walk_forward_config=WalkForwardConfig(
                train_months=6, test_months=1, step_months=1, min_train_months=3
            ),
            funding_rate=funding,
            feature_source_names={"BTC/USD|close": "coinbase_spot"},
            freq="1D",
            min_test_bars=10,
        )

        assert isinstance(result, BacktestResult)
        for field_name in BacktestResult.__dataclass_fields__:
            value = getattr(result, field_name)
            # Every field must be a real value — not None.
            assert value is not None, f"{field_name} is None on result"

    def test_factor_max_similarity_defaults_to_nan(self) -> None:
        """ "Not computed" (NaN) must be syntactically distinct from
        "computed and came out zero" (0.0). Per the architecture
        sketch amendment."""
        features, close, funding = _build_features_and_prices(n=600)
        regime_labels = build_default_labels(close_for_trend=close, funding_rate=funding)
        result = run_backtest(
            factor=_simple_factor(),
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=DataSplits(
                train_end=close.index[550],
                validation_end=close.index[580],
            ),
            cost_model=CostModel(),
            walk_forward_config=WalkForwardConfig(
                train_months=3, test_months=1, step_months=1, min_train_months=2
            ),
            funding_rate=funding,
            freq="1D",
            min_test_bars=10,
        )
        assert math.isnan(result.factor_max_similarity_to_zoo)

    def test_factor_max_similarity_passes_through_explicit_zero(self) -> None:
        """Caller can supply 0.0 explicitly — that's different from
        not computed (NaN)."""
        features, close, funding = _build_features_and_prices(n=600)
        regime_labels = build_default_labels(close_for_trend=close, funding_rate=funding)
        result = run_backtest(
            factor=_simple_factor(),
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=DataSplits(
                train_end=close.index[550],
                validation_end=close.index[580],
            ),
            cost_model=CostModel(),
            walk_forward_config=WalkForwardConfig(
                train_months=3, test_months=1, step_months=1, min_train_months=2
            ),
            funding_rate=funding,
            factor_max_similarity_to_zoo=0.0,
            freq="1D",
            min_test_bars=10,
        )
        assert result.factor_max_similarity_to_zoo == 0.0
        assert not math.isnan(result.factor_max_similarity_to_zoo)

    def test_factor_id_is_deterministic(self) -> None:
        features, close, funding = _build_features_and_prices(n=600)
        regime_labels = build_default_labels(close_for_trend=close, funding_rate=funding)
        splits = DataSplits(train_end=close.index[550], validation_end=close.index[580])
        wf = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_months=2)
        r1 = run_backtest(
            factor=_simple_factor(),
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=splits,
            cost_model=CostModel(),
            walk_forward_config=wf,
            funding_rate=funding,
            freq="1D",
            min_test_bars=10,
        )
        r2 = run_backtest(
            factor=_simple_factor(),
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=splits,
            cost_model=CostModel(),
            walk_forward_config=wf,
            funding_rate=funding,
            freq="1D",
            min_test_bars=10,
        )
        assert r1.factor_id == r2.factor_id
        assert r1.data_version == r2.data_version
