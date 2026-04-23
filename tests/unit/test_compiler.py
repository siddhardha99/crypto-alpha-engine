"""Compiler tests.

Covers:

1. Happy paths — flat op, nested, crypto alias, math on series + scalar.
2. The **memoisation verification test** — the architectural claim
   that subtree evaluation is cached within one call has a concrete
   runtime check: wrap ``ts_mean`` with a call counter and assert it
   runs once, not twice, when referenced from two sibling positions.
3. Arg-resolution dispatch — series vs series_or_scalar vs scalar,
   string vs sub-FactorNode.
4. Error paths — unknown operator, missing feature, arity mismatch.
5. Fixture-based integration — compile + run on real Phase-2 parquets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Importing the operators package triggers registration.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.compiler import CompiledFactor, compile_factor
from crypto_alpha_engine.factor.parser import parse_string
from crypto_alpha_engine.operators.timeseries import ts_mean
from crypto_alpha_engine.types import Factor, FactorNode


def _sample(n: int = 60, *, seed: int = 1, mean: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(mean, 1.0, size=n), index=idx, name="x")


# ---------------------------------------------------------------------------
# 1. Happy paths
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_simple_op_with_feature_lookup(self) -> None:
        node = parse_string('ts_mean("BTC/USD|close", 5)')
        compiled = compile_factor(node)
        features = {"BTC/USD|close": _sample()}
        out = compiled(features)
        pd.testing.assert_series_equal(out, ts_mean(features["BTC/USD|close"], 5))

    def test_nested_factor_via_subnode_recursion(self) -> None:
        node = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        compiled = compile_factor(node)
        features = {"x|close": _sample(n=60)}
        out = compiled(features)
        # Expected: ts_zscore of the ts_mean output.
        from crypto_alpha_engine.operators.timeseries import ts_mean, ts_zscore

        expected = ts_zscore(ts_mean(features["x|close"], 5), 10)
        pd.testing.assert_series_equal(out, expected)

    def test_crypto_alias_compiles_and_matches_ts_equivalent(self) -> None:
        """funding_z ≡ ts_zscore on the same series + window."""
        funding_node = parse_string('funding_z("BTC/USDT:USDT|funding_rate", 10)')
        zscore_node = parse_string('ts_zscore("BTC/USDT:USDT|funding_rate", 10)')
        features = {"BTC/USDT:USDT|funding_rate": _sample(n=60, mean=0.0001)}
        fz = compile_factor(funding_node)(features)
        tz = compile_factor(zscore_node)(features)
        pd.testing.assert_series_equal(fz, tz)

    def test_series_or_scalar_accepts_scalar(self) -> None:
        """add(x, 5) — scalar in a series_or_scalar position passes through."""
        node = parse_string('add("x|close", 5)')
        compiled = compile_factor(node)
        features = {"x|close": _sample(n=10)}
        out = compiled(features)
        pd.testing.assert_series_equal(out, features["x|close"] + 5)

    def test_series_or_scalar_accepts_series(self) -> None:
        """add(x, y) — two series in series_or_scalar positions."""
        node = parse_string('add("x|close", "y|close")')
        compiled = compile_factor(node)
        features = {"x|close": _sample(n=10, seed=1), "y|close": _sample(n=10, seed=2)}
        out = compiled(features)
        pd.testing.assert_series_equal(out, features["x|close"] + features["y|close"])

    def test_accepts_Factor_wrapper(self) -> None:
        node = parse_string('ts_mean("x|close", 5)')
        factor = Factor(name="f", description="d", hypothesis="h", root=node)
        compiled = compile_factor(factor)
        assert compiled.factor_id == compile_factor(node).factor_id


# ---------------------------------------------------------------------------
# 2. Memoisation verification — the architectural claim has a runtime check
# ---------------------------------------------------------------------------


class TestMemoisation:
    def test_shared_subtree_evaluates_once_not_twice(self) -> None:
        """Per docs/factor_design.md §3: identical sub-trees are cached.

        Factor: ``add(ts_mean("x|close", 20), ts_mean("x|close", 20))``
        — the ``ts_mean`` sub-expression appears twice. The compiler
        must call the kernel exactly once and reuse the result.
        """
        calls: list[tuple[str, int]] = []

        def counting_ts_mean(series: pd.Series, window: int) -> pd.Series:
            calls.append(("ts_mean", window))
            result: pd.Series = ts_mean(series, window)
            return result

        node = parse_string('add(ts_mean("x|close", 20), ts_mean("x|close", 20))')
        compiled = compile_factor(
            node,
            operator_override={"ts_mean": counting_ts_mean},
        )
        features = {"x|close": _sample(n=60)}
        out = compiled(features)

        # The architectural claim: ts_mean called exactly once.
        assert calls == [("ts_mean", 20)], (
            f"subtree memoisation broken: ts_mean called {len(calls)} times, "
            f"expected exactly 1. calls={calls}"
        )
        # And the result is correct: add(X, X) = 2*X.
        expected_inner = ts_mean(features["x|close"], 20)
        pd.testing.assert_series_equal(out, expected_inner + expected_inner)

    def test_different_subtrees_both_evaluate(self) -> None:
        """Sanity check: different sub-trees should each evaluate once."""
        calls: list[tuple[str, int]] = []

        def counting_ts_mean(series: pd.Series, window: int) -> pd.Series:
            calls.append(("ts_mean", window))
            result: pd.Series = ts_mean(series, window)
            return result

        node = parse_string('add(ts_mean("x|close", 10), ts_mean("x|close", 20))')
        compiled = compile_factor(
            node,
            operator_override={"ts_mean": counting_ts_mean},
        )
        compiled({"x|close": _sample(n=60)})
        # Two different windows → two distinct sub-trees → two calls.
        assert len(calls) == 2
        assert {c[1] for c in calls} == {10, 20}

    def test_cache_scope_is_per_call_not_global(self) -> None:
        """A second call with the same CompiledFactor runs the kernel again
        (no across-call caching — fresh evaluation per call)."""
        calls: list[tuple[str, int]] = []

        def counting_ts_mean(series: pd.Series, window: int) -> pd.Series:
            calls.append(("ts_mean", window))
            result: pd.Series = ts_mean(series, window)
            return result

        node = parse_string('ts_mean("x|close", 20)')
        compiled = compile_factor(
            node,
            operator_override={"ts_mean": counting_ts_mean},
        )
        features = {"x|close": _sample(n=60)}
        compiled(features)
        compiled(features)
        assert len(calls) == 2  # once per call


# ---------------------------------------------------------------------------
# 3. Arg-resolution dispatch
# ---------------------------------------------------------------------------


class TestArgResolution:
    def test_series_arg_with_subnode(self) -> None:
        """A series position accepts a FactorNode (sub-expression)."""
        node = parse_string('ts_mean(ts_mean("x|close", 5), 10)')
        compiled = compile_factor(node)
        out = compiled({"x|close": _sample(n=60)})
        assert isinstance(out, pd.Series)

    def test_series_arg_with_string(self) -> None:
        node = parse_string('ts_mean("x|close", 5)')
        compiled = compile_factor(node)
        out = compiled({"x|close": _sample(n=30)})
        assert isinstance(out, pd.Series)

    def test_scalar_position_receives_literal(self) -> None:
        """int/float/bool positions receive their literal unchanged."""
        node = parse_string('ts_quantile("x|close", 10, 0.5)')
        compiled = compile_factor(node)
        out = compiled({"x|close": _sample(n=30)})
        assert isinstance(out, pd.Series)


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_unknown_operator_at_compile_time(self) -> None:
        # Can't go through parse_string (it rejects unknown ops); build
        # directly.
        bad = FactorNode(operator="made_up_op", args=("x|close", 20))
        with pytest.raises(ConfigError, match="unknown operator"):
            compile_factor(bad)

    def test_arity_mismatch_at_compile_time(self) -> None:
        bad = FactorNode(operator="ts_mean", args=("x|close",))  # missing window
        with pytest.raises(ConfigError, match="expected 2 positional args"):
            compile_factor(bad)

    def test_missing_feature_at_evaluate_time(self) -> None:
        node = parse_string('ts_mean("absent|close", 5)')
        compiled = compile_factor(node)
        with pytest.raises(ConfigError, match="not found in features dict"):
            compiled({"x|close": _sample()})

    def test_kernel_raises_on_bad_window(self) -> None:
        """A window-0 node is parseable (parser validates types, not values);
        the compiler lets the kernel raise at call time."""
        bad = FactorNode(operator="ts_mean", args=("x|close", 0))
        compiled = compile_factor(bad)
        with pytest.raises(ConfigError, match="window must be a positive int"):
            compiled({"x|close": _sample()})


# ---------------------------------------------------------------------------
# 5. Real-data integration on Phase-2 fixtures
# ---------------------------------------------------------------------------


FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestOnFixtures:
    def test_funding_z_on_real_bitmex_funding(self) -> None:
        df = pd.read_parquet(FIXTURES / "btc_funding_8h_bitmex_perp.parquet")
        funding = df.set_index("timestamp")["funding_rate"]
        features = {"BTC/USDT:USDT|funding_rate": funding}

        node = parse_string('funding_z("BTC/USDT:USDT|funding_rate", 10)')
        out = compile_factor(node)(features)
        assert len(out) == len(funding)
        assert out.iloc[:9].isna().all()  # warmup
        assert not out.iloc[9:].isna().any()

    def test_nested_factor_on_btc_close(self) -> None:
        df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
        close = df.set_index("timestamp")["close"]
        features = {"BTC/USD|close": close}

        # z-score of 24-hour return
        node = parse_string('ts_zscore(ts_pct_change("BTC/USD|close", 24), 72)')
        out = compile_factor(node)(features)
        assert len(out) == len(close)
        # 24 + 72 = 96 warmup bars; some leeway for pandas' rolling ddof.
        assert out.iloc[:95].isna().all()
        assert out.abs().max() < 20.0  # sanity bound

    def test_reused_subtree_on_real_data(self) -> None:
        """Memoisation still works when operating on real fixtures."""
        df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
        close = df.set_index("timestamp")["close"]

        calls: list[int] = []

        def counting_ts_mean(series: pd.Series, window: int) -> pd.Series:
            calls.append(window)
            result: pd.Series = ts_mean(series, window)
            return result

        node = parse_string('sub(ts_mean("BTC/USD|close", 24), ts_mean("BTC/USD|close", 24))')
        out = compile_factor(
            node,
            operator_override={"ts_mean": counting_ts_mean},
        )({"BTC/USD|close": close})

        assert len(calls) == 1
        # sub(X, X) = 0, except for NaN warmup.
        assert (out.dropna() == 0.0).all()


# ---------------------------------------------------------------------------
# 6. CompiledFactor is idiomatic
# ---------------------------------------------------------------------------


class TestCompiledFactorSurface:
    def test_has_factor_id_and_root(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        compiled = compile_factor(node)
        assert compiled.factor_id.startswith("f_")
        assert compiled.root is node

    def test_compiled_factor_is_reusable(self) -> None:
        node = parse_string('ts_mean("x|close", 5)')
        compiled = compile_factor(node)
        for _ in range(3):
            out = compiled({"x|close": _sample()})
            assert isinstance(out, pd.Series)

    def test_compiled_factor_is_runtime_type(self) -> None:
        node = parse_string('ts_mean("x|close", 5)')
        assert isinstance(compile_factor(node), CompiledFactor)
