"""End-to-end validation of the sealed backtest engine.

Three suites:

1. **Evil-factor canaries** — capstone validation that the two-layer
   causality architecture works at the ``run_backtest`` public-API
   level (not just on helpers called directly). Three variants:
   explicit acausal declaration (Layer 1), lying causal_safe=True
   with a ``.shift(-1)`` kernel (Layer 2), and a composition-style
   leak where the acausal behavior is buried in a non-shift kernel
   operation nested inside a safe outer operator (Layer 2).

2. **Real-fixture integration** — loads the Phase-2 committed
   parquets (``btc_usd_1h_coinbase_spot``, ``btc_funding_8h_bitmex_perp``),
   compiles a real factor, and runs the full ``simulate_fold``
   pipeline on actual 2022 market data. Proves Phases 2-5 compose
   on real bytes, not just on numpy.random.

3. **Synthetic long-horizon end-to-end** — a multi-year synthetic
   series sized for the month-based WalkForwardConfig. Runs the
   complete ``run_backtest`` pipeline (walk-forward + aggregation +
   assembly) and sanity-checks the result.

Fixture scope note
------------------

The committed Phase-2 fixtures are deliberate 1-month snippets
(January 2022, 744 bars) to keep repo size small. The month-based
WalkForwardConfig (SPEC §8) needs at least
``(min_train_months + test_months)`` months of data, so a full
walk-forward against the real fixture isn't feasible. The integration
test runs a single ``simulate_fold`` against real data; the
walk-forward path is exercised against a long synthetic series in
the third suite. A multi-year real fixture for end-to-end WF would
be a Phase 7 addition.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.engine import run_backtest
from crypto_alpha_engine.backtest.simulation import simulate_fold
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import LookAheadDetected
from crypto_alpha_engine.factor.compiler import compile_factor
from crypto_alpha_engine.operators import registry as op_registry
from crypto_alpha_engine.regime import build_default_labels
from crypto_alpha_engine.types import (
    BacktestResult,
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_btc_close() -> pd.Series:
    """Load the committed 1h BTC close series from Phase 2 fixtures."""
    df = pd.read_parquet(FIXTURES / "btc_usd_1h_coinbase_spot.parquet")
    df = df.set_index("timestamp")
    return df["close"].astype(float)


def _load_btc_funding() -> pd.Series:
    """Load the 8h BitMEX funding-rate series (Phase 2)."""
    df = pd.read_parquet(FIXTURES / "btc_funding_8h_bitmex_perp.parquet")
    df = df.set_index("timestamp")
    return df["funding_rate"].astype(float)


def _long_synthetic_series(n_days: int, *, seed: int = 7) -> pd.Series:
    """3+ years of daily BTC-like prices — enough for a multi-fold WF."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="1D", tz="UTC")
    returns = rng.normal(0.0005, 0.025, size=n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx, name="close")


def _synthetic_funding(close: pd.Series, *, seed: int = 8) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0001, 0.0001, size=len(close)), index=close.index)


# ---------------------------------------------------------------------------
# Suite 1: Evil-factor canaries through the public run_backtest API
# ---------------------------------------------------------------------------


def _run_evil_factor(factor: Factor, close: pd.Series) -> BacktestResult:
    """Shared driver: wraps run_backtest with a common config so each
    evil-factor test focuses only on the factor construction and the
    expected exception."""
    funding = _synthetic_funding(close)
    return run_backtest(
        factor=factor,
        features={"BTC/USD|close": close},
        prices=close,
        regime_labels=build_default_labels(close_for_trend=close, funding_rate=funding),
        funding_rate=funding,
        splits=DataSplits(
            train_end=close.index[int(len(close) * 0.9)],
            validation_end=close.index[int(len(close) * 0.95)],
        ),
        cost_model=CostModel(),
        walk_forward_config=WalkForwardConfig(
            train_months=3, test_months=1, step_months=1, min_train_months=2
        ),
        freq="1D",
        min_test_bars=10,
    )


class TestEvilFactorVariants:
    def test_variant_a_explicit_unsafe_operator_trips_layer_1(self) -> None:
        """Variant A: the operator is honestly declared
        ``causal_safe=False``. Layer 1 rejects the factor before any
        simulation — the shortest path through the two-layer gate.
        Must go through the public ``run_backtest`` API, not just
        ``_layer_1_causality_check``, to prove the public call wires
        Layer 1 in."""
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator(
                "evil_explicit_unsafe", arg_types=("series",), causal_safe=False
            )
            def _evil(x: pd.Series) -> pd.Series:
                return x.shift(-1)  # blatant lookahead

            factor = Factor(
                name="evil_explicit",
                description="evil",
                hypothesis="this should never reach the engine",
                root=FactorNode(operator="evil_explicit_unsafe", args=("BTC/USD|close",)),
            )
            close = _long_synthetic_series(400)
            with pytest.raises(LookAheadDetected, match="causal_safe"):
                _run_evil_factor(factor, close)
        finally:
            op_registry._restore_for_tests(snapshot)

    def test_variant_b_lying_causal_safe_shift_minus_one_trips_layer_2(
        self,
    ) -> None:
        """Variant B: the operator LIES — registered with
        ``causal_safe=True`` but its kernel does ``.shift(-1)``. Layer 1
        passes it (the annotation is True), so Layer 2's runtime
        perturbation must catch the lie. This is the core justification
        for having two layers."""
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator(
                "evil_lying_shift", arg_types=("series",), causal_safe=True
            )
            def _evil(x: pd.Series) -> pd.Series:
                # Looks ahead by one bar — lying declaration.
                return x.shift(-1)

            factor = Factor(
                name="evil_lying_shift",
                description="evil",
                hypothesis="Layer 2 must catch this",
                root=FactorNode(operator="evil_lying_shift", args=("BTC/USD|close",)),
            )
            close = _long_synthetic_series(400)
            with pytest.raises(LookAheadDetected, match="Layer 2"):
                _run_evil_factor(factor, close)
        finally:
            op_registry._restore_for_tests(snapshot)

    def test_variant_c_composition_leak_via_centered_window(self) -> None:
        """Variant C: composition-style leak. The leak is buried inside
        a rolling-mean operator using ``center=True`` (symmetric window,
        peeks into the future). Both the inner and the outer operator
        are registered with ``causal_safe=True``. The factor tree nests
        them: ``ts_zscore(centered_mean(close, 10), 20)``.

        Why this is a composition bug, not just a single-op lie: a
        casual code review of ``centered_mean`` might not spot the
        ``center=True`` subtlety (no ``.shift(-N)`` is used), and the
        outer ``ts_zscore`` would still claim causal_safe=True honestly.
        The combined factor nonetheless leaks — Layer 2's runtime
        perturbation is the only line of defense.
        """
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator(
                "centered_mean_leaky",
                arg_types=("series", "int"),
                causal_safe=True,  # lying
            )
            def _leaky(x: pd.Series, n: int) -> pd.Series:
                # center=True uses future bars symmetrically — acausal,
                # but registration claims otherwise.
                return x.rolling(window=n, center=True).mean()

            # Nest the leaky op inside a safe one (ts_zscore is a real,
            # genuinely-causal operator from Phase 3). If Layer 2 didn't
            # walk the full tree this composition would slip through.
            inner = FactorNode(
                operator="centered_mean_leaky",
                args=("BTC/USD|close", 10),
            )
            root = FactorNode(operator="ts_zscore", args=(inner, 20), kwargs={})
            factor = Factor(
                name="evil_composition",
                description="Layer 2's composition-style canary",
                hypothesis="Composed leakage should be caught even when "
                "individual operators look innocent",
                root=root,
            )
            close = _long_synthetic_series(400)
            with pytest.raises(LookAheadDetected, match="Layer 2"):
                _run_evil_factor(factor, close)
        finally:
            op_registry._restore_for_tests(snapshot)


# ---------------------------------------------------------------------------
# Suite 2: Real-fixture integration — Phase-2 parquets exercised through sim
# ---------------------------------------------------------------------------


class TestRealFixtureIntegration:
    def test_simulate_fold_on_real_btc_hourly_fixture(self) -> None:
        """Real 2022 January BTC close + real factor compilation +
        real simulate_fold. Scope: prove Phases 2-5 compose on actual
        market bytes, not just synthetic data."""
        close = _load_btc_close()
        features = {"BTC/USD|close": close}

        # ts_zscore(ts_pct_change(close, 24), 72) — a real factor from
        # the user's spec. Uses 72-hour z-score of 24h returns.
        inner = FactorNode(operator="ts_pct_change", args=("BTC/USD|close", 24), kwargs={})
        root = FactorNode(operator="ts_zscore", args=(inner, 72), kwargs={})
        factor = Factor(
            name="momentum_z",
            description="72h z-score of 24h pct-change — momentum-style",
            hypothesis="Recent returns predict near-term returns",
            root=root,
        )
        compiled = compile_factor(factor)
        values = compiled(features)
        assert len(values) == len(close)

        # Signal generation + shift applied manually (no run_backtest
        # here because the 1-month fixture is too short for WF).
        entries = (values > 0).fillna(False).astype(bool).shift(1)
        exits = (values < 0).fillna(False).astype(bool).shift(1)
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)
        entries.attrs["shifted"] = True
        exits.attrs["shifted"] = True

        result = simulate_fold(
            entries=entries,
            exits=exits,
            close=close,
            cost_model=CostModel(),
            initial_cash=10_000.0,
            freq="1h",
        )
        # Basic sanity — we got back a populated FoldResult.
        assert result.n_trades >= 0
        assert math.isfinite(result.fees_paid)
        assert len(result.net_returns) == len(close)

    def test_real_funding_loads_and_aligns(self) -> None:
        """Real funding fixture loads and covers the expected 2022-Jan window.
        Wiring sanity: the Phase-2 loader still produces a UTC-aware,
        float series consumable downstream."""
        funding = _load_btc_funding()
        assert isinstance(funding.index, pd.DatetimeIndex)
        assert funding.index.tz is not None
        assert funding.dtype.kind == "f"
        assert len(funding) > 0


# ---------------------------------------------------------------------------
# Suite 3: Synthetic long-horizon end-to-end
# ---------------------------------------------------------------------------


class TestLongHorizonEndToEnd:
    def test_full_pipeline_on_synthetic_multi_year_data(self) -> None:
        """The proof that all six phases compose into a working engine.

        Uses 3+ years of daily synthetic data (sized so the month-based
        WalkForwardConfig actually yields multiple folds). Runs the
        full ``run_backtest`` pipeline and checks the result passes
        basic sanity identities.
        """
        close = _long_synthetic_series(n_days=1200)  # 3.3 years
        funding = _synthetic_funding(close)
        features = {"BTC/USD|close": close}
        regime_labels = build_default_labels(close_for_trend=close, funding_rate=funding)

        # Same factor shape as the user's example in the commit-5 ask.
        inner = FactorNode(operator="ts_pct_change", args=("BTC/USD|close", 24), kwargs={})
        root = FactorNode(operator="ts_zscore", args=(inner, 72), kwargs={})
        factor = Factor(
            name="momentum_z",
            description="72d z-score of 24d pct-change",
            hypothesis="Recent-return momentum predicts near-term returns",
            root=root,
        )

        result = run_backtest(
            factor=factor,
            features=features,
            prices=close,
            regime_labels=regime_labels,
            splits=DataSplits(
                train_end=close.index[1100],
                validation_end=close.index[1150],
            ),
            cost_model=CostModel(),  # SPEC §8 defaults
            walk_forward_config=WalkForwardConfig(),  # SPEC §8 defaults
            funding_rate=funding,
            feature_source_names={"BTC/USD|close": "coinbase_spot"},
            freq="1D",
            min_test_bars=20,
        )

        # --- Returned without error; right type ---
        assert isinstance(result, BacktestResult)

        # --- Every field populated (not None) ---
        for fn in BacktestResult.__dataclass_fields__:
            assert getattr(result, fn) is not None, f"{fn} is None"

        # --- Sharpe is finite (either sign acceptable on synthetic noise) ---
        assert math.isfinite(result.sharpe), f"sharpe = {result.sharpe}"

        # --- At least one trade ---
        assert result.n_trades > 0, "expected some trades on 3-year synthetic data"

        # --- Costs paid (Principle 5 enforced in practice) ---
        assert result.total_fees_paid > 0, "fees must be nonzero if trades happened"
        assert result.total_slippage_paid > 0

        # --- gross_sharpe >= net_sharpe (costs can only hurt) ---
        # Use a small tolerance for floating-point noise on near-equal values.
        assert result.gross_sharpe >= result.net_sharpe - 1e-9, (
            f"gross ({result.gross_sharpe}) < net ({result.net_sharpe}); "
            f"costs should never improve Sharpe"
        )

        # --- Identity fields byte-stable ---
        assert result.factor_id.startswith("f_")
        assert result.data_version.startswith("sha256:")
        assert result.run_timestamp.tzinfo is not None
