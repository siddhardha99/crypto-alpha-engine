"""The engine entry point — ``run_backtest`` composes every Phase 2-5
primitive into a single sealed :class:`BacktestResult`.

Principle 1 (sealed engine) is enforced by the return type: the only
thing ``run_backtest`` returns is a :class:`BacktestResult`. No
``return_equity_curve=True`` parameter, no bar-level Series accessors,
no trade timestamps. Every internal bar-level series lives inside
private helper scope and is dropped after aggregation.

Principle 2 (causality is sacred) is enforced by two independent
layers:

* **Layer 1 — AST whitelist** (static). Walks the factor AST and
  rejects any operator registered with ``causal_safe=False``. Cheap,
  happens before any simulation runs.
* **Layer 2 — runtime perturbation** (dynamic). Runs the compiled
  factor on perturbed future-feature values at five random cutoffs
  and asserts that past output is byte-identical to baseline. Catches
  composition errors — operators individually causal, but combined
  in a way that leaks future data — that Layer 1 can't see.

Composition sketch
------------------

``run_backtest`` is a thin composer::

    _validate_inputs(...)
    _layer_1_causality_check(factor.root)
    compiled = compile_factor(factor)
    _layer_2_causality_check(compiled, features, factor.root)

    factor_values = compiled(features)
    raw_entries, raw_exits = signal_rule(factor_values)
    entries, exits = _shift_signals_by_one_bar(raw_entries, raw_exits)

    agg, folds = _run_walk_forward(...)
    in_sample_sharpe = _compute_in_sample_sharpe(...)
    regime_breakdown = _compute_regime_breakdown(...)
    dsr = _compute_deflated_sharpe(...)
    data_version = _compute_data_version(...)

    return _assemble_result(...)

Each private helper is testable in isolation — commit 5's evil-factor
end-to-end test inspects the causality layers specifically, which is
only possible because they are named, standalone helpers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from crypto_alpha_engine.backtest.metrics import (
    annualized_return,
    calmar,
    hit_rate,
    ic_ir,
    ic_mean,
    ic_std,
    kurt,
    max_drawdown,
    sharpe,
    skew,
    sortino,
    total_return,
)
from crypto_alpha_engine.backtest.simulation import FoldResult, simulate_fold
from crypto_alpha_engine.backtest.walk_forward import (
    AggregatedFolds,
    FoldSpec,
    aggregate_folds,
    iter_folds,
)
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import ConfigError, LookAheadDetected
from crypto_alpha_engine.factor.ast import factor_id_of, walk
from crypto_alpha_engine.factor.compiler import CompiledFactor, compile_factor
from crypto_alpha_engine.factor.complexity import factor_complexity, unique_features
from crypto_alpha_engine.operators.registry import get_operator_causal_safe
from crypto_alpha_engine.regime.breakdown import breakdown_by_regime
from crypto_alpha_engine.statistics.deflated_sharpe import deflated_sharpe_ratio
from crypto_alpha_engine.types import (
    BacktestResult,
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)

SignalRule = Callable[[pd.Series], tuple[pd.Series, pd.Series]]
"""Protocol for signal-generation rules.

Takes the factor-value Series, returns ``(raw_entries, raw_exits)`` —
both boolean Series, NOT yet shifted. The engine owns the +1-bar
shift; the signal_rule just maps factor values to entry/exit
booleans. See :func:`default_signal_rule` for the default.
"""

_LAYER_2_N_CUTOFFS: int = 5
_LAYER_2_RNG_SEED: int = 42  # deterministic per-run
_IN_SAMPLE_MIN_BARS: int = 24


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_backtest(
    *,
    factor: Factor,
    features: dict[str, pd.Series],
    prices: pd.Series,
    regime_labels: dict[str, pd.Series],
    splits: DataSplits,
    cost_model: CostModel,
    walk_forward_config: WalkForwardConfig,
    funding_rate: pd.Series | None = None,
    signal_rule: SignalRule | None = None,
    feature_source_names: dict[str, str] | None = None,
    ledger_prior_count: int = 1,
    ledger_sharpe_variance: float = 0.0,
    factor_max_similarity_to_zoo: float = float("nan"),
    initial_cash: float = 10_000.0,
    freq: str = "1h",
    min_test_bars: int = 24,
) -> BacktestResult:
    """Run a walk-forward backtest and return a sealed :class:`BacktestResult`.

    This is the one public entry point to the engine. Everything else
    in ``crypto_alpha_engine/backtest/`` is internal.

    Args:
        factor: The :class:`Factor` to backtest. Its AST is checked
            by Layer 1 before any simulation runs.
        features: The features dict the compiled factor consumes.
            Keys are pipe-separated ``"<symbol>|<column>"`` per SPEC §5.1.
        prices: Close-price series of the traded symbol. Separate
            argument because prices are execution prices, not signal
            inputs — lumping them into features requires implicit
            convention on "which key is the execution symbol."
        regime_labels: Caller-supplied dict of regime-label Series.
            Expected keys: ``"trend"``, ``"vol"``, ``"funding"``.
            Missing keys yield NaN Sharpes in the corresponding
            BacktestResult fields. Use ``regime.build_default_labels``
            for the standard SPEC §9 configuration.
        splits: Train/validation/test boundaries. The walk-forward
            loop stays within ``[data_start, splits.train_end)``.
        cost_model: Required. Drives all cost application.
        walk_forward_config: Fold geometry.
        funding_rate: Optional per-bar funding series for perp strategies.
        signal_rule: Optional callable mapping factor_values →
            (entries, exits). Default is sign-based (see
            :func:`default_signal_rule`).
        feature_source_names: Optional per-feature provenance strings.
            Included in ``data_version`` for reproducibility per
            SPEC §5.1. Missing keys default to ``"unknown"``.
        ledger_prior_count: Number of prior factor experiments. Used
            for DSR multiple-testing correction. Default 1 → DSR
            returns NaN (no meaningful correction possible from a
            single trial). Phase 7 wires this to the real ledger.
        ledger_sharpe_variance: Variance of Sharpe across prior
            trials. Default 0 collapses DSR to PSR.
        factor_max_similarity_to_zoo: Maximum cosine similarity to
            the factor zoo. Caller supplies because the engine doesn't
            own the zoo. Default ``NaN`` — syntactically distinct from
            "computed and came out zero."
        initial_cash: Passed to ``simulate_fold``.
        freq: Bar-frequency string passed to vectorbt.
        min_test_bars: Minimum bars required per fold test window.

    Returns:
        :class:`BacktestResult` with every SPEC §8 field populated.

    Raises:
        LookAheadDetected: If the factor trips Layer 1 or Layer 2.
        ConfigError: On input validation failures.
    """
    if signal_rule is None:
        signal_rule = default_signal_rule

    _validate_inputs(features=features, prices=prices, regime_labels=regime_labels)

    # --- Causality layers ---
    _layer_1_causality_check(factor.root)
    compiled = compile_factor(factor)
    _layer_2_causality_check(compiled, features=features, root=factor.root)

    # --- Signal pipeline ---
    factor_values = compiled(features)
    raw_entries, raw_exits = signal_rule(factor_values)
    entries, exits = _shift_signals_by_one_bar(raw_entries, raw_exits)
    # Align signal index to prices.index (may differ if the factor
    # produced NaN-leading output). Reindex + fillna(False) to match.
    entries = entries.reindex(prices.index).fillna(False).astype(bool)
    exits = exits.reindex(prices.index).fillna(False).astype(bool)
    entries.attrs["shifted"] = True
    exits.attrs["shifted"] = True

    # --- Walk-forward simulation ---
    agg, folds = _run_walk_forward(
        entries=entries,
        exits=exits,
        prices=prices,
        funding_rate=funding_rate,
        cost_model=cost_model,
        splits=splits,
        config=walk_forward_config,
        initial_cash=initial_cash,
        freq=freq,
        min_test_bars=min_test_bars,
    )

    # --- In-sample (single sim over pre-WF training region) ---
    in_sample_sharpe = _compute_in_sample_sharpe(
        entries=entries,
        exits=exits,
        prices=prices,
        funding_rate=funding_rate,
        cost_model=cost_model,
        folds=folds,
        initial_cash=initial_cash,
        freq=freq,
    )

    # --- Composed metrics over aggregated OOS returns ---
    net_returns = agg.net_returns
    gross_returns = agg.gross_returns

    # --- Regime breakdown ---
    regime_sharpes = _compute_regime_breakdown(net_returns, regime_labels)

    # --- Deflated Sharpe ---
    dsr = _compute_deflated_sharpe(
        net_returns,
        n_trials=ledger_prior_count,
        sharpe_variance=ledger_sharpe_variance,
    )

    # --- Factor properties ---
    complexity = factor_complexity(factor.root)

    # --- Data provenance ---
    data_version = _compute_data_version(features, feature_source_names)

    # --- Assemble ---
    return _assemble_result(
        factor=factor,
        data_version=data_version,
        net_returns=net_returns,
        gross_returns=gross_returns,
        factor_values=factor_values,
        agg=agg,
        in_sample_sharpe=in_sample_sharpe,
        regime_sharpes=regime_sharpes,
        dsr=dsr,
        complexity=complexity,
        ledger_prior_count=ledger_prior_count,
        factor_max_similarity_to_zoo=factor_max_similarity_to_zoo,
        prices=prices,
        freq=freq,
    )


# ---------------------------------------------------------------------------
# Signal-generation default
# ---------------------------------------------------------------------------


def default_signal_rule(
    factor_values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Default: entry when factor > 0, exit when factor < 0.

    Sign-based — the simplest rule that makes both long and neutral
    states possible from a bipolar factor. NaN factor values produce
    False entries and False exits (no action on warmup bars).

    Callers wanting threshold-based or quantile-based rules supply
    their own :data:`SignalRule` via ``run_backtest``'s
    ``signal_rule=`` parameter.
    """
    entries = (factor_values > 0).fillna(False).astype(bool)
    exits = (factor_values < 0).fillna(False).astype(bool)
    return entries, exits


# ---------------------------------------------------------------------------
# Helpers — each independently testable
# ---------------------------------------------------------------------------


def _validate_inputs(
    *,
    features: dict[str, pd.Series],
    prices: pd.Series,
    regime_labels: dict[str, pd.Series],
) -> None:
    """Cheap sanity checks before any expensive step."""
    if not features:
        raise ConfigError("run_backtest: features dict is empty")
    if len(prices) == 0:
        raise ConfigError("run_backtest: prices series is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ConfigError("run_backtest: prices.index must be a DatetimeIndex")
    if prices.index.tz is None:
        raise ConfigError("run_backtest: prices.index must be UTC-aware")


def _layer_1_causality_check(root: FactorNode) -> None:
    """Walk AST; raise if any operator has ``causal_safe=False``.

    Static check. Runs before compilation so a violating factor never
    reaches the simulation layer.
    """
    for node in walk(root):
        if not get_operator_causal_safe(node.operator):
            raise LookAheadDetected(
                f"Layer 1 causality check: operator {node.operator!r} is "
                f"registered with causal_safe=False. Cannot use in a "
                f"backtest. If this is a research-purpose acausal operator, "
                f"separate its evaluation from the sealed engine."
            )


def _layer_2_causality_check(
    compiled: CompiledFactor,
    *,
    features: dict[str, pd.Series],
    root: FactorNode,
    n_cutoffs: int = _LAYER_2_N_CUTOFFS,
) -> None:
    """Runtime perturbation: assert past output unchanged under future perturbation.

    Perturbs only the feature keys the factor reads (via
    ``unique_features(root)``), not every feature in the dict — an
    unread feature's perturbation couldn't possibly affect output, so
    including it would just slow the check.

    ``n_cutoffs`` random cutoffs in the middle 60% of the data range.
    Fixed RNG seed for reproducibility within a run (so the same
    factor produces the same check output every time).
    """
    reads = unique_features(root)
    perturb_keys = reads & set(features.keys())
    if not perturb_keys:
        return  # factor reads no features — nothing to perturb

    baseline = compiled(features)
    n = len(baseline)
    if n < 2 * n_cutoffs:
        return  # not enough data to pick meaningful cutoffs

    rng = np.random.default_rng(_LAYER_2_RNG_SEED)
    low = int(n * 0.2)
    high = int(n * 0.8)
    if high <= low:
        return
    cutoffs = rng.integers(low=low, high=high, size=n_cutoffs)

    for cut in cutoffs:
        ts = baseline.index[int(cut)]
        perturbed = dict(features)
        for k in perturb_keys:
            s = features[k].copy()
            s.loc[s.index >= ts] = 9999.0  # absurd future value
            perturbed[k] = s

        pert_out = compiled(perturbed)
        base_past = baseline.loc[baseline.index < ts]
        pert_past = pert_out.loc[pert_out.index < ts]

        # NaN-safe equality: fillna with a sentinel, compare
        sentinel = -1.2345678e99
        base_filled = base_past.fillna(sentinel)
        pert_filled = pert_past.fillna(sentinel)
        if not base_filled.equals(pert_filled):
            raise LookAheadDetected(
                f"Layer 2 causality check failed at cutoff {ts}. "
                f"Perturbing features at indices >= {ts} changed factor "
                f"output at indices < {ts}. Every operator individually "
                f"declared causal_safe=True, but their composition leaks "
                f"future data. Investigate the factor AST for a shift "
                f"with a negative window or an off-by-one error."
            )


def _shift_signals_by_one_bar(
    raw_entries: pd.Series,
    raw_exits: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Apply the +1-bar execution-delay shift and stamp the ``attrs`` marker.

    Signals observed at bar t execute at bar t+1 close. After the
    shift, the first bar's entry/exit becomes NaN (pandas convention);
    we fillna(False) because "no signal" is the correct interpretation.
    """
    entries = raw_entries.shift(1).fillna(False).astype(bool)
    exits = raw_exits.shift(1).fillna(False).astype(bool)
    entries.attrs["shifted"] = True
    exits.attrs["shifted"] = True
    return entries, exits


def _run_walk_forward(
    *,
    entries: pd.Series,
    exits: pd.Series,
    prices: pd.Series,
    funding_rate: pd.Series | None,
    cost_model: CostModel,
    splits: DataSplits,
    config: WalkForwardConfig,
    initial_cash: float,
    freq: str,
    min_test_bars: int,
) -> tuple[AggregatedFolds, list[FoldSpec]]:
    """Loop iter_folds + simulate_fold, then aggregate_folds."""
    assert isinstance(prices.index, pd.DatetimeIndex)  # validated upstream
    folds = list(
        iter_folds(
            data_index=prices.index,
            splits=splits,
            config=config,
            min_test_bars=min_test_bars,
        )
    )
    if not folds:
        raise ConfigError(
            "run_backtest: no folds generated — data too short for "
            "walk_forward_config (min_train_months + test_months may "
            "exceed available data before splits.train_end)"
        )

    fold_results: list[FoldResult] = []
    for f in folds:
        e = entries.loc[f.test_start : f.test_end]
        x = exits.loc[f.test_start : f.test_end]
        c = prices.loc[f.test_start : f.test_end]
        fr_slice = funding_rate.loc[f.test_start : f.test_end] if funding_rate is not None else None
        # Defensive re-stamp after slicing.
        e.attrs["shifted"] = True
        x.attrs["shifted"] = True
        fold_results.append(
            simulate_fold(
                entries=e,
                exits=x,
                close=c,
                cost_model=cost_model,
                funding_rate=fr_slice,
                initial_cash=initial_cash,
                freq=freq,
            )
        )

    return aggregate_folds(fold_results), folds


def _compute_in_sample_sharpe(
    *,
    entries: pd.Series,
    exits: pd.Series,
    prices: pd.Series,
    funding_rate: pd.Series | None,
    cost_model: CostModel,
    folds: list[FoldSpec],
    initial_cash: float,
    freq: str,
) -> float:
    """Single sim over ``[data_start, first_test_start)``.

    For a static factor (Phase 6), this is the Sharpe achieved on data
    that came *before* any OOS window. Not a "fit" Sharpe — more a
    historical Sharpe for the factor's parameters.
    """
    if not folds:
        return float("nan")
    in_sample_end = folds[0].test_start
    data_start = prices.index[0]
    if data_start >= in_sample_end:
        return float("nan")

    e = entries.loc[data_start:in_sample_end]
    x = exits.loc[data_start:in_sample_end]
    c = prices.loc[data_start:in_sample_end]
    if len(c) < _IN_SAMPLE_MIN_BARS:
        return float("nan")
    fr = funding_rate.loc[data_start:in_sample_end] if funding_rate is not None else None
    e.attrs["shifted"] = True
    x.attrs["shifted"] = True
    result = simulate_fold(
        entries=e,
        exits=x,
        close=c,
        cost_model=cost_model,
        funding_rate=fr,
        initial_cash=initial_cash,
        freq=freq,
    )
    return sharpe(result.net_returns, periods_per_year=_periods_for_freq(freq))


def _compute_regime_breakdown(
    net_returns: pd.Series,
    regime_labels: dict[str, pd.Series],
) -> dict[str, float]:
    """Map regime breakdowns into the flat BacktestResult field names.

    Returns a dict with all nine regime-Sharpe field names. Missing
    dimensions or regime labels produce NaN — the engine never invents
    a number for an unobserved regime.
    """
    out: dict[str, float] = {}
    dimensions: list[tuple[str, dict[str, str]]] = [
        (
            "trend",
            {"bull": "bull_sharpe", "bear": "bear_sharpe", "crab": "crab_sharpe"},
        ),
        (
            "vol",
            {
                "high_vol": "high_vol_sharpe",
                "low_vol": "low_vol_sharpe",
                "normal_vol": "normal_vol_sharpe",
            },
        ),
        (
            "funding",
            {
                "euphoric": "euphoric_funding_sharpe",
                "fearful": "fearful_funding_sharpe",
                "neutral": "neutral_funding_sharpe",
            },
        ),
    ]
    for dim_key, label_to_field in dimensions:
        if dim_key not in regime_labels:
            for field_name in label_to_field.values():
                out[field_name] = float("nan")
            continue
        per_regime = breakdown_by_regime(net_returns, regime_labels[dim_key], sharpe)
        for label_key, field_name in label_to_field.items():
            out[field_name] = per_regime.get(label_key, float("nan"))
    return out


def _compute_deflated_sharpe(
    net_returns: pd.Series,
    *,
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """Deflated Sharpe on aggregated OOS returns.

    With defaults (n_trials=1, sharpe_variance=0), DSR returns NaN.
    Phase 7 will thread the real ledger values in.
    """
    arr = net_returns.dropna().to_numpy()
    if len(arr) < 4:
        return float("nan")
    if n_trials < 1:
        return float("nan")
    # DSR rejects n_trials=1 — the no-multi-testing degenerate case.
    # Surface as NaN rather than patching; phase 6 caller default
    # (n_trials=1) means "ledger not wired, DSR N/A."
    if n_trials < 2:
        return float("nan")
    return deflated_sharpe_ratio(
        observed_sharpe=sharpe(net_returns),
        n_trials=n_trials,
        returns_skew=skew(net_returns),
        returns_kurt=kurt(net_returns) + 3.0,  # raw from excess
        n_observations=len(arr),
        sharpe_variance_across_trials=sharpe_variance,
    )


def _compute_data_version(
    features: dict[str, pd.Series],
    feature_source_names: dict[str, str] | None,
) -> str:
    """Deterministic hash over sorted ``(key, feature_hash, source_name)``.

    Byte-stable for byte-stable inputs: same features + source names
    produce the same string across runs, across processes, across
    machines. Phase 7's ledger keys on this for reproducibility — a
    silent drift here would corrupt the ledger.
    """
    sources = feature_source_names or {}
    parts: list[str] = []
    for key in sorted(features):
        series = features[key]
        h = hashlib.sha256()
        # Values as float64 bytes — byte-stable.
        h.update(np.ascontiguousarray(series.to_numpy(dtype=np.float64)).tobytes())
        # Index as int64 ns since epoch — byte-stable for both tz-aware
        # and tz-naive DatetimeIndex. `.to_numpy()` on tz-aware returns
        # object array with identity-based bytes (non-deterministic);
        # `.asi8` / int64 view is stable.
        idx = series.index
        if isinstance(idx, pd.DatetimeIndex):
            # asi8 is the canonical int64 ns-since-epoch view. Access
            # via getattr to satisfy strict typing (pandas-stubs doesn't
            # expose .asi8 on the public type).
            ns_view = getattr(idx, "asi8")  # noqa: B009
            h.update(np.ascontiguousarray(ns_view).tobytes())
        else:
            h.update(np.ascontiguousarray(idx.to_numpy()).tobytes())
        feature_hash = h.hexdigest()[:16]
        source = sources.get(key, "unknown")
        parts.append(f"{key}|{feature_hash}|{source}")
    blob = "\n".join(parts).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return f"sha256:{digest[:32]}"


def _periods_for_freq(freq: str) -> float:
    """Bar-frequency string → periods_per_year."""
    table = {"1h": 8760.0, "4h": 2190.0, "1D": 365.0, "1d": 365.0, "1w": 52.0}
    return table.get(freq, 8760.0)


def _assemble_result(
    *,
    factor: Factor,
    data_version: str,
    net_returns: pd.Series,
    gross_returns: pd.Series,
    factor_values: pd.Series,
    agg: AggregatedFolds,
    in_sample_sharpe: float,
    regime_sharpes: dict[str, float],
    dsr: float,
    complexity: dict[str, float | int],
    ledger_prior_count: int,
    factor_max_similarity_to_zoo: float,
    prices: pd.Series,
    freq: str,
) -> BacktestResult:
    """Populate every :class:`BacktestResult` field. Gaps here surface
    as missing values downstream — every field must get a real
    assignment, no placeholders."""
    ppy = _periods_for_freq(freq)

    # IC metrics: factor_values vs forward returns of the prices series.
    fwd_returns = prices.pct_change().shift(-1)  # one-bar-ahead returns
    # Note: this shift is inside IC computation only, NOT in the
    # signal path. fwd_returns[t] is the realized return from t to t+1,
    # which is the correct alignment for IC on factor[t].

    # Cost-adjusted Sharpes on aggregated OOS returns.
    headline_sharpe = sharpe(net_returns, periods_per_year=ppy)
    gross_sharpe_val = sharpe(gross_returns, periods_per_year=ppy)
    headline_sortino = sortino(net_returns, periods_per_year=ppy)
    headline_calmar = calmar(net_returns, periods_per_year=ppy)
    headline_mdd = max_drawdown(net_returns)
    headline_ar = annualized_return(net_returns, periods_per_year=ppy)
    headline_tr = total_return(net_returns)
    headline_hit = hit_rate(net_returns)
    headline_ic_mean = ic_mean(factor_values, fwd_returns)
    headline_ic_std = ic_std(factor_values, fwd_returns)
    headline_ic_ir = ic_ir(factor_values, fwd_returns)

    wf_mean = float(np.mean(agg.per_fold_sharpes)) if agg.per_fold_sharpes else float("nan")
    wf_std = (
        float(np.std(agg.per_fold_sharpes, ddof=1))
        if len(agg.per_fold_sharpes) > 1
        else float("nan")
    )

    turnover_annual = agg.total_turnover * (ppy / max(len(net_returns), 1))

    return BacktestResult(
        # Identity
        factor_id=factor_id_of(factor),
        factor_name=factor.name,
        run_timestamp=datetime.now(UTC),
        data_version=data_version,
        # Headline
        sharpe=headline_sharpe,
        sortino=headline_sortino,
        calmar=headline_calmar,
        max_drawdown=headline_mdd,
        annualized_return=headline_ar,
        total_return=headline_tr,
        # Predictive power
        ic_mean=headline_ic_mean,
        ic_std=headline_ic_std,
        ic_ir=headline_ic_ir,
        hit_rate=headline_hit,
        # Cost breakdown
        gross_sharpe=gross_sharpe_val,
        net_sharpe=headline_sharpe,
        turnover_annual=float(turnover_annual),
        total_fees_paid=agg.total_fees_paid,
        total_slippage_paid=agg.total_slippage_paid,
        # Regime breakdown
        bull_sharpe=regime_sharpes["bull_sharpe"],
        bear_sharpe=regime_sharpes["bear_sharpe"],
        crab_sharpe=regime_sharpes["crab_sharpe"],
        high_vol_sharpe=regime_sharpes["high_vol_sharpe"],
        low_vol_sharpe=regime_sharpes["low_vol_sharpe"],
        normal_vol_sharpe=regime_sharpes["normal_vol_sharpe"],
        euphoric_funding_sharpe=regime_sharpes["euphoric_funding_sharpe"],
        fearful_funding_sharpe=regime_sharpes["fearful_funding_sharpe"],
        neutral_funding_sharpe=regime_sharpes["neutral_funding_sharpe"],
        # Robustness
        walk_forward_sharpe_mean=wf_mean,
        walk_forward_sharpe_std=wf_std,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=headline_sharpe,
        deflated_sharpe_ratio=dsr,
        n_experiments_in_ledger=ledger_prior_count,
        complexity_scalar=float(complexity["scalar"]),
        # Factor properties
        factor_ast_depth=int(complexity["ast_depth"]),
        factor_node_count=int(complexity["node_count"]),
        factor_max_similarity_to_zoo=factor_max_similarity_to_zoo,
        # Trade statistics
        n_trades=agg.n_trades_total,
        avg_trade_duration_hours=agg.avg_trade_duration_hours,
        avg_position_size=agg.avg_position_size,
        max_leverage_used=agg.max_leverage_used,
    )


# ---------------------------------------------------------------------------
# Internal: unused import guard
# ---------------------------------------------------------------------------

# Suppress unused-import warning for `replace` — kept in the toolbelt
# for future BacktestResult field mutation in tests without touching
# the engine.
_ = replace
