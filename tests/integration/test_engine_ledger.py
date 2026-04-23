"""Engine ↔ ledger integration tests (commit 3 of Phase 7).

These tests drive ``run_backtest`` with and without a ``Ledger``
argument and verify:

* First run against an empty ledger writes successfully.
* Second identical run raises :class:`DuplicateFactor` when
  ``on_duplicate="raise"``.
* Second identical run returns the prior :class:`BacktestResult`
  when ``on_duplicate="skip"``.
* ``skip_duplicate_check=True`` bypasses the check entirely — both
  runs write.
* ``DuplicateCheckSaturated`` fires regardless of ``on_duplicate``
  (saturation is caller-attention-required).
* DSR inputs pulled from ledger when provided; explicit kwargs
  ignored in that case.
* ``factor_max_similarity_to_zoo`` auto-populated from the
  duplicate check's structural score when ledger + not skipped;
  NaN otherwise.
* Evil factors (from commit 5 of Phase 6) still trip causality
  BEFORE any ledger interaction — the ledger never sees a cheating
  run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.backtest.engine import run_backtest
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import (
    DuplicateCheckSaturated,
    DuplicateFactor,
    LookAheadDetected,
)
from crypto_alpha_engine.ledger.ledger import Ledger
from crypto_alpha_engine.operators import registry as op_registry
from crypto_alpha_engine.regime import build_default_labels
from crypto_alpha_engine.types import (
    BacktestResult,
    CostModel,
    Factor,
    FactorNode,
    WalkForwardConfig,
)


def _long_synthetic_series(n_days: int = 1200, *, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="1D", tz="UTC")
    returns = rng.normal(0.0005, 0.025, size=n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx, name="close")


def _momentum_factor(window: int = 20, name: str = "momentum") -> Factor:
    """Simple ts_mean factor for the integration tests."""
    return Factor(
        name=f"{name}_{window}",
        description=f"{window}-day moving average of close",
        hypothesis="Prices mean-revert around the MA",
        root=FactorNode(operator="ts_mean", args=("BTC/USD|close", window), kwargs={}),
    )


def _run(
    factor: Factor,
    close: pd.Series,
    *,
    ledger: Ledger | None = None,
    on_duplicate: Literal["raise", "skip"] = "raise",
    skip_duplicate_check: bool = False,
    ledger_prior_count: int = 1,
    ledger_sharpe_variance: float = 0.0,
) -> BacktestResult:
    """Driver wrapping run_backtest with the common E2E config."""
    funding = pd.Series(
        np.random.default_rng(0).normal(0.0001, 0.0001, size=len(close)),
        index=close.index,
    )
    labels = build_default_labels(close_for_trend=close, funding_rate=funding)
    return run_backtest(
        factor=factor,
        features={"BTC/USD|close": close},
        prices=close,
        regime_labels=labels,
        funding_rate=funding,
        splits=DataSplits(
            train_end=close.index[1100],
            validation_end=close.index[1150],
        ),
        cost_model=CostModel(),
        walk_forward_config=WalkForwardConfig(
            train_months=3, test_months=1, step_months=1, min_train_months=2
        ),
        freq="1D",
        min_test_bars=10,
        ledger=ledger,
        on_duplicate=on_duplicate,
        skip_duplicate_check=skip_duplicate_check,
        ledger_prior_count=ledger_prior_count,
        ledger_sharpe_variance=ledger_sharpe_variance,
    )


# ---------------------------------------------------------------------------
# First-write / duplicate-detection flow
# ---------------------------------------------------------------------------


class TestFirstWriteAndDuplicateDetection:
    def test_first_run_writes_to_ledger(self, tmp_path: Path) -> None:
        """Empty ledger, novel factor → run writes one entry."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        result = _run(_momentum_factor(), close, ledger=ledger)
        assert ledger.count_experiments() == 1
        (entry,) = list(ledger.read_all())
        assert entry.result.factor_id == result.factor_id

    def test_second_identical_run_raises_duplicate_on_raise(self, tmp_path: Path) -> None:
        """Second run with same factor + ledger + on_duplicate='raise'
        triggers DuplicateFactor. Ledger still has only 1 entry."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        _run(_momentum_factor(), close, ledger=ledger)

        with pytest.raises(DuplicateFactor, match="structural"):
            _run(_momentum_factor(), close, ledger=ledger, on_duplicate="raise")
        assert ledger.count_experiments() == 1

    def test_second_identical_run_returns_prior_result_on_skip(self, tmp_path: Path) -> None:
        """on_duplicate='skip' returns the prior BacktestResult. Ledger
        still has only 1 entry — skip does NOT rewrite."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        first = _run(_momentum_factor(), close, ledger=ledger)
        second = _run(_momentum_factor(), close, ledger=ledger, on_duplicate="skip")
        # Prior result returned verbatim — same factor_id, same data_version.
        assert second.factor_id == first.factor_id
        assert second.data_version == first.data_version
        assert ledger.count_experiments() == 1


# ---------------------------------------------------------------------------
# skip_duplicate_check bypass
# ---------------------------------------------------------------------------


class TestSkipDuplicateCheck:
    def test_skip_duplicate_check_allows_repeat_writes(self, tmp_path: Path) -> None:
        """With skip_duplicate_check=True, the same factor can be
        written twice. Intended for reproducibility workflows that
        deliberately re-run a prior factor."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        _run(_momentum_factor(), close, ledger=ledger, skip_duplicate_check=True)
        _run(_momentum_factor(), close, ledger=ledger, skip_duplicate_check=True)
        assert ledger.count_experiments() == 2


# ---------------------------------------------------------------------------
# Saturation — always raises, regardless of on_duplicate
# ---------------------------------------------------------------------------


class TestCapSaturation:
    def _populate_saturating_ledger(
        self, tmp_path: Path, close: pd.Series, *, n_priors: int = 25
    ) -> Ledger:
        """Write n_priors structurally-identical ts_mean factors with
        different windows, all against DIFFERENT feature keys so
        behavioral similarity stays low."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        # First populate by running n_priors distinct ts_mean factors
        # with different windows so they're structurally similar but
        # produce uncorrelated outputs.
        for i in range(n_priors):
            factor = _momentum_factor(window=5 + i * 3, name=f"spam_{i}")
            _run(factor, close, ledger=ledger, skip_duplicate_check=True)
        return ledger

    def test_saturation_raises_on_raise(self, tmp_path: Path) -> None:
        # run_backtest doesn't expose hard_cap, so we saturate the
        # default cap (20) with 25 structurally-identical ts_mean priors.
        close_big = _long_synthetic_series(n_days=1500)
        ledger_big = Ledger(tmp_path / "big.jsonl")
        for i in range(25):
            f = _momentum_factor(window=5 + i, name=f"prior_{i}")
            _run(f, close_big, ledger=ledger_big, skip_duplicate_check=True)
        # A candidate structurally identical but with a novel window
        # will hit all 25 priors; default cap=20 → saturation.
        candidate = _momentum_factor(window=200, name="novel")
        with pytest.raises(DuplicateCheckSaturated, match="hard cap"):
            _run(candidate, close_big, ledger=ledger_big, on_duplicate="raise")

    def test_saturation_also_raises_on_skip(self, tmp_path: Path) -> None:
        """on_duplicate='skip' doesn't rescue saturation — we don't
        know what to skip to."""
        close_big = _long_synthetic_series(n_days=1500)
        ledger_big = Ledger(tmp_path / "big.jsonl")
        for i in range(25):
            f = _momentum_factor(window=5 + i, name=f"prior_{i}")
            _run(f, close_big, ledger=ledger_big, skip_duplicate_check=True)
        candidate = _momentum_factor(window=200, name="novel")
        with pytest.raises(DuplicateCheckSaturated):
            _run(candidate, close_big, ledger=ledger_big, on_duplicate="skip")


# ---------------------------------------------------------------------------
# DSR inputs auto-wiring
# ---------------------------------------------------------------------------


class TestDSRWiring:
    def test_dsr_inputs_come_from_ledger_when_provided(self, tmp_path: Path) -> None:
        """run_backtest's ledger_prior_count/ledger_sharpe_variance
        kwargs are ignored when a ledger is supplied. The result's
        n_experiments_in_ledger reflects the ledger's count, not the
        kwarg."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        # Seed the ledger with 3 entries via skip_duplicate_check.
        for window in (10, 15, 25):
            _run(
                _momentum_factor(window=window, name=f"seed_{window}"),
                close,
                ledger=ledger,
                skip_duplicate_check=True,
            )
        # Now run a factor STRUCTURALLY DIFFERENT from the seeds so it
        # isn't rejected as a duplicate (all seeds are ts_mean; use log).
        novel = Factor(
            name="log_factor",
            description="",
            hypothesis="",
            root=FactorNode(operator="log", args=("BTC/USD|close",), kwargs={}),
        )
        result = _run(
            novel,
            close,
            ledger=ledger,
            ledger_prior_count=999,  # should be overridden
            ledger_sharpe_variance=999.0,  # should be overridden
        )
        # The ledger's finite count is <= literal count = 3.
        assert result.n_experiments_in_ledger <= 3
        assert result.n_experiments_in_ledger != 999

    def test_dsr_inputs_from_kwargs_when_no_ledger(self, tmp_path: Path) -> None:
        """Without a ledger, the explicit kwargs are used."""
        close = _long_synthetic_series()
        result = _run(
            _momentum_factor(),
            close,
            ledger=None,
            ledger_prior_count=42,
            ledger_sharpe_variance=0.5,
        )
        assert result.n_experiments_in_ledger == 42


# ---------------------------------------------------------------------------
# factor_max_similarity_to_zoo wiring
# ---------------------------------------------------------------------------


class TestSimilarityFieldWiring:
    def test_similarity_nan_when_no_ledger(self, tmp_path: Path) -> None:
        close = _long_synthetic_series()
        result = _run(_momentum_factor(), close, ledger=None)
        assert math.isnan(result.factor_max_similarity_to_zoo)

    def test_similarity_nan_when_skip_duplicate_check(self, tmp_path: Path) -> None:
        """skip_duplicate_check=True must leave factor_max_similarity
        at its default NaN, even with a ledger present."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        _run(
            _momentum_factor(name="seed"),
            close,
            ledger=ledger,
            skip_duplicate_check=True,
        )
        result = _run(
            _momentum_factor(name="second", window=30),
            close,
            ledger=ledger,
            skip_duplicate_check=True,
        )
        assert math.isnan(result.factor_max_similarity_to_zoo)

    def test_similarity_populated_from_ledger_when_check_runs(self, tmp_path: Path) -> None:
        """With a ledger and no skip, factor_max_similarity_to_zoo is
        the duplicate check's structural score — even when the
        candidate is novel (no match)."""
        ledger = Ledger(tmp_path / "ledger.jsonl")
        close = _long_synthetic_series()
        # Seed with a ts_mean factor.
        _run(
            _momentum_factor(window=20, name="seed"),
            close,
            ledger=ledger,
            skip_duplicate_check=True,
        )
        # Novel factor using a different operator → structural < 1 but > 0.
        novel = Factor(
            name="log_thing",
            description="",
            hypothesis="",
            root=FactorNode(operator="log", args=("BTC/USD|close",), kwargs={}),
        )
        result = _run(novel, close, ledger=ledger, on_duplicate="raise")
        # Must be a real number (NOT NaN) because a check ran.
        assert not math.isnan(result.factor_max_similarity_to_zoo)
        # Novel factor → some similarity but less than 1.0 (it's not a
        # duplicate of ts_mean-based seed).
        assert 0.0 <= result.factor_max_similarity_to_zoo < 1.0


# ---------------------------------------------------------------------------
# Causality BEFORE ledger — evil factors never reach the ledger
# ---------------------------------------------------------------------------


class TestCausalityBeforeLedger:
    def test_evil_factor_fails_before_ledger_interaction(self, tmp_path: Path) -> None:
        """A factor that trips Layer 1 must raise BEFORE any write to
        the ledger. The ledger stays empty."""
        snapshot = op_registry._snapshot_for_tests()
        try:

            @op_registry.register_operator(
                "evil_unsafe_field", arg_types=("series",), causal_safe=False
            )
            def _evil(x: pd.Series) -> pd.Series:
                return x.shift(-1)

            factor = Factor(
                name="evil",
                description="evil",
                hypothesis="ledger must never see this",
                root=FactorNode(operator="evil_unsafe_field", args=("BTC/USD|close",)),
            )
            close = _long_synthetic_series()
            ledger = Ledger(tmp_path / "ledger.jsonl")
            with pytest.raises(LookAheadDetected):
                _run(factor, close, ledger=ledger)
            # Ledger file either never created or empty — no corrupt entry.
            assert ledger.count_experiments() == 0
        finally:
            op_registry._restore_for_tests(snapshot)
