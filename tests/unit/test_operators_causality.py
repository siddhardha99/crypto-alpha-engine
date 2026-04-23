"""Registry-walking causality canary.

This file serves two roles:

1. **Coverage canary.** Every operator registered in
   :mod:`crypto_alpha_engine.operators.registry` must appear in the
   ``_CAUSALITY_CATALOGUE`` below. If a new operator gets registered
   but the catalogue isn't updated, the canary test fires. This is
   the architectural guarantee from SPEC §13 that no operator slips
   past causality testing.

2. **Automated causality check.** For every entry in the catalogue,
   the generic :func:`assert_causal` helper runs the SPEC §13
   "modify-future-input, past-unchanged" test. Complements (never
   replaces) the per-operator tests in the ``test_operators_*.py``
   files, which cover edge cases this generic check doesn't.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

# Importing the operators package triggers registration.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.operators.causality_check import (
    assert_causal,
    synthetic_series,
)
from crypto_alpha_engine.operators.registry import get_operator, list_operators

# ---------------------------------------------------------------------------
# Catalogue: operator name → input factory
# ---------------------------------------------------------------------------


def _single_series_window(window: int) -> Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]:
    def factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (synthetic_series(n=50),), {"window": window}

    return factory


def _single_series_lag(lag: int) -> Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]:
    def factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (synthetic_series(n=50),), {"lag": lag}

    return factory


def _single_series_halflife(halflife: int) -> Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]:
    def factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (synthetic_series(n=50),), {"halflife": halflife}

    return factory


def _two_series_window(window: int) -> Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]:
    def factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (synthetic_series(n=50, seed=1), synthetic_series(n=50, seed=2)), {"window": window}

    return factory


def _single_series_only() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50),), {}


def _binary_arithmetic() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50, seed=1), synthetic_series(n=50, seed=2)), {}


def _quantile_factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50),), {"window": 5, "q": 0.75}


def _power_factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50).abs() + 0.1,), {"p": 2.0}


def _clip_factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50),), {"lo": 95.0, "hi": 105.0}


def _if_else_factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
    x = synthetic_series(n=50, seed=1)
    cond = x > 100.0
    a = synthetic_series(n=50, seed=2)
    b = synthetic_series(n=50, seed=3)
    return (cond, a, b), {}


def _comparison_scalar() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50),), {"y": 100.0}


def _logical_binary() -> tuple[tuple[Any, ...], dict[str, Any]]:
    a = synthetic_series(n=50, seed=1) > 100.0
    b = synthetic_series(n=50, seed=2) > 50.0
    # _perturb_series_args_in_place mutates pd.Series values; on a bool
    # Series the sentinel coerces to True — fine for a causality check.
    return (a, b), {}


def _logical_unary() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (synthetic_series(n=50) > 100.0,), {}


def _pct_change_series_factory() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """For *_change ops which call pct_change — positional int 'window'."""
    return (synthetic_series(n=50),), {"window": 5}


_CAUSALITY_CATALOGUE: dict[str, Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]] = {
    # ---- timeseries -----------------------------------------------
    "ts_mean": _single_series_window(5),
    "ts_std": _single_series_window(5),
    "ts_min": _single_series_window(5),
    "ts_max": _single_series_window(5),
    "ts_rank": _single_series_window(5),
    "ts_zscore": _single_series_window(5),
    "ts_diff": _single_series_lag(3),
    "ts_pct_change": _single_series_lag(3),
    "ts_skew": _single_series_window(8),
    "ts_kurt": _single_series_window(8),
    "ts_quantile": _quantile_factory,
    "ts_corr": _two_series_window(5),
    "ts_cov": _two_series_window(5),
    "ts_argmax": _single_series_window(5),
    "ts_argmin": _single_series_window(5),
    "ts_decay_linear": _single_series_window(5),
    "ts_ema": _single_series_halflife(4),
    # ---- math -----------------------------------------------------
    "add": _binary_arithmetic,
    "sub": _binary_arithmetic,
    "mul": _binary_arithmetic,
    "div": _binary_arithmetic,
    "log": _single_series_only,  # applied to positive series → well-defined
    "exp": _single_series_only,
    "abs": _single_series_only,
    "sign": _single_series_only,
    "sqrt": _single_series_only,
    "power": _power_factory,
    "tanh": _single_series_only,
    "sigmoid": _single_series_only,
    "clip": _clip_factory,
    # ---- conditional ---------------------------------------------
    "if_else": _if_else_factory,
    "greater_than": _comparison_scalar,
    "less_than": _comparison_scalar,
    "equal": _comparison_scalar,
    "and_": _logical_binary,
    "or_": _logical_binary,
    "not_": _logical_unary,
    # ---- crypto ---------------------------------------------------
    "funding_z": _single_series_window(5),
    "oi_change": _pct_change_series_factory,
    "fear_greed": _single_series_window(5),
    "btc_dominance_change": _pct_change_series_factory,
    "stablecoin_mcap_change": _pct_change_series_factory,
    "active_addresses_change": _pct_change_series_factory,
    "hashrate_change": _pct_change_series_factory,
    "dxy_change": _pct_change_series_factory,
    "spy_correlation": _two_series_window(5),
}


# ---------------------------------------------------------------------------
# Canary: every registered operator must be in the catalogue.
# ---------------------------------------------------------------------------


def test_every_registered_operator_is_catalogued() -> None:
    """Coverage canary. Fails if an operator is registered without a causality entry."""
    registered = set(list_operators())
    catalogued = set(_CAUSALITY_CATALOGUE)
    missing = registered - catalogued
    assert not missing, (
        f"Operators registered but missing from causality catalogue: "
        f"{sorted(missing)}. Add entries to _CAUSALITY_CATALOGUE in "
        f"tests/unit/test_operators_causality.py."
    )
    # Reverse check: no catalogue entries for operators that don't exist.
    extra = catalogued - registered
    assert not extra, (
        f"Causality catalogue has entries for operators not in the registry: " f"{sorted(extra)}"
    )


# ---------------------------------------------------------------------------
# Automated causality sweep — one test per catalogued operator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", sorted(_CAUSALITY_CATALOGUE.keys()))
def test_operator_is_causal(op_name: str) -> None:
    """SPEC §13 causality check, applied programmatically."""
    fn = get_operator(op_name)
    factory = _CAUSALITY_CATALOGUE[op_name]
    assert_causal(fn, inputs_factory=factory, name=op_name)


# ---------------------------------------------------------------------------
# Determinism canary — same input always returns the same output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", sorted(_CAUSALITY_CATALOGUE.keys()))
def test_operator_is_deterministic(op_name: str) -> None:
    """SPEC §13 determinism check."""
    fn = get_operator(op_name)
    args1, kwargs1 = _CAUSALITY_CATALOGUE[op_name]()
    args2, kwargs2 = _CAUSALITY_CATALOGUE[op_name]()
    out1 = fn(*args1, **kwargs1)
    out2 = fn(*args2, **kwargs2)
    pd.testing.assert_series_equal(out1, out2, check_names=False)
