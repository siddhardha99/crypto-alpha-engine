"""Programmatic causality verifier — the canary against operator drift.

The rule (SPEC §6, Principle 2): for any operator ``op`` and input
series ``x``, modifying ``x`` at indices ``>= k`` must NOT change any
output at indices ``< k``. This module ships the primitive that checks
it.

Per-operator tests written by hand (``tests/unit/test_operators_*.py``)
give specific error messages and cover edge cases. This module
provides the generic check used by the registry-walking canary in
``tests/unit/test_operators_causality.py``, which fires if a new
operator is registered without an entry in the causality test
catalogue.

Design choice: the check is in the library module (not the test
module) so downstream projects that add their own operators can run
the same check against their custom kernels — the extensibility use
case from SPEC §5.1 and ``docs/adding_custom_sources.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def synthetic_series(
    n: int = 50,
    *,
    seed: int = 42,
    name: str = "x",
) -> pd.Series:
    """Deterministic UTC-indexed series for causality / determinism checks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(100.0, 1.5, size=n), index=idx, name=name)


def assert_causal(
    fn: Callable[..., pd.Series],
    *,
    inputs_factory: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]],
    cutoff: int = 25,
    name: str | None = None,
) -> None:
    """Assert that ``fn`` doesn't look ahead.

    Calls ``fn`` twice: once with the full synthetic input, once with
    the input perturbed from ``cutoff`` forward. Every output at an
    index ``< cutoff`` must be identical between the two runs.

    Args:
        fn: Operator kernel to test.
        inputs_factory: Zero-arg callable returning ``(args, kwargs)``
            for ``fn``. Returning a factory (not a static value) lets
            us mutate the args for the perturbation pass without
            affecting the clean pass.
        cutoff: Index past which the perturbation is applied. Defaults
            to roughly mid-series.
        name: Optional label used in the assertion error message.

    Raises:
        AssertionError: If any output at ``< cutoff`` differs between
            the clean and perturbed runs.
    """
    clean_args, clean_kwargs = inputs_factory()
    clean_out = fn(*clean_args, **clean_kwargs)

    perturbed_args, perturbed_kwargs = inputs_factory()
    _perturb_series_args_in_place(perturbed_args, cutoff=cutoff)
    perturbed_out = fn(*perturbed_args, **perturbed_kwargs)

    label = name or getattr(fn, "__name__", "operator")
    pd.testing.assert_series_equal(
        clean_out.iloc[:cutoff],
        perturbed_out.iloc[:cutoff],
        check_names=False,
        obj=f"{label} (causality check)",
    )


def _perturb_series_args_in_place(args: tuple[Any, ...], *, cutoff: int) -> None:
    """Set values at ``>= cutoff`` in every Series arg to an obvious sentinel.

    Uses a dtype-appropriate sentinel: ``999.0`` for numerics, flipped
    value for bool. This way the check works uniformly across
    comparison / logical / arithmetic operators.
    """
    for a in args:
        if not isinstance(a, pd.Series):
            continue
        if a.dtype == bool:
            a.iloc[cutoff:] = ~a.iloc[cutoff:]
        else:
            a.iloc[cutoff:] = 999.0
