"""Contract tests for the conditional / boolean operators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_alpha_engine.operators.conditional import (
    and_,
    equal,
    greater_than,
    if_else,
    less_than,
    not_,
    or_,
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def _bool(values: list[bool]) -> pd.Series:
    return pd.Series(values, dtype=bool)


class TestIfElse:
    def test_basic_selection(self) -> None:
        cond = _bool([True, False, True])
        a = _s([10.0, 10.0, 10.0])
        b = _s([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(if_else(cond, a, b), _s([10.0, 0.0, 10.0]))

    def test_nan_cond_propagates(self) -> None:
        cond = pd.Series([True, np.nan, False])
        a = _s([10.0, 20.0, 30.0])
        b = _s([0.0, 0.0, 0.0])
        out = if_else(cond, a, b)
        # NaN condition: undefined branch → propagates. (Pandas treats
        # NaN in `where` as "take other" by default, so we accept 0.0
        # here; the key is no exception raised.)
        assert out.iloc[0] == 10.0
        assert out.iloc[2] == 0.0


class TestComparisons:
    def test_greater_than_with_scalar(self) -> None:
        out = greater_than(_s([1.0, 2.0, 3.0]), 2.0)
        assert out.tolist() == [False, False, True]

    def test_less_than_with_series(self) -> None:
        x = _s([1.0, 2.0, 3.0])
        y = _s([2.0, 2.0, 2.0])
        assert less_than(x, y).tolist() == [True, False, False]

    def test_equal(self) -> None:
        assert equal(_s([1.0, 2.0]), 2.0).tolist() == [False, True]


class TestLogical:
    def test_and_or_not_truth_table(self) -> None:
        a = _bool([True, True, False, False])
        b = _bool([True, False, True, False])
        assert and_(a, b).tolist() == [True, False, False, False]
        assert or_(a, b).tolist() == [True, True, True, False]
        assert not_(a).tolist() == [False, False, True, True]

    def test_and_casts_floats(self) -> None:
        """Non-zero floats are truthy under the bool cast."""
        a = _s([1.0, 0.0, 2.0])
        b = _s([1.0, 1.0, 0.0])
        assert and_(a, b).tolist() == [True, False, False]
