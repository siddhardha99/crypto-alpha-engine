"""Contract tests for the math operators.

Math operators are element-wise and trivially causal (no rolling
window, no lag). The interesting properties to verify are domain-edge
behavior (div-by-zero → NaN, log of non-positive → NaN) and
determinism.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_engine.operators.math import (
    abs_,
    add,
    clip,
    div,
    exp,
    log,
    mul,
    power,
    sigmoid,
    sign,
    sqrt,
    sub,
    tanh,
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float, name="x")


class TestBinaryArithmetic:
    def test_add_sub_mul(self) -> None:
        a = _s([1.0, 2.0, 3.0])
        b = _s([10.0, 20.0, 30.0])
        assert add(a, b).tolist() == [11.0, 22.0, 33.0]
        assert sub(a, b).tolist() == [-9.0, -18.0, -27.0]
        assert mul(a, b).tolist() == [10.0, 40.0, 90.0]

    def test_div_by_zero_is_nan(self) -> None:
        a = _s([1.0, 2.0, 3.0])
        b = _s([1.0, 0.0, 0.5])
        out = div(a, b)
        assert out.iloc[0] == 1.0
        assert np.isnan(out.iloc[1])  # 2/0 → NaN, not inf
        assert out.iloc[2] == 6.0

    def test_binary_with_scalar(self) -> None:
        a = _s([1.0, 2.0, 3.0])
        assert mul(a, 2.0).tolist() == [2.0, 4.0, 6.0]


class TestUnaryTransforms:
    def test_log_non_positive_is_nan(self) -> None:
        x = _s([1.0, 0.0, -1.0, np.e])
        out = log(x)
        assert out.iloc[0] == 0.0
        assert np.isnan(out.iloc[1])  # log(0) → NaN (not -inf)
        assert np.isnan(out.iloc[2])  # log(-1) → NaN
        assert out.iloc[3] == pytest.approx(1.0)

    def test_exp_roundtrip_with_log(self) -> None:
        x = _s([0.5, 1.0, 2.0])
        pd.testing.assert_series_equal(log(exp(x)), x, check_names=False)

    def test_abs(self) -> None:
        assert abs_(_s([-1.0, 2.0, -3.0])).tolist() == [1.0, 2.0, 3.0]

    def test_sign(self) -> None:
        assert sign(_s([-5.0, 0.0, 5.0])).tolist() == [-1.0, 0.0, 1.0]

    def test_sqrt_negative_is_nan(self) -> None:
        out = sqrt(_s([4.0, 0.0, -1.0]))
        assert out.iloc[0] == 2.0
        assert out.iloc[1] == 0.0
        assert np.isnan(out.iloc[2])

    def test_power(self) -> None:
        out = power(_s([2.0, 3.0, 4.0]), 2)
        assert out.tolist() == [4.0, 9.0, 16.0]

    def test_tanh_saturates(self) -> None:
        out = tanh(_s([-100.0, 0.0, 100.0]))
        assert out.iloc[0] == pytest.approx(-1.0)
        assert out.iloc[1] == 0.0
        assert out.iloc[2] == pytest.approx(1.0)

    def test_sigmoid_range(self) -> None:
        """Sigmoid is in [0, 1]; large magnitudes saturate to exactly 0/1 in float64."""
        out = sigmoid(_s([-100.0, 0.0, 100.0]))
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()
        assert out.iloc[1] == 0.5


class TestClip:
    def test_clip_bounds(self) -> None:
        out = clip(_s([-5.0, 0.0, 5.0, 10.0]), lo=0.0, hi=5.0)
        assert out.tolist() == [0.0, 0.0, 5.0, 5.0]

    def test_clip_raises_on_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="lo"):
            clip(_s([1.0]), lo=5.0, hi=3.0)


class TestDeterminism:
    """SPEC §13: every operator is deterministic."""

    @pytest.mark.parametrize(
        ("fn", "args"),
        [
            (add, (_s([1.0, 2.0]), _s([3.0, 4.0]))),
            (sub, (_s([1.0, 2.0]), _s([3.0, 4.0]))),
            (mul, (_s([1.0, 2.0]), _s([3.0, 4.0]))),
            (div, (_s([1.0, 2.0]), _s([3.0, 4.0]))),
            (log, (_s([1.0, 2.0, 3.0]),)),
            (exp, (_s([0.1, 0.2, 0.3]),)),
            (abs_, (_s([-1.0, 2.0]),)),
            (sign, (_s([-1.0, 0.0, 1.0]),)),
            (sqrt, (_s([1.0, 4.0]),)),
            (tanh, (_s([0.1, 0.5]),)),
            (sigmoid, (_s([0.1, 0.5]),)),
        ],
    )
    def test_same_input_same_output(self, fn: Any, args: tuple[Any, ...]) -> None:
        pd.testing.assert_series_equal(fn(*args), fn(*args))
