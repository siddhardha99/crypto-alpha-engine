"""Tests for the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

Coverage:

1. Output is in ``[0, 1]`` (it's a probability).
2. Monotonicity in ``n_trials`` — DSR decreases as trial count grows,
   holding other inputs fixed (SPEC §13).
3. Hand-computed reference value for one canonical input set.
4. Degenerate cases return NaN (n_trials=1; negative denom on
   extreme skew/kurt; NaN variance).
5. ConfigError on programmer errors (negative variance, bad counts).
"""

from __future__ import annotations

import math

import pytest

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.statistics.deflated_sharpe import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


class TestOutputRange:
    def test_dsr_is_probability(self) -> None:
        """DSR is Φ(z), so it's in [0, 1]."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=100,
            returns_skew=-0.2,
            returns_kurt=4.0,
            n_observations=2000,
            sharpe_variance_across_trials=0.25,
        )
        assert 0.0 <= dsr <= 1.0

    def test_large_observed_sharpe_pushes_dsr_high(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=5.0,
            n_trials=100,
            returns_skew=0.0,
            returns_kurt=3.0,
            n_observations=2000,
            sharpe_variance_across_trials=0.25,
        )
        assert dsr > 0.99


# ---------------------------------------------------------------------------
# Monotonicity in n_trials (the headline SPEC §13 property)
# ---------------------------------------------------------------------------


class TestMonotonicInNTrials:
    def test_dsr_decreases_as_n_trials_increases(self) -> None:
        """Holding the observed Sharpe and other inputs fixed, more trials
        means more multiple-testing penalty, which means a lower DSR."""
        kwargs = {
            "observed_sharpe": 1.5,
            "returns_skew": 0.0,
            "returns_kurt": 3.0,
            "n_observations": 2000,
            "sharpe_variance_across_trials": 0.25,
        }
        dsr_low = deflated_sharpe_ratio(n_trials=10, **kwargs)
        dsr_mid = deflated_sharpe_ratio(n_trials=100, **kwargs)
        dsr_high = deflated_sharpe_ratio(n_trials=10_000, **kwargs)
        assert dsr_low > dsr_mid > dsr_high

    def test_dsr_unchanged_when_sr_variance_is_zero(self) -> None:
        """If the variance of Sharpe across trials is zero, the
        multiple-testing penalty collapses: SR₀ = 0 regardless of N."""
        base_kwargs = {
            "observed_sharpe": 1.5,
            "returns_skew": 0.0,
            "returns_kurt": 3.0,
            "n_observations": 2000,
            "sharpe_variance_across_trials": 0.0,
        }
        dsr_small = deflated_sharpe_ratio(n_trials=10, **base_kwargs)
        dsr_big = deflated_sharpe_ratio(n_trials=10_000, **base_kwargs)
        assert dsr_small == pytest.approx(dsr_big, rel=1e-12)


# ---------------------------------------------------------------------------
# Hand-computed reference value
# ---------------------------------------------------------------------------


class TestReferenceValue:
    def test_matches_hand_computation(self) -> None:
        """Spot-check against manual arithmetic for one set of inputs.

        Inputs: SR=1.5, N=100, skew=0, kurt=3, T=2000, V[SR]=0.25.

        SR₀ = √0.25 · [(1 - EM) · Φ⁻¹(0.99) + EM · Φ⁻¹(1 - 1/(100·e))]
            = 0.5 · [0.4228 · 2.3264 + 0.5772 · 2.1868]     (approx)
            ≈ 0.5 · [0.9836 + 1.2622]
            ≈ 0.5 · 2.2458
            ≈ 1.1229

        σ²(SR) = 1 - 0 + ((3 - 1)/4) · 1.5²
               = 1 + 0.5 · 2.25
               = 2.125

        z = (1.5 - 1.1229) · √1999 / √2.125
          = 0.3771 · 44.7102 / 1.4577
          ≈ 11.566

        Φ(11.566) ≈ 1.0 to float precision.
        """
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=100,
            returns_skew=0.0,
            returns_kurt=3.0,
            n_observations=2000,
            sharpe_variance_across_trials=0.25,
        )
        # With 2000 observations and SR=1.5 well above SR₀, Φ saturates.
        assert dsr > 0.999

    def test_euler_mascheroni_constant_value(self) -> None:
        """The module's EM constant is pinned to the standard value."""
        assert EULER_MASCHERONI == pytest.approx(0.5772156649, rel=1e-9)


# ---------------------------------------------------------------------------
# Degenerate cases → NaN (documented in docstring)
# ---------------------------------------------------------------------------


class TestDegenerateCases:
    def test_n_trials_one_returns_nan(self) -> None:
        """Φ⁻¹(1 - 1/1) = -∞; formula is undefined."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=1,
            returns_skew=0.0,
            returns_kurt=3.0,
            n_observations=2000,
            sharpe_variance_across_trials=0.25,
        )
        assert math.isnan(dsr)

    def test_extreme_kurt_negative_denom_returns_nan(self) -> None:
        """Very high kurt on a high Sharpe drives σ²(SR) positive
        comfortably; but an extreme *negative* skew with a very high
        Sharpe can drive the denominator down. We construct a case
        where denom_inner <= 0 and assert NaN."""
        # Pick skew, kurt, SR so that 1 - γ₃·SR + (γ₄-1)/4·SR² ≤ 0.
        # E.g. γ₃ = 10, SR = 1, (γ₄-1)/4·SR² = 0.05 → 1 - 10 + 0.05 < 0.
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=10,
            returns_skew=10.0,
            returns_kurt=1.2,
            n_observations=100,
            sharpe_variance_across_trials=0.1,
        )
        assert math.isnan(dsr)

    def test_nan_variance_returns_nan(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=100,
            returns_skew=0.0,
            returns_kurt=3.0,
            n_observations=2000,
            sharpe_variance_across_trials=float("nan"),
        )
        assert math.isnan(dsr)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_n_trials_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_trials"):
            deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=-1,
                returns_skew=0.0,
                returns_kurt=3.0,
                n_observations=100,
                sharpe_variance_across_trials=0.1,
            )

    def test_zero_n_trials_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_trials"):
            deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=0,
                returns_skew=0.0,
                returns_kurt=3.0,
                n_observations=100,
                sharpe_variance_across_trials=0.1,
            )

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_observations"):
            deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=10,
                returns_skew=0.0,
                returns_kurt=3.0,
                n_observations=1,
                sharpe_variance_across_trials=0.1,
            )

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ConfigError, match="variance"):
            deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=10,
                returns_skew=0.0,
                returns_kurt=3.0,
                n_observations=100,
                sharpe_variance_across_trials=-0.1,
            )
