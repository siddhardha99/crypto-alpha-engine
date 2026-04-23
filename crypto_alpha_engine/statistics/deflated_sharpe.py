"""Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014).

Reference:
    Bailey, D. H., & Lopez de Prado, M. (2014).
    *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Non-Normality.*
    Journal of Portfolio Management, 40 (5), 94-107.

The DSR is a probability in [0, 1]. It answers: "Given that we searched
over N candidate strategies with some cross-trial Sharpe variance, how
likely is the observed Sharpe to be genuinely significant rather than
the best-of-N outlier?" DSR > 0.95 is the SPEC §10 publication
threshold.

Formula
-------

Let:
    SR            = observed annualised Sharpe
    V[SR]         = variance of Sharpe across prior trials
    N             = number of prior trials (ledger count)
    T             = number of observations backing the Sharpe
    γ₃            = returns skewness
    γ₄            = returns *raw* (Pearson) kurtosis — 3 for a normal
                    distribution. Callers passing excess kurt must add 3.
    EM            = Euler-Mascheroni constant ≈ 0.5772

Expected maximum Sharpe under the null of zero true Sharpe:

    SR₀ = √V[SR] · [ (1 - EM) · Φ⁻¹(1 - 1/N)
                     + EM       · Φ⁻¹(1 - 1/(N·e)) ]

Non-normality-adjusted standard error:

    σ(SR)² = 1 - γ₃·SR + ((γ₄ - 1)/4) · SR²

Deflated Sharpe Ratio:

    DSR = Φ( (SR - SR₀) · √(T - 1) / √σ(SR)² )

Returns Φ(z) ∈ [0, 1].

Kurtosis convention
-------------------

BLP's γ₄ is the raw fourth standardised moment (Pearson kurtosis):
3 for a normal distribution. Python's pandas `.kurt()` and
`scipy.stats.kurtosis(..., fisher=True)` both return *excess*
kurtosis: 0 for a normal distribution. If a caller has excess kurt
on hand, they must add 3 before passing it to
:func:`deflated_sharpe_ratio`.

Phase 5 scope
-------------

This module ships the closed-form DSR. The ``n_trials`` and
``sharpe_variance_across_trials`` inputs come from the experiment
ledger (Phase 7). Phase 5 tests cover the math against hand-computed
reference values + the monotonicity property SPEC §13 requires.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from crypto_alpha_engine.exceptions import ConfigError

EULER_MASCHERONI: float = 0.5772156649015329


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    returns_skew: float,
    returns_kurt: float,
    n_observations: int,
    sharpe_variance_across_trials: float,
) -> float:
    """Compute the Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    Args:
        observed_sharpe: The factor's measured annualised Sharpe.
        n_trials: Number of prior strategies considered (from the
            experiment ledger). Must be ≥ 1. For ``n_trials == 1``,
            ``Φ⁻¹(0)`` is ``-inf`` and the formula degenerates; the
            function returns ``nan`` in that case. Supply ``n_trials
            ≥ 2`` for meaningful output.
        returns_skew: Third standardised moment of the returns series.
        returns_kurt: *Raw* (Pearson) fourth standardised moment;
            **3 for a normal distribution**. If you have excess
            kurtosis (0 for normal), add 3 before passing it in.
        n_observations: Number of return observations backing
            ``observed_sharpe``. Must be ≥ 2.
        sharpe_variance_across_trials: ``V[SR]`` — variance of Sharpe
            ratios observed across the prior trials. Must be ≥ 0.

    Returns:
        Deflated Sharpe Ratio ∈ [0, 1]. Returns ``nan`` when the
        formula is degenerate (``n_trials == 1``; the non-normality
        denominator σ²(SR) ≤ 0; ``sharpe_variance_across_trials`` is
        NaN).

    Raises:
        ConfigError: On programmer errors — ``n_trials < 1``,
            ``n_observations < 2``, or a negative variance.

    Example:
        >>> dsr = deflated_sharpe_ratio(
        ...     observed_sharpe=1.5,
        ...     n_trials=100,
        ...     returns_skew=-0.2,
        ...     returns_kurt=4.0,               # raw (Pearson), not excess
        ...     n_observations=2000,
        ...     sharpe_variance_across_trials=0.25,
        ... )
        >>> 0.0 <= dsr <= 1.0
        True
    """
    _validate(
        n_trials=n_trials,
        n_observations=n_observations,
        sharpe_variance_across_trials=sharpe_variance_across_trials,
    )

    if n_trials == 1:
        # Φ⁻¹(1 - 1/1) = Φ⁻¹(0) = -inf; SR₀ degenerates. With only one
        # trial there's no multiple-testing bias to correct — but BLP's
        # formula can't express that; we return nan.
        return float("nan")

    if math.isnan(sharpe_variance_across_trials):
        return float("nan")

    # Expected max Sharpe under null (SR₀).
    sr0 = math.sqrt(sharpe_variance_across_trials) * (
        (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    # Non-normality denominator σ²(SR).
    denom_inner = (
        1.0 - returns_skew * observed_sharpe + ((returns_kurt - 1.0) / 4.0) * observed_sharpe**2
    )
    if denom_inner <= 0:
        # Extreme skew/kurt can drive the denominator negative — the
        # BLP formula is undefined there; we surface NaN rather than a
        # nonsense value.
        return float("nan")
    denom = math.sqrt(denom_inner)

    z = (observed_sharpe - sr0) * math.sqrt(n_observations - 1) / denom
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    *,
    n_trials: int,
    n_observations: int,
    sharpe_variance_across_trials: float,
) -> None:
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ConfigError(f"n_trials must be a positive int, got {n_trials!r}")
    if not isinstance(n_observations, int) or n_observations < 2:
        raise ConfigError(f"n_observations must be an int >= 2, got {n_observations!r}")
    if sharpe_variance_across_trials < 0 and not math.isnan(sharpe_variance_across_trials):
        raise ConfigError(
            f"sharpe_variance_across_trials must be >= 0, " f"got {sharpe_variance_across_trials!r}"
        )


# Re-export numpy to keep the import used at module-load time; scipy's
# `stats.norm.ppf` / `.cdf` are what we actually use.
_ = np
