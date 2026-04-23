"""Factor-similarity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Importing the operators package triggers registration.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.factor.parser import parse_string
from crypto_alpha_engine.factor.similarity import (
    ast_similarity,
    behavioural_similarity,
    is_too_similar,
)
from crypto_alpha_engine.types import FactorNode

# ---------------------------------------------------------------------------
# Structural (ast_similarity)
# ---------------------------------------------------------------------------


class TestAstSimilarity:
    def test_identical_trees_score_one(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_mean("x|close", 20)')
        assert ast_similarity(a, b) == 1.0

    def test_score_is_in_zero_one_range(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_zscore("y|close", 5)')
        s = ast_similarity(a, b)
        assert 0.0 <= s <= 1.0

    def test_symmetric(self) -> None:
        a = parse_string('ts_mean(funding_z("f|fr", 24), 7)')
        b = parse_string('ts_mean("x|close", 5)')
        assert ast_similarity(a, b) == ast_similarity(b, a)

    def test_same_shape_different_constants_score_high(self) -> None:
        """ts_mean(x, 20) vs ts_mean(x, 21) — structural shape is identical."""
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_mean("x|close", 21)')
        # Structural signature: (ts_mean, 2, (str, int)) — same for both.
        assert ast_similarity(a, b) == 1.0

    def test_different_operators_score_low(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_std("x|close", 20)')
        # ts_mean ≠ ts_std so the root-signature differs; shapes below
        # are leaf-equivalent but the root node is distinct.
        s = ast_similarity(a, b)
        assert s < 0.6  # substantially less than 1

    def test_deeply_nested_identical_scores_one(self) -> None:
        a = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        b = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        assert ast_similarity(a, b) == 1.0

    def test_partial_overlap_is_nonzero_less_than_one(self) -> None:
        """Factor A has ts_mean inside ts_zscore; factor B has ts_mean standalone."""
        a = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        b = parse_string('ts_mean("x|close", 5)')
        s = ast_similarity(a, b)
        assert 0.0 < s < 1.0


class TestIsTooSimilar:
    def test_default_threshold_seven_tenths(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_mean("x|close", 20)')
        assert is_too_similar(a, b)  # 1.0 >= 0.7

    def test_disjoint_factors_not_flagged(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('funding_z("y|fr", 5)')
        assert not is_too_similar(a, b)

    def test_threshold_tunable(self) -> None:
        a = parse_string('ts_mean("x|close", 20)')
        b = parse_string('ts_std("x|close", 20)')
        s = ast_similarity(a, b)
        # At a very low threshold, even dissimilar factors flag.
        assert is_too_similar(a, b, threshold=0.0)
        # At 1.0, only identical trees.
        assert not is_too_similar(a, b, threshold=1.0)
        # Sanity: at exactly s, the >= comparison flags.
        assert is_too_similar(a, b, threshold=s)


# ---------------------------------------------------------------------------
# Behavioural (Pearson correlation on outputs)
# ---------------------------------------------------------------------------


def _s(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


class TestBehaviouralSimilarity:
    def test_identical_series_score_one(self) -> None:
        rng = np.random.default_rng(0)
        s = _s(rng.normal(0, 1, size=50).tolist())
        assert behavioural_similarity(s, s) == pytest.approx(1.0)

    def test_negatively_correlated_score_one_absolute(self) -> None:
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1, size=50).tolist()
        s = _s(vals)
        neg = _s([-v for v in vals])
        assert behavioural_similarity(s, neg) == pytest.approx(1.0)

    def test_uncorrelated_scores_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        a = _s(rng.normal(0, 1, size=200).tolist())
        b = _s(rng.normal(0, 1, size=200).tolist())
        # Independent draws → near-zero correlation, loose bound.
        assert behavioural_similarity(a, b) < 0.25

    def test_empty_overlap_returns_zero(self) -> None:
        a = _s([1.0, 2.0, 3.0])
        # Disjoint index: shift b's index by 100 hours.
        b_idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        b = pd.Series([1.0, 2.0, 3.0], index=b_idx)
        assert behavioural_similarity(a, b) == 0.0

    def test_zero_variance_returns_zero(self) -> None:
        """If one series is constant, Pearson is undefined — clamp to 0."""
        a = _s([1.0, 2.0, 3.0, 4.0, 5.0])
        b = _s([3.0, 3.0, 3.0, 3.0, 3.0])
        assert behavioural_similarity(a, b) == 0.0

    def test_nans_dropped_before_correlation(self) -> None:
        import math

        a = pd.Series(
            [1.0, 2.0, math.nan, 4.0, 5.0],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        b = pd.Series(
            [1.0, 2.1, 3.2, 4.3, 5.4],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        # Should compute correlation on the 4 non-NaN pairs only.
        score = behavioural_similarity(a, b)
        assert 0.9 <= score <= 1.0


# ---------------------------------------------------------------------------
# Direct FactorNode construction (not via parser) — signatures still work
# ---------------------------------------------------------------------------


def test_direct_factornode_construction() -> None:
    """Similarity works on hand-built FactorNodes, not just parser output."""
    a = FactorNode(operator="ts_mean", args=("x|close", 20))
    b = FactorNode(operator="ts_mean", args=("x|close", 20))
    assert ast_similarity(a, b) == 1.0
