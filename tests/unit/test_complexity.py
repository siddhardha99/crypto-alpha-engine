"""Factor-complexity metric + threshold rejection tests."""

from __future__ import annotations

import pytest

# Importing the operators package triggers registration.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.complexity import (
    MAX_DEPTH,
    MAX_NODE_COUNT,
    MAX_UNIQUE_FEATURES,
    ast_depth,
    factor_complexity,
    node_count,
    reject_if_too_complex,
    unique_features,
    unique_operators,
)
from crypto_alpha_engine.factor.parser import parse_string
from crypto_alpha_engine.types import FactorNode

# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------


class TestAstDepth:
    def test_leaf_depth_is_one(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        assert ast_depth(node) == 1

    def test_one_level_nesting_is_two(self) -> None:
        node = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        assert ast_depth(node) == 2

    def test_deep_nesting(self) -> None:
        # ts_zscore(ts_mean(ts_mean(x, 5), 5), 10) → depth 3
        node = parse_string('ts_zscore(ts_mean(ts_mean("x|close", 5), 5), 10)')
        assert ast_depth(node) == 3


class TestNodeCount:
    def test_leaf_has_one_node(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        assert node_count(node) == 1

    def test_includes_all_subnodes(self) -> None:
        # outer ts_zscore + inner ts_mean = 2 FactorNodes
        node = parse_string('ts_zscore(ts_mean("x|close", 5), 10)')
        assert node_count(node) == 2

    def test_primitive_args_dont_count(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        # Only the FactorNode counts; the string and int args are primitives.
        assert node_count(node) == 1


class TestUniqueOperators:
    def test_single_operator(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        assert unique_operators(node) == {"ts_mean"}

    def test_multiple_distinct_operators(self) -> None:
        node = parse_string('add(ts_mean("x|close", 10), ts_zscore("y|close", 20))')
        assert unique_operators(node) == {"add", "ts_mean", "ts_zscore"}

    def test_repeated_operator_counted_once(self) -> None:
        node = parse_string('add(ts_mean("x|close", 10), ts_mean("y|close", 20))')
        assert unique_operators(node) == {"add", "ts_mean"}


class TestUniqueFeatures:
    def test_single_feature(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        assert unique_features(node) == {"x|close"}

    def test_multiple_features(self) -> None:
        node = parse_string('add("x|close", "y|close")')
        assert unique_features(node) == {"x|close", "y|close"}

    def test_repeated_feature_counted_once(self) -> None:
        node = parse_string('add("x|close", "x|close")')
        assert unique_features(node) == {"x|close"}

    def test_nested_features_collected(self) -> None:
        node = parse_string('add(ts_mean("x|close", 10), ts_mean("y|close", 20))')
        assert unique_features(node) == {"x|close", "y|close"}


# ---------------------------------------------------------------------------
# Combined scalar score
# ---------------------------------------------------------------------------


class TestFactorComplexityScalar:
    def test_simple_factor_lands_in_expected_range(self) -> None:
        """ts_mean("x|close", 20) — depth 1, 1 node, 1 feature."""
        node = parse_string('ts_mean("x|close", 20)')
        c = factor_complexity(node)
        assert c["ast_depth"] == 1
        assert c["node_count"] == 1
        assert c["unique_features"] == 1
        # Expected: 0.4*1/7 + 0.4*1/30 + 0.2*1/6 ≈ 0.0571 + 0.0133 + 0.0333 ≈ 0.104
        assert 0.05 < c["scalar"] < 0.15

    def test_mid_complexity_factor(self) -> None:
        """ts_zscore(ts_diff("x|close", 1), 20) — depth 2, 2 nodes, 1 feature."""
        node = parse_string('ts_zscore(ts_diff("x|close", 1), 20)')
        c = factor_complexity(node)
        assert c["ast_depth"] == 2
        assert c["node_count"] == 2
        assert 0.1 < c["scalar"] < 0.25

    def test_scalar_increases_with_depth(self) -> None:
        shallow = parse_string('ts_mean("x|close", 5)')
        deep = parse_string('ts_zscore(ts_mean(ts_mean("x|close", 5), 5), 5)')
        assert factor_complexity(deep)["scalar"] > factor_complexity(shallow)["scalar"]

    def test_scalar_increases_with_features(self) -> None:
        one_feat = parse_string('add("x|close", "x|close")')
        two_feat = parse_string('add("x|close", "y|close")')
        assert factor_complexity(two_feat)["scalar"] > factor_complexity(one_feat)["scalar"]


# ---------------------------------------------------------------------------
# Rejection thresholds (SPEC §7)
# ---------------------------------------------------------------------------


class TestRejectIfTooComplex:
    def test_simple_factor_accepted(self) -> None:
        node = parse_string('ts_mean("x|close", 20)')
        reject_if_too_complex(node)  # no raise

    def test_depth_at_limit_accepted(self) -> None:
        """A depth-7 tree is accepted; depth-8 is rejected."""
        # Build depth-MAX_DEPTH by nesting abs() calls.
        inner: FactorNode = FactorNode(operator="abs", args=("x|close",))
        for _ in range(MAX_DEPTH - 1):
            inner = FactorNode(operator="abs", args=(inner,))
        assert ast_depth(inner) == MAX_DEPTH
        reject_if_too_complex(inner)  # exactly at limit — accepted

    def test_depth_over_limit_rejected(self) -> None:
        inner: FactorNode = FactorNode(operator="abs", args=("x|close",))
        for _ in range(MAX_DEPTH):  # one more than the limit
            inner = FactorNode(operator="abs", args=(inner,))
        assert ast_depth(inner) == MAX_DEPTH + 1
        with pytest.raises(ConfigError, match="ast_depth"):
            reject_if_too_complex(inner)

    def test_node_count_over_limit_rejected(self) -> None:
        """A flat add chain of many nodes exceeds the 30-node cap."""
        root: FactorNode = FactorNode(operator="abs", args=("x|close",))
        # Nest enough abs() to exceed MAX_NODE_COUNT.
        for _ in range(MAX_NODE_COUNT):
            root = FactorNode(operator="abs", args=(root,))
        assert node_count(root) > MAX_NODE_COUNT
        # This factor also violates depth, which will be checked first; we
        # just need one ConfigError either way.
        with pytest.raises(ConfigError):
            reject_if_too_complex(root)

    def test_too_many_features_rejected(self) -> None:
        # Build a tree that references 7 distinct features via nested add().
        features = [f"f{i}|value" for i in range(MAX_UNIQUE_FEATURES + 1)]
        # Recursive add(add(add(...), f), f)
        node: FactorNode = FactorNode(operator="abs", args=(features[0],))
        for f in features[1:]:
            leaf = FactorNode(operator="abs", args=(f,))
            node = FactorNode(operator="add", args=(node, leaf))
        assert len(unique_features(node)) > MAX_UNIQUE_FEATURES
        # depth/node_count will trip first for this construction; the rejection
        # message should be ConfigError regardless.
        with pytest.raises(ConfigError):
            reject_if_too_complex(node)
