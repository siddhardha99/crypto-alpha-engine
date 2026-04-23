"""Factor-complexity metrics and rejection thresholds.

Per SPEC §7 and ``docs/factor_design.md`` §5, every submitted factor
is measured on four dimensions. Each has a hard rejection limit; the
scalar :func:`factor_complexity` output is what Phase 5's Deflated-
Sharpe adjustment consumes.

========================  ========  ===========
Component                 Typical   Reject at
========================  ========  ===========
``ast_depth``             2–5       > 7
``node_count``            3–15      > 30
``unique_operators``      2–5       (soft)
``unique_features``       1–3       > 6
========================  ========  ===========

Scalar score::

    complexity = 0.4 * (depth / 7)
               + 0.4 * (node_count / 30)
               + 0.2 * (unique_features / 6)

Typical accepted factors land in ``[0.15, 0.85]``.
"""

from __future__ import annotations

from typing import Any

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.ast import walk
from crypto_alpha_engine.types import FactorNode

# SPEC §7 rejection thresholds — hard caps.
MAX_DEPTH: int = 7
MAX_NODE_COUNT: int = 30
MAX_UNIQUE_FEATURES: int = 6


# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------


def ast_depth(node: FactorNode) -> int:
    """Return the maximum nesting depth of the AST.

    A leaf operator (no :class:`FactorNode` children) has depth 1; the
    root plus one sub-node nested inside it has depth 2; and so on.
    """
    child_depths = [ast_depth(arg) for arg in node.args if isinstance(arg, FactorNode)]
    return 1 + max(child_depths, default=0)


def node_count(node: FactorNode) -> int:
    """Return the total count of :class:`FactorNode` instances in the tree."""
    return sum(1 for _ in walk(node))


def unique_operators(node: FactorNode) -> set[str]:
    """Return the set of distinct operator names used in the tree."""
    return {n.operator for n in walk(node)}


def unique_features(node: FactorNode) -> set[str]:
    """Return the set of distinct feature-name strings referenced.

    A "feature" is any ``str`` argument anywhere in the tree. Per the
    vocabulary-closure rule (``docs/factor_design.md`` §1), strings
    in :attr:`FactorNode.args` are always feature-name lookups — so
    collecting them gives us the distinct-data-sources count.
    """
    features: set[str] = set()
    for n in walk(node):
        for arg in n.args:
            if isinstance(arg, str):
                features.add(arg)
    return features


# ---------------------------------------------------------------------------
# Combined scoring + rejection
# ---------------------------------------------------------------------------


def factor_complexity(node: FactorNode) -> dict[str, Any]:
    """Return all complexity metrics and the combined scalar score.

    The returned dict has keys:

    * ``"ast_depth"`` — int
    * ``"node_count"`` — int
    * ``"unique_operators"`` — int
    * ``"unique_features"`` — int
    * ``"scalar"`` — float in ``[0, ∞)``; clamped to ``[0, 1]`` for
      factors that pass :func:`reject_if_too_complex`.
    """
    depth = ast_depth(node)
    nodes = node_count(node)
    ops = unique_operators(node)
    feats = unique_features(node)
    scalar = (
        0.4 * (depth / MAX_DEPTH)
        + 0.4 * (nodes / MAX_NODE_COUNT)
        + 0.2 * (len(feats) / MAX_UNIQUE_FEATURES)
    )
    return {
        "ast_depth": depth,
        "node_count": nodes,
        "unique_operators": len(ops),
        "unique_features": len(feats),
        "scalar": scalar,
    }


def reject_if_too_complex(node: FactorNode) -> None:
    """Raise :class:`ConfigError` if any hard threshold is exceeded.

    SPEC §7 thresholds:

    * ``ast_depth > 7`` → reject.
    * ``node_count > 30`` → reject.
    * ``unique_features > 6`` → reject.

    ``unique_operators`` has no hard cap; it's only reported.
    """
    depth = ast_depth(node)
    if depth > MAX_DEPTH:
        raise ConfigError(
            f"factor rejected: ast_depth={depth} exceeds limit {MAX_DEPTH} "
            f"(SPEC §7; deep nesting is an overfit risk)"
        )
    nodes = node_count(node)
    if nodes > MAX_NODE_COUNT:
        raise ConfigError(
            f"factor rejected: node_count={nodes} exceeds limit {MAX_NODE_COUNT} " f"(SPEC §7)"
        )
    feats = len(unique_features(node))
    if feats > MAX_UNIQUE_FEATURES:
        raise ConfigError(
            f"factor rejected: unique_features={feats} exceeds limit "
            f"{MAX_UNIQUE_FEATURES} (SPEC §7; too many data sources = "
            "multi-comparison / data-dredging risk)"
        )
