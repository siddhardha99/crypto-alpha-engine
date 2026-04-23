"""Factor-similarity metrics.

Two forms, as spec'd in ``docs/factor_design.md`` §4:

* :func:`ast_similarity` — **structural**, normalised largest-common-
  subtree in ``[0, 1]``. Symmetric; 1.0 for identical trees; close to 0
  for disjoint operators. Cheap — runs in a few milliseconds on any
  factor that passes complexity limits. Ships in Phase 4 as the
  primary duplicate-detection mechanism.
* :func:`behavioural_similarity` — **behavioural**, Pearson correlation
  between two compiled factors' output series over a validation
  window. More expensive (needs compile + evaluate), catches
  semantic duplicates that have different tree shapes but produce
  correlated outputs. Signature ships here; full wiring to the ledger
  lands with Phase 6+7.

Rejection workflow (used by the ledger on factor submission):

* If ``ast_similarity(candidate, existing) >= 0.7`` for any existing
  factor, flag as structural duplicate → reject. SPEC §7 threshold.
* Below 0.7 structural but above a behavioural threshold (0.9 on
  validation-window correlation): semantic duplicate → reject.
  Behavioural check happens only after structural pre-filter clears,
  so it's typically skipped.

Only structural similarity ships in Phase 4. Behavioural takes compiled
factor outputs, which the ledger (Phase 7) will produce.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_engine.factor.ast import walk
from crypto_alpha_engine.types import FactorNode

# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------


def ast_similarity(a: FactorNode, b: FactorNode) -> float:
    """Return the structural similarity of two ASTs in ``[0, 1]``.

    Metric: the number of *matching node signatures* between the two
    trees divided by the size of the larger tree. A node's signature
    is ``(operator, arity, tuple_of_child_signatures)`` — so two
    nodes match iff they have the same operator, same number of args,
    and identical sub-tree shapes recursively. Leaf primitives
    (strings, numbers) don't produce their own node signatures;
    they're absorbed into the parent's arity.

    Properties:

    * Symmetric: ``ast_similarity(a, b) == ast_similarity(b, a)``.
    * Identical trees: ``1.0``.
    * Completely disjoint operators: ``0.0``.
    * Two trees that differ only in a constant (``ts_mean(x, 20)``
      vs ``ts_mean(x, 21)``): high — the structural signature is
      the same; only the literal differs.

    Args:
        a, b: Two :class:`FactorNode` trees to compare.

    Returns:
        A float in ``[0.0, 1.0]``.

    Example:
        >>> from crypto_alpha_engine.types import FactorNode
        >>> a = FactorNode(operator="ts_mean", args=("x|close", 20))
        >>> b = FactorNode(operator="ts_mean", args=("x|close", 20))
        >>> ast_similarity(a, b)
        1.0
    """
    sigs_a = _collect_signatures(a)
    sigs_b = _collect_signatures(b)
    if not sigs_a or not sigs_b:
        return 0.0
    # Normalise by the larger set — prevents trivially-similar small
    # sub-trees from dominating the score against a deeply-nested tree.
    denom = max(len(sigs_a), len(sigs_b))
    # Count signatures present in both (multiset intersection).
    from collections import Counter

    intersect = Counter(sigs_a) & Counter(sigs_b)
    shared = sum(intersect.values())
    return shared / denom


def is_too_similar(
    candidate: FactorNode,
    existing: FactorNode,
    *,
    threshold: float = 0.7,
) -> bool:
    """Return ``True`` if the candidate is a structural near-duplicate.

    Uses :func:`ast_similarity` against the given threshold (default
    0.7, per SPEC §7). The caller decides what to do with the result;
    the ledger rejects, research tools might just warn.
    """
    return ast_similarity(candidate, existing) >= threshold


def _collect_signatures(node: FactorNode) -> list[tuple[object, ...]]:
    """Return the multiset of node signatures in a tree (as a list)."""
    out: list[tuple[object, ...]] = []
    for n in walk(node):
        out.append(_node_signature(n))
    return out


def _node_signature(node: FactorNode) -> tuple[object, ...]:
    """A node's signature: ``(operator, arity, child_shapes...)``.

    Two nodes share a signature iff they have the same operator, the
    same number of args, AND the same pattern of child shapes
    (sub-FactorNode vs literal-primitive). Literal values aren't
    captured — two ``ts_mean(x, 20)`` and ``ts_mean(x, 21)`` share a
    signature, which is what we want: structurally near-duplicate.
    """
    child_shapes = tuple(
        _node_signature(arg) if isinstance(arg, FactorNode) else type(arg).__name__
        for arg in node.args
    )
    return (node.operator, len(node.args), child_shapes)


# ---------------------------------------------------------------------------
# Behavioural similarity (hook; full wiring in Phase 6/7)
# ---------------------------------------------------------------------------


def behavioural_similarity(
    output_a: pd.Series,
    output_b: pd.Series,
) -> float:
    """Return ``|Pearson corr|`` between two compiled-factor output series.

    The two series are aligned on their shared index; only the
    intersection where both are non-NaN is used. If the intersection
    is empty or zero-length, returns ``0.0`` (no meaningful signal).

    Args:
        output_a, output_b: The :class:`pd.Series` produced by
            evaluating two compiled factors on the same validation
            window.

    Returns:
        A float in ``[0.0, 1.0]`` — the absolute value of the Pearson
        correlation. A score above ``~0.9`` indicates a semantic
        duplicate even if the AST structures differ.
    """
    # Align on the index intersection; drop any row where either has NaN.
    common_index = output_a.index.intersection(output_b.index)
    if len(common_index) == 0:
        return 0.0
    a = output_a.loc[common_index]
    b = output_b.loc[common_index]
    mask = ~(a.isna() | b.isna())
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return 0.0
    # pandas' .corr ignores NaN but we've already stripped them; it
    # also returns NaN for zero-variance inputs — clamp that to 0 and
    # suppress the numpy divide-by-zero RuntimeWarning that comes with it.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = a.corr(b)
    if pd.isna(corr):
        return 0.0
    return float(abs(corr))
