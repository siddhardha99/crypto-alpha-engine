"""Factor-AST helpers on top of :class:`FactorNode` and :class:`Factor`.

The dataclasses themselves live in :mod:`crypto_alpha_engine.types`
(carried over from Phase 1 as the engine's core data contracts). This
module adds the AST-specific logic the compiler, serialiser, and
similarity/complexity modules rely on:

* :data:`ArgValue` — the exact type union a ``FactorNode.args``
  element may take.
* :func:`walk` — pre-order iterator over every :class:`FactorNode` in
  a tree.
* :func:`canonical_form` — deterministic nested-dict form of the
  tree, suitable for hashing and for the JSON ledger.
* :func:`factor_id` — the ``"f_<16-hex>"`` SHA-256-derived stable
  identifier for an AST. Same function is used for
  :attr:`BacktestResult.factor_id` and for ledger deduplication.

Vocabulary closure rule
-----------------------

The :mod:`operators.registry` vocabulary for ``arg_types`` is:

* ``"series"``
* ``"int"``
* ``"float"``
* ``"bool"``
* ``"series_or_scalar"``

**There is NO ``"string"`` tag.** No operator may take a literal
string as a positional argument. Strings appearing inside
``FactorNode.args`` are *always* feature-name lookups — the compiler
resolves them via the engine's ``features`` dict at evaluation time.
This rule is load-bearing for the compiler's dispatch on arg type;
documenting it as prose would invite drift, so
``tests/unit/test_ast.py`` contains a canary that fails if any
registered operator's ``arg_types`` ever contains ``"string"``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from crypto_alpha_engine.types import Factor, FactorNode

ArgValue = FactorNode | str | int | float | bool
"""The exact type union a :attr:`FactorNode.args` element may take.

* :class:`FactorNode` — a sub-expression; the compiler recurses.
* ``str`` — a feature-name lookup (never a literal string parameter).
* ``int`` / ``float`` / ``bool`` — literal scalar passed to the kernel
  unchanged.
"""


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def walk(node: FactorNode) -> Iterator[FactorNode]:
    """Yield every :class:`FactorNode` in the tree, pre-order (parent first).

    Example:
        >>> inner = FactorNode(operator="ts_mean", args=("x|close",), kwargs={"window": 20})
        >>> outer = FactorNode(operator="sub", args=("x|close", inner))
        >>> [n.operator for n in walk(outer)]
        ['sub', 'ts_mean']
    """
    yield node
    for arg in node.args:
        if isinstance(arg, FactorNode):
            yield from walk(arg)


# ---------------------------------------------------------------------------
# Canonical form / hashing
# ---------------------------------------------------------------------------


def canonical_form(node: FactorNode) -> dict[str, Any]:
    """Return a deterministic nested-dict representation of the tree.

    Serialising the returned dict with ``json.dumps(..., sort_keys=True)``
    yields the stable byte sequence used to compute :func:`factor_id`.
    Two ASTs with the same structure — regardless of the ordering of
    keys inside ``kwargs`` dicts — produce byte-identical JSON.

    Args:
        node: Root of the tree to canonicalise.

    Returns:
        Nested dict with keys ``"op"``, ``"args"``, ``"kwargs"``.
        ``args`` is a list (JSON has no tuple); each element is either
        a nested canonical dict (for sub-nodes) or a primitive scalar.

    Raises:
        TypeError: If any ``FactorNode.args`` element is not a valid
            :data:`ArgValue`.
    """
    return {
        "op": node.operator,
        "args": [_canonical_arg(a) for a in node.args],
        "kwargs": dict(node.kwargs),
    }


def _canonical_arg(arg: Any) -> Any:
    if isinstance(arg, FactorNode):
        return canonical_form(arg)
    if isinstance(arg, str | int | float | bool):
        return arg
    raise TypeError(
        f"invalid FactorNode arg type {type(arg).__name__}; "
        f"expected one of FactorNode, str, int, float, bool; got {arg!r}"
    )


def factor_id(node: FactorNode) -> str:
    """Return ``"f_<16-hex>"``, the stable identifier for an AST.

    Computation: SHA-256 of the UTF-8 bytes of
    ``json.dumps(canonical_form(node), sort_keys=True,
    separators=(",", ":"))``. First 16 hex chars are used — 64 bits
    of entropy, well beyond the collision floor for a 30-node AST
    cap (SPEC §7).

    Same value is used for :attr:`BacktestResult.factor_id` and for
    the ledger's factor-deduplication key.

    Example:
        >>> node = FactorNode(operator="ts_mean", args=("x|close", 20))
        >>> fid = factor_id(node)
        >>> fid.startswith("f_") and len(fid) == 18
        True
        >>> factor_id(node) == factor_id(node)          # deterministic
        True
    """
    canonical = canonical_form(node)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"f_{digest[:16]}"


def factor_id_of(factor: Factor) -> str:
    """Convenience: return :func:`factor_id` for a :class:`Factor`'s root."""
    return factor_id(factor.root)
