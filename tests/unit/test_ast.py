"""Tests for the factor-AST helpers and the vocabulary-closure canary."""

from __future__ import annotations

import json

import pytest

# Importing the operators package triggers registration of every built-in op,
# so the canary below sees the full set.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.factor.ast import (
    ArgValue,
    canonical_form,
    factor_id,
    factor_id_of,
    walk,
)
from crypto_alpha_engine.operators.registry import (
    get_operator_arg_types,
    list_operators,
)
from crypto_alpha_engine.types import Factor, FactorNode

# ---------------------------------------------------------------------------
# The vocabulary-closure canary
# ---------------------------------------------------------------------------


def test_no_registered_operator_takes_a_literal_string_arg() -> None:
    """Vocabulary-closure rule from ``docs/factor_design.md``.

    The arg_types vocabulary is ``{series, int, float, bool,
    series_or_scalar}`` — deliberately no ``"string"`` tag. Strings
    inside ``FactorNode.args`` are always feature-name lookups; the
    compiler's dispatch depends on this.

    If any contributor (or future Claude Code) adds an operator with
    ``arg_types=("string",)`` — even if the registry's
    ``VALID_ARG_TYPES`` check were later broadened — this test fires
    before the change can merge.
    """
    forbidden = "string"
    offenders: list[tuple[str, tuple[str, ...]]] = []
    for name in list_operators():
        arg_types = get_operator_arg_types(name)
        if forbidden in arg_types:
            offenders.append((name, arg_types))
    assert not offenders, (
        f"Operators declaring {forbidden!r} in arg_types violate the "
        f"vocabulary-closure rule (see docs/factor_design.md §1): "
        f"{offenders!r}"
    )


# ---------------------------------------------------------------------------
# ArgValue union & walk
# ---------------------------------------------------------------------------


class TestWalk:
    def test_single_node_yields_itself(self) -> None:
        node = FactorNode(operator="ts_mean", args=("x|close", 20))
        assert list(walk(node)) == [node]

    def test_preorder_parent_before_children(self) -> None:
        inner = FactorNode(
            operator="ts_mean",
            args=("x|close",),
            kwargs={"window": 20},
        )
        outer = FactorNode(operator="sub", args=("x|close", inner))
        ops = [n.operator for n in walk(outer)]
        assert ops == ["sub", "ts_mean"]

    def test_deeply_nested(self) -> None:
        a = FactorNode(operator="ts_mean", args=("x|close", 10))
        b = FactorNode(operator="ts_std", args=("x|close", 10))
        c = FactorNode(operator="div", args=(a, b))
        d = FactorNode(operator="sign", args=(c,))
        ops = [n.operator for n in walk(d)]
        assert ops == ["sign", "div", "ts_mean", "ts_std"]

    def test_primitive_args_do_not_appear_in_walk(self) -> None:
        node = FactorNode(operator="ts_mean", args=("x|close", 5, 3.14, True))
        assert list(walk(node)) == [node]


# ---------------------------------------------------------------------------
# canonical_form
# ---------------------------------------------------------------------------


class TestCanonicalForm:
    def test_simple_leaf(self) -> None:
        node = FactorNode(operator="ts_mean", args=("x|close", 20))
        assert canonical_form(node) == {
            "op": "ts_mean",
            "args": ["x|close", 20],
            "kwargs": {},
        }

    def test_nested(self) -> None:
        inner = FactorNode(operator="ts_mean", args=("x|close", 20))
        outer = FactorNode(operator="sub", args=("x|close", inner))
        assert canonical_form(outer) == {
            "op": "sub",
            "args": [
                "x|close",
                {"op": "ts_mean", "args": ["x|close", 20], "kwargs": {}},
            ],
            "kwargs": {},
        }

    def test_round_trips_through_json(self) -> None:
        """Canonical form is JSON-serialisable and round-trips byte-identical."""
        inner = FactorNode(operator="ts_mean", args=("x|close", 20))
        outer = FactorNode(operator="sub", args=("x|close", inner))
        payload = json.dumps(canonical_form(outer), sort_keys=True)
        assert json.loads(payload) == canonical_form(outer)

    def test_invalid_arg_type_raises(self) -> None:
        # A tuple isn't a legal ArgValue.
        node = FactorNode(operator="ts_mean", args=(("nested-tuple",), 20))
        with pytest.raises(TypeError, match="tuple"):
            canonical_form(node)


# ---------------------------------------------------------------------------
# factor_id
# ---------------------------------------------------------------------------


class TestFactorId:
    def test_format_is_f_plus_16_hex(self) -> None:
        fid = factor_id(FactorNode(operator="ts_mean", args=("x|close", 20)))
        assert fid.startswith("f_")
        assert len(fid) == 2 + 16
        body = fid[2:]
        assert all(c in "0123456789abcdef" for c in body)

    def test_deterministic(self) -> None:
        node = FactorNode(operator="ts_mean", args=("x|close", 20))
        assert factor_id(node) == factor_id(node)

    def test_differs_for_different_trees(self) -> None:
        a = FactorNode(operator="ts_mean", args=("x|close", 20))
        b = FactorNode(operator="ts_mean", args=("x|close", 21))
        assert factor_id(a) != factor_id(b)

    def test_insensitive_to_kwargs_insertion_order(self) -> None:
        """sort_keys=True means kwargs-ordering can't change the id."""
        a = FactorNode(operator="clip", args=("x|close",), kwargs={"lo": 0.0, "hi": 1.0})
        b = FactorNode(operator="clip", args=("x|close",), kwargs={"hi": 1.0, "lo": 0.0})
        assert factor_id(a) == factor_id(b)

    def test_factor_id_of_delegates_to_root(self) -> None:
        root = FactorNode(operator="ts_mean", args=("x|close", 20))
        factor = Factor(
            name="f",
            description="d",
            hypothesis="h",
            root=root,
        )
        assert factor_id_of(factor) == factor_id(root)

    def test_distinguishes_structurally_different_but_same_text(self) -> None:
        """Two trees that happen to pretty-print similarly still hash distinctly."""
        a = FactorNode(
            operator="add",
            args=(FactorNode(operator="ts_mean", args=("x|close", 5)), 1),
        )
        b = FactorNode(
            operator="add",
            args=(1, FactorNode(operator="ts_mean", args=("x|close", 5))),
        )
        assert factor_id(a) != factor_id(b)


# ---------------------------------------------------------------------------
# ArgValue type alias is usable
# ---------------------------------------------------------------------------


def test_argvalue_covers_expected_types() -> None:
    """Smoke test that ArgValue isn't accidentally empty or overly narrow."""
    # At runtime, `ArgValue` is a `types.UnionType`. `isinstance` works.
    samples: list[ArgValue] = [
        FactorNode(operator="ts_mean", args=()),
        "x|close",
        5,
        3.14,
        True,
    ]
    for s in samples:
        assert isinstance(s, FactorNode | str | int | float | bool)
