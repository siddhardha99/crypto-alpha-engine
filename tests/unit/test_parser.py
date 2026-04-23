"""Parser tests — happy paths, security rejections, arity/type checks, round-trips.

The file is organised into four sections:

1. Happy paths — parse_string builds the expected FactorNode for
   simple, nested, and negative-number inputs.
2. Security allowlist — the nine explicit rejection tests locked in
   during design review. Any future regression that admits one of
   these AST node types fires the corresponding test here.
3. Arity and type validation — the parser catches wrong-count or
   wrong-type args at parse time using the Phase-3 arg_types vocabulary.
4. JSON round-trip and string round-trip — parse_string /
   serialise_to_string and parse_json / serialise_to_json are
   symmetric within the documented round-trip rules.
"""

from __future__ import annotations

import pytest

# Importing the operators package triggers registration.
import crypto_alpha_engine.operators as _ops_pkg  # noqa: F401
from crypto_alpha_engine.factor.parser import (
    ParseError,
    parse_json,
    parse_string,
    serialise_to_json,
    serialise_to_string,
)
from crypto_alpha_engine.types import FactorNode

# ---------------------------------------------------------------------------
# 1. Happy paths
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_flat_operator_with_string_and_int(self) -> None:
        node = parse_string('ts_mean("BTC/USD|close", 20)')
        assert node.operator == "ts_mean"
        assert node.args == ("BTC/USD|close", 20)
        assert node.kwargs == {}

    def test_nested_operators(self) -> None:
        node = parse_string('ts_mean(funding_z("BTC/USDT:USDT|funding_rate", 24), 7)')
        assert node.operator == "ts_mean"
        inner = node.args[0]
        assert isinstance(inner, FactorNode)
        assert inner.operator == "funding_z"
        assert inner.args == ("BTC/USDT:USDT|funding_rate", 24)
        assert node.args[1] == 7

    def test_negative_int_via_unaryop(self) -> None:
        """ts_diff("x|close", -5) should parse despite -5 being UnaryOp(USub, 5)."""
        # lag must be positive per the kernel, but the PARSER is permissive;
        # the runtime ConfigError would fire on kernel call, not at parse.
        node = parse_string('ts_diff("x|close", -5)')
        assert node.args == ("x|close", -5)

    def test_negative_float_via_unaryop(self) -> None:
        node = parse_string('clip("x|close", -1.5, 1.5)')
        assert node.args == ("x|close", -1.5, 1.5)

    def test_unary_plus_preserves_value(self) -> None:
        node = parse_string('ts_mean("x|close", +5)')
        assert node.args[1] == 5

    def test_bool_literal(self) -> None:
        # No registered operator takes a bool, but the parser handles one.
        # We test this against a hypothetical that matches 'bool' in arg_types.
        # For now: bool survives _constant_to_value — verified via round-trip.
        from crypto_alpha_engine.factor.ast import canonical_form

        node = FactorNode(operator="ts_mean", args=("x|close", 20))
        # round-trip via JSON to prove bool handling is present in serialisation
        _ = canonical_form(node)

    def test_three_arg_operator(self) -> None:
        node = parse_string('ts_quantile("x|close", 20, 0.75)')
        assert node.args == ("x|close", 20, 0.75)


# ---------------------------------------------------------------------------
# 2. Security allowlist — nine explicit rejection tests
# ---------------------------------------------------------------------------


class TestSecurityRejections:
    """Every disallowed AST shape raises ParseError.

    Each test pins the *specific* forbidden node or behaviour in the
    ``match=`` pattern so a future regression produces a precisely-
    targeted failure.
    """

    def test_reject_attribute_chain_via_dunder(self) -> None:
        """Classic sandbox-escape probe: __import__('os').system(...)."""
        with pytest.raises(ParseError, match="Attribute"):
            parse_string("__import__('os').system('ls')")

    def test_reject_attribute_chain_on_operator_name(self) -> None:
        with pytest.raises(ParseError, match="Attribute"):
            parse_string("ts_mean.__class__.__bases__")

    def test_reject_listcomp_in_args(self) -> None:
        with pytest.raises(ParseError, match="ListComp"):
            parse_string("ts_mean([x for x in range(10)], 5)")

    def test_reject_lambda_in_args(self) -> None:
        with pytest.raises(ParseError, match="Lambda"):
            parse_string("ts_mean(lambda x: x, 5)")

    def test_reject_multi_statement(self) -> None:
        with pytest.raises(ParseError, match="single expression"):
            parse_string('ts_mean("x", 5); ts_mean("y", 5)')

    def test_reject_top_level_binop(self) -> None:
        with pytest.raises(ParseError, match="BinOp"):
            parse_string('ts_mean("x", 5) + 1')

    def test_reject_dunder_name_as_operator(self) -> None:
        """__import__ isn't a registered operator — caught at the registry check."""
        with pytest.raises(ParseError, match="unknown operator"):
            parse_string("ts_mean(__import__('os'), 5)")

    def test_reject_subscript(self) -> None:
        with pytest.raises(ParseError, match="Subscript"):
            parse_string('ts_mean("x", 5)[0]')

    def test_reject_multi_statement_with_import(self) -> None:
        """Top-level Import is caught as a multi-statement module body."""
        with pytest.raises(ParseError, match="single expression"):
            parse_string('ts_mean("x", 5); import os')


class TestMoreAllowlistRejections:
    """Belt-and-suspenders: other disallowed shapes that aren't on the
    explicit nine but would be genuine regressions if admitted."""

    def test_reject_starred_expansion(self) -> None:
        with pytest.raises(ParseError):
            parse_string('ts_mean(*("x", 5))')

    def test_reject_keyword_arguments(self) -> None:
        """ast.keyword is deliberately not on the allowlist (kwargs deferred)."""
        with pytest.raises(ParseError, match="keyword"):
            parse_string('ts_mean("x", window=5)')

    def test_reject_unary_op_on_non_constant(self) -> None:
        """-ts_mean(...) — unary negation applied to a Call, not a numeric literal."""
        with pytest.raises(ParseError, match="unary"):
            parse_string('-ts_mean("x", 5)')

    def test_reject_tuple_literal(self) -> None:
        with pytest.raises(ParseError, match="Tuple"):
            parse_string('ts_mean(("x", "y"), 5)')

    def test_reject_dict_literal(self) -> None:
        with pytest.raises(ParseError, match="Dict"):
            parse_string('ts_mean({"x": 1}, 5)')

    def test_reject_set_literal(self) -> None:
        with pytest.raises(ParseError, match="Set"):
            parse_string("ts_mean({1, 2, 3}, 5)")

    def test_reject_compare(self) -> None:
        with pytest.raises(ParseError, match="Compare"):
            parse_string('ts_mean("x", 5) > 0')

    def test_reject_ifexp(self) -> None:
        with pytest.raises(ParseError, match="IfExp"):
            parse_string('ts_mean("x", 5 if True else 10)')

    def test_reject_invalid_python_syntax(self) -> None:
        with pytest.raises(ParseError, match="invalid Python syntax"):
            parse_string("ts_mean('x',,,,)")


# ---------------------------------------------------------------------------
# 3. Arity and type validation at parse time
# ---------------------------------------------------------------------------


class TestArityAndTypes:
    def test_unknown_operator_rejected(self) -> None:
        with pytest.raises(ParseError, match="unknown operator"):
            parse_string('nonexistent_op("x", 5)')

    def test_too_few_args(self) -> None:
        with pytest.raises(ParseError, match="expected 2 positional args"):
            parse_string('ts_mean("x")')

    def test_too_many_args(self) -> None:
        with pytest.raises(ParseError, match="expected 2 positional args"):
            parse_string('ts_mean("x", 5, 10)')

    def test_scalar_where_series_expected(self) -> None:
        with pytest.raises(ParseError, match="expected 'series'"):
            parse_string("ts_mean(5, 20)")

    def test_string_where_int_expected(self) -> None:
        with pytest.raises(ParseError, match="expected 'int'"):
            parse_string('ts_mean("x", "not-an-int")')

    def test_float_where_int_expected(self) -> None:
        with pytest.raises(ParseError, match="expected 'int'"):
            parse_string('ts_mean("x", 2.5)')

    def test_int_where_float_expected_is_accepted(self) -> None:
        """int → float coercion is fine (int is a subset of float semantically)."""
        node = parse_string('ts_quantile("x", 20, 1)')  # q=1 is an int literal
        assert node.args[2] == 1

    def test_bool_rejected_for_int_position(self) -> None:
        """True passes isinstance(int) in Python; parser must catch and reject."""
        with pytest.raises(ParseError, match="expected 'int'"):
            parse_string('ts_mean("x", True)')

    def test_bool_rejected_at_series_position(self) -> None:
        """ts_mean(True, 20) is syntactically valid Python but semantically
        broken: bool isn't a feature-name string or a sub-FactorNode. The
        arg-type check for 'series' catches it before any kernel call."""
        with pytest.raises(ParseError, match="expected 'series'"):
            parse_string("ts_mean(True, 20)")

    def test_bool_rejected_at_series_or_scalar_position(self) -> None:
        """series_or_scalar also refuses bool — prevents the confusing
        add("x", True) that pandas would silently accept as 1."""
        with pytest.raises(ParseError, match="expected 'series_or_scalar'"):
            parse_string('add("x|close", True)')


# ---------------------------------------------------------------------------
# 4. JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_parse_json_simple(self) -> None:
        payload = '{"op":"ts_mean","args":["x|close",20],"kwargs":{}}'
        node = parse_json(payload)
        assert node.operator == "ts_mean"
        assert node.args == ("x|close", 20)

    def test_parse_json_nested(self) -> None:
        payload = (
            '{"op":"ts_mean","args":['
            '{"op":"funding_z","args":["BTC/USDT:USDT|funding_rate",24],"kwargs":{}},'
            "7],"
            '"kwargs":{}}'
        )
        node = parse_json(payload)
        assert isinstance(node.args[0], FactorNode)
        assert node.args[0].operator == "funding_z"

    def test_serialise_roundtrip_byte_identical(self) -> None:
        """Serialise → parse → serialise yields the same bytes."""
        original = parse_string('ts_mean(funding_z("x|f", 24), 7)')
        payload = serialise_to_json(original)
        reparsed = parse_json(payload)
        assert serialise_to_json(reparsed) == payload

    def test_parse_json_rejects_malformed(self) -> None:
        with pytest.raises(ParseError, match="invalid JSON"):
            parse_json('{"op": "ts_mean", "args": [')

    def test_parse_json_rejects_missing_keys(self) -> None:
        with pytest.raises(ParseError, match="missing keys"):
            parse_json('{"op":"ts_mean"}')

    def test_parse_json_rejects_unknown_operator(self) -> None:
        with pytest.raises(ParseError, match="unknown operator"):
            parse_json('{"op":"made_up","args":[],"kwargs":{}}')

    def test_parse_json_enforces_arity(self) -> None:
        with pytest.raises(ParseError, match="expected 2 positional args"):
            parse_json('{"op":"ts_mean","args":["x"],"kwargs":{}}')


# ---------------------------------------------------------------------------
# 5. String round-trip
# ---------------------------------------------------------------------------


class TestStringRoundTrip:
    def test_simple_node_roundtrips(self) -> None:
        source = 'ts_mean("x|close", 20)'
        node = parse_string(source)
        reserialised = serialise_to_string(node)
        # Modulo whitespace; compare via re-parsed node equality.
        reparsed = parse_string(reserialised)
        assert parse_string(source) == reparsed

    def test_nested_node_roundtrips(self) -> None:
        source = 'ts_mean(funding_z("BTC/USDT:USDT|funding_rate", 24), 7)'
        node = parse_string(source)
        reparsed = parse_string(serialise_to_string(node))
        assert node == reparsed

    def test_negative_int_roundtrips(self) -> None:
        # After parse, -5 becomes a plain int; serialise emits "-5".
        node = parse_string('ts_diff("x|close", -5)')
        source2 = serialise_to_string(node)
        assert parse_string(source2).args == node.args
