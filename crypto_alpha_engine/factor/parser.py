"""Factor parser and serialiser.

This parser uses ast.parse to read the syntax tree only. It does not
execute any Python code at any point. Factor strings are parsed
structurally against an allowlist of AST node types.

The four public entry points:

* :func:`parse_string` — human-authored Python-call syntax
  (``ts_mean(funding_z("BTC/USDT:USDT", 24), 7)``) → :class:`FactorNode`.
* :func:`parse_json` — canonical JSON form (ledger / AI-agent
  output) → :class:`FactorNode`.
* :func:`serialise_to_json` — :class:`FactorNode` → canonical JSON
  string (sorted keys, compact separators).
* :func:`serialise_to_string` — :class:`FactorNode` → authoring-form
  string. ``parse_string`` and ``serialise_to_string`` round-trip
  byte-identically modulo whitespace.

Security: every AST node is matched against :data:`_ALLOWED_NODE_TYPES`.
Anything else — attribute access, subscripting, lambdas, comprehensions,
stray binops, dunder names, multi-statement input — raises
:class:`ParseError` before any operator lookup or evaluation. The
allowlist is deliberately small; new Python AST features don't slip
through silently, they fail closed.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from crypto_alpha_engine.exceptions import CryptoAlphaEngineError
from crypto_alpha_engine.factor.ast import ArgValue
from crypto_alpha_engine.operators.registry import (
    get_operator_arg_types,
    has_operator,
)
from crypto_alpha_engine.types import FactorNode


class ParseError(CryptoAlphaEngineError):
    """Raised when a factor string doesn't parse to a valid AST."""


# ---------------------------------------------------------------------------
# Security allowlist — the heart of parser.py
# ---------------------------------------------------------------------------

# The only AST node types this parser admits. Every other node type —
# including any added to the ast module in a future Python version —
# raises ParseError at visit time. Additions to this tuple require a
# deliberate code change; forgetting one is a safe (reject) failure.
_ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Module,  # root from ast.parse(mode="exec")
    ast.Expression,  # root from ast.parse(mode="eval") — alt entry
    ast.Expr,  # an expression used as a statement (Module body member)
    ast.Call,  # function call — the only shape a FactorNode takes
    ast.Name,  # operator names; validated against the registry
    ast.Constant,  # literal int/float/str/bool/None
    ast.Load,  # context on Name; required by Python AST semantics
    ast.UnaryOp,  # ONLY for +/- over a numeric Constant (see visitor)
    ast.USub,
    ast.UAdd,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_string(source: str) -> FactorNode:
    """Parse a human-authored factor string into a :class:`FactorNode`.

    Example:
        >>> node = parse_string('ts_mean("BTC/USD|close", 20)')
        >>> node.operator
        'ts_mean'
        >>> node.args
        ('BTC/USD|close', 20)

    Raises:
        ParseError: If the source fails to parse, contains a
            disallowed AST node, references an unknown operator, or
            mismatches an operator's positional arity / arg types.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as err:
        raise ParseError(f"invalid Python syntax: {err.msg} at line {err.lineno}") from err

    # Module body must be exactly one expression statement.
    if not isinstance(tree, ast.Module) or len(tree.body) != 1:
        raise ParseError(
            f"factor source must be a single expression; got "
            f"{len(tree.body)} top-level statement(s)"
        )
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr):
        raise ParseError(
            f"top-level must be an expression-statement; got " f"{type(stmt).__name__}"
        )

    visitor = _AllowListVisitor()
    visitor.visit(tree)  # raises on any disallowed node
    return _expr_to_factor_node(stmt.value)


def parse_json(source: str) -> FactorNode:
    """Parse the canonical JSON form into a :class:`FactorNode`.

    The wire format is the nested dict returned by
    :func:`crypto_alpha_engine.factor.ast.canonical_form`.

    Example:
        >>> parse_json('{"op":"ts_mean","args":["x|close",20],"kwargs":{}}').operator
        'ts_mean'

    Raises:
        ParseError: If the JSON is malformed or doesn't match the
            canonical shape.
    """
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as err:
        raise ParseError(f"invalid JSON: {err.msg} at pos {err.pos}") from err
    return _json_to_factor_node(payload)


def serialise_to_json(node: FactorNode) -> str:
    """Serialise a :class:`FactorNode` to canonical JSON.

    Sorted keys, compact separators. Deterministic byte-for-byte:
    the same tree always produces the same string. Round-trips
    through :func:`parse_json`.
    """
    from crypto_alpha_engine.factor.ast import canonical_form

    return json.dumps(
        canonical_form(node),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def serialise_to_string(node: FactorNode) -> str:
    """Serialise a :class:`FactorNode` back to authoring-form syntax.

    Round-trips through :func:`parse_string` modulo whitespace.
    Non-idiomatic float representations (``1.0`` vs ``1``) may vary
    between the input string and this output, but the parsed tree
    is identical in either direction.
    """
    return _node_to_string(node)


# ---------------------------------------------------------------------------
# Allowlist visitor
# ---------------------------------------------------------------------------


class _AllowListVisitor(ast.NodeVisitor):
    """Traverses the AST and rejects any node not on the allowlist.

    Additionally enforces that ``UnaryOp`` is used only to negate a
    numeric literal (so ``ts_mean(x, -1)`` parses but ``-ts_mean(...)``
    does not), and that every ``Name`` in call position is a known
    operator.
    """

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ParseError(
                f"disallowed AST node type {type(node).__name__}: "
                f"parser only admits {sorted(n.__name__ for n in _ALLOWED_NODE_TYPES)}"
            )
        super().generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # noqa: N802
        if not isinstance(node.op, ast.UAdd | ast.USub):
            raise ParseError(f"only unary +/- are allowed; got {type(node.op).__name__}")
        if not (
            isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int | float)
        ):
            raise ParseError("unary +/- may only be applied directly to a numeric literal")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Reject kwargs (ast.keyword) — they are not on the security allowlist.
        if node.keywords:
            raise ParseError("keyword arguments are not supported; use positional args only")
        # Reject *args / **kwargs — would arrive as ast.Starred in node.args
        # or as ast.keyword(arg=None); both are caught elsewhere.
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# AST → FactorNode
# ---------------------------------------------------------------------------


def _expr_to_factor_node(expr: ast.expr) -> FactorNode:
    """Convert a validated AST expression into a :class:`FactorNode`.

    Assumes :class:`_AllowListVisitor` has already traversed the tree
    and rejected anything disallowed; we re-check critical shapes
    here anyway because this function is the only path from external
    input to engine-visible data.
    """
    if not isinstance(expr, ast.Call):
        raise ParseError(
            f"expected a function-call expression at the root; got " f"{type(expr).__name__}"
        )
    if not isinstance(expr.func, ast.Name):
        raise ParseError(f"operator must be a bare name, got {type(expr.func).__name__}")
    op_name = expr.func.id
    if not has_operator(op_name):
        raise ParseError(f"unknown operator {op_name!r}")

    args = tuple(_arg_expr_to_value(a) for a in expr.args)
    _validate_arity_and_types(op_name, args)
    return FactorNode(operator=op_name, args=args, kwargs={})


def _arg_expr_to_value(expr: ast.expr) -> ArgValue:
    """Resolve one AST arg expression to its :data:`ArgValue` form."""
    if isinstance(expr, ast.Call):
        return _expr_to_factor_node(expr)
    if isinstance(expr, ast.Constant):
        return _constant_to_value(expr.value)
    if isinstance(expr, ast.UnaryOp):
        # AllowListVisitor already proved operand is numeric Constant.
        assert isinstance(expr.operand, ast.Constant)  # noqa: S101
        value = expr.operand.value
        sign = -1 if isinstance(expr.op, ast.USub) else 1
        if isinstance(value, int):
            return sign * value
        if isinstance(value, float):
            return sign * value
        # Unreachable given visitor guard.
        raise ParseError(f"unary +/- applied to non-numeric constant {type(value).__name__}")
    raise ParseError(
        f"unexpected arg expression {type(expr).__name__} — should have been "
        "rejected by the allowlist"
    )


def _constant_to_value(value: Any) -> ArgValue:
    if isinstance(value, bool):  # bool is a subclass of int in Python; check first
        return value
    if isinstance(value, int | float | str):
        return value
    raise ParseError(
        f"unsupported literal type {type(value).__name__}: allowed are " "str, int, float, bool"
    )


def _validate_arity_and_types(op_name: str, args: tuple[ArgValue, ...]) -> None:
    """Check positional arity and per-position type compatibility."""
    expected = get_operator_arg_types(op_name)
    if len(args) != len(expected):
        raise ParseError(
            f"{op_name}: expected {len(expected)} positional args ({expected!r}), "
            f"got {len(args)}"
        )
    for i, (arg, tag) in enumerate(zip(args, expected, strict=True)):
        _check_arg_compatible(op_name, i, arg, tag)


def _check_arg_compatible(op_name: str, pos: int, arg: ArgValue, tag: str) -> None:
    """Raise ParseError if the AST-level type of ``arg`` doesn't fit ``tag``.

    The rules:

    * ``"series"`` accepts FactorNode (sub-expression) OR str
      (feature-name lookup).
    * ``"int"`` accepts int only (bool is NOT accepted — it's a
      distinct semantic).
    * ``"float"`` accepts int or float (int promotes).
    * ``"bool"`` accepts bool only.
    * ``"series_or_scalar"`` accepts any of: FactorNode, str, int,
      float (not bool — prevents confusing ``greater_than(x, True)``).
    """
    if tag == "series":
        if not isinstance(arg, FactorNode | str):
            raise ParseError(
                f"{op_name} arg {pos}: expected 'series' (FactorNode or "
                f"feature-name str); got {type(arg).__name__}"
            )
        return
    if tag == "int":
        # Reject bool explicitly — isinstance(True, int) is True in Python.
        if isinstance(arg, bool) or not isinstance(arg, int):
            raise ParseError(f"{op_name} arg {pos}: expected 'int'; got {type(arg).__name__}")
        return
    if tag == "float":
        if isinstance(arg, bool) or not isinstance(arg, int | float):
            raise ParseError(f"{op_name} arg {pos}: expected 'float'; got {type(arg).__name__}")
        return
    if tag == "bool":
        if not isinstance(arg, bool):
            raise ParseError(f"{op_name} arg {pos}: expected 'bool'; got {type(arg).__name__}")
        return
    if tag == "series_or_scalar":
        if isinstance(arg, bool) or not isinstance(arg, FactorNode | str | int | float):
            raise ParseError(
                f"{op_name} arg {pos}: expected 'series_or_scalar'; " f"got {type(arg).__name__}"
            )
        return
    # Defensive — should never trip since VALID_ARG_TYPES is a closed set.
    raise ParseError(f"unknown arg_type tag {tag!r} on operator {op_name}")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


_REQUIRED_JSON_KEYS: frozenset[str] = frozenset({"op", "args", "kwargs"})


def _json_to_factor_node(payload: Any) -> FactorNode:
    if not isinstance(payload, dict):
        raise ParseError(f"expected JSON object for factor node; got {type(payload).__name__}")
    missing = _REQUIRED_JSON_KEYS - set(payload)
    if missing:
        raise ParseError(f"factor-node JSON missing keys: {sorted(missing)!r}")
    op = payload["op"]
    if not isinstance(op, str):
        raise ParseError(f"factor-node 'op' must be str; got {type(op).__name__}")
    if not has_operator(op):
        raise ParseError(f"unknown operator {op!r}")

    raw_args = payload["args"]
    if not isinstance(raw_args, list):
        raise ParseError(f"factor-node 'args' must be list; got {type(raw_args).__name__}")
    args = tuple(_json_to_arg_value(a) for a in raw_args)

    raw_kwargs = payload["kwargs"]
    if not isinstance(raw_kwargs, dict):
        raise ParseError(f"factor-node 'kwargs' must be dict; got {type(raw_kwargs).__name__}")

    _validate_arity_and_types(op, args)
    return FactorNode(operator=op, args=args, kwargs=dict(raw_kwargs))


def _json_to_arg_value(arg: Any) -> ArgValue:
    if isinstance(arg, dict):
        return _json_to_factor_node(arg)
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, int | float | str):
        return arg
    raise ParseError(
        f"unsupported JSON arg type {type(arg).__name__}; "
        "allowed: object (sub-node), str, int, float, bool"
    )


# ---------------------------------------------------------------------------
# FactorNode → source string
# ---------------------------------------------------------------------------


def _node_to_string(node: FactorNode) -> str:
    parts = [_arg_to_source(a) for a in node.args]
    return f"{node.operator}({', '.join(parts)})"


def _arg_to_source(arg: ArgValue) -> str:
    if isinstance(arg, FactorNode):
        return _node_to_string(arg)
    if isinstance(arg, bool):
        return "True" if arg else "False"
    if isinstance(arg, str):
        return json.dumps(arg)  # uses " and escapes \ / " / \n correctly
    # int / float: repr() gives faithful Python syntax.
    return repr(arg)
