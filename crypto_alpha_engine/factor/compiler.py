"""Compile a :class:`FactorNode` into an evaluable function.

The compiler bridges the AST layer (what users write, what the ledger
stores) to the kernel layer (pure functions on ``pd.Series``).

Two-stage design
----------------

::

    compiled = compile_factor(factor)          # once, at submission time
    series_a = compiled(features_for_window_a)  # per walk-forward window
    series_b = compiled(features_for_window_b)

Compilation validates the AST, resolves every operator against the
registry, and prepares the subtree-memoisation cache. Evaluation is
per-window: the same compiled factor runs against many sliding data
windows during a walk-forward backtest, and we want to amortise the
validation cost across those runs.

Arg resolution
--------------

For each positional argument in a :class:`FactorNode`, the compiler
looks up the operator's registered ``arg_types`` (see Phase 3) and
dispatches on the tag:

==========================  ==================================================
``arg_type``                Resolution
==========================  ==================================================
``"series"`` + FactorNode   recursively evaluate the sub-node
``"series"`` + str          look up ``features[string]``
``"int"`` / ``"float"`` / ``"bool"`` + literal  pass through unchanged
``"series_or_scalar"``      dispatch on the runtime type of the arg
==========================  ==================================================

``kwargs`` are forwarded to the kernel verbatim.

features dict format
--------------------

``features: dict[str, pd.Series]`` with **pipe-separated** keys of the
form ``"<symbol>|<column>"``. Colon is reserved — ccxt perp symbols
like ``BTC/USDT:USDT`` already contain one, so ``key.rsplit("|", 1)``
is the only unambiguous split. Examples::

    "BTC/USD|close"
    "BTC/USDT:USDT|funding_rate"
    "fear_greed|value"

Subtree memoisation
-------------------

Within a single :meth:`CompiledFactor.__call__`, identical sub-trees
are evaluated exactly once. The cache is keyed by
:func:`factor_id` of each sub-node, so a factor that references
``ts_mean("x|close", 20)`` five times triggers one kernel call, not
five. The cache is scoped to a single evaluation call — there is no
global cache across factors or across windows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from crypto_alpha_engine.exceptions import ConfigError
from crypto_alpha_engine.factor.ast import factor_id
from crypto_alpha_engine.operators.registry import (
    get_operator,
    get_operator_arg_types,
    has_operator,
)
from crypto_alpha_engine.types import Factor, FactorNode


class CompiledFactor:
    """Callable wrapper around a validated :class:`FactorNode` tree.

    Produced by :func:`compile_factor`. Calling the instance with a
    ``features`` dict runs the factor once.

    Attributes:
        factor_id: The ``"f_<16-hex>"`` identifier for the root node;
            also used as the ledger key.
        root: The original :class:`FactorNode` root, exposed for
            introspection.
        _operator_override: Optional dict mapping operator name →
            kernel. Used by tests to wrap a kernel with a counter for
            the memoisation verification test; production code passes
            ``None``.
    """

    __slots__ = ("factor_id", "root", "_operator_override")

    def __init__(
        self,
        root: FactorNode,
        *,
        operator_override: dict[str, Callable[..., pd.Series]] | None = None,
    ) -> None:
        self.root = root
        self.factor_id = factor_id(root)
        self._operator_override = operator_override or {}

    def __call__(self, features: dict[str, pd.Series]) -> pd.Series:
        """Evaluate the compiled factor against a features dict.

        Args:
            features: Map from ``"<symbol>|<column>"`` keys to
                time-series data. Every string argument in the AST
                must have a matching key.

        Returns:
            The :class:`pd.Series` computed by the factor.

        Raises:
            ConfigError: If a string arg references a feature not in
                ``features``.
        """
        cache: dict[str, pd.Series] = {}
        return self._eval_node(self.root, features, cache)

    def _eval_node(
        self,
        node: FactorNode,
        features: dict[str, pd.Series],
        cache: dict[str, pd.Series],
    ) -> pd.Series:
        node_id = factor_id(node)
        if node_id in cache:
            return cache[node_id]

        arg_types = get_operator_arg_types(node.operator)
        resolved = tuple(
            self._resolve_arg(arg, tag, features, cache)
            for arg, tag in zip(node.args, arg_types, strict=True)
        )
        kernel = self._get_kernel(node.operator)
        result = kernel(*resolved, **node.kwargs)
        cache[node_id] = result
        return result

    def _resolve_arg(
        self,
        arg: Any,
        tag: str,
        features: dict[str, pd.Series],
        cache: dict[str, pd.Series],
    ) -> Any:
        if tag == "series":
            return self._resolve_series(arg, features, cache)
        if tag == "series_or_scalar":
            if isinstance(arg, FactorNode | str):
                return self._resolve_series(arg, features, cache)
            return arg  # int/float/bool pass through
        # Scalar tags — arg is already a literal; parse-time validation
        # guaranteed the type.
        return arg

    def _resolve_series(
        self,
        arg: Any,
        features: dict[str, pd.Series],
        cache: dict[str, pd.Series],
    ) -> pd.Series:
        if isinstance(arg, FactorNode):
            return self._eval_node(arg, features, cache)
        if isinstance(arg, str):
            try:
                return features[arg]
            except KeyError as err:
                raise ConfigError(
                    f"feature {arg!r} not found in features dict; "
                    f"available: {sorted(features)!r}"
                ) from err
        raise ConfigError(
            f"expected FactorNode or feature-name str at series position; "
            f"got {type(arg).__name__}"
        )

    def _get_kernel(self, op_name: str) -> Callable[..., pd.Series]:
        if op_name in self._operator_override:
            return self._operator_override[op_name]
        return get_operator(op_name)


# ---------------------------------------------------------------------------
# Compilation entry points
# ---------------------------------------------------------------------------


def compile_factor(
    factor: Factor | FactorNode,
    *,
    operator_override: dict[str, Callable[..., pd.Series]] | None = None,
) -> CompiledFactor:
    """Validate a factor's AST and return a callable :class:`CompiledFactor`.

    Accepts either a :class:`Factor` wrapper or a bare
    :class:`FactorNode`. The argument shape doesn't affect the
    compiled output.

    Args:
        factor: The factor to compile.
        operator_override: Optional dict mapping operator names to
            alternative kernels. Intended for tests (wrap a kernel
            with a call counter, stub a kernel, etc.); production
            code passes ``None``.

    Returns:
        :class:`CompiledFactor` ready to invoke against a features dict.

    Raises:
        ConfigError: If the AST references an operator not in the
            registry, or if an operator's positional arity is wrong.
    """
    root = factor.root if isinstance(factor, Factor) else factor
    _validate_ast(root)
    return CompiledFactor(root, operator_override=operator_override)


def _validate_ast(node: FactorNode) -> None:
    """Recursively check that every node in the tree is compilable."""
    if not has_operator(node.operator):
        raise ConfigError(f"unknown operator {node.operator!r}")
    expected = get_operator_arg_types(node.operator)
    if len(node.args) != len(expected):
        raise ConfigError(
            f"{node.operator}: expected {len(expected)} positional args "
            f"({expected!r}), got {len(node.args)}"
        )
    for arg in node.args:
        if isinstance(arg, FactorNode):
            _validate_ast(arg)
