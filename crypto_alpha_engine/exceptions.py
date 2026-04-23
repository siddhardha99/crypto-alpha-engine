"""Custom exception hierarchy for crypto-alpha-engine.

Per SPEC §15 and CLAUDE.md Pitfall 7, every failure mode raises a specific,
well-named exception with a human-readable message — never ``Exception`` or
bare ``ValueError``. Catching ``CryptoAlphaEngineError`` matches anything the
engine raises; catching a subclass matches a specific failure mode.

Phase 1 defines only the three exceptions explicitly referenced by SPEC and
by Phase 1 code paths. Later phases will extend this module with phase-local
exceptions (e.g. ``SplitLeakageError`` in Phase 2, ``CausalityError`` in
Phase 3). Every new exception must subclass ``CryptoAlphaEngineError``.
"""

from __future__ import annotations


class CryptoAlphaEngineError(Exception):
    """Common base for every exception the engine raises.

    User code that wants to catch "anything the engine could fail with"
    should catch this. User code that wants to handle a specific failure
    (a schema violation vs. a look-ahead) should catch the subclass.
    """


class ConfigError(CryptoAlphaEngineError):
    """A configuration value is invalid or forbidden by the engine's rules.

    Raised for invariant violations on constructor inputs: zero or negative
    costs, walk-forward windows that don't add up, timezone-naive timestamps,
    and similar guard-rail failures.

    Example:
        >>> from crypto_alpha_engine.types import CostModel
        >>> try:
        ...     CostModel(taker_bps=0)
        ... except ConfigError as err:
        ...     assert "taker_bps" in str(err)
    """


class LookAheadDetected(CryptoAlphaEngineError):
    """A factor, operator, or backtest step referenced data from the future.

    Principle 2 (all operators are causal) is non-negotiable. Any code path
    that can produce output at time ``t`` depending on input at time ``t+k``
    for positive ``k`` must raise this. The canary test in
    ``tests/integration/test_look_ahead_detection.py`` deliberately crafts a
    cheating factor and asserts this exception fires.

    Example:
        >>> raise LookAheadDetected(
        ...     "Factor momentum_24h references index 101 at time 100"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        crypto_alpha_engine.exceptions.LookAheadDetected: ...
    """


class DataSchemaViolation(CryptoAlphaEngineError):
    """A dataset failed its Pandera schema at load time.

    Raised by ``crypto_alpha_engine.data.loader`` (Phase 2) when a parquet
    file's contents don't match the schema for its dataset type — missing
    columns, out-of-range values, duplicate timestamps, non-UTC index, etc.
    The engine refuses to proceed rather than silently operating on bad
    data.

    Example:
        >>> raise DataSchemaViolation(
        ...     "Column 'close' has 3 negative values in BTC/USDT"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        crypto_alpha_engine.exceptions.DataSchemaViolation: ...
    """


class DuplicateFactor(CryptoAlphaEngineError):
    """A candidate factor duplicates an existing entry in the ledger.

    Raised by ``run_backtest`` when ``on_duplicate="raise"`` and
    :func:`check_duplicate` found a match clearing both SPEC §7
    thresholds (structural ≥ 0.7 AND behavioral ≥ 0.9). The message
    includes the matched ledger_line_id and both similarity scores so
    the caller can locate the prior entry for comparison.
    """


class DuplicateCheckSaturated(CryptoAlphaEngineError):
    """Duplicate detection exceeded its hard cap without a verdict.

    Raised when :func:`check_duplicate` processed ``hard_cap``
    structurally-qualifying priors behaviorally without finding a
    match. The verdict is indeterminate — the candidate might or
    might not be a duplicate of a prior beyond the cap. Always raised
    regardless of ``on_duplicate``, because "skip" would need a
    specific target entry to skip to, which saturation can't supply.

    Typical remedies: prune the ledger, raise ``hard_cap``, or
    ``skip_duplicate_check=True`` if the caller is consciously
    overriding.
    """
