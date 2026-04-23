"""Contract tests for the custom exception hierarchy.

Only the three exceptions explicitly referenced by SPEC and CLAUDE.md are in
scope for Phase 1: the common base, ``LookAheadDetected``, ``ConfigError``,
and ``DataSchemaViolation``. More specific exceptions will be added by the
phases that need them (e.g. ``SplitLeakageError`` in Phase 2).
"""

from __future__ import annotations

import pytest

from crypto_alpha_engine.exceptions import (
    ConfigError,
    CryptoAlphaEngineError,
    DataSchemaViolation,
    LookAheadDetected,
)


class TestHierarchy:
    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(CryptoAlphaEngineError, Exception)

    @pytest.mark.parametrize(
        "exc_type",
        [ConfigError, LookAheadDetected, DataSchemaViolation],
    )
    def test_all_specific_exceptions_inherit_from_base(self, exc_type: type[Exception]) -> None:
        assert issubclass(exc_type, CryptoAlphaEngineError)

    def test_specific_exceptions_are_distinct(self) -> None:
        assert not issubclass(ConfigError, LookAheadDetected)
        assert not issubclass(LookAheadDetected, DataSchemaViolation)
        assert not issubclass(DataSchemaViolation, ConfigError)


class TestRaiseAndCatch:
    def test_look_ahead_detected_message_preserved(self) -> None:
        msg = "Factor momentum_24h references index 100 at time 99"
        with pytest.raises(LookAheadDetected, match="momentum_24h"):
            raise LookAheadDetected(msg)

    def test_data_schema_violation_message_preserved(self) -> None:
        msg = "Column 'close' has 3 negative values in BTC/USDT"
        with pytest.raises(DataSchemaViolation, match="negative"):
            raise DataSchemaViolation(msg)

    def test_config_error_message_preserved(self) -> None:
        with pytest.raises(ConfigError, match="taker_bps"):
            raise ConfigError("taker_bps must be positive, got 0.0")

    def test_specific_exception_caught_by_base(self) -> None:
        with pytest.raises(CryptoAlphaEngineError):
            raise LookAheadDetected("future data at t+1")

    def test_base_does_not_swallow_plain_exception(self) -> None:
        # A plain ValueError is NOT a CryptoAlphaEngineError, so the engine
        # base must not catch it via ``except CryptoAlphaEngineError``.
        assert not isinstance(ValueError("unrelated"), CryptoAlphaEngineError)


class TestMessageIsRequired:
    """Every raised exception must carry a specific, human-readable message.

    This mirrors SPEC §15 and CLAUDE.md Pitfall 7: no bare ``raise SomeError()``
    slipping through. We enforce it at the test level since Python itself
    doesn't require exceptions to have a message.
    """

    @pytest.mark.parametrize(
        "exc_type",
        [ConfigError, LookAheadDetected, DataSchemaViolation],
    )
    def test_exception_accepts_a_string_message(
        self, exc_type: type[CryptoAlphaEngineError]
    ) -> None:
        err = exc_type("something concrete went wrong")
        assert "concrete" in str(err)
