"""Contract tests for the data-source registry."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data import registry as registry_mod
from crypto_alpha_engine.data.protocol import DataType
from crypto_alpha_engine.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _MockSource:
    def __init__(self, name: str, data_type: DataType, symbols: list[str]) -> None:
        self.name = name
        self.data_type = data_type
        self.symbols = symbols

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        _ = symbol, start, end, freq
        return pd.DataFrame()

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2020-01-01", tz="UTC")


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    """Each test starts fresh and restores the registry on teardown.

    The built-in sources (auto-registered on import) would otherwise leak
    between tests and shadow our mock sources. Every test starts empty
    and re-imports the sources package on teardown so later modules see
    the built-ins again.
    """
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()
    # Re-register the built-ins for any subsequent test module that
    # depends on them (e.g. test_coinbase_spot.py).
    import importlib

    import crypto_alpha_engine.data.sources as _sources_pkg

    importlib.reload(_sources_pkg)


# ---------------------------------------------------------------------------
# register_source / get_source
# ---------------------------------------------------------------------------


class TestRegisterAndGet:
    def test_register_then_get_returns_same_instance(self) -> None:
        registry_mod._reset_for_tests()
        s = _MockSource("t_a", DataType.SENTIMENT, ["BTC"])
        registry_mod.register_source(s)
        assert registry_mod.get_source("t_a") is s

    def test_duplicate_name_raises(self) -> None:
        registry_mod._reset_for_tests()
        registry_mod.register_source(_MockSource("t_dup", DataType.SENTIMENT, ["BTC"]))
        with pytest.raises(ConfigError, match="t_dup"):
            registry_mod.register_source(_MockSource("t_dup", DataType.ONCHAIN, ["BTC"]))

    def test_get_missing_raises(self) -> None:
        registry_mod._reset_for_tests()
        with pytest.raises(ConfigError, match="no source registered"):
            registry_mod.get_source("does_not_exist")


# ---------------------------------------------------------------------------
# list_sources_for / default_for
# ---------------------------------------------------------------------------


class TestLookup:
    def test_list_sources_filters_by_data_type(self) -> None:
        registry_mod._reset_for_tests()
        registry_mod.register_source(_MockSource("a", DataType.OHLCV, ["BTC/USD"]))
        registry_mod.register_source(_MockSource("b", DataType.SENTIMENT, ["BTC"]))
        registry_mod.register_source(_MockSource("c", DataType.OHLCV, ["ETH/USD"]))

        ohlcv = registry_mod.list_sources_for(DataType.OHLCV)
        names = [s.name for s in ohlcv]
        assert names == ["a", "c"]

    def test_list_sources_filters_by_symbol(self) -> None:
        registry_mod._reset_for_tests()
        registry_mod.register_source(_MockSource("a", DataType.OHLCV, ["BTC/USD", "ETH/USD"]))
        registry_mod.register_source(_MockSource("b", DataType.OHLCV, ["SOL/USD"]))

        btc_sources = registry_mod.list_sources_for(DataType.OHLCV, symbol="BTC/USD")
        assert [s.name for s in btc_sources] == ["a"]

    def test_default_for_returns_first_registered(self) -> None:
        registry_mod._reset_for_tests()
        registry_mod.register_source(_MockSource("primary", DataType.OHLCV, ["BTC/USD"]))
        registry_mod.register_source(_MockSource("secondary", DataType.OHLCV, ["BTC/USD"]))

        assert registry_mod.default_for(DataType.OHLCV, "BTC/USD").name == "primary"

    def test_default_for_unknown_raises(self) -> None:
        registry_mod._reset_for_tests()
        with pytest.raises(ConfigError, match="no registered source"):
            registry_mod.default_for(DataType.OHLCV, "NONEXISTENT/PAIR")


# ---------------------------------------------------------------------------
# Built-in auto-registration
# ---------------------------------------------------------------------------


class TestBuiltInSources:
    def test_coinbase_spot_auto_registered_on_import(self) -> None:
        """Importing the sources package registers built-ins (A1 ships only coinbase)."""
        registry_mod._reset_for_tests()
        # Re-import the sources package to trigger registration.
        import importlib

        import crypto_alpha_engine.data.sources as _sources_pkg

        importlib.reload(_sources_pkg)
        src = registry_mod.get_source("coinbase_spot")
        assert src.data_type == DataType.OHLCV
        assert set(src.symbols) == {"BTC/USD", "ETH/USD"}
