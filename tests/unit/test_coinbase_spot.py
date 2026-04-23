"""Tests for the built-in Coinbase spot OHLCV source.

Uses an injected FakeExchange — no real ccxt client, no network. The
tests exercise the source's Protocol conformance and its ccxt-pagination
plumbing end-to-end via :func:`run_source` for the orchestration
invariants (resume, dedup, quarantine).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data import registry as registry_mod
from crypto_alpha_engine.data.downloader import run_source
from crypto_alpha_engine.data.loader import load_ohlcv
from crypto_alpha_engine.data.protocol import DataSource, DataType
from crypto_alpha_engine.data.sources.coinbase_spot import CoinbaseSpotSource
from crypto_alpha_engine.exceptions import DataSchemaViolation

# ---------------------------------------------------------------------------
# FakeExchange — implements the fetch_ohlcv bit of the ccxt Exchange API.
# ---------------------------------------------------------------------------


class FakeExchange:
    def __init__(
        self,
        bars: list[list[float]] | None = None,
        fail_mode: str | None = None,
    ) -> None:
        self.bars = list(bars) if bars is not None else []
        self.calls: list[dict[str, Any]] = []
        self.fail_mode = fail_mode

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        self.calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        bars = [b for b in self.bars if since is None or b[0] >= since]
        if limit is not None:
            bars = bars[:limit]
        if self.fail_mode == "corrupt":
            bars = [[b[0], b[1], b[2], b[3], b[4], -1.0] for b in bars]
        return bars

    # The full protocol requires these even if our source doesn't use them.
    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        _ = symbol, since, limit
        return []

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = symbol, timeframe, since, limit
        return []

    def parse_timeframe(self, timeframe: str) -> int:
        if timeframe == "1h":
            return 3600
        if timeframe == "1d":
            return 86400
        raise ValueError(f"unsupported timeframe {timeframe!r}")


def _make_bars(start_ms: int, count: int, step_ms: int) -> list[list[float]]:
    return [
        [start_ms + i * step_ms, 100.0 + i, 110.0 + i, 95.0 + i, 105.0 + i, 1000.0 + i]
        for i in range(count)
    ]


_REF_MS = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
_HOUR_MS = 3_600_000


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_coinbase_source_implements_protocol(self) -> None:
        src = CoinbaseSpotSource(exchange=FakeExchange())
        assert isinstance(src, DataSource)
        assert src.name == "coinbase_spot"
        assert src.data_type == DataType.OHLCV
        assert "BTC/USD" in src.symbols
        assert "ETH/USD" in src.symbols

    def test_earliest_available_returns_known_dates(self) -> None:
        src = CoinbaseSpotSource(exchange=FakeExchange())
        assert src.earliest_available("BTC/USD") == pd.Timestamp("2017-01-01", tz="UTC")
        assert src.earliest_available("ETH/USD") == pd.Timestamp("2017-08-01", tz="UTC")

    def test_earliest_available_raises_for_unknown_symbol(self) -> None:
        src = CoinbaseSpotSource(exchange=FakeExchange())
        with pytest.raises(ValueError, match="SOL/USD"):
            src.earliest_available("SOL/USD")

    def test_fetch_unsupported_symbol_raises(self) -> None:
        src = CoinbaseSpotSource(exchange=FakeExchange())
        with pytest.raises(ValueError, match="SOL/USD"):
            src.fetch("SOL/USD", start=pd.Timestamp("2024-01-01", tz="UTC"))


# ---------------------------------------------------------------------------
# End-to-end through run_source: the orchestrator's invariants
# ---------------------------------------------------------------------------


class TestOrchestrationInvariants:
    def test_fresh_download_writes_validated_parquet(self, tmp_path: Path) -> None:
        ex = FakeExchange(bars=_make_bars(_REF_MS, 24, _HOUR_MS))
        src = CoinbaseSpotSource(exchange=ex)
        registry_mod.register_source(src)

        out = run_source(
            src,
            "BTC/USD",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-02", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )
        assert out is not None

        df = load_ohlcv("BTC/USD", "1h", source="coinbase_spot", data_dir=tmp_path)
        assert len(df) == 24

    def test_idempotent_resume(self, tmp_path: Path) -> None:
        src = CoinbaseSpotSource(exchange=FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS)))
        registry_mod.register_source(src)
        run_source(
            src,
            "BTC/USD",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 10:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )

        # Second run: exchange now has 20 bars; should only fetch the new 10.
        ex2 = FakeExchange(bars=_make_bars(_REF_MS, 20, _HOUR_MS))
        src2 = CoinbaseSpotSource(exchange=ex2)
        # Re-register under a different temporary instance. (The registry
        # disallows duplicate names; unregister first.)
        registry_mod.unregister_source("coinbase_spot")
        registry_mod.register_source(src2)
        run_source(
            src2,
            "BTC/USD",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 20:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )

        df = load_ohlcv("BTC/USD", "1h", source="coinbase_spot", data_dir=tmp_path)
        assert len(df) == 20
        assert df["timestamp"].is_unique
        # Resume fetch asked for data strictly after the 10th existing bar.
        first_extend = ex2.calls[0]
        assert first_extend["since"] is not None
        assert first_extend["since"] > _REF_MS + 9 * _HOUR_MS

    def test_up_to_date_short_circuits_exchange(self, tmp_path: Path) -> None:
        # First pass seeds the file.
        src = CoinbaseSpotSource(exchange=FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS)))
        registry_mod.register_source(src)
        run_source(
            src,
            "BTC/USD",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 10:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )

        # Second pass with the same window: ccxt must not be called.
        ex2 = FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS))
        src2 = CoinbaseSpotSource(exchange=ex2)
        registry_mod.unregister_source("coinbase_spot")
        registry_mod.register_source(src2)
        run_source(
            src2,
            "BTC/USD",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 10:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )
        assert ex2.calls == []

    def test_validation_failure_quarantines(self, tmp_path: Path) -> None:
        ex = FakeExchange(bars=_make_bars(_REF_MS, 5, _HOUR_MS), fail_mode="corrupt")
        src = CoinbaseSpotSource(exchange=ex)
        registry_mod.register_source(src)

        with pytest.raises(DataSchemaViolation):
            run_source(
                src,
                "BTC/USD",
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
                freq="1h",
                data_dir=tmp_path,
            )

        # No final file.
        assert not (tmp_path / "ohlcv" / "coinbase_spot").exists() or not list(
            (tmp_path / "ohlcv" / "coinbase_spot").glob("*.parquet")
        )
        # Quarantine populated.
        q = tmp_path / "quarantine"
        assert q.exists()
        assert list(q.glob("*.parquet"))
        assert list(q.glob("*.error.txt"))
