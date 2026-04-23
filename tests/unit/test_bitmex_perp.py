"""Tests for the BitMEX perp funding-rate source."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataSource, DataType
from crypto_alpha_engine.data.sources.bitmex_perp import BitmexPerpFundingSource


class FakeBitmexExchange:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = list(records) if records is not None else []
        self.calls: list[dict[str, Any]] = []

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"symbol": symbol, "since": since, "limit": limit})
        recs = [r for r in self.records if since is None or int(r["timestamp"]) >= since]
        if limit is not None:
            recs = recs[:limit]
        return recs

    # Unused for funding but required by the Protocol surface.
    def fetch_ohlcv(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        _ = args, kwargs
        return []

    def fetch_open_interest_history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        _ = args, kwargs
        return []

    def parse_timeframe(self, timeframe: str) -> int:
        if timeframe == "8h":
            return 8 * 3600
        raise ValueError(f"unsupported timeframe {timeframe!r}")


def _records(start_ms: int, count: int) -> list[dict[str, Any]]:
    step = 8 * 3_600_000  # 8h in ms
    return [
        {"timestamp": start_ms + i * step, "fundingRate": 0.0001 + i * 1e-5} for i in range(count)
    ]


_REF_MS = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)


class TestProtocolConformance:
    def test_implements_protocol(self) -> None:
        src = BitmexPerpFundingSource(exchange=FakeBitmexExchange())
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.FUNDING
        assert src.symbols == ["BTC/USD:BTC", "ETH/USD:BTC"]

    def test_earliest_available_from_live_probe(self) -> None:
        src = BitmexPerpFundingSource(exchange=FakeBitmexExchange())
        assert src.earliest_available("BTC/USD:BTC") == pd.Timestamp(
            "2016-05-14T12:00:00", tz="UTC"
        )
        assert src.earliest_available("ETH/USD:BTC") == pd.Timestamp(
            "2018-08-02T12:00:00", tz="UTC"
        )

    def test_unknown_symbol_raises(self) -> None:
        src = BitmexPerpFundingSource(exchange=FakeBitmexExchange())
        with pytest.raises(ValueError, match="SOL/USD:BTC"):
            src.earliest_available("SOL/USD:BTC")
        with pytest.raises(ValueError, match="SOL/USD:BTC"):
            src.fetch("SOL/USD:BTC", start=pd.Timestamp("2024-01-01", tz="UTC"))


class TestFetch:
    def test_happy_path_conforms_to_funding_schema(self) -> None:
        ex = FakeBitmexExchange(records=_records(_REF_MS, 5))
        src = BitmexPerpFundingSource(exchange=ex)
        df = src.fetch(
            "BTC/USD:BTC",
            start=pd.Timestamp("2023-01-01", tz="UTC"),
            end=pd.Timestamp("2023-01-02 16:00", tz="UTC"),
        )
        CANONICAL_SCHEMAS[DataType.FUNDING].validate(df)
        assert len(df) == 5
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)

    def test_since_is_passed_through(self) -> None:
        ex = FakeBitmexExchange(records=_records(_REF_MS, 10))
        src = BitmexPerpFundingSource(exchange=ex)
        src.fetch(
            "BTC/USD:BTC",
            start=pd.Timestamp("2023-01-01", tz="UTC"),
            end=pd.Timestamp("2023-01-05", tz="UTC"),
        )
        assert ex.calls[0]["since"] == _REF_MS

    def test_empty_response_returns_empty_df(self) -> None:
        ex = FakeBitmexExchange(records=[])
        src = BitmexPerpFundingSource(exchange=ex)
        df = src.fetch(
            "BTC/USD:BTC",
            start=pd.Timestamp("2023-01-01", tz="UTC"),
            end=pd.Timestamp("2023-01-02", tz="UTC"),
        )
        assert df.empty
        assert list(df.columns) == ["timestamp", "funding_rate"]
