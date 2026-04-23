"""Tests for all HTTP-based built-in sources (no ccxt, no yfinance).

Each source is exercised via an injected fake ``http_get``; no real
network. The five sources covered here share the resilient-fetch
pattern, and this file is where that pattern is policed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataSource, DataType
from crypto_alpha_engine.data.sources.alternative_fear_greed import (
    AlternativeFearGreedSource,
)
from crypto_alpha_engine.data.sources.blockchain_com import BlockchainComSource
from crypto_alpha_engine.data.sources.coingecko_dominance import CoinGeckoDominanceSource
from crypto_alpha_engine.data.sources.cryptopanic_news import CryptoPanicNewsSource
from crypto_alpha_engine.data.sources.defillama_stablecoin import DefiLlamaStablecoinSource

# ---------------------------------------------------------------------------
# Fake HTTP
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, json_data: Any, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._json


def _make_getter(
    *,
    response: FakeResponse | None = None,
    error: Exception | None = None,
) -> Any:
    def _getter(
        url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> FakeResponse:
        _ = url, params, timeout
        if error is not None:
            raise error
        assert response is not None
        return response

    return _getter


START = pd.Timestamp("2024-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# alternative_fear_greed
# ---------------------------------------------------------------------------


class TestAlternativeFearGreed:
    def test_protocol_conformance(self) -> None:
        src = AlternativeFearGreedSource(http_get=_make_getter(response=FakeResponse({"data": []})))
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.SENTIMENT
        assert src.symbols == ["fear_greed"]

    def test_happy_path_conforms_to_canonical_schema(self) -> None:
        payload = {
            "data": [
                {"value": "42", "timestamp": "1704067200"},  # 2024-01-01
                {"value": "55", "timestamp": "1703980800"},  # 2023-12-31
            ]
        }
        src = AlternativeFearGreedSource(http_get=_make_getter(response=FakeResponse(payload)))
        df = src.fetch("fear_greed", start=START)
        CANONICAL_SCHEMAS[DataType.SENTIMENT].validate(df)
        assert df["value"].tolist() == [55.0, 42.0]  # sorted ascending

    def test_http_error_propagates(self) -> None:
        src = AlternativeFearGreedSource(
            http_get=_make_getter(error=RuntimeError("503 Service Unavailable"))
        )
        with pytest.raises(RuntimeError, match="503"):
            src.fetch("fear_greed", start=START)

    def test_empty_response_returns_empty_df(self) -> None:
        src = AlternativeFearGreedSource(
            http_get=_make_getter(response=FakeResponse({"unexpected": "shape"}))
        )
        df = src.fetch("fear_greed", start=START)
        assert df.empty
        assert list(df.columns) == ["timestamp", "value"]

    def test_out_of_range_value_raises(self) -> None:
        """Source-specific 0..100 bound fires even though canonical SENTIMENT is unbounded."""
        payload = {"data": [{"value": "200", "timestamp": "1704067200"}]}
        src = AlternativeFearGreedSource(http_get=_make_getter(response=FakeResponse(payload)))
        with pytest.raises(ValueError, match="0..100"):
            src.fetch("fear_greed", start=START)

    def test_unsupported_symbol_raises(self) -> None:
        src = AlternativeFearGreedSource(http_get=_make_getter(response=FakeResponse({"data": []})))
        with pytest.raises(ValueError, match="BTC"):
            src.fetch("BTC", start=START)


# ---------------------------------------------------------------------------
# blockchain_com
# ---------------------------------------------------------------------------


class TestBlockchainCom:
    def test_protocol_conformance_and_symbols(self) -> None:
        src = BlockchainComSource(http_get=_make_getter(response=FakeResponse({"values": []})))
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.ONCHAIN
        assert "btc_active_addresses" in src.symbols
        assert "btc_hashrate" in src.symbols

    def test_active_addresses_happy_path(self) -> None:
        payload = {
            "values": [
                {"x": 1703980800, "y": 940000.5},
                {"x": 1704067200, "y": 950000.0},
            ]
        }
        src = BlockchainComSource(http_get=_make_getter(response=FakeResponse(payload)))
        df = src.fetch("btc_active_addresses", start=START)
        CANONICAL_SCHEMAS[DataType.ONCHAIN].validate(df)
        assert len(df) == 2

    def test_hashrate_happy_path(self) -> None:
        payload = {"values": [{"x": 1704067200, "y": 5e20}]}
        src = BlockchainComSource(http_get=_make_getter(response=FakeResponse(payload)))
        df = src.fetch("btc_hashrate", start=START)
        CANONICAL_SCHEMAS[DataType.ONCHAIN].validate(df)
        assert df["value"].iloc[0] == 5e20

    def test_timeout_propagates(self) -> None:
        src = BlockchainComSource(http_get=_make_getter(error=TimeoutError("read timeout")))
        with pytest.raises(TimeoutError):
            src.fetch("btc_active_addresses", start=START)

    def test_unsupported_symbol_raises(self) -> None:
        src = BlockchainComSource(http_get=_make_getter(response=FakeResponse({"values": []})))
        with pytest.raises(ValueError, match="eth_balance"):
            src.fetch("eth_balance", start=START)


# ---------------------------------------------------------------------------
# defillama_stablecoin
# ---------------------------------------------------------------------------


class TestDefiLlamaStablecoin:
    def test_protocol_conformance(self) -> None:
        src = DefiLlamaStablecoinSource(http_get=_make_getter(response=FakeResponse([])))
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.MACRO
        assert src.symbols == ["stablecoin_mcap"]

    def test_happy_path_conforms_to_schema(self) -> None:
        payload = [
            {"date": 1703980800, "totalCirculatingUSD": {"peggedUSD": 140_000_000_000.0}},
            {"date": 1704067200, "totalCirculatingUSD": {"peggedUSD": 141_500_000_000.0}},
        ]
        src = DefiLlamaStablecoinSource(http_get=_make_getter(response=FakeResponse(payload)))
        df = src.fetch("stablecoin_mcap", start=START)
        CANONICAL_SCHEMAS[DataType.MACRO].validate(df)
        assert len(df) == 2
        assert df["value"].iloc[1] == 141_500_000_000.0

    def test_empty_returns_empty_df(self) -> None:
        src = DefiLlamaStablecoinSource(http_get=_make_getter(response=FakeResponse([])))
        df = src.fetch("stablecoin_mcap", start=START)
        assert df.empty


# ---------------------------------------------------------------------------
# coingecko_dominance
# ---------------------------------------------------------------------------


class TestCoinGeckoDominance:
    def test_protocol_conformance(self) -> None:
        src = CoinGeckoDominanceSource(
            http_get=_make_getter(
                response=FakeResponse({"data": {"market_cap_percentage": {"btc": 52.1}}})
            )
        )
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.MACRO
        assert src.symbols == ["btc_dominance"]

    def test_current_snapshot_writes_single_row(self) -> None:
        payload = {"data": {"market_cap_percentage": {"btc": 52.3}}}
        src = CoinGeckoDominanceSource(http_get=_make_getter(response=FakeResponse(payload)))
        df = src.fetch("btc_dominance", start=START)
        CANONICAL_SCHEMAS[DataType.MACRO].validate(df)
        assert len(df) == 1
        assert df["value"].iloc[0] == 52.3

    def test_error_propagates(self) -> None:
        src = CoinGeckoDominanceSource(http_get=_make_getter(error=RuntimeError("502 Bad Gateway")))
        with pytest.raises(RuntimeError):
            src.fetch("btc_dominance", start=START)


# ---------------------------------------------------------------------------
# cryptopanic_news
# ---------------------------------------------------------------------------


class TestCryptoPanic:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            CryptoPanicNewsSource(api_key="")

    def test_protocol_conformance(self) -> None:
        src = CryptoPanicNewsSource(
            api_key="dummy",
            http_get=_make_getter(response=FakeResponse({"results": []})),
        )
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.SENTIMENT
        assert src.symbols == ["cryptopanic"]

    def test_posts_become_presence_indicators(self) -> None:
        payload = {
            "results": [
                {"published_at": "2024-01-01T00:00:00Z", "title": "a"},
                {"published_at": "2024-01-01T01:00:00Z", "title": "b"},
            ]
        }
        src = CryptoPanicNewsSource(
            api_key="dummy", http_get=_make_getter(response=FakeResponse(payload))
        )
        df = src.fetch("cryptopanic", start=START)
        CANONICAL_SCHEMAS[DataType.SENTIMENT].validate(df)
        assert df["value"].tolist() == [1.0, 1.0]

    def test_empty_returns_empty_df(self) -> None:
        src = CryptoPanicNewsSource(
            api_key="dummy", http_get=_make_getter(response=FakeResponse({"results": []}))
        )
        df = src.fetch("cryptopanic", start=START)
        assert df.empty
