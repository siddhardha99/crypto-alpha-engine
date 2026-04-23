"""Contract tests for the free data sources.

The key invariants, shared across every source:

* API failures (network, non-2xx, JSON mismatch) must NOT raise — they log
  a warning and return ``None`` so the download pipeline continues.
* Missing credentials are silent skips (warning + ``None``), not errors.
* Successful responses round-trip through the Pandera schema just like
  the exchange downloader; validation failures land in
  ``<data_dir>/quarantine/`` instead of the final path.

Tests use a hand-rolled ``FakeResponse`` and inject it through each
function's ``http_get=`` parameter, so nothing here hits the real network.
The yfinance sources are covered via ``yfinance_mod=`` injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data.free_sources import (
    download_btc_active_addresses,
    download_cryptopanic_news,
    download_dxy,
    download_fear_greed,
)
from crypto_alpha_engine.data.loader import load_fear_greed, load_index_value
from crypto_alpha_engine.data.schemas import IndexValueSchema

# ---------------------------------------------------------------------------
# Fake HTTP response and getter
# ---------------------------------------------------------------------------


class FakeResponse:
    """Stand-in for ``requests.Response``."""

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
    """Return an object callable via ``http_get(url, params=..., timeout=...)``."""

    def _getter(
        url: str, *, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> FakeResponse:
        _ = url, params, timeout  # silence unused-var lint
        if error is not None:
            raise error
        assert response is not None
        return response

    return _getter


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------


class TestFearGreed:
    def test_happy_path_writes_valid_parquet(self, tmp_path: Path) -> None:
        payload = {
            "data": [
                {"value": "42", "timestamp": "1704067200"},  # 2024-01-01
                {"value": "55", "timestamp": "1703980800"},  # 2023-12-31
            ]
        }
        out = download_fear_greed(
            data_dir=tmp_path,
            http_get=_make_getter(response=FakeResponse(payload)),
        )
        assert out is not None
        df = load_fear_greed(data_dir=tmp_path)
        assert len(df) == 2
        assert df["value"].tolist() == [55, 42]  # sorted ascending

    def test_http_error_returns_none(self, tmp_path: Path) -> None:
        out = download_fear_greed(
            data_dir=tmp_path,
            http_get=_make_getter(error=RuntimeError("503 Service Unavailable")),
        )
        assert out is None
        assert not (tmp_path / "free").exists()

    def test_empty_response_returns_none_no_file_written(self, tmp_path: Path) -> None:
        # API returned no rows (shape mismatch or quiet day). Must not write
        # an empty parquet; must not raise. Pipeline keeps going.
        out = download_fear_greed(
            data_dir=tmp_path,
            http_get=_make_getter(response=FakeResponse({"unexpected": "shape"})),
        )
        assert out is None
        assert not (tmp_path / "free" / "fear_greed.parquet").exists()

    def test_invalid_value_lands_in_quarantine(self, tmp_path: Path) -> None:
        """A value of 200 fails FearGreedSchema (range [0,100])."""
        payload = {"data": [{"value": "200", "timestamp": "1704067200"}]}
        out = download_fear_greed(
            data_dir=tmp_path,
            http_get=_make_getter(response=FakeResponse(payload)),
        )
        assert out is None
        # No final file.
        assert not (tmp_path / "free" / "fear_greed.parquet").exists()
        # A quarantined parquet + sidecar should exist.
        quarantine = tmp_path / "quarantine"
        assert quarantine.exists()
        assert list(quarantine.glob("*.parquet"))
        assert list(quarantine.glob("*.error.txt"))


# ---------------------------------------------------------------------------
# Blockchain.com (exercised via active addresses)
# ---------------------------------------------------------------------------


class TestBlockchainMetric:
    def test_active_addresses_happy_path(self, tmp_path: Path) -> None:
        payload = {
            "values": [
                {"x": 1704067200, "y": 950000.0},
                {"x": 1703980800, "y": 940000.5},
            ]
        }
        out = download_btc_active_addresses(
            data_dir=tmp_path,
            http_get=_make_getter(response=FakeResponse(payload)),
        )
        assert out is not None
        df = load_index_value("btc_active_addresses", data_dir=tmp_path)
        IndexValueSchema.validate(df)
        assert len(df) == 2

    def test_timeout_returns_none(self, tmp_path: Path) -> None:
        out = download_btc_active_addresses(
            data_dir=tmp_path,
            http_get=_make_getter(error=TimeoutError("read timeout")),
        )
        assert out is None


# ---------------------------------------------------------------------------
# CryptoPanic (API key gating — resilience requirement)
# ---------------------------------------------------------------------------


class TestCryptoPanic:
    def test_missing_key_is_silent_skip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CRYPTOPANIC_API_KEY", raising=False)
        out = download_cryptopanic_news(
            data_dir=tmp_path,
            http_get=_make_getter(error=AssertionError("should never be called")),
        )
        assert out is None


# ---------------------------------------------------------------------------
# yfinance (dependency injection, not real network)
# ---------------------------------------------------------------------------


class TestDxy:
    def test_happy_path_writes_parquet(self, tmp_path: Path) -> None:
        class _FakeYfinance:
            @staticmethod
            def download(
                ticker: str,
                period: str = "max",
                interval: str = "1d",
                auto_adjust: bool = False,
                progress: bool = False,
            ) -> pd.DataFrame:
                _ = ticker, period, interval, auto_adjust, progress
                idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
                return pd.DataFrame({"Close": [103.5, 104.2, 103.9]}, index=idx)

        out = download_dxy(data_dir=tmp_path, yfinance_mod=_FakeYfinance())
        assert out is not None
        df = load_index_value("dxy", data_dir=tmp_path)
        assert df["value"].tolist() == [103.5, 104.2, 103.9]

    def test_empty_yfinance_response_returns_none(self, tmp_path: Path) -> None:
        class _EmptyYfinance:
            @staticmethod
            def download(
                ticker: str,
                period: str = "max",
                interval: str = "1d",
                auto_adjust: bool = False,
                progress: bool = False,
            ) -> pd.DataFrame:
                _ = ticker, period, interval, auto_adjust, progress
                return pd.DataFrame()

        out = download_dxy(data_dir=tmp_path, yfinance_mod=_EmptyYfinance())
        assert out is None
