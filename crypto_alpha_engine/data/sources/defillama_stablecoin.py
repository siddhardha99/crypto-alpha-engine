"""DefiLlama total-stablecoin-market-cap source.

API: ``GET https://stablecoins.llama.fi/stablecoincharts/all``

Returns a list of ``{"date": <unix_seconds_str>, "totalCirculatingUSD":
{"peggedUSD": <float>, ...}}``. We extract ``peggedUSD`` (USD-pegged
stablecoins) as the canonical value under ``IndexValueSchema``.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType

_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
_DEFAULT_TIMEOUT_S = 30.0


class _HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


class DefiLlamaStablecoinSource:
    """``DataSource`` for DefiLlama's total stablecoin market cap."""

    name: str = "defillama_stablecoin"
    data_type: DataType = DataType.MACRO
    symbols: list[str] = ["stablecoin_mcap"]

    def __init__(self, http_get: _HttpGetter | None = None) -> None:
        self._http_get: _HttpGetter = http_get if http_get is not None else requests.get

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        if symbol not in self.symbols:
            raise ValueError(
                f"DefiLlamaStablecoinSource does not serve {symbol!r}; "
                f"supported: {self.symbols!r}"
            )
        _ = start, end, freq  # API returns the full series.
        resp = self._http_get(_URL, params=None, timeout=_DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [int(r["date"]) for r in payload], unit="s", utc=True
                ).astype("datetime64[us, UTC]"),
                "value": [float(r["totalCirculatingUSD"]["peggedUSD"]) for r in payload],
            }
        )
        return df.sort_values("timestamp", ignore_index=True)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2019-01-01", tz="UTC")
