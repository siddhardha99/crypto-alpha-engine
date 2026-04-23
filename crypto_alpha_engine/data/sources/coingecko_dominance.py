"""CoinGecko BTC dominance source (current-value snapshot).

CoinGecko's free public API exposes the global metrics snapshot
(``/api/v3/global``) but not a historical dominance series. Each
``fetch()`` call returns a single row for today — the orchestrator's
merge-with-existing behaviour means the series accumulates over time
as the downloader is re-run.

This limitation is documented so contributors aren't surprised by the
sparse history on a fresh clone.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType

_URL = "https://api.coingecko.com/api/v3/global"
_DEFAULT_TIMEOUT_S = 30.0


class _HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


class CoinGeckoDominanceSource:
    """``DataSource`` for BTC dominance as a daily index."""

    name: str = "coingecko_dominance"
    data_type: DataType = DataType.MACRO
    symbols: list[str] = ["btc_dominance"]

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
                f"CoinGeckoDominanceSource does not serve {symbol!r}; "
                f"supported: {self.symbols!r}"
            )
        _ = start, end, freq  # API returns the current snapshot only.
        resp = self._http_get(_URL, params=None, timeout=_DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        pct = float(data["market_cap_percentage"]["btc"])
        today = pd.Timestamp.now(tz="UTC").floor("D")
        ts = pd.to_datetime([today], utc=True).astype("datetime64[us, UTC]")
        return pd.DataFrame({"timestamp": ts, "value": [pct]})

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        # The API serves only "now" on the free tier — use today as the
        # honest floor rather than a fictitious deep history.
        return pd.Timestamp.now(tz="UTC").floor("D")
