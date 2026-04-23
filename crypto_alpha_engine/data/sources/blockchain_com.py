"""Blockchain.com on-chain metrics source.

Serves two symbols through a single source (one per Blockchain.com
charts endpoint):

* ``btc_active_addresses`` → ``/charts/n-unique-addresses``
* ``btc_hashrate``         → ``/charts/hash-rate``

Both return ``{"values": [{"x": <unix_seconds>, "y": <float>}, ...]}``
and validate under the canonical ONCHAIN schema (``IndexValueSchema``).
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType

_BASE = "https://api.blockchain.info/charts"
_DEFAULT_TIMEOUT_S = 30.0

_SYMBOL_TO_METRIC: dict[str, str] = {
    "btc_active_addresses": "n-unique-addresses",
    "btc_hashrate": "hash-rate",
}


class _HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


class BlockchainComSource:
    """``DataSource`` for Blockchain.com daily on-chain metrics."""

    name: str = "blockchain_com"
    data_type: DataType = DataType.ONCHAIN
    symbols: list[str] = list(_SYMBOL_TO_METRIC.keys())

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
                f"BlockchainComSource does not serve {symbol!r}; " f"supported: {self.symbols!r}"
            )
        _ = start, end, freq  # API returns the full series by default.
        metric = _SYMBOL_TO_METRIC[symbol]
        resp = self._http_get(
            f"{_BASE}/{metric}",
            params={"timespan": "all", "format": "json"},
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        values = resp.json().get("values", [])
        if not values:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [int(v["x"]) for v in values], unit="s", utc=True
                ).astype("datetime64[us, UTC]"),
                "value": [float(v["y"]) for v in values],
            }
        )
        return df.sort_values("timestamp", ignore_index=True)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        if symbol not in self.symbols:
            raise ValueError(f"unknown symbol {symbol!r}; supported: {self.symbols!r}")
        # Both metrics go back to Bitcoin genesis, but meaningful values
        # only appear after a few months. 2010-01-01 is a safe floor.
        return pd.Timestamp("2010-01-01", tz="UTC")
