"""CryptoPanic news source — optional, requires ``CRYPTOPANIC_API_KEY``.

Registered conditionally in ``sources/__init__.py``: if the env var is
missing at import time, the source is simply not registered. The
registry therefore has no record of it, scripts iterating the registry
skip it cleanly, and no warning noise is produced downstream. This is
the pattern SPEC §5.1 calls out for gated sources.

Phase-2 shape: each post becomes one row at its ``published_at``
timestamp, ``value=1.0`` as a presence indicator. Later phases may
replace this with a richer sentiment scoring — the DataSource interface
lets that happen as a new source class without touching the engine.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType

_URL = "https://cryptopanic.com/api/v1/posts/"
_DEFAULT_TIMEOUT_S = 30.0


class _HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


class CryptoPanicNewsSource:
    """``DataSource`` for CryptoPanic news posts (presence indicator)."""

    name: str = "cryptopanic_news"
    data_type: DataType = DataType.SENTIMENT
    symbols: list[str] = ["cryptopanic"]

    def __init__(
        self,
        api_key: str,
        http_get: _HttpGetter | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("CryptoPanicNewsSource requires a non-empty api_key")
        self._key = api_key
        self._http_get: _HttpGetter = http_get if http_get is not None else requests.get

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        if symbol not in self.symbols:
            raise ValueError(
                f"CryptoPanicNewsSource does not serve {symbol!r}; " f"supported: {self.symbols!r}"
            )
        _ = start, end, freq
        resp = self._http_get(
            _URL,
            params={"auth_token": self._key, "public": "true"},
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        posts = resp.json().get("results", [])
        if not posts:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([p["published_at"] for p in posts], utc=True).astype(
                    "datetime64[us, UTC]"
                ),
                "value": [1.0 for _ in posts],
            }
        )
        return df.sort_values("timestamp", ignore_index=True)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        # CryptoPanic's API exposes a rolling recent window, not deep
        # history. Registrants accumulate the series by running the
        # downloader on a schedule.
        return pd.Timestamp.now(tz="UTC").floor("D")
