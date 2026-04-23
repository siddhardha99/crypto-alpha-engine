"""Alternative.me Fear & Greed index — a 0..100 integer sentiment indicator.

The API publishes the full history in one call (``?limit=0``). We cast
integer values to float to match the canonical ``IndexValueSchema`` and
let the source impose the 0..100 bound as an internal sanity check (the
canonical SENTIMENT schema is permissive, per SPEC §5.1 — sources may
refine).
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
import requests

from crypto_alpha_engine.data.protocol import DataType

_URL = "https://api.alternative.me/fng/"
_DEFAULT_TIMEOUT_S = 30.0


class _HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


class AlternativeFearGreedSource:
    """``DataSource`` for the Alternative.me Fear & Greed index."""

    name: str = "alternative_fear_greed"
    data_type: DataType = DataType.SENTIMENT
    symbols: list[str] = ["fear_greed"]

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
                f"AlternativeFearGreedSource does not serve {symbol!r}; "
                f"supported: {self.symbols!r}"
            )
        _ = start, end, freq  # API returns full history; orchestrator merges.
        resp = self._http_get(
            _URL, params={"limit": 0, "format": "json"}, timeout=_DEFAULT_TIMEOUT_S
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return pd.DataFrame(columns=["timestamp", "value"])

        ts = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s", utc=True)
        values = [int(r["value"]) for r in rows]
        if any(v < 0 or v > 100 for v in values):
            raise ValueError(
                "alternative.me returned a value outside the documented 0..100 range; "
                "treating as corrupt data rather than writing to disk"
            )
        df = pd.DataFrame(
            {
                "timestamp": ts.astype("datetime64[us, UTC]"),
                "value": [float(v) for v in values],  # canonical schema: float
            }
        )
        return df.sort_values("timestamp", ignore_index=True)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2018-02-01", tz="UTC")
