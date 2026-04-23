"""Coinbase spot OHLCV — the primary built-in OHLCV source.

Chosen because Coinbase is reachable from every region we've tested
(US, EU, Asia), requires no API key, and has 1h bars back to
2017-01-01 for BTC/USD and 2017-08-01 for ETH/USD — the history depth
SPEC §5 requires.

See :mod:`crypto_alpha_engine.data.protocol` for the Protocol contract.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_alpha_engine.data.downloader import (
    _timestamp_to_ms,
    bars_to_ohlcv_df,
    paginated_ccxt_ohlcv,
)
from crypto_alpha_engine.data.protocol import DataType

# Per our live probe at Phase-2 close: Coinbase has 1h bars for each
# symbol back to these dates. Hardcoded rather than queried so the
# source is importable without a network call.
_EARLIEST: dict[str, pd.Timestamp] = {
    "BTC/USD": pd.Timestamp("2017-01-01", tz="UTC"),
    "ETH/USD": pd.Timestamp("2017-08-01", tz="UTC"),
}


class CoinbaseSpotSource:
    """``DataSource`` wrapping ``ccxt.coinbase()`` for spot OHLCV.

    Args:
        exchange: Injectable ccxt-compatible client. Tests pass a fake;
            production defaults to the real :func:`ccxt.coinbase`
            client, created lazily on first :meth:`fetch` call so
            ``import coinbase_spot`` never touches the network.
    """

    name: str = "coinbase_spot"
    data_type: DataType = DataType.OHLCV
    symbols: list[str] = ["BTC/USD", "ETH/USD"]

    def __init__(self, exchange: Any | None = None) -> None:
        self._exchange: Any | None = exchange  # None means "lazy-make on first use"

    def _get_exchange(self) -> Any:
        if self._exchange is None:
            import ccxt  # noqa: PLC0415 — lazy import so tests don't need network

            self._exchange = ccxt.coinbase({"enableRateLimit": True})
        return self._exchange

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Paginated OHLCV fetch. Returns a frame conforming to ``OHLCVSchema``.

        Args:
            symbol: One of :attr:`symbols`.
            start: UTC-aware earliest timestamp to fetch.
            end: Optional exclusive upper bound (UTC-aware).
            freq: ccxt timeframe string (``"1h"`` or ``"1d"``).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close,
            volume]``. Empty (but correctly shaped) if the exchange
            returns nothing.
        """
        if symbol not in self.symbols:
            raise ValueError(
                f"CoinbaseSpotSource does not serve {symbol!r}; " f"supported: {self.symbols!r}"
            )
        exchange = self._get_exchange()
        bars = paginated_ccxt_ohlcv(
            exchange,
            symbol,
            freq,
            since_ms=_timestamp_to_ms(start),
            end_ms=_timestamp_to_ms(end) if end is not None else None,
        )
        return bars_to_ohlcv_df(bars)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        """Earliest timestamp this source promises to return for ``symbol``."""
        try:
            return _EARLIEST[symbol]
        except KeyError as err:
            raise ValueError(f"unknown symbol {symbol!r}; supported: {self.symbols!r}") from err
