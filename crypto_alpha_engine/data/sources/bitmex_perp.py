"""BitMEX perpetual-futures funding-rate source.

BitMEX was chosen because it's the deepest US-reachable source for
inverse perp funding: BTC history starts 2016-05-14, ETH starts
2018-08-02 (both confirmed by live probe at Phase-2 close). Since we
use funding as a *signal*, not a PnL source, the inverse-vs-linear
distinction is orthogonal — see ``docs/methodology.md``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_alpha_engine.data.downloader import (
    _timestamp_to_ms,
    funding_records_to_df,
    paginated_ccxt_funding,
)
from crypto_alpha_engine.data.protocol import DataType

_EARLIEST: dict[str, pd.Timestamp] = {
    "BTC/USD:BTC": pd.Timestamp("2016-05-14T12:00:00", tz="UTC"),
    "ETH/USD:BTC": pd.Timestamp("2018-08-02T12:00:00", tz="UTC"),
}

# BitMEX's funding endpoint caps paginated responses at 500 records —
# asking for more returns an HTTPError. Ccxt does not normalise this,
# so each source using BitMEX has to honor the cap.
_BITMEX_PAGE_LIMIT = 500


class BitmexPerpFundingSource:
    """``DataSource`` wrapping ``ccxt.bitmex()`` for perp funding rates."""

    name: str = "bitmex_perp"
    data_type: DataType = DataType.FUNDING
    symbols: list[str] = ["BTC/USD:BTC", "ETH/USD:BTC"]

    def __init__(self, exchange: Any | None = None) -> None:
        self._exchange: Any | None = exchange

    def _get_exchange(self) -> Any:
        if self._exchange is None:
            import ccxt  # noqa: PLC0415

            self._exchange = ccxt.bitmex({"enableRateLimit": True})
        return self._exchange

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "8h",
    ) -> pd.DataFrame:
        """Paginated funding-rate fetch conforming to ``FundingRateSchema``."""
        if symbol not in self.symbols:
            raise ValueError(
                f"BitmexPerpFundingSource does not serve {symbol!r}; "
                f"supported: {self.symbols!r}"
            )
        _ = freq  # BitMEX funding cadence is fixed at 8h; freq is informational.
        records = paginated_ccxt_funding(
            self._get_exchange(),
            symbol,
            since_ms=_timestamp_to_ms(start),
            end_ms=_timestamp_to_ms(end) if end is not None else None,
            limit=_BITMEX_PAGE_LIMIT,
        )
        return funding_records_to_df(records)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        try:
            return _EARLIEST[symbol]
        except KeyError as err:
            raise ValueError(f"unknown symbol {symbol!r}; supported: {self.symbols!r}") from err
