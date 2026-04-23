"""yfinance macro source — DXY (US Dollar Index) and SPY (S&P 500 ETF).

Both are daily close series, returned by yfinance as full-history
DataFrames. We keep only the ``Close`` column and project it onto the
canonical ``IndexValueSchema``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_alpha_engine.data.protocol import DataType

_SYMBOL_TO_TICKER: dict[str, str] = {
    "dxy": "DX-Y.NYB",
    "spy": "SPY",
}


class YFinanceMacroSource:
    """``DataSource`` for macro daily closes via yfinance."""

    name: str = "yfinance_macro"
    data_type: DataType = DataType.MACRO
    symbols: list[str] = list(_SYMBOL_TO_TICKER.keys())

    def __init__(self, yfinance_mod: Any | None = None) -> None:
        self._yf: Any | None = yfinance_mod

    def _get_yf(self) -> Any:
        if self._yf is None:
            import yfinance  # noqa: PLC0415

            self._yf = yfinance
        return self._yf

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
                f"YFinanceMacroSource does not serve {symbol!r}; " f"supported: {self.symbols!r}"
            )
        _ = start, end, freq  # yfinance returns full history; orchestrator merges.
        ticker = _SYMBOL_TO_TICKER[symbol]
        yf = self._get_yf()
        raw = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", "value"])
        close_col: pd.Series = raw["Close"].squeeze()
        ts = pd.to_datetime(close_col.index, utc=True).astype("datetime64[us, UTC]")
        return pd.DataFrame(
            {"timestamp": ts, "value": close_col.astype(float).to_numpy()}
        ).sort_values("timestamp", ignore_index=True)

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        if symbol not in self.symbols:
            raise ValueError(f"unknown symbol {symbol!r}; supported: {self.symbols!r}")
        return pd.Timestamp("2015-01-01", tz="UTC")
