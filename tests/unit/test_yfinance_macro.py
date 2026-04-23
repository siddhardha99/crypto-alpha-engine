"""Tests for the yfinance macro source (DXY + SPY)."""

from __future__ import annotations

import pandas as pd
import pytest

from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataSource, DataType
from crypto_alpha_engine.data.sources.yfinance_macro import YFinanceMacroSource


class _FakeYfinance:
    """Stand-in for the ``yfinance`` module — only ``.download`` is used."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self._frame = frame if frame is not None else pd.DataFrame(columns=["Open", "Close"])
        self.calls: list[tuple[str, str, str, bool, bool]] = []

    def download(
        self,
        ticker: str,
        period: str = "max",
        interval: str = "1d",
        auto_adjust: bool = False,
        progress: bool = False,
    ) -> pd.DataFrame:
        self.calls.append((ticker, period, interval, auto_adjust, progress))
        return self._frame


def _frame_with_closes(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame({"Close": values}, index=idx)


START = pd.Timestamp("2024-01-01", tz="UTC")


class TestProtocolConformance:
    def test_implements_protocol(self) -> None:
        src = YFinanceMacroSource(yfinance_mod=_FakeYfinance())
        assert isinstance(src, DataSource)
        assert src.data_type == DataType.MACRO
        assert set(src.symbols) == {"dxy", "spy"}

    def test_earliest_available(self) -> None:
        src = YFinanceMacroSource(yfinance_mod=_FakeYfinance())
        assert src.earliest_available("dxy") == pd.Timestamp("2015-01-01", tz="UTC")
        assert src.earliest_available("spy") == pd.Timestamp("2015-01-01", tz="UTC")

    def test_unknown_symbol_raises(self) -> None:
        src = YFinanceMacroSource(yfinance_mod=_FakeYfinance())
        with pytest.raises(ValueError, match="GLD"):
            src.earliest_available("GLD")
        with pytest.raises(ValueError, match="GLD"):
            src.fetch("GLD", start=START)


class TestFetch:
    def test_dxy_happy_path(self) -> None:
        fake = _FakeYfinance(frame=_frame_with_closes([103.5, 104.2, 103.9]))
        src = YFinanceMacroSource(yfinance_mod=fake)
        df = src.fetch("dxy", start=START)
        CANONICAL_SCHEMAS[DataType.MACRO].validate(df)
        assert df["value"].tolist() == [103.5, 104.2, 103.9]
        # Source translates the short symbol name to the yfinance ticker.
        assert fake.calls[0][0] == "DX-Y.NYB"

    def test_spy_happy_path(self) -> None:
        fake = _FakeYfinance(frame=_frame_with_closes([470.0, 472.5]))
        src = YFinanceMacroSource(yfinance_mod=fake)
        df = src.fetch("spy", start=START)
        CANONICAL_SCHEMAS[DataType.MACRO].validate(df)
        assert fake.calls[0][0] == "SPY"

    def test_empty_response_returns_empty_df(self) -> None:
        fake = _FakeYfinance(frame=pd.DataFrame())
        src = YFinanceMacroSource(yfinance_mod=fake)
        df = src.fetch("dxy", start=START)
        assert df.empty
        assert list(df.columns) == ["timestamp", "value"]
