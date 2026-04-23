"""Contract tests for the Pandera schemas in ``crypto_alpha_engine.data.schemas``.

Schema tests exercise static validation rules on small hand-built frames.
Realistic market dynamics are not in scope here — these tests verify that
malformed inputs are caught at the edge of the data layer, per SPEC §5 and
Principle 8 (idempotent data layer with schema-level rejection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from crypto_alpha_engine.data.schemas import (
    FearGreedSchema,
    FundingRateSchema,
    IndexValueSchema,
    OHLCVSchema,
    OpenInterestSchema,
)

UTC_DTYPE = "datetime64[ns, UTC]"


def _ohlcv_row(
    ts: pd.Timestamp,
    *,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 95.0,
    close: float = 105.0,
    volume: float = 1000.0,
) -> dict[str, object]:
    return {
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _ohlcv_df(n: int = 3) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame([_ohlcv_row(t) for t in ts])


# ---------------------------------------------------------------------------
# OHLCVSchema
# ---------------------------------------------------------------------------


class TestOHLCVSchema:
    def test_valid_frame_passes(self) -> None:
        df = _ohlcv_df(5)
        OHLCVSchema.validate(df)  # should not raise

    def test_missing_column_fails(self) -> None:
        df = _ohlcv_df(3).drop(columns=["close"])
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_extra_column_fails_under_strict(self) -> None:
        df = _ohlcv_df(3).assign(extra=1.0)
        # Pandera raises SchemaErrors (plural) when an extra column triggers
        # strict-mode collection rather than a single SchemaError. Accept
        # either flavor of the pandera exception.
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            OHLCVSchema.validate(df)

    def test_duplicate_timestamp_fails(self) -> None:
        df = _ohlcv_df(3)
        df.loc[2, "timestamp"] = df.loc[0, "timestamp"]
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_naive_timestamp_fails(self) -> None:
        df = _ohlcv_df(3)
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_non_utc_timestamp_fails(self) -> None:
        df = _ohlcv_df(3)
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_timestamps_out_of_order_fails(self) -> None:
        df = _ohlcv_df(3)
        df = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    @pytest.mark.parametrize("col", ["open", "high", "low", "close"])
    def test_non_positive_price_fails(self, col: str) -> None:
        df = _ohlcv_df(3)
        df.loc[1, col] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_negative_volume_fails(self) -> None:
        df = _ohlcv_df(3)
        df.loc[1, "volume"] = -1.0
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_zero_volume_allowed(self) -> None:
        """Thin-liquidity bars happen on real exchanges; don't reject them."""
        df = _ohlcv_df(3)
        df.loc[1, "volume"] = 0.0
        OHLCVSchema.validate(df)

    def test_high_less_than_open_fails(self) -> None:
        df = _ohlcv_df(3)
        df.loc[1, "open"] = 200.0  # above high=110
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_low_greater_than_close_fails(self) -> None:
        df = _ohlcv_df(3)
        df.loc[1, "low"] = 200.0  # above close=105
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_nan_price_fails(self) -> None:
        df = _ohlcv_df(3)
        df.loc[1, "close"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)


# ---------------------------------------------------------------------------
# FundingRateSchema
# ---------------------------------------------------------------------------


class TestFundingRateSchema:
    @staticmethod
    def _df(rates: list[float]) -> pd.DataFrame:
        ts = pd.date_range("2020-01-01", periods=len(rates), freq="8h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "funding_rate": rates})

    def test_valid_positive_and_negative_rates(self) -> None:
        df = self._df([0.0001, -0.0002, 0.00005])
        FundingRateSchema.validate(df)

    def test_duplicate_timestamp_fails(self) -> None:
        df = self._df([0.0001, 0.0002, 0.0003])
        df.loc[2, "timestamp"] = df.loc[0, "timestamp"]
        with pytest.raises(pa.errors.SchemaError):
            FundingRateSchema.validate(df)

    def test_naive_timestamp_fails(self) -> None:
        df = self._df([0.0001, 0.0002])
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        with pytest.raises(pa.errors.SchemaError):
            FundingRateSchema.validate(df)

    def test_nan_funding_rate_fails(self) -> None:
        df = self._df([0.0001, np.nan])
        with pytest.raises(pa.errors.SchemaError):
            FundingRateSchema.validate(df)


# ---------------------------------------------------------------------------
# OpenInterestSchema
# ---------------------------------------------------------------------------


class TestOpenInterestSchema:
    @staticmethod
    def _df(values: list[float]) -> pd.DataFrame:
        ts = pd.date_range("2020-01-01", periods=len(values), freq="h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open_interest": values})

    def test_valid_frame_passes(self) -> None:
        OpenInterestSchema.validate(self._df([1000.0, 1200.0, 1100.0]))

    def test_zero_oi_allowed(self) -> None:
        OpenInterestSchema.validate(self._df([0.0, 100.0]))

    def test_negative_oi_fails(self) -> None:
        with pytest.raises(pa.errors.SchemaError):
            OpenInterestSchema.validate(self._df([100.0, -1.0]))


# ---------------------------------------------------------------------------
# IndexValueSchema
# ---------------------------------------------------------------------------


class TestIndexValueSchema:
    @staticmethod
    def _df(values: list[float], *, freq: str = "D") -> pd.DataFrame:
        ts = pd.date_range("2020-01-01", periods=len(values), freq=freq, tz="UTC")
        return pd.DataFrame({"timestamp": ts, "value": values})

    def test_valid_frame_passes(self) -> None:
        IndexValueSchema.validate(self._df([42.0, 43.5, 41.9]))

    def test_negative_values_allowed(self) -> None:
        """DXY deltas, change rates etc. can legitimately go negative."""
        IndexValueSchema.validate(self._df([-1.5, 0.0, 2.3]))

    def test_nan_value_fails(self) -> None:
        with pytest.raises(pa.errors.SchemaError):
            IndexValueSchema.validate(self._df([42.0, np.nan, 41.0]))


# ---------------------------------------------------------------------------
# FearGreedSchema
# ---------------------------------------------------------------------------


class TestFearGreedSchema:
    @staticmethod
    def _df(values: list[int]) -> pd.DataFrame:
        ts = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "value": values})

    def test_valid_values_pass(self) -> None:
        FearGreedSchema.validate(self._df([0, 50, 100]))

    def test_value_over_100_fails(self) -> None:
        with pytest.raises(pa.errors.SchemaError):
            FearGreedSchema.validate(self._df([50, 101]))

    def test_negative_value_fails(self) -> None:
        with pytest.raises(pa.errors.SchemaError):
            FearGreedSchema.validate(self._df([-1, 50]))
