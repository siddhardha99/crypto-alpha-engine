"""Pandera schemas for every dataset loaded from disk.

Principle 8 ("idempotent data layer") depends on the guarantee that no bytes
from disk flow into the engine without passing a schema first. These schemas
define what valid data looks like for each dataset; the loader (and the
download pipeline that writes the files) is responsible for running them and
rejecting anything that doesn't conform.

Naming convention: one ``DataFrameModel`` per distinct on-disk shape. Datasets
that share a shape share a schema — e.g. the Fear & Greed index, hashrate,
and stablecoin mcap all use :class:`IndexValueSchema` because they all boil
down to a timestamp + a scalar value.

Invariants enforced across all timeseries schemas
-------------------------------------------------
Every schema here checks, via ``_TimestampBaseSchema``:

* The ``timestamp`` column exists.
* Its dtype is ``datetime64[ns, UTC]`` — explicitly UTC-aware
  (CLAUDE.md Pitfall 3).
* Values are unique (no duplicate bars).
* Values are sorted ascending (on-disk order is chronological).
* No NaT timestamps.

Strict mode
-----------
All schemas are ``strict=True``: extra columns fail. This is deliberate. If
a new column is legitimately needed, add it to the schema explicitly rather
than letting it slip through silently.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class _TimestampBaseSchema(pa.DataFrameModel):
    """Shared validation for the ``timestamp`` column across all timeseries.

    Subclasses add their own data columns. This base is not meant to be used
    directly — it describes only the timestamp contract.

    Timestamp precision is pinned to microseconds (``datetime64[us, UTC]``).
    That's the canonical pyarrow / parquet / pandas-3.0 default and is what
    every downloader and loader must produce — any ``ns``-precision series
    should be down-cast to ``us`` before validation.
    """

    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(
        unique=True,
        nullable=False,
        dtype_kwargs={"unit": "us", "tz": "UTC"},
    )

    class Config:
        strict = True
        coerce = False

    @pa.dataframe_check
    @classmethod
    def _timestamp_is_sorted_ascending(cls, df: pd.DataFrame) -> bool:
        return bool(df["timestamp"].is_monotonic_increasing)


class OHLCVSchema(_TimestampBaseSchema):
    """Open / high / low / close / volume bars.

    Used for every candles dataset — BTC/ETH spot, perps, macro (DXY, SPY).
    Prices are strictly positive; volume is non-negative (zero-volume bars
    happen on thin markets). Cross-column invariants: the high is at least
    the max of open, close, low; the low is at most the min of open, close,
    high.
    """

    open: Series[float] = pa.Field(gt=0, nullable=False)
    high: Series[float] = pa.Field(gt=0, nullable=False)
    low: Series[float] = pa.Field(gt=0, nullable=False)
    close: Series[float] = pa.Field(gt=0, nullable=False)
    volume: Series[float] = pa.Field(ge=0, nullable=False)

    @pa.dataframe_check
    @classmethod
    def _high_dominates_other_prices(cls, df: pd.DataFrame) -> pd.Series:
        return df["high"] >= df[["open", "close", "low"]].max(axis=1)

    @pa.dataframe_check
    @classmethod
    def _low_dominated_by_other_prices(cls, df: pd.DataFrame) -> pd.Series:
        return df["low"] <= df[["open", "close", "high"]].min(axis=1)


class FundingRateSchema(_TimestampBaseSchema):
    """Perpetual-futures funding rate.

    The rate is a fractional number (``0.0001`` = 1 bp per period) and
    frequently goes negative in bearish regimes — no sign constraint here.
    """

    funding_rate: Series[float] = pa.Field(nullable=False)


class OpenInterestSchema(_TimestampBaseSchema):
    """Aggregate open interest for a perpetual-futures market.

    Units are exchange-specific (contracts or quote-currency notional); the
    schema only requires a non-negative float. Callers interpret the unit.
    """

    open_interest: Series[float] = pa.Field(ge=0, nullable=False)


class IndexValueSchema(_TimestampBaseSchema):
    """Generic daily scalar index: (timestamp, value).

    Covers datasets without a more specific shape — BTC dominance, total
    stablecoin market cap, active addresses, hashrate, macro indices (DXY,
    SPY close), etc. Callers are expected to know what the ``value`` column
    means for a given dataset; sign and range constraints that vary by
    dataset are imposed at the downloader/loader layer, not here.
    """

    value: Series[float] = pa.Field(nullable=False)


class FearGreedSchema(_TimestampBaseSchema):
    """Alternative.me Fear & Greed index: integer score 0..100.

    Separate from :class:`IndexValueSchema` because the (integer, bounded)
    range is a meaningful invariant — an out-of-range value means the
    source returned something malformed and we should refuse it.
    """

    value: Series[int] = pa.Field(ge=0, le=100, nullable=False)
