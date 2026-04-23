"""Live-API integration test for every auto-registered DataSource.

Marked ``@pytest.mark.integration`` and additionally skipped unless the
environment variable ``RUN_LIVE_DOWNLOAD_TESTS=1`` is set. CI does not
run this; contributors run it by hand when touching source code::

    RUN_LIVE_DOWNLOAD_TESTS=1 uv run pytest tests/integration/test_live_sources.py

For each registered source, we ask for a small recent window
(≈ 7 days) and assert the returned DataFrame validates against the
canonical schema for its ``data_type``.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

# Import the sources package so every built-in source auto-registers.
import crypto_alpha_engine.data.sources as _sources_pkg  # noqa: F401
from crypto_alpha_engine.data import registry
from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataSource, DataType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_DOWNLOAD_TESTS") != "1",
        reason="set RUN_LIVE_DOWNLOAD_TESTS=1 to run live-API tests",
    ),
]


def _recent_window(source: DataSource, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A 7-day window ending now, clipped to what ``source`` actually has."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=7)
    earliest = source.earliest_available(symbol)
    if start < earliest:
        start = earliest
    return start, end


def _freq_for(data_type: DataType) -> str:
    return {
        DataType.OHLCV: "1h",
        DataType.FUNDING: "8h",
        DataType.SENTIMENT: "1d",
        DataType.ONCHAIN: "1d",
        DataType.MACRO: "1d",
        DataType.OPEN_INTEREST: "1h",
    }[data_type]


def _params() -> list[object]:
    out: list[object] = []
    for source in registry.iter_all_sources():
        for symbol in source.symbols:
            out.append(
                pytest.param(
                    source.name,
                    symbol,
                    id=f"{source.name}::{symbol}",
                )
            )
    return out


@pytest.mark.parametrize(("source_name", "symbol"), _params())
def test_live_source_returns_schema_valid_frame(source_name: str, symbol: str) -> None:
    source = registry.get_source(source_name)
    start, end = _recent_window(source, symbol)
    freq = _freq_for(source.data_type)
    df = source.fetch(symbol, start=start, end=end, freq=freq)

    # An empty frame is acceptable — the API may genuinely have nothing
    # for a 7-day window (e.g. CoinGecko dominance returns current-only).
    # What we verify is: if the source returned rows, they validate.
    if df.empty:
        pytest.skip(f"{source_name}/{symbol}: empty response in recent window")
    CANONICAL_SCHEMAS[source.data_type].validate(df)
