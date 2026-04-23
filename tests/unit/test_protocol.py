"""Contract tests for the DataSource Protocol and DataType enum."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from crypto_alpha_engine.data.protocol import (
    CANONICAL_SCHEMAS,
    DataSource,
    DataType,
)
from crypto_alpha_engine.data.schemas import (
    FundingRateSchema,
    IndexValueSchema,
    OHLCVSchema,
    OpenInterestSchema,
)


class TestDataType:
    def test_values_used_as_directory_segments(self) -> None:
        """Enum values double as on-disk directory names (SPEC §5.1)."""
        assert DataType.OHLCV.value == "ohlcv"
        assert DataType.FUNDING.value == "funding"
        assert DataType.OPEN_INTEREST.value == "open_interest"
        assert DataType.ONCHAIN.value == "onchain"
        assert DataType.SENTIMENT.value == "sentiment"
        assert DataType.MACRO.value == "macro"

    def test_all_members_have_distinct_values(self) -> None:
        values = [t.value for t in DataType]
        assert len(values) == len(set(values))


class TestCanonicalSchemas:
    def test_every_data_type_has_a_canonical_schema(self) -> None:
        """Schema ownership rule (SPEC §5.1): one schema per DataType, defined by the engine."""
        for dt in DataType:
            assert dt in CANONICAL_SCHEMAS, (
                f"DataType {dt!r} has no canonical schema — adding a DataType "
                "requires choosing its schema in protocol.py"
            )

    def test_expected_mappings(self) -> None:
        assert CANONICAL_SCHEMAS[DataType.OHLCV] is OHLCVSchema
        assert CANONICAL_SCHEMAS[DataType.FUNDING] is FundingRateSchema
        assert CANONICAL_SCHEMAS[DataType.OPEN_INTEREST] is OpenInterestSchema
        # ONCHAIN / SENTIMENT / MACRO share the generic value-series schema.
        assert CANONICAL_SCHEMAS[DataType.ONCHAIN] is IndexValueSchema
        assert CANONICAL_SCHEMAS[DataType.SENTIMENT] is IndexValueSchema
        assert CANONICAL_SCHEMAS[DataType.MACRO] is IndexValueSchema


class TestProtocolConformance:
    """A minimal source class demonstrates Protocol conformance at runtime."""

    def test_runtime_checkable_accepts_valid_shape(self) -> None:
        class _Good:
            name = "good_mock"
            data_type = DataType.SENTIMENT
            symbols = ["BTC"]

            def fetch(
                self,
                symbol: str,
                *,
                start: pd.Timestamp,
                end: pd.Timestamp | None = None,
                freq: str = "1h",
            ) -> pd.DataFrame:
                return pd.DataFrame({"timestamp": [], "value": []})

            def earliest_available(self, symbol: str) -> pd.Timestamp:
                return pd.Timestamp("2020-01-01", tz="UTC")

        assert isinstance(_Good(), DataSource)

    def test_runtime_checkable_rejects_missing_method(self) -> None:
        class _MissingFetch:
            name = "bad_mock"
            data_type = DataType.SENTIMENT
            symbols = ["BTC"]

            # No fetch() or earliest_available()

        # ``@runtime_checkable`` checks method existence on the instance.
        assert not isinstance(_MissingFetch(), DataSource)

    def test_source_returns_ohlcv_schema_conforming_frame(self) -> None:
        """A minimum-shape OHLCV source builds frames that validate."""

        class _MockOhlcv:
            name = "mock_ohlcv"
            data_type = DataType.OHLCV
            symbols = ["BTC/USD"]

            def fetch(
                self,
                symbol: str,
                *,
                start: pd.Timestamp,
                end: pd.Timestamp | None = None,
                freq: str = "1h",
            ) -> pd.DataFrame:
                ts = pd.date_range(start, periods=3, freq="h", tz="UTC").astype(
                    "datetime64[us, UTC]"
                )
                return pd.DataFrame(
                    {
                        "timestamp": ts,
                        "open": [100.0, 101.0, 102.0],
                        "high": [110.0, 111.0, 112.0],
                        "low": [95.0, 96.0, 97.0],
                        "close": [105.0, 106.0, 107.0],
                        "volume": [1000.0, 1001.0, 1002.0],
                    }
                )

            def earliest_available(self, symbol: str) -> pd.Timestamp:
                return pd.Timestamp("2020-01-01", tz="UTC")

        src = _MockOhlcv()
        start = pd.Timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        df = src.fetch("BTC/USD", start=start)
        # The frame conforms to its canonical schema — that's the Protocol's
        # central promise (SPEC §5.1 "Schema ownership").
        CANONICAL_SCHEMAS[src.data_type].validate(df)
