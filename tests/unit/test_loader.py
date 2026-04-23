"""Contract tests for the parquet loader.

The loader is the only path by which on-disk data flows into the engine, so
every test here pins a specific responsibility: schema validation raising
``DataSchemaViolation``, clear file-not-found errors, deterministic file
hashing, and the split-aware helpers that sit on top of the low-level
loaders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crypto_alpha_engine.data.loader import (
    OHLCV_FILENAME_TEMPLATE,
    compute_file_hash,
    load_fear_greed,
    load_funding,
    load_ohlcv,
    load_open_interest,
    load_train,
    load_train_plus_validation,
    load_validation,
)
from crypto_alpha_engine.data.splits import DataSplits
from crypto_alpha_engine.exceptions import DataSchemaViolation

# ---------------------------------------------------------------------------
# Fixtures — write small valid parquets into a tmp data_dir.
# ---------------------------------------------------------------------------


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def _make_ohlcv_df(n: int, start: str = "2023-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC").astype("datetime64[us, UTC]")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + i for i in range(n)],
            "high": [110.0 + i for i in range(n)],
            "low": [95.0 + i for i in range(n)],
            "close": [105.0 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


@pytest.fixture
def data_dir_with_ohlcv(tmp_path: Path) -> Path:
    """A data directory containing BTC/USDT 1h OHLCV parquet for 48 hours."""
    # Frame spans 2023-12-30 00:00 through 2024-01-01 23:00: straddles the
    # train/validation boundary (2024-01-01) so split tests have work to do.
    ts = pd.date_range("2023-12-30 00:00:00", periods=48, freq="h", tz="UTC").astype(
        "datetime64[us, UTC]"
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 110.0,
            "low": 95.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    )
    dest = tmp_path / "binance" / "ohlcv"
    dest.mkdir(parents=True)
    file = dest / OHLCV_FILENAME_TEMPLATE.format(base="BTC", quote="USDT", interval="1h")
    df.to_parquet(file, index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_is_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"alpha")
        b.write_bytes(b"beta")
        assert compute_file_hash(a) != compute_file_hash(b)

    def test_starts_with_sha256_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"x")
        assert compute_file_hash(f).startswith("sha256:")


# ---------------------------------------------------------------------------
# load_ohlcv
# ---------------------------------------------------------------------------


class TestLoadOhlcv:
    def test_loads_valid_parquet(self, data_dir_with_ohlcv: Path) -> None:
        df = load_ohlcv("BTC/USDT", "1h", data_dir=data_dir_with_ohlcv)
        assert len(df) == 48
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BTC/USDT"):
            load_ohlcv("BTC/USDT", "1h", data_dir=tmp_path)

    def test_invalid_schema_raises_data_schema_violation(self, tmp_path: Path) -> None:
        """Files that slip in without a column must fail loudly."""
        bad = _make_ohlcv_df(3).drop(columns=["close"])
        dest = tmp_path / "binance" / "ohlcv"
        dest.mkdir(parents=True)
        file = dest / OHLCV_FILENAME_TEMPLATE.format(base="BTC", quote="USDT", interval="1h")
        bad.to_parquet(file, index=False)

        with pytest.raises(DataSchemaViolation, match="close"):
            load_ohlcv("BTC/USDT", "1h", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# load_funding, load_open_interest, load_fear_greed
# ---------------------------------------------------------------------------


class TestLoadFunding:
    def test_roundtrip(self, tmp_path: Path) -> None:
        ts = pd.date_range("2022-01-01", periods=5, freq="8h", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        df = pd.DataFrame({"timestamp": ts, "funding_rate": [0.0001] * 5})
        dest = tmp_path / "binance" / "funding"
        dest.mkdir(parents=True)
        df.to_parquet(dest / "BTC_USDT.parquet", index=False)

        out = load_funding("BTC/USDT", data_dir=tmp_path)
        assert len(out) == 5


class TestLoadOpenInterest:
    def test_roundtrip(self, tmp_path: Path) -> None:
        ts = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        df = pd.DataFrame({"timestamp": ts, "open_interest": [1000.0] * 5})
        dest = tmp_path / "binance" / "open_interest"
        dest.mkdir(parents=True)
        df.to_parquet(dest / "BTC_USDT.parquet", index=False)

        out = load_open_interest("BTC/USDT", data_dir=tmp_path)
        assert len(out) == 5


class TestLoadFearGreed:
    def test_roundtrip(self, tmp_path: Path) -> None:
        ts = pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        df = pd.DataFrame({"timestamp": ts, "value": [25, 50, 75, 100]})
        dest = tmp_path / "free"
        dest.mkdir(parents=True)
        df.to_parquet(dest / "fear_greed.parquet", index=False)

        out = load_fear_greed(data_dir=tmp_path)
        assert len(out) == 4
        assert out["value"].tolist() == [25, 50, 75, 100]

    def test_out_of_range_rejected(self, tmp_path: Path) -> None:
        ts = pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        # F&G must be in [0, 100]; 200 is a bug in the data source.
        df = pd.DataFrame({"timestamp": ts, "value": [50, 200]})
        dest = tmp_path / "free"
        dest.mkdir(parents=True)
        df.to_parquet(dest / "fear_greed.parquet", index=False)

        with pytest.raises(DataSchemaViolation):
            load_fear_greed(data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Split-aware helpers (compose loader + splits).
# ---------------------------------------------------------------------------


class TestSplitAwareLoaders:
    def test_load_train_slices_to_pre_train_end(self, data_dir_with_ohlcv: Path) -> None:
        splits = DataSplits(
            train_end=_utc("2024-01-01"),
            validation_end=_utc("2024-01-01 12:00"),
        )
        out = load_train(
            ["BTC/USDT"],
            "1h",
            data_dir=data_dir_with_ohlcv,
            splits=splits,
        )
        assert set(out.keys()) == {"BTC/USDT"}
        assert out["BTC/USDT"]["timestamp"].max() < splits.train_end

    def test_load_validation_slices_to_validation_zone(self, data_dir_with_ohlcv: Path) -> None:
        splits = DataSplits(
            train_end=_utc("2023-12-31"),
            validation_end=_utc("2024-01-01"),
        )
        out = load_validation(["BTC/USDT"], "1h", data_dir=data_dir_with_ohlcv, splits=splits)
        ts = out["BTC/USDT"]["timestamp"]
        assert ts.min() >= splits.train_end
        assert ts.max() < splits.validation_end

    def test_load_train_plus_validation_covers_both_zones(self, data_dir_with_ohlcv: Path) -> None:
        splits = DataSplits(
            train_end=_utc("2023-12-31"),
            validation_end=_utc("2024-01-01"),
        )
        out = load_train_plus_validation(
            ["BTC/USDT"],
            "1h",
            data_dir=data_dir_with_ohlcv,
            splits=splits,
        )
        assert out["BTC/USDT"]["timestamp"].max() < splits.validation_end
