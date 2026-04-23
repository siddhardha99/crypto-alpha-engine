"""Tests for loader helpers not covered by the source-specific tests.

The main load_ohlcv / load_by_source behaviors are exercised by:

* ``test_coinbase_spot.py`` — load_ohlcv via the built-in source
* ``test_custom_source_registration.py`` — load_by_source end-to-end
* ``test_source_provenance.py`` — metadata mismatch detection

This file covers the file-hash utility and split-aware helpers that sit
on top of load_ohlcv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data import registry as registry_mod
from crypto_alpha_engine.data.downloader import run_source
from crypto_alpha_engine.data.loader import (
    compute_file_hash,
    load_train,
    load_train_plus_validation,
    load_validation,
)
from crypto_alpha_engine.data.protocol import DataType
from crypto_alpha_engine.data.splits import DataSplits

# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_is_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello world")
        assert compute_file_hash(f) == compute_file_hash(f)

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
# Split-aware loaders — compose load_ohlcv with split_* slicers.
# ---------------------------------------------------------------------------


class _MockOhlcvSource:
    """OHLCV source that spans 2023-12-30 → 2024-01-01 for split-boundary testing."""

    name = "tso_mock_ohlcv"
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
        _ = symbol, start, end, freq
        ts = pd.date_range("2023-12-30 00:00:00", periods=48, freq="h", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 105.0,
                "volume": 1000.0,
            }
        )

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2020-01-01", tz="UTC")


@pytest.fixture(autouse=True)
def _registry() -> Any:
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()


@pytest.fixture
def seeded_data_dir(tmp_path: Path) -> Path:
    src = _MockOhlcvSource()
    registry_mod.register_source(src)
    run_source(
        src,
        "BTC/USD",
        start=pd.Timestamp("2023-12-30", tz="UTC"),
        end=pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        freq="1h",
        data_dir=tmp_path,
    )
    # The fetched frame spans [2023-12-30, 2024-01-01] and doesn't depend
    # on the (start, end) window — the mock is window-insensitive.
    return tmp_path


class TestSplitAwareLoaders:
    def test_load_train_slices_to_pre_train_end(self, seeded_data_dir: Path) -> None:
        splits = DataSplits(
            train_end=pd.Timestamp("2024-01-01", tz="UTC"),
            validation_end=pd.Timestamp("2024-01-01 12:00", tz="UTC"),
        )
        out = load_train(
            ["BTC/USD"],
            "1h",
            data_dir=seeded_data_dir,
            splits=splits,
        )
        assert set(out.keys()) == {"BTC/USD"}
        assert out["BTC/USD"]["timestamp"].max() < splits.train_end

    def test_load_validation_slices_to_validation_zone(self, seeded_data_dir: Path) -> None:
        splits = DataSplits(
            train_end=pd.Timestamp("2023-12-31", tz="UTC"),
            validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        out = load_validation(["BTC/USD"], "1h", data_dir=seeded_data_dir, splits=splits)
        ts = out["BTC/USD"]["timestamp"]
        assert ts.min() >= splits.train_end
        assert ts.max() < splits.validation_end

    def test_load_train_plus_validation_covers_both_zones(self, seeded_data_dir: Path) -> None:
        splits = DataSplits(
            train_end=pd.Timestamp("2023-12-31", tz="UTC"),
            validation_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        out = load_train_plus_validation(
            ["BTC/USD"],
            "1h",
            data_dir=seeded_data_dir,
            splits=splits,
        )
        assert out["BTC/USD"]["timestamp"].max() < splits.validation_end
