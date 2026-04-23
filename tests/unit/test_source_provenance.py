"""Provenance invariants: every published parquet identifies its source.

Two architectural rules from SPEC §5.1 are enforced here by canary tests:

1. Every parquet written via ``run_source`` carries a ``source_name``
   entry in its pyarrow file-level metadata.
2. The loader refuses to read a file whose metadata disagrees with the
   source the caller asked for — "sources must not share files".
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data import registry as registry_mod
from crypto_alpha_engine.data.downloader import (
    SOURCE_NAME_METADATA_KEY,
    canonical_path,
    read_source_name,
    run_source,
)
from crypto_alpha_engine.data.loader import load_by_source
from crypto_alpha_engine.data.protocol import DataType
from crypto_alpha_engine.exceptions import DataSchemaViolation


class _MockSentSource:
    """Minimal sentiment source with controllable name for provenance tests."""

    data_type = DataType.SENTIMENT
    symbols = ["BTC"]

    def __init__(self, name: str, offset: float = 0.0) -> None:
        self.name = name
        self._offset = offset

    def fetch(
        self,
        symbol: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        _ = symbol
        if end is None:
            end = start + pd.Timedelta("3h")
        ts = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC").astype(
            "datetime64[us, UTC]"
        )
        return pd.DataFrame(
            {"timestamp": ts, "value": [0.5 + self._offset + 0.01 * i for i in range(len(ts))]}
        )

    def earliest_available(self, symbol: str) -> pd.Timestamp:
        _ = symbol
        return pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()


class TestProvenanceMetadata:
    def test_parquet_carries_source_name_in_file_metadata(self, tmp_path: Path) -> None:
        src = _MockSentSource(name="prov_test_v1")
        registry_mod.register_source(src)
        out = run_source(
            src,
            "BTC",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )
        assert out is not None
        name = read_source_name(out)
        assert name == "prov_test_v1"

    def test_read_source_name_raw_key(self, tmp_path: Path) -> None:
        """Metadata key ``b"crypto_alpha_engine.source_name"`` is stable for external tools."""
        src = _MockSentSource(name="prov_test_raw")
        registry_mod.register_source(src)
        out = run_source(
            src,
            "BTC",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 03:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )
        assert out is not None

        import pyarrow.parquet as pq

        meta = pq.read_schema(out).metadata or {}  # type: ignore[no-untyped-call]
        assert meta.get(SOURCE_NAME_METADATA_KEY) == b"prov_test_raw"


class TestNoMixingSourcesInAFile:
    """SPEC §5.1: two sources must never share a file, even if hand-moved."""

    def test_loader_rejects_swapped_file(self, tmp_path: Path) -> None:
        """If someone renames source-A's parquet into source-B's directory,
        the loader catches the mismatch via file metadata."""
        src_a = _MockSentSource(name="src_alpha", offset=0.0)
        src_b = _MockSentSource(name="src_beta", offset=0.2)
        registry_mod.register_source(src_a)
        registry_mod.register_source(src_b)

        # Only src_a has actually downloaded.
        out = run_source(
            src_a,
            "BTC",
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 03:00", tz="UTC"),
            freq="1h",
            data_dir=tmp_path,
        )
        assert out is not None

        # Hand-copy src_a's file into src_b's canonical path — the metadata
        # still identifies source as "src_alpha", so loading as "src_beta"
        # must fail.
        dest = canonical_path(tmp_path, DataType.SENTIMENT, src_b.name, "BTC", "1h")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dest)

        with pytest.raises(DataSchemaViolation, match="provenance mismatch"):
            load_by_source(
                DataType.SENTIMENT,
                "BTC",
                source="src_beta",
                data_dir=tmp_path,
            )

    def test_canonical_paths_differ_per_source(self, tmp_path: Path) -> None:
        """Same (data_type, symbol, freq) but different sources → different paths."""
        p1 = canonical_path(tmp_path, DataType.OHLCV, "coinbase_spot", "BTC/USD", "1h")
        p2 = canonical_path(tmp_path, DataType.OHLCV, "kraken_spot", "BTC/USD", "1h")
        assert p1 != p2
        assert p1.parent != p2.parent
