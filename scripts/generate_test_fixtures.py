"""Slice small real-data parquets into ``tests/fixtures/``.

CLAUDE.md requires tests to use real market data for anything that
exercises market behavior. The full download (``data/``) is gitignored
and lives outside the repo; we slice a short window and commit the
slices.

Window: **January 2022**, ~720 hourly rows. Small enough to fit in the
repo (~50 KB per parquet), chaotic enough to exercise schema validation,
quiet enough to avoid 2021/2022 shock regimes that might not match the
behaviour a Phase-3 operator expects.

Usage::

    uv run python scripts/download_all_data.py
    uv run python scripts/generate_test_fixtures.py

Outputs (each carries ``source_name`` in its parquet metadata, same as
the full downloads):

* ``tests/fixtures/btc_usd_1h_coinbase_spot.parquet``
* ``tests/fixtures/eth_usd_1h_coinbase_spot.parquet``
* ``tests/fixtures/btc_funding_8h_bitmex_perp.parquet``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Import the sources package so built-ins auto-register.
import crypto_alpha_engine.data.sources as _sources_pkg  # noqa: F401
from crypto_alpha_engine.data.downloader import canonical_path
from crypto_alpha_engine.data.protocol import DataType

_WINDOW_START = pd.Timestamp("2022-01-01", tz="UTC")
_WINDOW_END = pd.Timestamp("2022-02-01", tz="UTC")


@dataclass(frozen=True)
class _Slice:
    out_filename: str
    data_type: DataType
    source: str
    symbol: str
    freq: str


_PLAN: list[_Slice] = [
    _Slice(
        out_filename="btc_usd_1h_coinbase_spot.parquet",
        data_type=DataType.OHLCV,
        source="coinbase_spot",
        symbol="BTC/USD",
        freq="1h",
    ),
    _Slice(
        out_filename="eth_usd_1h_coinbase_spot.parquet",
        data_type=DataType.OHLCV,
        source="coinbase_spot",
        symbol="ETH/USD",
        freq="1h",
    ),
    _Slice(
        out_filename="btc_funding_8h_bitmex_perp.parquet",
        data_type=DataType.FUNDING,
        source="bitmex_perp",
        symbol="BTC/USD:BTC",
        freq="8h",
    ),
]


def _slice_window(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["timestamp"] >= _WINDOW_START) & (df["timestamp"] < _WINDOW_END)
    return df.loc[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="root of the downloaded dataset tree",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="output directory for sliced fixtures",
    )
    args = parser.parse_args(argv)
    data_dir: Path = args.data_dir
    fixtures_dir: Path = args.fixtures_dir
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for s in _PLAN:
        src_path = canonical_path(data_dir, s.data_type, s.source, s.symbol, s.freq)
        if not src_path.exists():
            print(f"[SKIP] {s.out_filename}: upstream missing at {src_path!s}")
            all_ok = False
            continue
        full = pd.read_parquet(src_path)
        sliced = _slice_window(full)
        if sliced.empty:
            print(f"[SKIP] {s.out_filename}: no rows in " f"[{_WINDOW_START} .. {_WINDOW_END})")
            all_ok = False
            continue
        out_path = fixtures_dir / s.out_filename
        sliced.to_parquet(out_path, index=False)
        print(f"[OK]   {s.out_filename}  rows={len(sliced)}  path={out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
