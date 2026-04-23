"""Produce small real-data parquet fixtures for tests/fixtures/.

CLAUDE.md requires tests to use *real* market data, not synthetic series,
for anything that exercises market behavior. The full download lives
outside the repo (gitignored), so we slice a small window of real bars
and commit it.

Current fixture slice: **January 2022**, which is a relatively calm
month and avoids the 2021 bull-top aftermath and 2022 LUNA/FTX shocks —
fine for schema tests, split tests, and early operator tests. Any
behaviour test that needs a volatile regime will pull its own window.

Usage::

    uv run python scripts/download_all_data.py --data-dir ./data
    uv run python scripts/generate_test_fixtures.py \
        --data-dir ./data --fixtures-dir tests/fixtures

Generated files (~100 KB each):

* ``tests/fixtures/btc_1h_sample.parquet``
* ``tests/fixtures/eth_1h_sample.parquet``
* ``tests/fixtures/funding_sample.parquet``  (BTC perp)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from crypto_alpha_engine.data.loader import (
    load_funding,
    load_ohlcv,
)

_WINDOW_START = pd.Timestamp("2022-01-01", tz="UTC")
_WINDOW_END = pd.Timestamp("2022-02-01", tz="UTC")


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

    def _btc_1h() -> pd.DataFrame:
        return load_ohlcv("BTC/USDT", "1h", data_dir=data_dir)

    def _eth_1h() -> pd.DataFrame:
        return load_ohlcv("ETH/USDT", "1h", data_dir=data_dir)

    def _funding() -> pd.DataFrame:
        return load_funding("BTC/USDT", data_dir=data_dir)

    plan: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("btc_1h_sample.parquet", _btc_1h),
        ("eth_1h_sample.parquet", _eth_1h),
        ("funding_sample.parquet", _funding),
    ]

    all_ok = True
    for filename, loader in plan:
        try:
            full = loader()
        except FileNotFoundError as err:
            print(f"[SKIP] {filename}: upstream data missing ({err})")
            all_ok = False
            continue
        sliced = _slice_window(full)
        if sliced.empty:
            print(f"[SKIP] {filename}: no rows in window [{_WINDOW_START} .. {_WINDOW_END})")
            all_ok = False
            continue
        out = fixtures_dir / filename
        sliced.to_parquet(out, index=False)
        print(f"[OK]   {filename}  rows={len(sliced)}  path={out}")

    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
