"""Read parquet files from disk, validate them, and hand them to the engine.

This module is the only path by which disk bytes become DataFrames inside
the engine. Every load is paired with a Pandera schema; a file that fails
validation is surfaced as :class:`DataSchemaViolation` — never silently
returned. The upstream downloader (Phase 2) is expected to validate before
it writes, so a schema failure at load time usually indicates:

1. A file was edited or corrupted after download.
2. A schema was tightened since the file was written.
3. Something was dropped into ``data/`` by hand, outside the download
   pipeline.

In all three cases, refusing to proceed is safer than best-effort loading.

Public surface
--------------
Low-level loaders (one function per on-disk shape):

* :func:`load_ohlcv` — symbol + interval, e.g. ``("BTC/USDT", "1h")``.
* :func:`load_funding` — perp funding rate for a symbol.
* :func:`load_open_interest` — perp open interest for a symbol.
* :func:`load_fear_greed` — Alternative.me Fear & Greed daily index.
* :func:`load_index_value` — generic loader for any (timestamp, value)
  dataset (``btc_dominance``, ``stablecoin_mcap``, hashrate, macro, etc.).

Split-aware helpers (compose loader + :mod:`splits`):

* :func:`load_train` / :func:`load_validation` / :func:`load_train_plus_validation`.
* :func:`load_test` — gated behind ``reveal_test_set=True`` plus a
  ``reason``; delegates to :func:`split_test` so every reveal is logged.

Hashing
-------
:func:`compute_file_hash` returns ``"sha256:<hex>"`` for any file and is
used by the engine to populate ``BacktestResult.data_version`` (Phase 6),
so every backtest record is pinned to the exact bytes it ran on.

Directory layout
----------------
::

    <data_dir>/
    ├── binance/
    │   ├── ohlcv/<BASE>_<QUOTE>_<INTERVAL>.parquet
    │   ├── funding/<BASE>_<QUOTE>.parquet
    │   └── open_interest/<BASE>_<QUOTE>.parquet
    └── free/
        ├── fear_greed.parquet
        ├── btc_dominance.parquet
        ├── stablecoin_mcap.parquet
        ├── btc_active_addresses.parquet
        ├── btc_hashrate.parquet
        ├── dxy.parquet
        └── spy.parquet
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pandera.pandas as pa

from crypto_alpha_engine.data.schemas import (
    FearGreedSchema,
    FundingRateSchema,
    IndexValueSchema,
    OHLCVSchema,
    OpenInterestSchema,
)
from crypto_alpha_engine.data.splits import (
    DEFAULT_SPLITS,
    DataSplits,
    split_test,
    split_train,
    split_train_plus_validation,
    split_validation,
)
from crypto_alpha_engine.exceptions import DataSchemaViolation

if TYPE_CHECKING:
    pass

OHLCV_FILENAME_TEMPLATE = "{base}_{quote}_{interval}.parquet"
"""Filename template for OHLCV parquets. Filled from ``symbol.split('/')``."""

_DEFAULT_DATA_DIR = Path("./data")
_HASH_READ_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Return ``"sha256:<hex>"`` for the bytes at ``path``.

    Used to pin :attr:`BacktestResult.data_version` to the exact parquet(s)
    the engine ran against. Streams the file in chunks so multi-GB datasets
    don't blow up memory.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        A ``sha256:<64-hex-chars>`` string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_READ_CHUNK):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------


def _read_validated_parquet(path: Path, schema: type[pa.DataFrameModel]) -> pd.DataFrame:
    """Read a parquet, validate against the schema, wrap any failure.

    Pandera raises its own exception types on validation failure. We catch
    those and re-raise as :class:`DataSchemaViolation` so callers get a
    single, engine-native exception type to handle.
    """
    if not path.exists():
        raise FileNotFoundError(f"expected parquet at {path!s}")
    df = pd.read_parquet(path)
    try:
        schema.validate(df, lazy=False)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        raise DataSchemaViolation(
            f"{path.name} failed {schema.__name__}: {err}"
        ) from err
    return df


def load_ohlcv(
    symbol: str,
    interval: str,
    *,
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load an OHLCV parquet for ``symbol`` at the given ``interval``.

    Args:
        symbol: Exchange symbol in ``BASE/QUOTE`` form (e.g. ``"BTC/USDT"``).
        interval: Bar interval (``"1h"``, ``"1d"``, ...).
        data_dir: Root data directory. Defaults to ``./data``.

    Returns:
        Schema-validated DataFrame with columns
        ``[timestamp, open, high, low, close, volume]``.

    Raises:
        FileNotFoundError: If the expected parquet doesn't exist.
        DataSchemaViolation: If the parquet fails :class:`OHLCVSchema`.

    Example:
        >>> # df = load_ohlcv("BTC/USDT", "1h", data_dir=Path("./data"))
    """
    base, quote = symbol.split("/", 1)
    path = (
        data_dir
        / "binance"
        / "ohlcv"
        / OHLCV_FILENAME_TEMPLATE.format(base=base, quote=quote, interval=interval)
    )
    if not path.exists():
        raise FileNotFoundError(f"no OHLCV parquet for {symbol} {interval} at {path!s}")
    return _read_validated_parquet(path, OHLCVSchema)


def load_funding(symbol: str, *, data_dir: Path = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the perpetual-futures funding rate series for ``symbol``."""
    base, quote = symbol.split("/", 1)
    path = data_dir / "binance" / "funding" / f"{base}_{quote}.parquet"
    return _read_validated_parquet(path, FundingRateSchema)


def load_open_interest(symbol: str, *, data_dir: Path = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the perpetual-futures open-interest series for ``symbol``."""
    base, quote = symbol.split("/", 1)
    path = data_dir / "binance" / "open_interest" / f"{base}_{quote}.parquet"
    return _read_validated_parquet(path, OpenInterestSchema)


def load_fear_greed(*, data_dir: Path = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the Alternative.me Fear & Greed daily index."""
    path = data_dir / "free" / "fear_greed.parquet"
    return _read_validated_parquet(path, FearGreedSchema)


def load_index_value(name: str, *, data_dir: Path = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load a generic ``(timestamp, value)`` dataset by file name.

    Used for BTC dominance, stablecoin mcap, hashrate, active addresses,
    macro indices (DXY, SPY close), and any future dataset that fits the
    generic shape.

    Args:
        name: Filename stem (without ``.parquet``). Matches the free-source
            filenames documented in the module docstring.
        data_dir: Root data directory.
    """
    path = data_dir / "free" / f"{name}.parquet"
    return _read_validated_parquet(path, IndexValueSchema)


# ---------------------------------------------------------------------------
# Split-aware helpers
# ---------------------------------------------------------------------------


def _load_all_ohlcv(
    symbols: list[str], interval: str, *, data_dir: Path
) -> dict[str, pd.DataFrame]:
    return {sym: load_ohlcv(sym, interval, data_dir=data_dir) for sym in symbols}


def load_train(
    symbols: list[str],
    interval: str,
    *,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for each symbol, sliced to the train window.

    Args:
        symbols: List of exchange symbols (``["BTC/USDT", "ETH/USDT"]``).
        interval: Bar interval (``"1h"``, ``"1d"``, ...).
        data_dir: Root data directory.
        splits: Partition boundaries. Defaults to ``DEFAULT_SPLITS``.

    Returns:
        Dict mapping symbol to its train-zone DataFrame. Each DataFrame is
        schema-validated and has its index reset to ``0..n-1``.
    """
    raw = _load_all_ohlcv(symbols, interval, data_dir=data_dir)
    return {sym: split_train(df, splits=splits) for sym, df in raw.items()}


def load_validation(
    symbols: list[str],
    interval: str,
    *,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for each symbol, sliced to the validation window."""
    raw = _load_all_ohlcv(symbols, interval, data_dir=data_dir)
    return {sym: split_validation(df, splits=splits) for sym, df in raw.items()}


def load_train_plus_validation(
    symbols: list[str],
    interval: str,
    *,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV sliced to train ∪ validation — for final-model fit."""
    raw = _load_all_ohlcv(symbols, interval, data_dir=data_dir)
    return {sym: split_train_plus_validation(df, splits=splits) for sym, df in raw.items()}


def load_test(
    symbols: list[str],
    interval: str,
    *,
    reveal_test_set: bool,
    reason: str,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV sliced to the test window — gated by reveal flag.

    Delegates to :func:`split_test`, which raises if the flag is missing or
    the reason is empty, and logs every successful reveal.

    Raises:
        ConfigError: Passed through from :func:`split_test` if the reveal
            flag or reason is missing.
    """
    raw = _load_all_ohlcv(symbols, interval, data_dir=data_dir)
    return {
        sym: split_test(df, reveal_test_set=reveal_test_set, reason=reason, splits=splits)
        for sym, df in raw.items()
    }
