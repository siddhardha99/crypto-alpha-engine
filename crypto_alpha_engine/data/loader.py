"""Read parquet files from disk, validate them, and hand them to the engine.

Every load goes through a source-aware path: either the caller names the
source explicitly (``source="coinbase_spot"``) or the loader asks the
registry for the default source for a ``(data_type, symbol)`` pair. The
underlying file is always at the canonical location documented in
:mod:`crypto_alpha_engine.data.downloader`.

On every read:

1. The parquet's file-level metadata is checked; ``source_name`` must
   match the requested source. A mismatch means the file was
   hand-moved or tampered with, and raises
   :class:`DataSchemaViolation`.
2. The DataFrame is validated against the canonical schema for the
   source's ``data_type``. Any failure re-raises as
   :class:`DataSchemaViolation`.

If either check fails, the engine refuses the read rather than silently
proceeding on bad data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from crypto_alpha_engine.data.downloader import (
    canonical_path,
    read_source_name,
)
from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataType
from crypto_alpha_engine.data.registry import default_for, get_source
from crypto_alpha_engine.data.splits import (
    DEFAULT_SPLITS,
    DataSplits,
    split_test,
    split_train,
    split_train_plus_validation,
    split_validation,
)
from crypto_alpha_engine.exceptions import DataSchemaViolation

_DEFAULT_DATA_DIR = Path("./data")
_HASH_READ_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Hashing (unchanged)
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Return ``"sha256:<hex>"`` for the bytes at ``path``.

    Streams the file in chunks so multi-GB datasets don't blow up memory.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_READ_CHUNK):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


# ---------------------------------------------------------------------------
# Source-aware reader
# ---------------------------------------------------------------------------


def _read_and_validate(
    path: Path,
    *,
    schema: type[pa.DataFrameModel],
    expected_source_name: str,
) -> pd.DataFrame:
    """Read parquet at ``path``, verify provenance, validate schema.

    Raises:
        FileNotFoundError: If the file is missing.
        DataSchemaViolation: If the provenance metadata disagrees with
            ``expected_source_name`` or the Pandera schema fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"expected parquet at {path!s}")

    actual_source = read_source_name(path)
    if actual_source is not None and actual_source != expected_source_name:
        raise DataSchemaViolation(
            f"provenance mismatch at {path.name}: file metadata says "
            f"source_name={actual_source!r}, loader asked for "
            f"{expected_source_name!r}. Sources must not share files — "
            "see SPEC §5.1."
        )

    df = pd.read_parquet(path)
    try:
        schema.validate(df, lazy=False)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        raise DataSchemaViolation(f"{path.name} failed {schema.__name__}: {err}") from err
    return df


def load_by_source(
    data_type: DataType,
    symbol: str,
    *,
    freq: str = "1h",
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Generic loader — resolves source via registry and reads its canonical file.

    Args:
        data_type: The :class:`DataType` to load.
        symbol: Symbol string as registered with the source.
        freq: Bar frequency string used to compute the filename.
        source: Source name. If ``None``, uses
            :func:`registry.default_for(data_type, symbol)`.
        data_dir: Root data directory.

    Returns:
        Schema-validated DataFrame.

    Raises:
        ConfigError: Passed through if no source matches.
        FileNotFoundError: If the target parquet doesn't exist.
        DataSchemaViolation: Provenance or schema failure.
    """
    resolved = get_source(source) if source is not None else default_for(data_type, symbol)
    path = canonical_path(data_dir, data_type, resolved.name, symbol, freq)
    schema = CANONICAL_SCHEMAS[data_type]
    return _read_and_validate(path, schema=schema, expected_source_name=resolved.name)


def load_ohlcv(
    symbol: str,
    interval: str,
    *,
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load an OHLCV parquet for ``symbol`` at ``interval``.

    Args:
        symbol: Exchange symbol (``"BTC/USD"``, ``"ETH/USD"``, ...).
        interval: Bar interval (``"1h"``, ``"1d"``).
        source: Optional source name (``"coinbase_spot"``, ...). If
            ``None``, the registry's default OHLCV source for ``symbol``
            is used.
        data_dir: Root data directory.

    Returns:
        Schema-validated DataFrame with the standard OHLCV columns.

    Raises:
        ConfigError: If no source serves ``(OHLCV, symbol)``.
        FileNotFoundError: If the target parquet doesn't exist.
        DataSchemaViolation: Provenance or schema failure.
    """
    return load_by_source(
        DataType.OHLCV,
        symbol,
        freq=interval,
        source=source,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# Split-aware helpers
# ---------------------------------------------------------------------------


def _load_all_ohlcv(
    symbols: list[str],
    interval: str,
    *,
    source: str | None,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    return {sym: load_ohlcv(sym, interval, source=source, data_dir=data_dir) for sym in symbols}


def load_train(
    symbols: list[str],
    interval: str,
    *,
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for each symbol, sliced to the train window."""
    raw = _load_all_ohlcv(symbols, interval, source=source, data_dir=data_dir)
    return {sym: split_train(df, splits=splits) for sym, df in raw.items()}


def load_validation(
    symbols: list[str],
    interval: str,
    *,
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for each symbol, sliced to the validation window."""
    raw = _load_all_ohlcv(symbols, interval, source=source, data_dir=data_dir)
    return {sym: split_validation(df, splits=splits) for sym, df in raw.items()}


def load_train_plus_validation(
    symbols: list[str],
    interval: str,
    *,
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV sliced to train + validation — for final-model fit."""
    raw = _load_all_ohlcv(symbols, interval, source=source, data_dir=data_dir)
    return {sym: split_train_plus_validation(df, splits=splits) for sym, df in raw.items()}


def load_test(
    symbols: list[str],
    interval: str,
    *,
    reveal_test_set: bool,
    reason: str,
    source: str | None = None,
    data_dir: Path = _DEFAULT_DATA_DIR,
    splits: DataSplits = DEFAULT_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV sliced to the test window — gated by reveal flag."""
    raw = _load_all_ohlcv(symbols, interval, source=source, data_dir=data_dir)
    return {
        sym: split_test(df, reveal_test_set=reveal_test_set, reason=reason, splits=splits)
        for sym, df in raw.items()
    }
