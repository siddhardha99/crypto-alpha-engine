"""Source-orchestrated download pipeline.

The public entry point is :func:`run_source`. Callers pass a
:class:`DataSource` (see :mod:`crypto_alpha_engine.data.protocol`) and a
(symbol, start, end, freq) window; this module owns the rest:

1. **Resume point** — reads the existing parquet at
   ``<data_dir>/<data_type>/<source_name>/<symbol>_<freq>.parquet`` and
   computes the next timestamp to fetch (Principle 8 — idempotent).
2. **Short-circuit** — if the existing file already covers the requested
   window, returns without calling the source.
3. **Fetch** — delegates to ``source.fetch()``, which is responsible for
   pagination, retries, and shaping output to the canonical schema for
   the source's ``data_type``.
4. **Merge + dedup** — appends new rows to existing, dedupes on
   ``timestamp``.
5. **Validate + publish** — tempfile → round-trip parquet → Pandera →
   atomic rename to the final path. On failure: quarantine the tempfile
   with an ``.error.txt`` sidecar and re-raise
   :class:`DataSchemaViolation`.
6. **Provenance** — every published parquet carries a ``source_name``
   entry in its pyarrow file-level metadata. The loader verifies on read;
   a mismatch raises :class:`DataSchemaViolation`.

Helpers exported for source classes
-----------------------------------

Sources that use ccxt for pagination can reuse:

* :func:`paginated_ccxt_ohlcv` — for OHLCV bars.
* :func:`paginated_ccxt_funding` — for funding records.
* :func:`paginated_ccxt_oi` — for open-interest records.
* :func:`bars_to_ohlcv_df`, :func:`funding_records_to_df`,
  :func:`oi_records_to_df` — record → DataFrame with canonical dtypes.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pandera.pandas as pa
import pyarrow as pyarrow_pa
import pyarrow.parquet as pq
import structlog

from crypto_alpha_engine.data.protocol import CANONICAL_SCHEMAS, DataSource, DataType
from crypto_alpha_engine.exceptions import DataSchemaViolation

_logger = structlog.get_logger(__name__)

QUARANTINE_DIR_NAME = "quarantine"
"""Subdirectory of ``data_dir`` that holds files which failed validation."""

SOURCE_NAME_METADATA_KEY = b"crypto_alpha_engine.source_name"
"""Parquet file-level metadata key for source provenance."""

_MAX_PAGES = 10_000  # safety: ~10M rows per call
_DEFAULT_LIMIT = 1000


# ---------------------------------------------------------------------------
# ccxt Protocol (subset the shared helpers need)
# ---------------------------------------------------------------------------


class _CcxtExchange(Protocol):
    """Subset of ``ccxt.Exchange`` the paginated helpers use."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = ...,
        limit: int | None = ...,
    ) -> list[list[float]]: ...

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = ...,
        limit: int | None = ...,
    ) -> list[dict[str, Any]]: ...

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = ...,
        limit: int | None = ...,
    ) -> list[dict[str, Any]]: ...

    def parse_timeframe(self, timeframe: str) -> int: ...


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def run_source(
    source: DataSource,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    freq: str = "1h",
    data_dir: Path,
) -> Path | None:
    """Idempotent download through ``source`` into the canonical path.

    Args:
        source: A registered :class:`DataSource`.
        symbol: The symbol to fetch (must be in ``source.symbols``).
        start: Earliest timestamp of interest (UTC-aware). Only used on
            fresh runs; subsequent runs compute their own ``since`` from
            the existing parquet.
        end: Optional exclusive upper bound (UTC-aware). ``None`` means
            "fetch up to whatever the source serves now."
        freq: Bar frequency — OHLCV-style strings like ``"1h"``, or
            source-specific labels (``"8h"`` for funding, ``"1d"`` for
            daily indices).
        data_dir: Root data directory.

    Returns:
        Path of the published parquet, or ``None`` if the source returned
        nothing and no file previously existed.

    Raises:
        DataSchemaViolation: Combined (existing + fetched) frame fails
            the canonical schema for the source's ``data_type``. The
            offending frame is quarantined and the existing final file
            is left unchanged.
        ValueError: Programmer error (naive ``start``/``end``, symbol not
            in ``source.symbols``).

    Example:
        >>> # from crypto_alpha_engine.data.registry import get_source
        >>> # out = run_source(
        >>> #     get_source("coinbase_spot"),
        >>> #     "BTC/USD",
        >>> #     start=pd.Timestamp("2024-01-01", tz="UTC"),
        >>> #     data_dir=Path("./data"),
        >>> # )
    """
    _require_utc(start, "start")
    if end is not None:
        _require_utc(end, "end")
    if symbol not in source.symbols:
        raise ValueError(
            f"symbol {symbol!r} is not in source {source.name!r} "
            f"supported list {source.symbols!r}"
        )

    final_path = canonical_path(data_dir, source.data_type, source.name, symbol, freq)
    existing = _read_existing_or_none(final_path)
    since = _compute_since(existing, start, freq)

    end_ms = _timestamp_to_ms(end) if end is not None else None
    since_ms = _timestamp_to_ms(since)
    if existing is not None and end_ms is not None and since_ms >= end_ms:
        _logger.info(
            "run_source_up_to_date",
            source=source.name,
            symbol=symbol,
            rows=len(existing),
        )
        return final_path

    new_df = source.fetch(symbol, start=since, end=end, freq=freq)

    if new_df.empty and existing is None:
        _logger.info("run_source_empty_response", source=source.name, symbol=symbol)
        return None

    combined = _concat_dedup(existing, new_df)
    schema = CANONICAL_SCHEMAS[source.data_type]
    _validate_then_publish(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=schema,
        source_name=source.name,
    )
    _logger.info(
        "run_source_download_complete",
        source=source.name,
        symbol=symbol,
        new_rows=len(new_df),
        total_rows=len(combined),
        path=str(final_path),
    )
    return final_path


def canonical_path(
    data_dir: Path,
    data_type: DataType,
    source_name: str,
    symbol: str,
    freq: str,
) -> Path:
    """Compute the canonical on-disk path for a (source, symbol, freq) tuple.

    Layout: ``<data_dir>/<data_type>/<source_name>/<symbol_slug>_<freq>.parquet``.
    ``symbol_slug`` replaces ``/`` and ``:`` with ``_``, so
    ``BTC/USD:BTC`` becomes ``BTC_USD_BTC``.
    """
    slug = symbol.replace("/", "_").replace(":", "_")
    return data_dir / data_type.value / source_name / f"{slug}_{freq}.parquet"


# ---------------------------------------------------------------------------
# Shared primitives (used by run_source and by source classes)
# ---------------------------------------------------------------------------


def _require_utc(ts: pd.Timestamp, name: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be UTC-aware; got naive {ts!r}")


def _timestamp_to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _read_existing_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = df["timestamp"].astype("datetime64[us, UTC]")
    return df


def read_source_name(path: Path) -> str | None:
    """Read the ``source_name`` embedded in a parquet's file metadata.

    Returns ``None`` if the file was written without the metadata (legacy
    or hand-authored). The loader uses this for the "no mixing sources"
    provenance check.
    """
    if not path.exists():
        return None
    meta = pq.read_schema(path).metadata or {}  # type: ignore[no-untyped-call]
    raw = meta.get(SOURCE_NAME_METADATA_KEY)
    return raw.decode("utf-8") if raw is not None else None


def _compute_since(existing: pd.DataFrame | None, start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Resume strictly after the last existing row, else return ``start``.

    The step is parsed from ``freq``; unknown freqs default to 1h.
    """
    if existing is None or existing.empty:
        return start
    last = existing["timestamp"].max()
    step = _freq_to_timedelta(freq)
    return pd.Timestamp(last + step)


def _freq_to_timedelta(freq: str) -> pd.Timedelta:
    """Best-effort freq parser for the resume step."""
    try:
        return pd.Timedelta(freq)
    except ValueError:
        return pd.Timedelta("1h")


def _concat_dedup(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = new
    elif new.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    return combined.sort_values("timestamp", ignore_index=True)


def _validate_then_publish(
    *,
    df: pd.DataFrame,
    final_path: Path,
    data_dir: Path,
    schema: type[pa.DataFrameModel],
    source_name: str,
) -> None:
    """Validate, atomically publish with provenance metadata, or quarantine."""
    final_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{final_path.stem}_",
        suffix=".parquet",
        dir=final_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    # Write with provenance metadata embedded in the parquet schema so
    # the loader can verify "this file came from <source_name>" on read.
    table = pyarrow_pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, SOURCE_NAME_METADATA_KEY: source_name.encode("utf-8")}
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, tmp_path)  # type: ignore[no-untyped-call]

    try:
        round_tripped = pd.read_parquet(tmp_path)
        schema.validate(round_tripped, lazy=False)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        _quarantine(tmp_path=tmp_path, final_name=final_path.name, data_dir=data_dir, err=err)
        raise DataSchemaViolation(
            f"{final_path.name} failed {schema.__name__} on download: {err}"
        ) from err
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(final_path)


def _quarantine(
    *,
    tmp_path: Path,
    final_name: str,
    data_dir: Path,
    err: Exception,
) -> None:
    quarantine_dir = data_dir / QUARANTINE_DIR_NAME
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    q_parquet = quarantine_dir / f"{stamp}_{final_name}"
    q_error = quarantine_dir / f"{stamp}_{final_name}.error.txt"
    tmp_path.replace(q_parquet)
    q_error.write_text(
        "Validation failed on download.\n"
        f"Timestamp: {stamp}\n"
        f"Target: {final_name}\n"
        f"Error type: {type(err).__name__}\n"
        f"Error:\n{err}\n",
        encoding="utf-8",
    )
    _logger.error(
        "download_quarantined",
        parquet=str(q_parquet),
        error_file=str(q_error),
        error_type=type(err).__name__,
    )


# ---------------------------------------------------------------------------
# ccxt-backed pagination helpers
# ---------------------------------------------------------------------------


def paginated_ccxt_ohlcv(
    exchange: _CcxtExchange,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    end_ms: int | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[list[float]]:
    """Page through ``exchange.fetch_ohlcv`` until exhausted or past ``end_ms``.

    Termination is by empty-response or stuck-cursor, NOT by
    ``len(page) < limit``. Some exchanges (Coinbase, notably) return
    partial pages mid-history when data is sparse, and using "partial
    page" as an end-of-history signal causes silent truncation.
    """
    out: list[list[float]] = []
    cursor = since_ms
    for _ in range(_MAX_PAGES):
        page = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not page:
            break
        if end_ms is not None:
            page = [b for b in page if b[0] < end_ms]
        if not page:
            break
        out.extend(page)
        next_cursor = int(page[-1][0]) + 1
        if next_cursor <= cursor:
            # Cursor didn't advance — defensive guard against an API
            # echoing the same page indefinitely.
            break
        cursor = next_cursor
        if end_ms is not None and cursor >= end_ms:
            break
    return out


def paginated_ccxt_funding(
    exchange: _CcxtExchange,
    symbol: str,
    *,
    since_ms: int,
    end_ms: int | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """See :func:`paginated_ccxt_ohlcv` for termination semantics."""
    out: list[dict[str, Any]] = []
    cursor = since_ms
    for _ in range(_MAX_PAGES):
        page = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=limit)
        if not page:
            break
        if end_ms is not None:
            page = [r for r in page if int(r["timestamp"]) < end_ms]
        if not page:
            break
        out.extend(page)
        next_cursor = int(page[-1]["timestamp"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if end_ms is not None and cursor >= end_ms:
            break
    return out


def paginated_ccxt_oi(
    exchange: _CcxtExchange,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    end_ms: int | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """See :func:`paginated_ccxt_ohlcv` for termination semantics."""
    out: list[dict[str, Any]] = []
    cursor = since_ms
    for _ in range(_MAX_PAGES):
        page = exchange.fetch_open_interest_history(symbol, timeframe, since=cursor, limit=limit)
        if not page:
            break
        if end_ms is not None:
            page = [r for r in page if int(r["timestamp"]) < end_ms]
        if not page:
            break
        out.extend(page)
        next_cursor = int(page[-1]["timestamp"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if end_ms is not None and cursor >= end_ms:
            break
    return out


# ---------------------------------------------------------------------------
# Record → DataFrame shapers (canonical dtype coercion)
# ---------------------------------------------------------------------------


def bars_to_ohlcv_df(bars: list[list[float]]) -> pd.DataFrame:
    """Shape ccxt OHLCV bar-lists into the canonical OHLCV DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(
        "datetime64[us, UTC]"
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def funding_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Shape ccxt funding records into the canonical (timestamp, funding_rate) df."""
    if not records:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(r["timestamp"]) for r in records], unit="ms", utc=True
            ).astype("datetime64[us, UTC]"),
            "funding_rate": [float(r["fundingRate"]) for r in records],
        }
    )


def oi_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Shape ccxt OI records into the canonical (timestamp, open_interest) df."""
    if not records:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(r["timestamp"]) for r in records], unit="ms", utc=True
            ).astype("datetime64[us, UTC]"),
            "open_interest": [float(r["openInterestAmount"]) for r in records],
        }
    )
