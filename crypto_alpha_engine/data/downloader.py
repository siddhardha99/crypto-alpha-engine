"""Idempotent Binance downloaders for OHLCV, funding, and open interest.

Principle 8 ("idempotent data layer") and the user's Phase 2 asks drive
everything here:

1. **No re-fetch.** If the target parquet exists, the downloader reads the
   max timestamp on disk and asks the exchange only for bars strictly after
   that. Running :func:`download_ohlcv` ten times in a row on unchanged
   data produces one real network call on the first run and zero on the
   next nine.

2. **Append-only on success.** New bars are appended to the existing
   parquet in chronological order, with a final ``drop_duplicates`` on
   timestamp so any overlap returned by the exchange is resolved
   deterministically (first copy wins).

3. **Validate before publishing.** Every downloaded frame is validated
   against its Pandera schema *before* it lands in the canonical
   destination. On failure, the frame is written to
   ``<data_dir>/quarantine/<stamp>_<name>.parquet`` with a sidecar
   ``.error.txt`` describing the violation, and a
   :class:`DataSchemaViolation` is raised.

4. **Injectable exchange.** Every downloader accepts an ``exchange``
   argument (any object that quacks like ``ccxt.binance()``). Unit tests
   pass a fake; production uses :func:`make_binance_spot` /
   :func:`make_binance_perp`.

Three downloaders ship here: :func:`download_ohlcv`,
:func:`download_funding`, and :func:`download_open_interest`. They share
the publish / validate / quarantine machinery via
:func:`_validate_then_publish` and only differ in how they page through
the exchange and how they shape records into a DataFrame.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
import pandera.pandas as pa
import structlog

from crypto_alpha_engine.data.loader import OHLCV_FILENAME_TEMPLATE
from crypto_alpha_engine.data.schemas import (
    FundingRateSchema,
    OHLCVSchema,
    OpenInterestSchema,
)
from crypto_alpha_engine.exceptions import DataSchemaViolation

if TYPE_CHECKING:
    pass

_logger = structlog.get_logger(__name__)

QUARANTINE_DIR_NAME = "quarantine"
"""Subdirectory of ``data_dir`` that holds files which failed validation."""

_MAX_PAGES = 10_000  # hard cap: ~10M bars, safety net against runaway loops
_DEFAULT_LIMIT = 1000  # ccxt's typical per-call max for Binance


# ---------------------------------------------------------------------------
# Exchange Protocol
# ---------------------------------------------------------------------------


class _ExchangeProtocol(Protocol):
    """The subset of ``ccxt.Exchange`` this module needs.

    Each downloader uses a different subset: OHLCV uses
    :meth:`fetch_ohlcv`; funding uses :meth:`fetch_funding_rate_history`;
    open interest uses :meth:`fetch_open_interest_history`. All three
    use :meth:`parse_timeframe`.
    """

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
# Public entrypoint
# ---------------------------------------------------------------------------


def download_ohlcv(
    symbol: str,
    interval: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    data_dir: Path,
    exchange: _ExchangeProtocol,
    limit: int = _DEFAULT_LIMIT,
) -> Path | None:
    """Idempotently fetch OHLCV bars for ``symbol`` over ``[start, end]``.

    On first run, fetches the full range from the exchange and writes a new
    parquet. On subsequent runs, reads the existing file, resumes from the
    last on-disk bar, and appends only new data.

    Args:
        symbol: Exchange symbol in ``BASE/QUOTE`` form (e.g. ``"BTC/USDT"``).
        interval: Bar interval (``"1h"`` or ``"1d"`` in Phase 2).
        start: Earliest timestamp to fetch (UTC-aware). Only used on fresh
            downloads; resume runs compute their own ``since`` from the
            existing file.
        end: Optional exclusive upper bound. If ``None``, fetch up to the
            present. Must be UTC-aware when provided.
        data_dir: Root data directory; canonical location is
            ``<data_dir>/binance/ohlcv/<BASE>_<QUOTE>_<INTERVAL>.parquet``.
        exchange: Object conforming to :class:`_ExchangeProtocol` — in
            practice, ``ccxt.binance()`` or a test double.
        limit: Maximum bars to request per exchange call.

    Returns:
        Path to the canonical parquet on success, or ``None`` if the
        exchange returned no bars and no file previously existed.

    Raises:
        DataSchemaViolation: If the combined (existing + fetched) frame
            fails :class:`OHLCVSchema`. The offending frame is quarantined
            and the final file is left unchanged.

    Example:
        >>> # import ccxt
        >>> # exchange = ccxt.binance()
        >>> # download_ohlcv("BTC/USDT", "1h",
        >>> #                 start=pd.Timestamp("2023-01-01", tz="UTC"),
        >>> #                 data_dir=Path("./data"),
        >>> #                 exchange=exchange)
    """
    _require_utc(start, "start")
    if end is not None:
        _require_utc(end, "end")

    base, quote = symbol.split("/", 1)
    final_path = (
        data_dir
        / "binance"
        / "ohlcv"
        / OHLCV_FILENAME_TEMPLATE.format(base=base, quote=quote, interval=interval)
    )

    existing = _read_existing_or_none(final_path)
    since_ms = _resume_since_ms(existing, start, interval, exchange)
    end_ms = _timestamp_to_ms(end) if end is not None else None

    if existing is not None and end_ms is not None and since_ms >= end_ms:
        # The next bar we would need already sits at or past the requested
        # end — existing file is complete. Short-circuit the network call.
        _logger.info(
            "ohlcv_up_to_date",
            symbol=symbol,
            interval=interval,
            rows=len(existing),
        )
        return final_path

    fetched = _fetch_paginated(
        exchange=exchange,
        symbol=symbol,
        timeframe=interval,
        since_ms=since_ms,
        end_ms=end_ms,
        limit=limit,
    )

    if not fetched and existing is None:
        _logger.info("ohlcv_empty_response", symbol=symbol, interval=interval)
        return None

    new_df = _bars_to_df(fetched)
    combined = _concat_dedup(existing, new_df)

    _validate_then_publish(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=OHLCVSchema,
    )
    _logger.info(
        "ohlcv_download_complete",
        symbol=symbol,
        interval=interval,
        new_rows=len(new_df),
        total_rows=len(combined),
        path=str(final_path),
    )
    return final_path


# ---------------------------------------------------------------------------
# Internal helpers
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
    # Ensure canonical dtype on resume; silently migrates older files.
    df["timestamp"] = df["timestamp"].astype("datetime64[us, UTC]")
    return df


def _resume_since_ms(
    existing: pd.DataFrame | None,
    start: pd.Timestamp,
    interval: str,
    exchange: _ExchangeProtocol,
) -> int:
    """Resume strictly after the last existing bar, else use ``start``."""
    if existing is None or existing.empty:
        return _timestamp_to_ms(start)
    interval_ms = exchange.parse_timeframe(interval) * 1000
    last_ms = int(existing["timestamp"].max().timestamp() * 1000)
    return last_ms + interval_ms


def _fetch_paginated(
    *,
    exchange: _ExchangeProtocol,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int | None,
    limit: int,
) -> list[list[float]]:
    """Walk the exchange's paginated OHLCV API until caught up.

    Termination: the exchange returns fewer than ``limit`` rows, or the
    latest bar's timestamp reaches ``end_ms`` (if provided), or the hard
    page cap is hit (defensive).
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
        last_ts = int(page[-1][0])
        if len(page) < limit:
            break
        cursor = last_ts + 1
        if end_ms is not None and cursor >= end_ms:
            break
    return out


def _bars_to_df(bars: list[list[float]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(
        "datetime64[us, UTC]"
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


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
) -> None:
    """Validate, atomically publish, or quarantine on failure.

    Writes to a tempfile inside ``final_path``'s parent directory, then
    validates the tempfile's read-back contents against ``schema``. On
    success, atomically renames the tempfile to ``final_path``. On
    failure, the tempfile moves to ``<data_dir>/quarantine/`` with a
    sidecar ``.error.txt`` and a :class:`DataSchemaViolation` is raised.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist to a tempfile first so we never publish a file unless it
    # successfully round-trips through parquet + validation.
    with tempfile.NamedTemporaryFile(
        prefix=f".{final_path.stem}_",
        suffix=".parquet",
        dir=final_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)

    try:
        round_tripped = pd.read_parquet(tmp_path)
        schema.validate(round_tripped, lazy=False)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as err:
        _quarantine(
            tmp_path=tmp_path,
            final_name=final_path.name,
            data_dir=data_dir,
            err=err,
        )
        raise DataSchemaViolation(
            f"{final_path.name} failed {schema.__name__} on download: {err}"
        ) from err
    except Exception:
        # Any unexpected failure: remove tempfile, re-raise. Nothing
        # published, nothing quarantined (no signal to future runs).
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
    """Move a failed tempfile to the quarantine directory with an error sidecar."""
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
        "ohlcv_quarantined",
        parquet=str(q_parquet),
        error_file=str(q_error),
        error_type=type(err).__name__,
    )


# ---------------------------------------------------------------------------
# Exchange factories
# ---------------------------------------------------------------------------


def make_binance_spot() -> _ExchangeProtocol:
    """Return a ccxt Binance spot client with sensible defaults.

    Kept as a factory so tests don't need ccxt on import. Users who want
    a real exchange call this; tests inject a fake exchange directly.
    """
    import ccxt  # noqa: PLC0415  — lazy import so tests don't need the network

    # ccxt logs a lot at default levels; cap it to warnings.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    exchange: Any = ccxt.binance({"enableRateLimit": True})
    return exchange  # type: ignore[no-any-return]


def make_binance_perp() -> _ExchangeProtocol:
    """Return a ccxt Binance USDⓈ-M futures client.

    Used by :func:`download_funding` and :func:`download_open_interest`
    — funding data and OI live on the perp markets, not spot.
    """
    import ccxt  # noqa: PLC0415

    logging.getLogger("ccxt").setLevel(logging.WARNING)
    exchange: Any = ccxt.binanceusdm({"enableRateLimit": True})
    return exchange  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# download_funding
# ---------------------------------------------------------------------------

_FUNDING_INTERVAL_MS = 8 * 3_600_000  # Binance perp funding settles every 8h


def download_funding(
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    data_dir: Path,
    exchange: _ExchangeProtocol,
    limit: int = _DEFAULT_LIMIT,
) -> Path | None:
    """Idempotently fetch the perp funding-rate history for ``symbol``.

    Funding rates settle every 8h on Binance USDⓈ-M. Same invariants as
    :func:`download_ohlcv`: existing file → resume from max(timestamp)
    + 8h; validate before publish; failures go to
    ``<data_dir>/quarantine/``.

    Args:
        symbol: Perpetual symbol string (ccxt form, e.g.
            ``"BTC/USDT:USDT"``). The parquet filename uses only
            ``BASE_QUOTE`` to match the loader's convention.
        start: Earliest UTC-aware timestamp to fetch on a fresh run.
        end: Optional exclusive upper bound (UTC-aware).
        data_dir: Root data directory.
        exchange: Perp-market exchange (use :func:`make_binance_perp`).
        limit: Maximum records per exchange call.

    Returns:
        Path to the canonical parquet, or ``None`` if the exchange
        returned nothing and no file existed previously.

    Raises:
        DataSchemaViolation: Combined frame fails
            :class:`FundingRateSchema`.
    """
    _require_utc(start, "start")
    if end is not None:
        _require_utc(end, "end")

    base, quote_full = symbol.split("/", 1)
    quote = quote_full.split(":", 1)[0]
    final_path = data_dir / "binance" / "funding" / f"{base}_{quote}.parquet"

    existing = _read_existing_or_none(final_path)
    since_ms = _resume_from_existing_or_start(existing, start, _FUNDING_INTERVAL_MS)
    end_ms = _timestamp_to_ms(end) if end is not None else None

    if existing is not None and end_ms is not None and since_ms >= end_ms:
        _logger.info("funding_up_to_date", symbol=symbol, rows=len(existing))
        return final_path

    records = _fetch_funding_paginated(
        exchange=exchange,
        symbol=symbol,
        since_ms=since_ms,
        end_ms=end_ms,
        limit=limit,
    )
    if not records and existing is None:
        _logger.info("funding_empty_response", symbol=symbol)
        return None

    new_df = _funding_records_to_df(records)
    combined = _concat_dedup(existing, new_df)

    _validate_then_publish(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=FundingRateSchema,
    )
    _logger.info(
        "funding_download_complete",
        symbol=symbol,
        new_rows=len(new_df),
        total_rows=len(combined),
        path=str(final_path),
    )
    return final_path


def _fetch_funding_paginated(
    *,
    exchange: _ExchangeProtocol,
    symbol: str,
    since_ms: int,
    end_ms: int | None,
    limit: int,
) -> list[dict[str, Any]]:
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
        if len(page) < limit:
            break
        cursor = int(page[-1]["timestamp"]) + 1
        if end_ms is not None and cursor >= end_ms:
            break
    return out


def _funding_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# download_open_interest
# ---------------------------------------------------------------------------


def download_open_interest(
    symbol: str,
    interval: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    data_dir: Path,
    exchange: _ExchangeProtocol,
    limit: int = _DEFAULT_LIMIT,
) -> Path | None:
    """Idempotently fetch perp open-interest history for ``symbol``.

    Args:
        symbol: Perpetual symbol string (``"BTC/USDT:USDT"``).
        interval: Bar interval for the OI series (``"1h"`` in Phase 2).
        start, end, data_dir, exchange, limit: See :func:`download_funding`.

    Returns:
        Path to the canonical parquet, or ``None`` on empty-response +
        no-existing-file.

    Raises:
        DataSchemaViolation: Combined frame fails
            :class:`OpenInterestSchema`.
    """
    _require_utc(start, "start")
    if end is not None:
        _require_utc(end, "end")

    base, quote_full = symbol.split("/", 1)
    quote = quote_full.split(":", 1)[0]
    final_path = data_dir / "binance" / "open_interest" / f"{base}_{quote}.parquet"

    existing = _read_existing_or_none(final_path)
    interval_ms = exchange.parse_timeframe(interval) * 1000
    since_ms = _resume_from_existing_or_start(existing, start, interval_ms)
    end_ms = _timestamp_to_ms(end) if end is not None else None

    if existing is not None and end_ms is not None and since_ms >= end_ms:
        _logger.info("oi_up_to_date", symbol=symbol, rows=len(existing))
        return final_path

    records = _fetch_oi_paginated(
        exchange=exchange,
        symbol=symbol,
        timeframe=interval,
        since_ms=since_ms,
        end_ms=end_ms,
        limit=limit,
    )
    if not records and existing is None:
        _logger.info("oi_empty_response", symbol=symbol)
        return None

    new_df = _oi_records_to_df(records)
    combined = _concat_dedup(existing, new_df)

    _validate_then_publish(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=OpenInterestSchema,
    )
    _logger.info(
        "oi_download_complete",
        symbol=symbol,
        interval=interval,
        new_rows=len(new_df),
        total_rows=len(combined),
        path=str(final_path),
    )
    return final_path


def _fetch_oi_paginated(
    *,
    exchange: _ExchangeProtocol,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int | None,
    limit: int,
) -> list[dict[str, Any]]:
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
        if len(page) < limit:
            break
        cursor = int(page[-1]["timestamp"]) + 1
        if end_ms is not None and cursor >= end_ms:
            break
    return out


def _oi_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Shared resume helper
# ---------------------------------------------------------------------------


def _resume_from_existing_or_start(
    existing: pd.DataFrame | None,
    start: pd.Timestamp,
    interval_ms: int,
) -> int:
    """Resume strictly after the last existing row, else use ``start``.

    Shared by :func:`download_funding` and :func:`download_open_interest`;
    :func:`download_ohlcv` uses its own :func:`_resume_since_ms` which
    takes the exchange directly to parse the timeframe.
    """
    if existing is None or existing.empty:
        return _timestamp_to_ms(start)
    last_ms = int(existing["timestamp"].max().timestamp() * 1000)
    return last_ms + interval_ms
