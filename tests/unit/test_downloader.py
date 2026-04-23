"""Contract tests for the Binance downloader.

Key behaviors exercised (all via an injectable fake ``ccxt.Exchange``, no
network):

* Fresh download: no file exists → fetch, validate, write.
* Idempotence (Principle 8): file exists → only fetch bars strictly after
  the last on-disk timestamp, and always deduplicate on timestamp before
  appending.
* Up-to-date short-circuit: no-op if the file already covers the requested
  range. Exchange must not be called.
* Validation failure → quarantine: bad bars land in
  ``<data_dir>/quarantine/`` with a sidecar ``.error.txt`` and never touch
  the final destination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crypto_alpha_engine.data.downloader import (
    QUARANTINE_DIR_NAME,
    download_funding,
    download_ohlcv,
    download_open_interest,
)
from crypto_alpha_engine.data.loader import (
    OHLCV_FILENAME_TEMPLATE,
    load_funding,
    load_ohlcv,
    load_open_interest,
)
from crypto_alpha_engine.exceptions import DataSchemaViolation

# ---------------------------------------------------------------------------
# Fake exchange — just enough of ccxt.Exchange to drive the downloader.
# ---------------------------------------------------------------------------


class FakeExchange:
    """Stand-in for ``ccxt.binance()`` / ``ccxt.binanceusdm()`` for tests.

    Supports the three ccxt methods the downloader uses: ``fetch_ohlcv``,
    ``fetch_funding_rate_history``, ``fetch_open_interest_history``. Each
    method's served payload is configured via the constructor; every call
    is logged to the ``calls`` list so tests can assert call counts and
    ``since`` arguments.

    Attributes:
        bars: Full list of [ts_ms, open, high, low, close, volume] to
            return from ``fetch_ohlcv``.
        funding_records: List of funding dicts (``{"timestamp", "fundingRate"}``).
        oi_records: List of OI dicts (``{"timestamp", "openInterestAmount"}``).
        calls: Every method call is appended here as a dict.
        fail_mode: ``"corrupt"`` makes ``fetch_ohlcv`` return bars with
            negative volume so validation fails.
    """

    def __init__(
        self,
        bars: list[list[float]] | None = None,
        funding_records: list[dict[str, Any]] | None = None,
        oi_records: list[dict[str, Any]] | None = None,
        fail_mode: str | None = None,
    ) -> None:
        self.bars = list(bars) if bars is not None else []
        self.funding_records = list(funding_records) if funding_records is not None else []
        self.oi_records = list(oi_records) if oi_records is not None else []
        self.calls: list[dict[str, Any]] = []
        self.fail_mode = fail_mode

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "method": "fetch_ohlcv",
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "limit": limit,
            }
        )
        bars = [b for b in self.bars if since is None or b[0] >= since]
        if limit is not None:
            bars = bars[:limit]
        if self.fail_mode == "corrupt":
            bars = [[b[0], b[1], b[2], b[3], b[4], -1.0] for b in bars]  # negative volume
        return bars

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "fetch_funding_rate_history",
                "symbol": symbol,
                "since": since,
                "limit": limit,
            }
        )
        records = [r for r in self.funding_records if since is None or int(r["timestamp"]) >= since]
        if limit is not None:
            records = records[:limit]
        return records

    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "fetch_open_interest_history",
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "limit": limit,
            }
        )
        records = [r for r in self.oi_records if since is None or int(r["timestamp"]) >= since]
        if limit is not None:
            records = records[:limit]
        return records

    def parse_timeframe(self, timeframe: str) -> int:
        """Return the timeframe in seconds, mirroring ccxt's helper."""
        if timeframe == "1h":
            return 3600
        if timeframe == "8h":
            return 8 * 3600
        if timeframe == "1d":
            return 86400
        raise ValueError(f"FakeExchange: unsupported timeframe {timeframe!r}")


def _make_bars(start_ms: int, count: int, step_ms: int) -> list[list[float]]:
    """A sequence of synthetic-but-valid OHLCV rows, ascending in time."""
    return [
        [start_ms + i * step_ms, 100.0 + i, 110.0 + i, 95.0 + i, 105.0 + i, 1000.0 + i]
        for i in range(count)
    ]


# 2023-06-01 00:00:00 UTC in ms
_REF_MS = int(pd.Timestamp("2023-06-01", tz="UTC").timestamp() * 1000)
_HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# Fresh download
# ---------------------------------------------------------------------------


class TestFreshDownload:
    def test_writes_validated_parquet(self, tmp_path: Path) -> None:
        ex = FakeExchange(bars=_make_bars(_REF_MS, 24, _HOUR_MS))
        out = download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-02", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex,
        )
        # File exists at the canonical path and loads cleanly under the schema.
        expected = (
            tmp_path
            / "binance"
            / "ohlcv"
            / OHLCV_FILENAME_TEMPLATE.format(base="BTC", quote="USDT", interval="1h")
        )
        assert out == expected
        df = load_ohlcv("BTC/USDT", "1h", data_dir=tmp_path)
        assert len(df) == 24

    def test_no_bars_returned_creates_no_file(self, tmp_path: Path) -> None:
        ex = FakeExchange(bars=[])
        out = download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-02", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex,
        )
        assert out is None
        assert (
            not (tmp_path / "binance").exists()
            or list((tmp_path / "binance").rglob("*.parquet")) == []
        )


# ---------------------------------------------------------------------------
# Idempotence (Principle 8)
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_resume_fetches_only_new_bars(self, tmp_path: Path) -> None:
        # Seed: file has 10 hourly bars starting at _REF_MS.
        ex_initial = FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 10:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_initial,
        )

        # Extend: exchange now has 20 bars; downloader should fetch from bar
        # 10 onwards (strict ">"), not re-fetch bars 0..9.
        ex_extend = FakeExchange(bars=_make_bars(_REF_MS, 20, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 20:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_extend,
        )

        df = load_ohlcv("BTC/USDT", "1h", data_dir=tmp_path)
        assert len(df) == 20  # all 20 unique bars, no duplicates
        # The resume call must have started after bar 9's timestamp.
        first_extend_call = ex_extend.calls[0]
        assert first_extend_call["since"] is not None
        assert first_extend_call["since"] > _REF_MS + 9 * _HOUR_MS

    def test_already_up_to_date_does_not_call_exchange(self, tmp_path: Path) -> None:
        # Seed with 10 bars covering [REF, REF+9h].
        ex_seed = FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 10:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_seed,
        )

        # Re-run asking for the same window; exchange must not be called.
        ex_again = FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 10:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_again,
        )
        assert ex_again.calls == [], "up-to-date file should short-circuit network fetch"

    def test_dedup_on_overlapping_exchange_response(self, tmp_path: Path) -> None:
        """Even if the exchange returns bars we already have, they're dropped."""
        ex_seed = FakeExchange(bars=_make_bars(_REF_MS, 5, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 05:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_seed,
        )

        # Second exchange returns 10 bars starting at REF — overlapping the
        # first 5 we already have. Downloader must not double-write any bar.
        ex_overlap = FakeExchange(bars=_make_bars(_REF_MS, 10, _HOUR_MS))
        download_ohlcv(
            "BTC/USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 10:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_overlap,
        )

        df = load_ohlcv("BTC/USDT", "1h", data_dir=tmp_path)
        assert len(df) == 10
        assert df["timestamp"].is_unique


# ---------------------------------------------------------------------------
# Validation failure → quarantine
# ---------------------------------------------------------------------------


class TestQuarantine:
    def test_invalid_bars_go_to_quarantine_not_final_path(self, tmp_path: Path) -> None:
        ex = FakeExchange(bars=_make_bars(_REF_MS, 5, _HOUR_MS), fail_mode="corrupt")
        with pytest.raises(DataSchemaViolation):
            download_ohlcv(
                "BTC/USDT",
                "1h",
                start=pd.Timestamp("2023-06-01", tz="UTC"),
                end=pd.Timestamp("2023-06-01 05:00", tz="UTC"),
                data_dir=tmp_path,
                exchange=ex,
            )

        # Final path must NOT exist.
        final = (
            tmp_path
            / "binance"
            / "ohlcv"
            / OHLCV_FILENAME_TEMPLATE.format(base="BTC", quote="USDT", interval="1h")
        )
        assert not final.exists()

        # A quarantined parquet + sidecar error file must exist.
        quarantine = tmp_path / QUARANTINE_DIR_NAME
        assert quarantine.exists()
        parquets = list(quarantine.glob("*.parquet"))
        errors = list(quarantine.glob("*.error.txt"))
        assert len(parquets) == 1
        assert len(errors) == 1
        # Sidecar content references the schema failure.
        sidecar_text = errors[0].read_text().lower()
        assert "volume" in sidecar_text or "schema" in sidecar_text


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def _funding_records(
    start_ms: int, count: int, step_ms: int = 8 * 3_600_000
) -> list[dict[str, Any]]:
    return [
        {"timestamp": start_ms + i * step_ms, "fundingRate": 0.0001 + i * 1e-5}
        for i in range(count)
    ]


class TestDownloadFunding:
    def test_fresh_download_writes_validated_parquet(self, tmp_path: Path) -> None:
        ex = FakeExchange(funding_records=_funding_records(_REF_MS, 6))
        out = download_funding(
            "BTC/USDT:USDT",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-03", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex,
        )
        assert out is not None
        df = load_funding("BTC/USDT", data_dir=tmp_path)
        assert len(df) == 6
        assert df["funding_rate"].tolist() == [0.0001 + i * 1e-5 for i in range(6)]

    def test_resume_skips_existing_rows(self, tmp_path: Path) -> None:
        ex_seed = FakeExchange(funding_records=_funding_records(_REF_MS, 3))
        download_funding(
            "BTC/USDT:USDT",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-02", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_seed,
        )

        ex_extend = FakeExchange(funding_records=_funding_records(_REF_MS, 6))
        download_funding(
            "BTC/USDT:USDT",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-03", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_extend,
        )

        df = load_funding("BTC/USDT", data_dir=tmp_path)
        assert len(df) == 6
        assert df["timestamp"].is_unique
        # First extend call must have resumed strictly after the third record.
        first_call = ex_extend.calls[0]
        assert first_call["since"] > _REF_MS + 2 * (8 * 3_600_000)


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


def _oi_records(start_ms: int, count: int) -> list[dict[str, Any]]:
    return [
        {"timestamp": start_ms + i * _HOUR_MS, "openInterestAmount": 1000.0 + i}
        for i in range(count)
    ]


class TestDownloadOpenInterest:
    def test_fresh_download_writes_validated_parquet(self, tmp_path: Path) -> None:
        ex = FakeExchange(oi_records=_oi_records(_REF_MS, 5))
        out = download_open_interest(
            "BTC/USDT:USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 05:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex,
        )
        assert out is not None
        df = load_open_interest("BTC/USDT", data_dir=tmp_path)
        assert len(df) == 5
        assert df["open_interest"].tolist() == [1000.0 + i for i in range(5)]

    def test_already_up_to_date_skips_exchange(self, tmp_path: Path) -> None:
        ex_seed = FakeExchange(oi_records=_oi_records(_REF_MS, 5))
        download_open_interest(
            "BTC/USDT:USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 05:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_seed,
        )

        ex_again = FakeExchange(oi_records=_oi_records(_REF_MS, 5))
        download_open_interest(
            "BTC/USDT:USDT",
            "1h",
            start=pd.Timestamp("2023-06-01", tz="UTC"),
            end=pd.Timestamp("2023-06-01 05:00", tz="UTC"),
            data_dir=tmp_path,
            exchange=ex_again,
        )
        # parse_timeframe may be called, but no fetch_open_interest_history.
        fetch_calls = [c for c in ex_again.calls if c["method"] == "fetch_open_interest_history"]
        assert fetch_calls == []
