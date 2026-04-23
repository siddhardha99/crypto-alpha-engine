"""Resilient downloaders for the free / macro data sources in SPEC §5.

Per the Phase 2 ask: **every source is optional and never blocks the
pipeline**. If an API is down, returns unexpected shapes, or its credentials
are missing, the source function logs a warning and returns ``None`` — it
does NOT raise. The only exception is programmer error (e.g. calling with
a non-UTC timestamp), which raises synchronously.

Data flow for each source:

1. Call the external API via :mod:`requests` (REST endpoints) or
   :mod:`yfinance` (macro series).
2. Coerce the response into a ``(timestamp, value)`` DataFrame with the
   canonical ``datetime64[us, UTC]`` dtype on ``timestamp``.
3. Merge with any existing parquet at the target path (last write wins on
   duplicate timestamps), sort, drop duplicates.
4. Round-trip through :func:`_validate_then_publish` (re-used from the
   :mod:`downloader` module): tempfile → parquet → Pandera → atomic rename.
   Validation failures quarantine the tempfile and return ``None`` +
   warning.

Sources shipped here (one function each):

* :func:`download_fear_greed` — alternative.me Fear & Greed.
* :func:`download_btc_active_addresses` — Blockchain.com.
* :func:`download_btc_hashrate` — Blockchain.com.
* :func:`download_btc_dominance` — CoinGecko global metrics.
* :func:`download_stablecoin_mcap` — DefiLlama charts.
* :func:`download_dxy` — yfinance ``DX-Y.NYB``.
* :func:`download_spy` — yfinance ``SPY``.
* :func:`download_cryptopanic_news` — CryptoPanic REST. Requires
  ``CRYPTOPANIC_API_KEY`` in the environment; returns ``None`` with a
  warning if missing (SPEC §5's flexibility about news coverage).

All sources take a ``data_dir`` argument and write to ``<data_dir>/free/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pandera.pandas as pa
import requests
import structlog

from crypto_alpha_engine.data.downloader import _validate_then_publish
from crypto_alpha_engine.data.schemas import FearGreedSchema, IndexValueSchema
from crypto_alpha_engine.exceptions import DataSchemaViolation

_logger = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT_S = 30.0


class _HttpGetter(Protocol):
    """Minimal subset of ``requests.get`` we rely on.

    Tests inject a fake; production code uses :func:`requests.get`.
    """

    def __call__(
        self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...
    ) -> requests.Response: ...


# ---------------------------------------------------------------------------
# Shared merge-and-publish helper
# ---------------------------------------------------------------------------


def _merge_with_existing(new_df: pd.DataFrame, final_path: Path) -> pd.DataFrame:
    """Merge ``new_df`` with any existing parquet at ``final_path``.

    Deduplicates on ``timestamp`` (last occurrence wins — free sources can
    revise past values so we trust the freshest fetch) and returns a sorted
    frame.
    """
    if not final_path.exists():
        return new_df
    existing = pd.read_parquet(final_path)
    existing["timestamp"] = existing["timestamp"].astype("datetime64[us, UTC]")
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    return combined.sort_values("timestamp", ignore_index=True)


def _publish_or_log(
    *,
    df: pd.DataFrame,
    final_path: Path,
    data_dir: Path,
    schema: type[pa.DataFrameModel],
    source_name: str,
) -> Path | None:
    """Validate + publish; on failure, log a warning and return None.

    Differs from the exchange downloader's behavior (which re-raises) by
    swallowing :class:`DataSchemaViolation` after quarantining. Free
    sources must never block the pipeline.
    """
    try:
        _validate_then_publish(
            df=df,
            final_path=final_path,
            data_dir=data_dir,
            schema=schema,
        )
    except DataSchemaViolation as err:
        _logger.warning(
            "free_source_validation_failed",
            source=source_name,
            error=str(err),
        )
        return None
    _logger.info("free_source_download_complete", source=source_name, rows=len(df))
    return final_path


def _log_failure_and_return_none(source: str, err: Exception) -> None:
    """Shared warning format so every free source logs consistently."""
    _logger.warning(
        "free_source_download_failed",
        source=source,
        error_type=type(err).__name__,
        error=str(err),
    )


# ---------------------------------------------------------------------------
# Fear & Greed (alternative.me)
# ---------------------------------------------------------------------------

_FEAR_GREED_URL = "https://api.alternative.me/fng/"


def download_fear_greed(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
) -> Path | None:
    """Fetch the full Fear & Greed history and merge into ``fear_greed.parquet``.

    Args:
        data_dir: Root data directory; file lands at
            ``<data_dir>/free/fear_greed.parquet``.
        http_get: Injectable HTTP getter for tests. Defaults to
            :func:`requests.get`.

    Returns:
        Path to the parquet on success, ``None`` on any failure.
    """
    final_path = data_dir / "free" / "fear_greed.parquet"
    try:
        resp = http_get(
            _FEAR_GREED_URL,
            params={"limit": 0, "format": "json"},
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
    except Exception as err:
        _log_failure_and_return_none("fear_greed", err)
        return None

    if not rows:
        _logger.info("free_source_empty", source="fear_greed")
        return None

    try:
        # API returns descending timestamps (newest first) as unix-seconds strings.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [int(r["timestamp"]) for r in rows], unit="s", utc=True
                ).astype("datetime64[us, UTC]"),
                "value": [int(r["value"]) for r in rows],
            }
        ).sort_values("timestamp", ignore_index=True)
    except Exception as err:
        _log_failure_and_return_none("fear_greed", err)
        return None

    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=FearGreedSchema,
        source_name="fear_greed",
    )


# ---------------------------------------------------------------------------
# Blockchain.com (active addresses + hashrate)
# ---------------------------------------------------------------------------

_BLOCKCHAIN_BASE = "https://api.blockchain.info/charts"


def _download_blockchain_metric(
    *,
    metric: str,
    filename: str,
    data_dir: Path,
    http_get: _HttpGetter,
) -> Path | None:
    final_path = data_dir / "free" / filename
    try:
        resp = http_get(
            f"{_BLOCKCHAIN_BASE}/{metric}",
            params={"timespan": "all", "format": "json"},
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        values = resp.json().get("values", [])
    except Exception as err:
        _log_failure_and_return_none(metric, err)
        return None

    if not values:
        _logger.info("free_source_empty", source=metric)
        return None

    try:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [int(v["x"]) for v in values], unit="s", utc=True
                ).astype("datetime64[us, UTC]"),
                "value": [float(v["y"]) for v in values],
            }
        ).sort_values("timestamp", ignore_index=True)
    except Exception as err:
        _log_failure_and_return_none(metric, err)
        return None

    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=IndexValueSchema,
        source_name=metric,
    )


def download_btc_active_addresses(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
) -> Path | None:
    """BTC unique active addresses per day (Blockchain.com)."""
    return _download_blockchain_metric(
        metric="n-unique-addresses",
        filename="btc_active_addresses.parquet",
        data_dir=data_dir,
        http_get=http_get,
    )


def download_btc_hashrate(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
) -> Path | None:
    """BTC network hashrate (Blockchain.com)."""
    return _download_blockchain_metric(
        metric="hash-rate",
        filename="btc_hashrate.parquet",
        data_dir=data_dir,
        http_get=http_get,
    )


# ---------------------------------------------------------------------------
# CoinGecko — BTC dominance
# ---------------------------------------------------------------------------


def download_btc_dominance(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
) -> Path | None:
    """BTC dominance percentage series (CoinGecko global snapshot).

    CoinGecko's free public API only exposes the *current* global metrics,
    not a historical series. We sample the current value and append a row
    for today — subsequent runs extend the series over time. Not ideal
    history but matches the practical reality of free CoinGecko access.
    """
    final_path = data_dir / "free" / "btc_dominance.parquet"
    try:
        resp = http_get(
            "https://api.coingecko.com/api/v3/global",
            params=None,
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        pct = float(data["market_cap_percentage"]["btc"])
    except Exception as err:
        _log_failure_and_return_none("btc_dominance", err)
        return None

    now_ts = pd.Timestamp.now(tz="UTC").floor("D")
    ts_col = pd.to_datetime([now_ts], utc=True).astype("datetime64[us, UTC]")
    df = pd.DataFrame({"timestamp": ts_col, "value": [pct]})
    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=IndexValueSchema,
        source_name="btc_dominance",
    )


# ---------------------------------------------------------------------------
# DefiLlama — stablecoin market cap
# ---------------------------------------------------------------------------


def download_stablecoin_mcap(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
) -> Path | None:
    """Total stablecoin market cap (DefiLlama)."""
    final_path = data_dir / "free" / "stablecoin_mcap.parquet"
    try:
        resp = http_get(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            params=None,
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as err:
        _log_failure_and_return_none("stablecoin_mcap", err)
        return None

    if not payload:
        _logger.info("free_source_empty", source="stablecoin_mcap")
        return None

    try:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [int(r["date"]) for r in payload], unit="s", utc=True
                ).astype("datetime64[us, UTC]"),
                "value": [float(r["totalCirculatingUSD"]["peggedUSD"]) for r in payload],
            }
        ).sort_values("timestamp", ignore_index=True)
    except Exception as err:
        _log_failure_and_return_none("stablecoin_mcap", err)
        return None

    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=IndexValueSchema,
        source_name="stablecoin_mcap",
    )


# ---------------------------------------------------------------------------
# yfinance — macro (DXY, SPY)
# ---------------------------------------------------------------------------


def _download_yfinance_close(
    *,
    ticker: str,
    filename: str,
    data_dir: Path,
    yfinance_mod: Any | None = None,
) -> Path | None:
    final_path = data_dir / "free" / filename
    try:
        if yfinance_mod is None:
            import yfinance  # noqa: PLC0415

            yf: Any = yfinance
        else:
            yf = yfinance_mod
        raw = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as err:
        _log_failure_and_return_none(ticker, err)
        return None

    try:
        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned empty frame for {ticker!r}")
        close_col: pd.Series = raw["Close"].squeeze()
        ts = pd.to_datetime(close_col.index, utc=True).astype("datetime64[us, UTC]")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "value": close_col.astype(float).to_numpy(),
            }
        ).sort_values("timestamp", ignore_index=True)
    except Exception as err:
        _log_failure_and_return_none(ticker, err)
        return None

    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=IndexValueSchema,
        source_name=ticker,
    )


def download_dxy(*, data_dir: Path, yfinance_mod: Any | None = None) -> Path | None:
    """DXY (US dollar index) daily close via yfinance."""
    return _download_yfinance_close(
        ticker="DX-Y.NYB",
        filename="dxy.parquet",
        data_dir=data_dir,
        yfinance_mod=yfinance_mod,
    )


def download_spy(*, data_dir: Path, yfinance_mod: Any | None = None) -> Path | None:
    """SPY (S&P 500 ETF) daily close via yfinance."""
    return _download_yfinance_close(
        ticker="SPY",
        filename="spy.parquet",
        data_dir=data_dir,
        yfinance_mod=yfinance_mod,
    )


# ---------------------------------------------------------------------------
# CryptoPanic (optional, requires API key)
# ---------------------------------------------------------------------------


def download_cryptopanic_news(
    *,
    data_dir: Path,
    http_get: _HttpGetter = requests.get,
    api_key_env_var: str = "CRYPTOPANIC_API_KEY",
) -> Path | None:
    """Optional CryptoPanic news feed.

    Returns ``None`` with a warning if ``CRYPTOPANIC_API_KEY`` is not in the
    environment — per the user's Phase 2 guidance that missing credentials
    must never block the pipeline.
    """
    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        _logger.warning(
            "free_source_skipped_no_key",
            source="cryptopanic",
            env_var=api_key_env_var,
        )
        return None

    final_path = data_dir / "free" / "cryptopanic_news.parquet"
    try:
        resp = http_get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": api_key, "public": "true"},
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        posts = resp.json().get("results", [])
    except Exception as err:
        _log_failure_and_return_none("cryptopanic", err)
        return None

    if not posts:
        _logger.info("free_source_empty", source="cryptopanic")
        return None

    try:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([p["published_at"] for p in posts], utc=True).astype(
                    "datetime64[us, UTC]"
                ),
                "value": [1.0 for _ in posts],  # count indicator; richer schema later
            }
        ).sort_values("timestamp", ignore_index=True)
    except Exception as err:
        _log_failure_and_return_none("cryptopanic", err)
        return None

    combined = _merge_with_existing(df, final_path)
    return _publish_or_log(
        df=combined,
        final_path=final_path,
        data_dir=data_dir,
        schema=IndexValueSchema,
        source_name="cryptopanic",
    )
