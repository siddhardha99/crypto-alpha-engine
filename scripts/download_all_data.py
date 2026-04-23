"""One-command data bootstrap for crypto-alpha-engine.

Runs every downloader in SPEC §5 against its real upstream source and
populates ``<repo>/data/`` (or the directory given by ``--data-dir``).

Invariants:

* **Idempotent.** Re-running is a no-op for unchanged data. Each
  downloader reads the existing parquet and resumes from the last bar.
* **Resilient.** Failures in any single source (API down, missing key,
  schema mismatch) are logged as warnings and the script keeps going.
  A final summary lists which sources succeeded and which didn't.
* **No secrets in logs.** The only credential this script touches is
  ``CRYPTOPANIC_API_KEY``; it's read from the environment but never
  printed.

Usage::

    uv run python scripts/download_all_data.py [--data-dir ./data]

Exit code is 0 even if some free sources failed — that's the Phase 2
resilience guarantee. The exchange downloads (OHLCV, funding, OI) are
considered mandatory; if any of *those* raises, the script exits 1 so
CI catches a real regression.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from crypto_alpha_engine.data.downloader import (
    download_funding,
    download_ohlcv,
    download_open_interest,
    make_binance_perp,
    make_binance_spot,
)
from crypto_alpha_engine.data.free_sources import (
    download_btc_active_addresses,
    download_btc_dominance,
    download_btc_hashrate,
    download_cryptopanic_news,
    download_dxy,
    download_fear_greed,
    download_spy,
    download_stablecoin_mcap,
)
from crypto_alpha_engine.exceptions import CryptoAlphaEngineError

_logger = structlog.get_logger("download_all_data")

# Per SPEC §5 — history windows per dataset. ``None`` for end means "up to now".
_BTC_START = pd.Timestamp("2017-01-01", tz="UTC")
_ETH_START = pd.Timestamp("2017-08-01", tz="UTC")
_FUNDING_START = pd.Timestamp("2019-09-01", tz="UTC")
_OI_START = pd.Timestamp("2020-01-01", tz="UTC")


@dataclass
class _Result:
    name: str
    ok: bool
    detail: str


def _run_mandatory(
    name: str,
    fn: Callable[[], Path | None],
    results: list[_Result],
) -> bool:
    """Run an exchange downloader; a failure aborts the script (exit 1)."""
    try:
        out = fn()
    except CryptoAlphaEngineError as err:
        # Engine-native schema failures — still mandatory, but log cleanly.
        _logger.error("mandatory_source_failed", source=name, error=str(err))
        results.append(_Result(name, ok=False, detail=str(err)))
        return False
    except Exception as err:  # network / upstream infra failure
        _logger.error(
            "mandatory_source_failed",
            source=name,
            error_type=type(err).__name__,
            error=str(err),
        )
        results.append(_Result(name, ok=False, detail=f"{type(err).__name__}: {err}"))
        return False
    results.append(_Result(name, ok=True, detail=str(out) if out else "(no new bars)"))
    return True


def _run_optional(
    name: str,
    fn: Callable[[], Path | None],
    results: list[_Result],
) -> None:
    """Run a free-source downloader; failures are warnings, never aborts."""
    out = fn()
    if out is None:
        results.append(_Result(name, ok=False, detail="source returned None (see logs)"))
    else:
        results.append(_Result(name, ok=True, detail=str(out)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="root directory for downloaded data (default: ./data)",
    )
    args = parser.parse_args(argv)
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    spot = make_binance_spot()
    perp = make_binance_perp()

    results: list[_Result] = []
    all_mandatory_ok = True

    # --- Mandatory: spot OHLCV for BTC/ETH at 1h + 1d ---
    spot_universe = [
        ("BTC/USDT", "1h", _BTC_START),
        ("BTC/USDT", "1d", _BTC_START),
        ("ETH/USDT", "1h", _ETH_START),
        ("ETH/USDT", "1d", _ETH_START),
    ]
    for symbol, interval, start in spot_universe:
        label = f"ohlcv:{symbol}:{interval}"

        def _fetch(sym: str = symbol, iv: str = interval, st: pd.Timestamp = start) -> Path | None:
            return download_ohlcv(sym, iv, start=st, data_dir=data_dir, exchange=spot)

        ok = _run_mandatory(label, _fetch, results)
        all_mandatory_ok &= ok

    # --- Mandatory: perp funding + OI for BTC/ETH ---
    perp_universe = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    for symbol in perp_universe:
        label = f"funding:{symbol}"

        def _fetch_funding(sym: str = symbol) -> Path | None:
            return download_funding(sym, start=_FUNDING_START, data_dir=data_dir, exchange=perp)

        ok = _run_mandatory(label, _fetch_funding, results)
        all_mandatory_ok &= ok

        label = f"oi:{symbol}"

        def _fetch_oi(sym: str = symbol) -> Path | None:
            return download_open_interest(
                sym, "1h", start=_OI_START, data_dir=data_dir, exchange=perp
            )

        ok = _run_mandatory(label, _fetch_oi, results)
        all_mandatory_ok &= ok

    # --- Optional: free sources ---
    _run_optional(
        "fear_greed",
        lambda: download_fear_greed(data_dir=data_dir),
        results,
    )
    _run_optional(
        "btc_active_addresses",
        lambda: download_btc_active_addresses(data_dir=data_dir),
        results,
    )
    _run_optional(
        "btc_hashrate",
        lambda: download_btc_hashrate(data_dir=data_dir),
        results,
    )
    _run_optional(
        "btc_dominance",
        lambda: download_btc_dominance(data_dir=data_dir),
        results,
    )
    _run_optional(
        "stablecoin_mcap",
        lambda: download_stablecoin_mcap(data_dir=data_dir),
        results,
    )
    _run_optional("dxy", lambda: download_dxy(data_dir=data_dir), results)
    _run_optional("spy", lambda: download_spy(data_dir=data_dir), results)
    _run_optional(
        "cryptopanic",
        lambda: download_cryptopanic_news(data_dir=data_dir),
        results,
    )

    # --- Summary ---
    print("\n=== download_all_data summary ===")
    for r in results:
        status = "OK " if r.ok else "SKIP/FAIL"
        print(f"  [{status}] {r.name:<32} {r.detail}")
    n_ok = sum(1 for r in results if r.ok)
    print(f"\n{n_ok}/{len(results)} sources succeeded.\n")

    return 0 if all_mandatory_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
