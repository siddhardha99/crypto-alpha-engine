"""One-command data bootstrap.

Iterates every source registered with the engine's data-source registry
and downloads its full history into ``<data_dir>/``. Per SPEC §5, the
pipeline is resilient: a failure in any single source logs a warning
and the script keeps going. The exit summary lists what succeeded and
what didn't.

Usage::

    uv run python scripts/download_all_data.py [--data-dir ./data]

Environment:
    CRYPTOPANIC_API_KEY  (optional) — enables the cryptopanic_news
        source. Unset → that source is silently skipped.

No API keys are required for the default six sources
(Coinbase, BitMEX, Alternative.me, Blockchain.com, DefiLlama,
CoinGecko, yfinance).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import structlog

# Import the sources package so every built-in source auto-registers.
import crypto_alpha_engine.data.sources as _sources_pkg  # noqa: F401
from crypto_alpha_engine.data import registry
from crypto_alpha_engine.data.downloader import run_source
from crypto_alpha_engine.data.protocol import DataType

_logger = structlog.get_logger("download_all_data")

# Which bar frequencies to download per data type. OHLCV is dual-rate
# (1h for research, 1d for long-horizon strategies); everything else
# has a single natural cadence.
_FREQS_PER_DATA_TYPE: dict[DataType, list[str]] = {
    DataType.OHLCV: ["1h", "1d"],
    DataType.FUNDING: ["8h"],
    DataType.OPEN_INTEREST: ["1h"],
    DataType.SENTIMENT: ["1d"],
    DataType.ONCHAIN: ["1d"],
    DataType.MACRO: ["1d"],
}


@dataclass
class _Result:
    source: str
    symbol: str
    freq: str
    ok: bool
    detail: str


def _run_one(
    source_name: str,
    symbol: str,
    freq: str,
    *,
    data_dir: Path,
) -> _Result:
    source = registry.get_source(source_name)
    start = source.earliest_available(symbol)
    try:
        out = run_source(
            source,
            symbol,
            start=start,
            freq=freq,
            data_dir=data_dir,
        )
    except Exception as err:  # noqa: BLE001 — resilience: log + continue
        _logger.error(
            "source_failed",
            source=source_name,
            symbol=symbol,
            freq=freq,
            error_type=type(err).__name__,
            error=str(err),
        )
        return _Result(
            source=source_name,
            symbol=symbol,
            freq=freq,
            ok=False,
            detail=f"{type(err).__name__}: {err}",
        )
    return _Result(
        source=source_name,
        symbol=symbol,
        freq=freq,
        ok=True,
        detail=str(out) if out else "(no rows returned)",
    )


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

    results: list[_Result] = []

    # Iterate in the DataType enum order so the summary is stable.
    for data_type in DataType:
        freqs = _FREQS_PER_DATA_TYPE.get(data_type, ["1d"])
        for source in registry.list_sources_for(data_type):
            for symbol in source.symbols:
                for freq in freqs:
                    results.append(_run_one(source.name, symbol, freq, data_dir=data_dir))

    print("\n=== download_all_data summary ===")
    for r in results:
        status = "OK " if r.ok else "FAIL"
        print(f"  [{status}] {r.source:<24} {r.symbol:<18} {r.freq:<4}  {r.detail}")
    n_ok = sum(1 for r in results if r.ok)
    print(f"\n{n_ok}/{len(results)} downloads succeeded.\n")

    # Exit 0 if at least one succeeded — a total failure (all zero) most
    # likely means no network, which is worth a non-zero code.
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
